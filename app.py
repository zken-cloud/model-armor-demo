from google import genai
from google.genai import types
import os
import re
import requests
import google.auth
from flask import Flask, render_template, request, jsonify
from google.cloud import modelarmor_v1
from dotenv import load_dotenv
import base64
from werkzeug.utils import secure_filename
import mimetypes
import json
import traceback
import asyncio
import hashlib
import time
from datetime import timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# --- Jinja2 Configuration for XSS Protection --
app.jinja_env.autoescape = True

project = os.getenv('GCP_PROJECT_ID')

# --- Authentication Token Cache ---
auth_token_cache = {}
TOKEN_TTL = 3300  # fallback only, for credentials that expose no expiry
TOKEN_REFRESH_MARGIN = 300  # refresh this many seconds before actual expiry

def _token_expiry_epoch(credentials, now):
    """Epoch seconds at which these credentials expire."""
    expiry = getattr(credentials, 'expiry', None)
    if not expiry:
        return now + TOKEN_TTL
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return expiry.timestamp()

def get_cached_auth_token(force_refresh=False):
    """Get a valid access token, refreshing shortly before it actually expires.

    The metadata server returns a token with whatever lifetime it has left
    rather than a fresh hour, so measuring a fixed TTL from the moment we
    cached it can keep serving a token that has already expired. Track the
    credential's real expiry instead.
    """
    current_time = time.time()

    cached = auth_token_cache.get('token_data')
    if cached and not force_refresh:
        token, expires_at = cached
        if current_time < expires_at - TOKEN_REFRESH_MARGIN:
            return token

    credentials, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)

    expires_at = _token_expiry_epoch(credentials, current_time)
    auth_token_cache['token_data'] = (credentials.token, expires_at)
    print(f"INFO: Authentication token refreshed, valid for {int(expires_at - current_time)}s")

    return credentials.token

# Add allowed file extensions
DOCUMENT_EXTENSIONS = {
    'pdf', 'docx', 'docm', 'dotx', 'dotm',
    'pptx', 'pptm', 'potx', 'pot',
    'xlsx', 'xlsm', 'xltx', 'xltm'
}
# Model Armor image screening accepts JPEG, PNG and BMP only.
IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_EXTENSIONS = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS

IMAGE_MIME_TYPES = {'image/png', 'image/jpeg', 'image/bmp'}

# Model Armor rejects images over 4MB.
MAX_IMAGE_BYTES = 4 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image_mime(mime_type):
    return mime_type in IMAGE_MIME_TYPES

def get_byte_data_type(mime_type):
    """Convert MIME type to Model Armor byteDataType. Returns None if unsupported."""
    mime_to_type = {
        'application/pdf': 'PDF',
        # Word
        'application/msword': 'WORD_DOCUMENT',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'WORD_DOCUMENT',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.template': 'WORD_DOCUMENT',
        'application/vnd.ms-word.document.macroEnabled.12': 'WORD_DOCUMENT',
        'application/vnd.ms-word.template.macroEnabled.12': 'WORD_DOCUMENT',
        # PowerPoint
        'application/vnd.ms-powerpoint': 'POWERPOINT_DOCUMENT',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'POWERPOINT_DOCUMENT',
        'application/vnd.openxmlformats-officedocument.presentationml.template': 'POWERPOINT_DOCUMENT',
        'application/vnd.ms-powerpoint.presentation.macroEnabled.12': 'POWERPOINT_DOCUMENT',
        # Excel
        'application/vnd.ms-excel': 'EXCEL_DOCUMENT',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'EXCEL_DOCUMENT',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.template': 'EXCEL_DOCUMENT',
        'application/vnd.ms-excel.sheet.macroEnabled.12': 'EXCEL_DOCUMENT',
        'application/vnd.ms-excel.template.macroEnabled.12': 'EXCEL_DOCUMENT',
        # Images and plain text
        'image/png': 'IMAGE',
        'image/jpeg': 'IMAGE',
        'image/bmp': 'IMAGE',
        'text/plain': 'TXT',
        'text/csv': 'CSV',
    }
    return mime_to_type.get(mime_type)

# --- Optimized HTTP Client ---
class OptimizedHTTPClient:
    def __init__(self):
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

# Global HTTP client
http_client = OptimizedHTTPClient()

# --- Caching Layer ---
model_armor_cache = {}
template_cache = {}
CACHE_TTL = 300  # 5 minutes

def get_file_data(file):
    """Read an upload and base64 encode it.

    Deliberately not cached by name: a "<filename>_<size>" key is shared by every
    caller, so one upload could be answered with a different upload's bytes and
    skip screening entirely.
    """
    file.seek(0)
    file_content = file.read()
    file_data_base64 = base64.b64encode(file_content).decode('utf-8')
    mime_type = mimetypes.guess_type(file.filename)[0]
    return file_data_base64, mime_type

def get_cache_key(data, template_name, location):
    """Generate cache key for Model Armor results"""
    if isinstance(data, dict):  # File data
        content_hash = hashlib.sha256(data['base64_data'].encode()).hexdigest()
        return f"file_{content_hash}_{template_name}_{location}"
    else:  # Text data
        content_hash = hashlib.sha256(data.encode()).hexdigest()
        return f"text_{content_hash}_{template_name}_{location}"

def get_cached_result(cache_key):
    """Get cached result if still valid"""
    if cache_key in model_armor_cache:
        result, timestamp = model_armor_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            print(f"Cache hit for {cache_key[:20]}...")
            return result
        else:
            del model_armor_cache[cache_key]
    return None

def cache_result(cache_key, result):
    """Cache the result"""
    model_armor_cache[cache_key] = (result, time.time())
    # Clean old cache entries periodically
    if len(model_armor_cache) > 100:
        current_time = time.time()
        expired_keys = [k for k, (_, timestamp) in model_armor_cache.items() 
                       if current_time - timestamp > CACHE_TTL]
        for k in expired_keys:
            del model_armor_cache[k]

# --- Client Caching ---
model_armor_clients = {}
genai_clients = {}

def get_model_armor_client(location, endpoint):
    if location not in model_armor_clients:
        model_armor_clients[location] = modelarmor_v1.ModelArmorClient(
            transport="rest", client_options={"api_endpoint": endpoint}
        )
    return model_armor_clients[location]

def get_genai_client(location):
    if location not in genai_clients:
        genai_clients[location] = genai.Client(vertexai=True, project=project, location=location)
    return genai_clients[location]

def pre_initialize_clients():
    """Pre-initializes all necessary API clients to prevent cold start issues."""
    print("INFO: Pre-initializing all API clients...")
    
    # Pre-warm Model Armor clients
    for endpoint_info in model_armor_endpoints:
        try:
            get_model_armor_client(endpoint_info['location'], endpoint_info['endpoint'])
            print(f"  - Successfully initialized Model Armor client for {endpoint_info['location']}")
        except Exception as e:
            print(f"  - WARNING: Failed to initialize Model Armor client for {endpoint_info['location']}: {e}")
            
    # Pre-warm Generative AI clients
    unique_locations = {model.get('location', 'us-central1') for model in foundation_models}
    for location in unique_locations:
        try:
            get_genai_client(location)
            print(f"  - Successfully initialized GenAI client for {location}")
        except Exception as e:
            print(f"  - WARNING: Failed to initialize GenAI client for {location}: {e}")
    
    print("INFO: All API clients pre-initialization complete.")


# Only the us and eu multi-regions are offered: image screening is unavailable
# on regional endpoints, which return invocationResult=FAILURE for IMAGE payloads.
#
# Advanced SDP needs the DLP templates to sit in the same location as the Model
# Armor template. Sensitive Data Protection has a "us" location but no "eu" one
# (its EU multi-region is called "europe"), so advanced SDP can be used from us
# but not from eu. sdp_location records the matching DLP location, or None when
# there is none.
model_armor_endpoints = [
    {"location": "us", "endpoint": "modelarmor.us.rep.googleapis.com",
     "display_name": "us (multi-region)", "supports_images": True, "sdp_location": "us"},
    {"location": "eu", "endpoint": "modelarmor.eu.rep.googleapis.com",
     "display_name": "eu (multi-region)", "supports_images": True, "sdp_location": None},
]

generation_config = types.GenerateContentConfig(
    max_output_tokens=2048, temperature=0.2, top_p=0.95, response_modalities=["TEXT"],
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
)

foundation_models = [
    {"name": "gemini-3.8-flash", "provider": "Google", "location": "global", "display_name": "gemini-3.8-flash"},
]

# Demo template defaults: every filter at "High and above", the latest filter
# model for PI/Jailbreak (and RAI), and both modalities where images are supported.
DEFAULT_CONFIDENCE_LEVEL = 'HIGH'
DEFAULT_FILTER_VERSION_ALIAS = 'FILTER_VERSION_ALIAS_LATEST'
DEFAULT_RAI_FILTER_TYPES = ['HATE_SPEECH', 'DANGEROUS', 'HARASSMENT', 'SEXUALLY_EXPLICIT']

def build_default_template_payload(supports_images):
    """Default demo template body. filterVersionSelector is REST-only (absent from the SDK)."""
    modalities = ['MODALITY_TEXT']
    if supports_images:
        modalities.append('MODALITY_IMAGE')

    return {
        'filterConfig': {
            'raiSettings': {
                'raiFilters': [
                    {'filterType': f, 'confidenceLevel': DEFAULT_CONFIDENCE_LEVEL}
                    for f in DEFAULT_RAI_FILTER_TYPES
                ]
            },
            'piAndJailbreakFilterSettings': {
                'filterEnforcement': 'ENABLED',
                'confidenceLevel': DEFAULT_CONFIDENCE_LEVEL,
            },
            'maliciousUriFilterSettings': {'filterEnforcement': 'ENABLED'},
            'sdpSettings': {'basicConfig': {'filterEnforcement': 'ENABLED'}},
        },
        'templateMetadata': {
            'modalities': modalities,
            'filterVersionSelector': {'alias': DEFAULT_FILTER_VERSION_ALIAS},
            'logTemplateOperations': True,
            'logSanitizeOperations': True,
        },
    }

def extract_unsupported_capabilities(resp):
    """Capability names from a CAPABILITY_NOT_SUPPORTED error, e.g. {'Malicious URI filter'}."""
    try:
        for detail in resp.json().get('error', {}).get('details', []):
            if detail.get('reason') == 'CAPABILITY_NOT_SUPPORTED':
                raw = detail.get('metadata', {}).get('invalid_capabilities', '')
                return {c.strip() for c in raw.split(',') if c.strip()}
    except (ValueError, AttributeError):
        pass
    return set()

def reconcile_demo_template(base_url, template_id, existing, supports_images, headers):
    """Bring an existing demo template up to the demo defaults.

    Only touches the fields the demo owns (RAI confidence, PI/Jailbreak, filter
    version, modalities). SDP and malicious-URI settings are left alone so any
    per-region customisation survives a restart.
    """
    defaults = build_default_template_payload(supports_images)
    existing_filters = existing.get('filterConfig', {})
    existing_meta = existing.get('templateMetadata', {})
    filter_config, metadata, update_mask = {}, {}, []

    rai_filters = existing_filters.get('raiSettings', {}).get('raiFilters', [])
    if not rai_filters or any(f.get('confidenceLevel') != DEFAULT_CONFIDENCE_LEVEL for f in rai_filters):
        filter_config['raiSettings'] = defaults['filterConfig']['raiSettings']
        update_mask.append('filterConfig.raiSettings')

    pi_jb = existing_filters.get('piAndJailbreakFilterSettings', {})
    if (pi_jb.get('confidenceLevel') != DEFAULT_CONFIDENCE_LEVEL
            or pi_jb.get('filterEnforcement') != 'ENABLED'):
        filter_config['piAndJailbreakFilterSettings'] = defaults['filterConfig']['piAndJailbreakFilterSettings']
        update_mask.append('filterConfig.piAndJailbreakFilterSettings')

    if existing_meta.get('filterVersionSelector', {}).get('alias') != DEFAULT_FILTER_VERSION_ALIAS:
        metadata['filterVersionSelector'] = defaults['templateMetadata']['filterVersionSelector']
        update_mask.append('templateMetadata.filterVersionSelector')

    wanted_modalities = defaults['templateMetadata']['modalities']
    if list(existing_meta.get('modalities', [])) != wanted_modalities:
        metadata['modalities'] = wanted_modalities
        update_mask.append('templateMetadata.modalities')

    if not update_mask:
        return None

    payload = {}
    if filter_config:
        payload['filterConfig'] = filter_config
    if metadata:
        payload['templateMetadata'] = metadata

    def patch():
        return http_client.session.patch(
            f"{base_url}/{template_id}?updateMask={','.join(update_mask)}",
            headers=headers, json=payload, timeout=(5, 30)
        )

    resp = patch()
    # A PATCH validates the whole template, so capabilities a region has since
    # dropped block the update even when we are not touching them. Drop those
    # and retry once.
    if resp.status_code == 400:
        unsupported = extract_unsupported_capabilities(resp)
        if 'Malicious URI filter' in unsupported:
            # A disabled malicious-URI block still counts as "requested", so the
            # whole filterConfig has to be rewritten without it. SDP is carried
            # over so per-region customisation is not lost.
            replacement = {k: v for k, v in defaults['filterConfig'].items()
                           if k != 'maliciousUriFilterSettings'}
            if existing_filters.get('sdpSettings'):
                replacement['sdpSettings'] = existing_filters['sdpSettings']
            payload['filterConfig'] = replacement
            update_mask = [m for m in update_mask if not m.startswith('filterConfig')]
            update_mask.append('filterConfig')
        if 'Multi-language detection' in unsupported:
            payload.setdefault('templateMetadata', {})['multiLanguageDetection'] = {'enableMultiLanguageDetection': False}
            update_mask.append('templateMetadata.multiLanguageDetection')
        if unsupported:
            print(f"  - {template_id}: region rejects {', '.join(sorted(unsupported))}; dropping and retrying")
            resp = patch()

    return resp, update_mask

def ensure_demo_templates_exist():
    print("INFO: Ensuring demo templates exist...")
    for endpoint_info in model_armor_endpoints:
        location = endpoint_info['location']
        endpoint = endpoint_info['endpoint']
        supports_images = endpoint_info.get('supports_images', False)
        try:
            token = get_cached_auth_token()
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            base_url = f"https://{endpoint}/v1/projects/{project}/locations/{location}/templates"

            list_resp = http_client.session.get(base_url, headers=headers, timeout=(5, 30))
            if list_resp.status_code != 200:
                print(f"  - ERROR: Failed to list templates in {location}: {list_resp.text}")
                continue
            existing = {t.get('name', '').split('/')[-1]: t for t in list_resp.json().get('templates', [])}

            payload = build_default_template_payload(supports_images)
            for template_id in ["modelarmor-demo-prompt", "modelarmor-demo-response"]:
                if template_id in existing:
                    outcome = reconcile_demo_template(
                        base_url, template_id, existing[template_id], supports_images, headers
                    )
                    if outcome is None:
                        print(f"  - {template_id} in {location} already matches defaults")
                    else:
                        resp, update_mask = outcome
                        if resp.status_code == 200:
                            print(f"  - Updated {template_id} in {location}: {', '.join(update_mask)}")
                        else:
                            print(f"  - ERROR: Failed to update {template_id} in {location}: {resp.text}")
                    continue

                print(f"  - Creating {template_id} in {location}...")
                resp = http_client.session.post(
                    f"{base_url}?template_id={template_id}", headers=headers, json=payload, timeout=(5, 30)
                )
                if resp.status_code == 200:
                    print(f"  - Successfully created {template_id} in {location} "
                          f"(confidence={DEFAULT_CONFIDENCE_LEVEL}, modalities={payload['templateMetadata']['modalities']})")
                else:
                    print(f"  - ERROR: Failed to create {template_id} in {location}: {resp.text}")
        except Exception as e:
            print(f"  - ERROR: Failed to ensure templates in {location}: {e}")

# --- Pre-initialize all clients on startup ---
pre_initialize_clients()
ensure_demo_templates_exist()

def _generate_with_sdk(prompt, model_info, system_instruction, file_data=None):
    """Original SDK approach - kept for fallback"""
    client_to_use = get_genai_client(model_info.get('location', 'us-central1'))
    parts = [{'text': prompt}]
    if file_data and model_info.get('provider') == 'Google':
        parts.append({
            'inline_data': {
                'mime_type': file_data['mime_type'],
                'data': file_data['base64_data']
            }
        })
        
    # Create a new config incorporating the system instruction
    config = types.GenerateContentConfig(
        max_output_tokens=generation_config.max_output_tokens,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        response_modalities=generation_config.response_modalities,
        safety_settings=generation_config.safety_settings,
        system_instruction=system_instruction if system_instruction else None
    )
    
    response = client_to_use.models.generate_content(
        model=model_info['name'],
        contents=[{'role': 'user', 'parts': parts}],
        config=config
    )

    if response.text:
        return response.text

    # A thinking model can decline by returning a candidate with no text parts
    # (and sometimes no finish reason at all). Returning None from here breaks
    # response screening and the UI, so always hand back a string.
    finish_reason = None
    if response.candidates:
        finish_reason = getattr(response.candidates[0], 'finish_reason', None)
    print(f"WARNING: {model_info['name']} returned no text (finish_reason={finish_reason}).")
    detail = f" (finish reason: {finish_reason})" if finish_reason else ""
    return f"[The model returned no content{detail}.]"

def generate_model_response(prompt, model_info, system_instruction, file_data=None):
    """Original generation function using SDK"""
    return _generate_with_sdk(prompt, model_info, system_instruction, file_data)

def serialize_template(template):
    """Helper to serialize template in a human-friendly format."""
    # Map filter type enum values to readable names, supporting both int and string keys
    FILTER_TYPE_MAP = {
        0: 'Unspecified', 'RAI_FILTER_TYPE_UNSPECIFIED': 'Unspecified',
        1: 'Prompt Injection',
        2: 'Hate Speech', 'HATE_SPEECH': 'Hate Speech',
        3: 'Dangerous', 'DANGEROUS': 'Dangerous',
        4: 'Jailbreak',
        5: 'Malicious URL',
        6: 'Harassment', 'HARASSMENT': 'Harassment',
        7: 'Multilanguage',
        17: 'Sexually Explicit', 'SEXUALLY_EXPLICIT': 'Sexually Explicit'
    }
    
    # Map confidence level enum values, supporting both int and string keys
    CONFIDENCE_MAP = {
        0: 'UNSPECIFIED', 'DETECTION_CONFIDENCE_LEVEL_UNSPECIFIED': 'UNSPECIFIED',
        1: 'LOW', 'LOW_AND_ABOVE': 'LOW',
        2: 'MEDIUM', 'MEDIUM_AND_ABOVE': 'MEDIUM',
        3: 'HIGH', 'HIGH': 'HIGH'
    }
    
    config = {
        'rai_filters': [],
        'rai_filters_structured': [],
        'detection_filters': [],  # For PI, jailbreak, malicious URL
        'sdp_settings': {},
        'other_settings': {}
    }
    
    try:
        def get_field(obj, name, default=None):
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)
            
        filter_config = get_field(template, 'filterConfig') or get_field(template, 'filter_config')
        
        if filter_config:
            # Extract RAI settings
            rai = get_field(filter_config, 'raiSettings') or get_field(filter_config, 'rai_settings')
            if rai:
                rai_filters = get_field(rai, 'raiFilters') or get_field(rai, 'rai_filters')
                if rai_filters:
                    for rai_filter in rai_filters:
                        filter_type_val = get_field(rai_filter, 'filterType') or get_field(rai_filter, 'filter_type', 0)
                        # An absent confidenceLevel means unspecified; report it as such
                        # so the editor shows the real setting instead of a blank default.
                        confidence_val = (get_field(rai_filter, 'confidenceLevel')
                                          or get_field(rai_filter, 'confidence_level')
                                          or UNSPECIFIED_CONFIDENCE)
                        
                        # Map to readable names
                        display_type = FILTER_TYPE_MAP.get(filter_type_val, f'Unknown Filter ({filter_type_val})')
                        threshold = CONFIDENCE_MAP.get(confidence_val, 'UNKNOWN')
                        
                        config['rai_filters'].append(f"{display_type}: {threshold}")
                        
                        # For structured data, try to keep it as int if possible, but strings are ok too
                        config['rai_filters_structured'].append({
                            'filter_type': filter_type_val,
                            'confidence_level': confidence_val
                        })
                        
            # Extract PI and Jailbreak filter settings
            pi_jb = get_field(filter_config, 'piAndJailbreakFilterSettings') or get_field(filter_config, 'pi_and_jailbreak_filter_settings')
            if pi_jb:
                enforcement = get_field(pi_jb, 'filterEnforcement') or get_field(pi_jb, 'filter_enforcement', 0)
                confidence_val = (get_field(pi_jb, 'confidenceLevel')
                                  or get_field(pi_jb, 'confidence_level')
                                  or UNSPECIFIED_CONFIDENCE)
                
                # Store raw confidence value for UI
                config['other_settings']['pi_jb_confidence'] = confidence_val
                
                if enforcement == 1 or enforcement == 'ENABLED':
                    threshold = CONFIDENCE_MAP.get(confidence_val, 'UNKNOWN')
                    config['detection_filters'].append(f"Prompt Injection & Jailbreak: {threshold}")
            
            # Extract Malicious URL filter settings
            mal_url = get_field(filter_config, 'maliciousUriFilterSettings') or get_field(filter_config, 'malicious_uri_filter_settings')
            if mal_url:
                enforcement = get_field(mal_url, 'filterEnforcement') or get_field(mal_url, 'filter_enforcement', 0)
                if enforcement == 1 or enforcement == 'ENABLED':
                    config['detection_filters'].append("Malicious URL: Enabled")
            
            # Extract SDP settings
            sdp = get_field(filter_config, 'sdpSettings') or get_field(filter_config, 'sdp_settings')
            if sdp:
                bc = get_field(sdp, 'basicConfig') or get_field(sdp, 'basic_config')
                if bc:
                    enforcement = get_field(bc, 'filterEnforcement') or get_field(bc, 'filter_enforcement', 0)
                    if enforcement == 1 or enforcement == 'ENABLED':
                        config['sdp_settings']['mode'] = 'Basic'
                        
                adv = get_field(sdp, 'advancedConfig') or get_field(sdp, 'advanced_config')
                if adv:
                    inspect_template = get_field(adv, 'inspectTemplate') or get_field(adv, 'inspect_template', '')
                    deidentify_template = get_field(adv, 'deidentifyTemplate') or get_field(adv, 'deidentify_template', '')
                    if inspect_template or deidentify_template:
                        config['sdp_settings']['mode'] = 'Advanced (DLP)'
                        config['sdp_settings']['inspect_template'] = inspect_template.split('/')[-1] if inspect_template else 'None'
                        config['sdp_settings']['deidentify_template'] = deidentify_template.split('/')[-1] if deidentify_template else 'None'
        
        # Add metadata if available
        template_metadata = get_field(template, 'templateMetadata') or get_field(template, 'template_metadata')
        if template_metadata:
            log_template_operations = get_field(template_metadata, 'logTemplateOperations') or get_field(template_metadata, 'log_template_operations')
            if log_template_operations is not None:
                config['other_settings']['logging_enabled'] = bool(log_template_operations)

            modalities = get_field(template_metadata, 'modalities')
            if modalities:
                config['other_settings']['modalities'] = list(modalities)

            version_selector = (get_field(template_metadata, 'filterVersionSelector')
                                or get_field(template_metadata, 'filter_version_selector'))
            if version_selector:
                config['other_settings']['filter_version'] = (
                    get_field(version_selector, 'version')
                    or get_field(version_selector, 'alias')
                    or ''
                )

        return config
    except Exception as e:
        print(f"Error serializing template: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def fetch_model_armor_templates(location, endpoint):
    # Check cache first
    cache_key = f"templates_{location}"
    if cache_key in template_cache:
        cached_data, timestamp = template_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return cached_data
    
    local_prompt_templates, local_response_templates = [], []
    try:
        import google.auth
        from google.auth.transport.requests import Request
        import requests
        
        credentials, _ = google.auth.default()
        credentials.refresh(Request())
        token = credentials.token
        
        url = f"https://{endpoint}/v1/projects/{project}/locations/{location}/templates"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        print(f"Fetching templates from {url}...", flush=True)
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"Failed to list templates from {location}: {resp.text}")
            return [], []
            
        data = resp.json()
        templates = data.get('templates', [])
        
        for template in templates:
            template_name = template.get('name', '').split('/')[-1]
            if not template_name.startswith('modelarmor-demo-'):
                continue
            config_dict = serialize_template(template)
            print(f"DEBUG template {template_name} config: {config_dict['sdp_settings']}", flush=True)
            
            # Parse updateTime from string to datetime object if needed, or just use string
            # The original code used template.update_time.strftime('%Y-%m-%d %H:%M:%S UTC')
            # REST API returns string like "2026-04-17T03:02:51.300961843Z"
            # Let's just use the string or parse it.
            update_time_str = template.get('updateTime', '')
            
            template_info = {
                'name': template_name, 
                'display_name': template_name, 
                'location': location, 
                'last_updated': update_time_str, # Just use the string from API
                'config': config_dict
            }
            if template_name.endswith('-prompt'):
                local_prompt_templates.append(template_info)
            elif template_name.endswith('-response'):
                local_response_templates.append(template_info)
        
        # Cache the result
        result = (local_prompt_templates, local_response_templates)
        template_cache[cache_key] = (result, time.time())
        return result
    except Exception as e:
        print(f"Error fetching templates from {location}: {e}")
        return [], []

FILTER_LABELS = {
    'rai': 'Responsible AI',
    'sdp': 'Sensitive Data',
    'pi_and_jailbreak': 'Jailbreak / PI',
    'malicious_uris': 'Malicious URLs',
    'csam': 'CSAM',
}

UNSPECIFIED_CONFIDENCE = 'DETECTION_CONFIDENCE_LEVEL_UNSPECIFIED'

def sanitize_result_to_dict(response):
    """Normalize an SDK sanitize response into the JSON shape the REST API returns."""
    return type(response).to_dict(
        response, preserving_proto_field_name=False, use_integers_for_enums=False
    )

def get_filter_verdict(filter_data, key):
    """The dict holding matchState/executionState for one filter result."""
    inner = next((v for v in filter_data.values() if isinstance(v, dict)), {})
    if key == 'sdp':
        return inner.get('inspectResult') or inner.get('deidentifyResult') or inner
    return inner

def describe_filter_match(key, verdict):
    """Human-readable detail for a matched filter, or '' when there is nothing to add."""
    if key == 'rai':
        matched = []
        for category, result in (verdict.get('raiFilterTypeResults') or {}).items():
            if result.get('matchState') != 'MATCH_FOUND':
                continue
            confidence = result.get('confidenceLevel')
            label = category.replace('_', ' ')
            matched.append(f"{label}: {confidence}" if confidence and confidence != UNSPECIFIED_CONFIDENCE else label)
        return ', '.join(matched)
    if key == 'sdp':
        info_types = []
        for finding in verdict.get('findings') or []:
            info_type = finding.get('infoType')
            info_types.append(info_type.get('name') if isinstance(info_type, dict) else info_type)
        info_types = [i for i in info_types if i]
        return 'Found: ' + ', '.join(sorted(set(info_types))) if info_types else ''
    if key == 'pi_and_jailbreak':
        confidence = verdict.get('confidenceLevel')
        return f"Confidence: {confidence}" if confidence and confidence != UNSPECIFIED_CONFIDENCE else ''
    if key == 'malicious_uris':
        matched = verdict.get('maliciousUriMatchedItems') or verdict.get('matchedUris') or []
        uris = [u.get('uri') for u in matched if u.get('uri')]
        return 'Matched URLs: ' + ', '.join(uris) if uris else ''
    return ''

def summarize_filter_results(response_data, scope=None):
    """Normalize one sanitization result into UI-ready cards.

    A skipped filter gets its own state rather than a pass, so the UI cannot
    show a green tick for a check that never ran.
    """
    results = response_data.get('sanitizationResult', {}).get('filterResults', {})
    cards = []
    for key, label in FILTER_LABELS.items():
        filter_data = results.get(key)
        if not filter_data:
            continue
        verdict = get_filter_verdict(filter_data, key)
        if verdict.get('executionState') == 'EXECUTION_SKIPPED':
            status = 'skipped'
        elif verdict.get('matchState') == 'MATCH_FOUND':
            status = 'fail'
        else:
            status = 'pass'
        cards.append({
            'key': key,
            'label': label,
            'status': status,
            'details': describe_filter_match(key, verdict) if status == 'fail' else '',
            'scope': scope,
        })
    return cards

def process_rest_api_results(response_data):
    """Parses the JSON response from the Model Armor REST API to find all filters with a MATCH_FOUND state."""
    filter_results = []
    try:
        results = response_data.get('sanitizationResult', {}).get('filterResults', {})
        for filter_name, filter_data in results.items():
            json_str = json.dumps(filter_data, separators=(',', ':'))
            if '"matchState":"MATCH_FOUND"' in json_str:
                filter_results.append(filter_name)
    except Exception as e:
        print(f"Error processing REST API results: {e}")
    return filter_results

def get_invocation_result(response_data):
    """SUCCESS / PARTIAL / FAILURE. FAILURE means nothing was actually screened."""
    return response_data.get('sanitizationResult', {}).get('invocationResult', '')

def get_skipped_filters(response_data):
    """Filters that did not execute. A skipped filter is not a pass."""
    skipped = []
    results = response_data.get('sanitizationResult', {}).get('filterResults', {})
    for filter_name, filter_data in results.items():
        for inner in filter_data.values():
            if isinstance(inner, dict) and inner.get('executionState') == 'EXECUTION_SKIPPED':
                skipped.append(filter_name)
                break
    return skipped

def get_filter_version(response_data):
    """The filter model version Model Armor actually used, e.g. 'v3 (FILTER_VERSION_ALIAS_LATEST)'."""
    cfg = (response_data.get('sanitizationResult', {})
           .get('sanitizationMetadata', {}).get('filterVersionConfig', {}))
    version, alias = cfg.get('filterVersion'), cfg.get('filterVersionAlias')
    if version and alias:
        return f"{version} ({alias})"
    return version or alias or None

def get_extracted_image_text(response_data):
    """OCR text Model Armor read out of a screened image."""
    try:
        return (response_data['sanitizationResult']['filterResults']['sdp']
                ['sdpFilterResult']['inspectResult'].get('extractedImageText'))
    except (KeyError, TypeError):
        return None

def check_sdp_transformation_for_file(response_data):
    """Extracts the transformed (redacted) text from a Model Armor file sanitization JSON result."""
    try:
        deidentify_result = response_data.get('sanitizationResult', {}).get('filterResults', {}).get('sdp', {}).get('sdpFilterResult', {}).get('deidentifyResult', {})
        if deidentify_result.get('matchState') == 'MATCH_FOUND':
            return deidentify_result.get('data', {}).get('text')
    except Exception as e:
        print(f"Error extracting SDP transformation from file JSON: {e}")
    return None

def analyze_response_with_template(response_text, template_name, location, modelarmor_client, use_default_response):
    template_display_name = template_name
    try:
        model_response_data = modelarmor_v1.DataItem()
        model_response_data.text = response_text
        response_sanitize_request = modelarmor_v1.SanitizeModelResponseRequest(name=get_template_path(template_name, location), model_response_data=model_response_data)
        response_check = modelarmor_client.sanitize_model_response(request=response_sanitize_request)
        result = sanitize_result_to_dict(response_check)
        output_str = json.dumps(result, indent=2)
        filter_results = process_rest_api_results(result)

        sdp_text = check_sdp_transformation_for_file(result)
        has_sdp = 'sdp' in filter_results

        if sdp_text and has_sdp and not use_default_response:
            response_text = sdp_text

        details = "❌ Violations found:\n" + "\n".join(f"• {result}" for result in filter_results) if filter_results else "✅ No template violations found"
        return {'response_text': response_text, 'analysis': {'template': template_display_name, 'status': 'fail' if filter_results else 'pass', 'details': details, 'matches': bool(filter_results), 'filter_results': filter_results, 'raw_output': output_str, 'filter_details': summarize_filter_results(result), 'filter_version': get_filter_version(result)}, 'has_violations': bool(filter_results), 'has_sdp': has_sdp}
    except Exception as e:
        print(f"Error in response analysis: {e}")
        return {'response_text': response_text, 'analysis': {'template': template_display_name, 'status': 'error', 'details': f'Error in Model Armor analysis: {e}', 'matches': False, 'raw_output': str(e)}, 'has_violations': False}

TEMPLATE_NAME_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,63}$')

def validate_template_name(template_name):
    """Reject anything that could escape the templates/ path segment."""
    if not template_name or not TEMPLATE_NAME_PATTERN.match(template_name):
        raise ValueError(f"Invalid template name: {template_name!r}")
    return template_name

RAI_FILTER_TYPES_BY_INT = {2: 'HATE_SPEECH', 3: 'DANGEROUS', 6: 'HARASSMENT', 17: 'SEXUALLY_EXPLICIT'}
CONFIDENCE_LEVELS_BY_INT = {
    0: UNSPECIFIED_CONFIDENCE, 1: 'LOW_AND_ABOVE', 2: 'MEDIUM_AND_ABOVE', 3: 'HIGH',
}
VALID_CONFIDENCE_LEVELS = set(CONFIDENCE_LEVELS_BY_INT.values())

def normalize_confidence(value):
    """Accept an enum name, or the legacy 0-3 index, and return the enum name."""
    if isinstance(value, str) and value in VALID_CONFIDENCE_LEVELS:
        return value
    try:
        return CONFIDENCE_LEVELS_BY_INT[int(value)]
    except (TypeError, ValueError, KeyError):
        return UNSPECIFIED_CONFIDENCE

def normalize_rai_filter_type(value):
    """Accept an enum name, or the legacy int, and return the enum name."""
    if isinstance(value, str) and value in RAI_FILTER_TYPES_BY_INT.values():
        return value
    try:
        return RAI_FILTER_TYPES_BY_INT.get(int(value))
    except (TypeError, ValueError):
        return None

def get_template_path(template_name, location):
    return f"projects/{project}/locations/{location}/templates/{validate_template_name(template_name)}"

def create_text_data_item(text):
    data_item = modelarmor_v1.DataItem()
    data_item.text = text
    return data_item

def sanitize_file_prompt_with_rest_api_optimized(file_data_base64, mime_type, template_name, location, endpoint):
    """Optimized version with caching and connection reuse"""
    # Check cache first
    cache_key = get_cache_key({'base64_data': file_data_base64, 'mime_type': mime_type}, template_name, location)
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result
    
    byte_data_type = get_byte_data_type(mime_type)
    if not byte_data_type:
        raise ValueError(f"Model Armor cannot screen '{mime_type}' content.")

    if byte_data_type == 'IMAGE':
        approx_bytes = (len(file_data_base64) * 3) // 4
        if approx_bytes > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image is ~{approx_bytes // (1024 * 1024)}MB. Model Armor screens images up to 4MB."
            )

    try:
        access_token = get_cached_auth_token()
        
        url = (f"https://{endpoint}/v1/projects/{project}/locations/{location}"
               f"/templates/{validate_template_name(template_name)}:sanitizeUserPrompt")
        
        payload = {
            "userPromptData": {
                "byteItem": {
                    "byteDataType": byte_data_type,
                    "byteData": file_data_base64
                }
            }
        }
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        response = http_client.session.post(url, headers=headers, json=payload, timeout=(5, 30))
        if response.status_code == 401:
            # The cached token was rejected. Force a refresh and try once more.
            print("WARNING: Model Armor returned 401; refreshing token and retrying.")
            headers["Authorization"] = f"Bearer {get_cached_auth_token(force_refresh=True)}"
            response = http_client.session.post(url, headers=headers, json=payload, timeout=(5, 30))
        response.raise_for_status()
        result = response.json()
        
        # Cache the result
        cache_result(cache_key, result)
        return result
        
    except Exception as e:
        print(f"Error in optimized file sanitization: {e}")
        raise e

def sanitize_text_prompt_optimized(text, template_name, location, endpoint_info):
    """Optimized text prompt sanitization with caching"""
    # Check cache first
    cache_key = get_cache_key(text, template_name, location)
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result
    
    try:
        modelarmor_client = get_model_armor_client(location, endpoint_info['endpoint'])
        prompt_data_item = create_text_data_item(text)
        prompt_sanitize_request = modelarmor_v1.SanitizeUserPromptRequest(
            name=get_template_path(template_name, location),
            user_prompt_data=prompt_data_item
        )
        prompt_check = modelarmor_client.sanitize_user_prompt(request=prompt_sanitize_request)
        result = sanitize_result_to_dict(prompt_check)
        
        # Cache the result
        cache_result(cache_key, result)
        return result
        
    except Exception as e:
        print(f"Error in optimized text sanitization: {e}")
        raise e

# --- Async Processing Functions ---
def describe_scan_problems(response_data, subject):
    """Return a message if Model Armor did not actually screen `subject`, else None.

    Both cases below come back as HTTP 200 with no matches, so treating them as a
    pass would let unscreened content through.
    """
    problems = []
    invocation = get_invocation_result(response_data)
    if invocation and invocation != 'SUCCESS':
        problems.append(f"Model Armor did not screen the {subject} (invocationResult={invocation}).")
    skipped = get_skipped_filters(response_data)
    if skipped:
        problems.append(f"Filters skipped on the {subject}: {', '.join(sorted(skipped))}.")
    if problems and subject == 'image':
        # A de-identify template disables image screening: every image filter is
        # skipped. An inspect-only advanced config screens images normally.
        problems.append("A De-identify template on the prompt template blocks image "
                        "screening. Clear that field to screen images; an Inspect "
                        "template on its own is fine, and De-identify still works for "
                        "text prompts and for the response template.")
    return " ".join(problems) if problems else None

def analyze_image_prompt(prompt, file_data, template_name, location, endpoint_info):
    """Screen an image, and its accompanying text separately.

    userPromptData is a oneof, so `text` and `byteItem` cannot go in one request;
    an image prompt with a caption therefore needs two sanitize calls.
    """
    if not endpoint_info.get('supports_images'):
        message = (f"Image screening is only supported in the us and eu multi-regions; "
                   f"'{location}' cannot screen images.")
        return {
            'output_str': message,
            'filter_results': [],
            'filter_details': [],
            'sdp_transformed_text': None,
            'is_file': True,
            'is_image': True,
            'extracted_image_text': None,
            'filter_version': None,
            'scan_error': message,
        }

    image_result = sanitize_file_prompt_with_rest_api_optimized(
        file_data['base64_data'], file_data['mime_type'],
        template_name, location, endpoint_info['endpoint']
    )
    scan_error = describe_scan_problems(image_result, 'image')
    filter_results = [f"{name} (image)" for name in process_rest_api_results(image_result)]
    filter_details = summarize_filter_results(image_result, scope='image')
    sections = ["=== IMAGE SCAN ===", json.dumps(image_result, indent=2)]

    text = (prompt or '').strip()
    if text:
        text_result = sanitize_text_prompt_optimized(text, template_name, location, endpoint_info)
        filter_results += [f"{name} (text)" for name in process_rest_api_results(text_result)]
        filter_details += summarize_filter_results(text_result, scope='text')
        sections += ["", "=== TEXT SCAN ===", json.dumps(text_result, indent=2)]

    return {
        'output_str': "\n".join(sections),
        'filter_results': filter_results,
        'filter_details': filter_details,
        # Basic SDP reports findings on images; it does not hand back a redacted image.
        'sdp_transformed_text': None,
        'is_file': True,
        'is_image': True,
        'extracted_image_text': get_extracted_image_text(image_result),
        'filter_version': get_filter_version(image_result),
        'scan_error': scan_error,
    }

async def analyze_prompt_async(prompt, file_data, prompt_template, location, endpoint_info):
    """Async wrapper for prompt analysis"""
    loop = asyncio.get_event_loop()
    
    def run_analysis():
        if file_data and file_data.get('is_image'):
            return analyze_image_prompt(prompt, file_data, prompt_template, location, endpoint_info)
        elif file_data:
            result = sanitize_file_prompt_with_rest_api_optimized(
                file_data['base64_data'], file_data['mime_type'], 
                prompt_template, location, endpoint_info['endpoint']
            )
            output_str = json.dumps(result, indent=2)
            filter_results = process_rest_api_results(result)
            sdp_transformed_text = check_sdp_transformation_for_file(result)
            return {
                'output_str': output_str,
                'filter_results': filter_results,
                'filter_details': summarize_filter_results(result),
                'sdp_transformed_text': sdp_transformed_text,
                'is_file': True,
                'is_image': False,
                'extracted_image_text': None,
                'filter_version': get_filter_version(result),
                'scan_error': describe_scan_problems(result, 'file'),
            }
        else:
            result = sanitize_text_prompt_optimized(prompt, prompt_template, location, endpoint_info)
            filter_results = process_rest_api_results(result)
            sdp_transformed_text = check_sdp_transformation_for_file(result)
            return {
                'output_str': json.dumps(result, indent=2),
                'filter_results': filter_results,
                'filter_details': summarize_filter_results(result),
                'filter_version': get_filter_version(result),
                'sdp_transformed_text': sdp_transformed_text,
                'is_file': False,
                'is_image': False,
                'extracted_image_text': None,
                'scan_error': None,
            }
    
    return await loop.run_in_executor(None, run_analysis)

async def generate_response_async(prompt, model_info, system_instruction, file_data):
    """Async wrapper for model generation"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_model_response, prompt, model_info, system_instruction, file_data)

async def analyze_response_async(response_text, response_template, location, endpoint_info, use_default_response):
    """Async wrapper for response analysis"""
    loop = asyncio.get_event_loop()
    
    def run_analysis():
        modelarmor_client = get_model_armor_client(location, endpoint_info['endpoint'])
        return analyze_response_with_template(response_text, response_template, location, modelarmor_client, use_default_response)
    
    return await loop.run_in_executor(None, run_analysis)

# START: CORRECTED SEQUENTIAL LOGIC WITH FILE HANDLING
async def process_chat_async(prompt, model_info, system_instruction, file_data, 
                           prompt_template, response_template, location, endpoint_info, 
                           use_default_response, default_response, prompt_text):
    """
    Process Model Armor and model generation sequentially, with special handling
    for redacted file content.
    """
    prompt_analysis = None
    prompt_has_violations = False
    
    # These will be the final inputs for the LLM call
    prompt_for_llm = prompt 
    file_data_for_llm = file_data

    # --- Step 1: Analyze the prompt/file FIRST ---
    if prompt_template:
        print("INFO: Analyzing prompt/file with Model Armor...")
        prompt_analysis_result = await analyze_prompt_async(prompt, file_data, prompt_template, location, endpoint_info)
        
        filter_results = prompt_analysis_result['filter_results']
        output_str = prompt_analysis_result['output_str']
        sdp_transformed_text = prompt_analysis_result.get('sdp_transformed_text')
        is_file_analysis = prompt_analysis_result.get('is_file', False)

        if filter_results:
            prompt_has_violations = True

        # *** START THE FIX for File vs. Text Redaction ***
        if is_file_analysis and sdp_transformed_text:
            # If a FILE was analyzed and redacted, we must construct a new text prompt
            # that combines the user's original question with the redacted file content.
            prompt_for_llm = f"{prompt}\n\n--- Redacted File Content ---\n{sdp_transformed_text}"
            
            # CRITICAL: We must now remove the original file data so it's not sent to the LLM.
            # The LLM will work with the redacted text we just added to the prompt.
            file_data_for_llm = None
            print("INFO: Constructed new prompt from redacted file content. Original file data will not be sent to LLM.")

        elif not is_file_analysis and sdp_transformed_text:
            # This is for the simple case where the user's TEXT prompt was redacted.
            prompt_for_llm = sdp_transformed_text
            print(f"INFO: Using redacted text prompt for LLM: '{sdp_transformed_text}'")
        # *** END THE FIX ***

        details = "❌ Violations found:\n" + "\n".join(f"• {result}" for result in filter_results) if filter_results else "✅ No template violations found"
        prompt_analysis = {
            'template': prompt_template, 'status': 'fail' if filter_results else 'pass',
            'details': details, 'matches': bool(filter_results),
            'filter_results': filter_results, 'raw_output': output_str,
            'filter_details': prompt_analysis_result.get('filter_details') or [],
            'extracted_image_text': prompt_analysis_result.get('extracted_image_text'),
            'filter_version': prompt_analysis_result.get('filter_version')
        }

        # Fail closed: content Model Armor did not actually screen must not reach the model.
        scan_error = prompt_analysis_result.get('scan_error')
        if scan_error:
            print(f"WARNING: {scan_error} Not sending the prompt to the model.")
            prompt_analysis['status'] = 'error'
            prompt_analysis['details'] = f"⚠️ {scan_error}"
            return {
                'response': f"Request blocked — {scan_error}",
                'prompt_analysis': prompt_analysis,
                'response_analysis': None
            }

    # If prompt had violations and we should use a default, we can stop here.
    if prompt_has_violations and use_default_response:
        print("INFO: Prompt violation found, using default response.")
        return {
            'response': default_response,
            'prompt_analysis': prompt_analysis,
            'response_analysis': None
        }

    # --- Step 2: Generate the model response using the corrected inputs ---
    print(f"INFO: Generating content with model '{model_info['name']}'.")
    model_response = await generate_response_async(prompt_for_llm, model_info, system_instruction, file_data_for_llm)
    
    # --- Step 3: Analyze the response (as before) ---
    response_analysis = None
    if response_template:
        print("INFO: Analyzing response with Model Armor...")
        response_result = await analyze_response_async(model_response, response_template, location, endpoint_info, use_default_response)
        response_analysis = response_result['analysis']
        if response_result['has_violations']:
            if use_default_response:
                print("INFO: Response violation found, using default response.")
                model_response = default_response
            else:
                # Use the redacted response if available
                model_response = response_result.get('response_text', model_response)
                print("INFO: Response violation found, using redacted response.")

    return {
        'response': model_response,
        'prompt_analysis': prompt_analysis,
        'response_analysis': response_analysis
    }
# END: CORRECTED SEQUENTIAL LOGIC WITH FILE HANDLING

@app.route('/')
def home():
    initial_location = model_armor_endpoints[0]['location']
    initial_endpoint = model_armor_endpoints[0]['endpoint']
    prompt_templates, response_templates = fetch_model_armor_templates(initial_location, initial_endpoint)
    return render_template('index.html', foundation_models=foundation_models, model_armor_endpoints=model_armor_endpoints, prompt_templates=prompt_templates, response_templates=response_templates)

DLP_ENDPOINT = 'dlp.googleapis.com'

def _dlp_info_types_from_inspect(template):
    """infoTypes an inspect template looks for. Empty list means DLP defaults."""
    info_types = (template.get('inspectConfig', {}) or {}).get('infoTypes') or []
    return [i.get('name') for i in info_types if i.get('name')]

def _dlp_info_types_from_deidentify(template):
    """infoTypes a deidentify template transforms.

    A transformation with no infoTypes applies to every finding, so report that
    rather than an empty list that would read as "nothing".
    """
    config = template.get('deidentifyConfig', {}) or {}
    transformations = (config.get('infoTypeTransformations', {}) or {}).get('transformations') or []
    names, applies_to_all = [], False
    for transformation in transformations:
        info_types = transformation.get('infoTypes') or []
        if not info_types:
            applies_to_all = True
        names.extend(i.get('name') for i in info_types if i.get('name'))
    if applies_to_all and not names:
        return ['(all findings from the inspect template)']
    return names

def fetch_dlp_templates(location):
    """List the SDP inspect and deidentify templates usable from a Model Armor location.

    Model Armor requires the DLP templates it references to live in the same
    location as the Model Armor template, so only that location is listed. When
    Sensitive Data Protection has no matching location, advanced SDP cannot be
    used at all and a note explains why.
    """
    cache_key = f"dlp_{location}"
    if cache_key in template_cache:
        cached_data, timestamp = template_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return cached_data

    endpoint_info = next((e for e in model_armor_endpoints if e['location'] == location), None)
    sdp_location = (endpoint_info or {}).get('sdp_location')
    if not sdp_location:
        return {
            'inspect_templates': [],
            'deidentify_templates': [],
            'note': (f"Sensitive Data Protection has no '{location}' location, and Model Armor "
                     f"requires the DLP templates to be in the same location as the template. "
                     f"Use Basic SDP here; Advanced SDP is only available from us."),
        }

    headers = {"Authorization": f"Bearer {get_cached_auth_token()}"}
    base = f"https://{DLP_ENDPOINT}/v2/projects/{project}/locations/{sdp_location}"
    result = {'inspect_templates': [], 'deidentify_templates': [], 'note': ''}

    for kind, list_field, extractor in (
        ('inspectTemplates', 'inspect_templates', _dlp_info_types_from_inspect),
        ('deidentifyTemplates', 'deidentify_templates', _dlp_info_types_from_deidentify),
    ):
        try:
            resp = http_client.session.get(f"{base}/{kind}", headers=headers, timeout=(5, 30))
            if resp.status_code != 200:
                # 404 simply means SDP has no such location (for example eu).
                print(f"INFO: No {kind} in {location} (HTTP {resp.status_code})")
                continue
            for template in resp.json().get(kind, []):
                template_id = template.get('name', '').split('/')[-1]
                result[list_field].append({
                    'id': template_id,
                    'display_name': template.get('displayName') or template_id,
                    'info_types': extractor(template),
                })
        except Exception as e:
            print(f"Error listing {kind} in {location}: {e}")

    template_cache[cache_key] = (result, time.time())
    return result

@app.route('/dlp_templates/<location>')
def get_dlp_templates_for_location(location):
    """SDP templates available for the Model Armor location, for the editor dropdowns."""
    if not any(e['location'] == location for e in model_armor_endpoints):
        return jsonify({'error': 'Invalid location'}), 400
    return jsonify(fetch_dlp_templates(location))

@app.route('/templates/<location>')
def get_templates_for_location(location):
    endpoint_info = next((e for e in model_armor_endpoints if e["location"] == location), None)
    if not endpoint_info: 
        return jsonify({'error': 'Invalid location'}), 404
    prompt_templates, response_templates = fetch_model_armor_templates(location, endpoint_info['endpoint'])
    return jsonify({'prompt_templates': prompt_templates, 'response_templates': response_templates})

# --- START: MODIFIED ENDPOINT FOR PROMPT ANALYSIS (NOW HANDLES FILES) ---
@app.route('/analyze_prompt', methods=['POST'])
def analyze_prompt():
    """
    A dedicated endpoint to only run prompt analysis and return quickly.
    Handles both text and file prompts.
    """
    try:
        prompt_analysis = None
        filter_results = []
        filter_details = []
        output_str = ""
        scan_error = None

        # Handle file upload scenario
        if 'file' in request.files:
            file = request.files.get('file')
            prompt_template = request.form.get('promptTemplate')
            location = request.form.get('location')
            
            if not file or not prompt_template or not location:
                return jsonify({'error': 'Missing file, template, or location for analysis'}), 400
            
            endpoint_info = next((e for e in model_armor_endpoints if e["location"] == location), None)
            if not endpoint_info:
                return jsonify({'error': 'Invalid location for analysis'}), 400

            # Security Check: Enforce file extensions
            if not allowed_file(file.filename):
                return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
                
            # Security Check: Sanitize filename
            file.filename = secure_filename(file.filename)
            
            file_data_base64, mime_type = get_file_data(file)

            if is_image_mime(mime_type):
                analysis = analyze_image_prompt(
                    request.form.get('prompt', ''),
                    {'base64_data': file_data_base64, 'mime_type': mime_type, 'is_image': True},
                    prompt_template, location, endpoint_info
                )
                output_str = analysis['output_str']
                filter_results = analysis['filter_results']
                filter_details = analysis['filter_details']
                scan_error = analysis['scan_error']
            else:
                result = sanitize_file_prompt_with_rest_api_optimized(
                    file_data_base64, mime_type, prompt_template, location, endpoint_info['endpoint']
                )
                output_str = json.dumps(result, indent=2)
                filter_results = process_rest_api_results(result)
                filter_details = summarize_filter_results(result)
                scan_error = describe_scan_problems(result, 'file')

        # Handle text-only scenario
        else:
            data = request.get_json()
            prompt = data.get('prompt', '')
            prompt_template = data.get('promptTemplate')
            location = data.get('location')

            if not prompt_template:
                return jsonify({'prompt_analysis': None}) # Nothing to do

            endpoint_info = next((e for e in model_armor_endpoints if e["location"] == location), None)
            if not endpoint_info:
                return jsonify({'error': 'Invalid location for analysis'}), 400
            
            result = sanitize_text_prompt_optimized(prompt, prompt_template, location, endpoint_info)
            output_str = json.dumps(result, indent=2)
            filter_results = process_rest_api_results(result)
            filter_details = summarize_filter_results(result)

        # Common response structure
        prompt_analysis = {
            'template': prompt_template,
            'status': 'error' if scan_error else ('fail' if filter_results else 'pass'),
            'filter_details': filter_details,
            'raw_output': f"⚠️ {scan_error}\n\n{output_str}" if scan_error else output_str
        }
        
        return jsonify({'prompt_analysis': prompt_analysis})

    except Exception:
        print(f"ERROR in /analyze_prompt: {traceback.format_exc()}")
        return jsonify({
            'error': 'An unexpected error occurred in prompt analysis.',
            'prompt_analysis': {
                'template': request.form.get('promptTemplate'),
                'status': 'error',
                'raw_output': 'Prompt analysis failed. See server logs for details.'
            }
        }), 500
# --- END: MODIFIED ENDPOINT ---

@app.route('/update_template', methods=['POST'])
def update_template():
    """Endpoint to update a dedicated demo template."""
    try:
        data = request.get_json()
        template_name = data.get('templateName')
        location = data.get('location')
        config_data = data.get('config')
        print(f"DEBUG config_data from frontend: {config_data}", flush=True)

        if not template_name or not location or not config_data:
            return jsonify({'error': 'Missing templateName, location, or config'}), 400

        if not template_name.startswith('modelarmor-demo-'):
            return jsonify({'error': 'Only dedicated demo templates can be modified'}), 403
        endpoint_info = next((e for e in model_armor_endpoints if e["location"] == location), None)
        if not endpoint_info:
            return jsonify({'error': 'Invalid location'}), 400

        # Construct template path
        name = f"projects/{project}/locations/{location}/templates/{template_name}"

        # Construct payload for REST API
        payload = {}
        filter_config = {}
        update_mask = []

        if 'pi_and_jailbreak' in config_data:
            pi_jb_settings = {
                'filterEnforcement': 'ENABLED' if config_data['pi_and_jailbreak'] == 'ENABLED' else 'DISABLED'
            }
            
            if 'pi_jb_confidence' in config_data:
                pi_jb_settings['confidenceLevel'] = normalize_confidence(config_data['pi_jb_confidence'])
                
            filter_config['piAndJailbreakFilterSettings'] = pi_jb_settings
            update_mask.append('filterConfig.piAndJailbreakFilterSettings')
            
        if 'malicious_uris' in config_data:
            filter_config['maliciousUriFilterSettings'] = {
                'filterEnforcement': 'ENABLED' if config_data['malicious_uris'] == 'ENABLED' else 'DISABLED'
            }
            update_mask.append('filterConfig.maliciousUriFilterSettings')
            
        if 'rai_filters' in config_data:
            rai_filters = []
            for f in config_data['rai_filters']:
                str_type = normalize_rai_filter_type(f.get('filter_type'))
                if str_type:
                    rai_filters.append({
                        'filterType': str_type,
                        'confidenceLevel': normalize_confidence(f.get('confidence_level')),
                    })
            filter_config['raiSettings'] = {'raiFilters': rai_filters}
            update_mask.append('filterConfig.raiSettings')
            
        if 'sdp_settings' in config_data:
            sdp_data = config_data['sdp_settings']
            sdp_config = {}
            
            if sdp_data.get('mode') == 'Basic':
                sdp_config['basicConfig'] = {'filterEnforcement': 'ENABLED'}
                sdp_config['advancedConfig'] = None
            elif sdp_data.get('mode') == 'Advanced':
                advanced_config = {}
                if 'inspect_template' in sdp_data:
                    inspect_template = sdp_data['inspect_template']
                    if inspect_template and not inspect_template.startswith('projects/'):
                        inspect_template = f"projects/{project}/locations/{location}/inspectTemplates/{inspect_template}"
                    advanced_config['inspectTemplate'] = inspect_template
                    
                if 'deidentify_template' in sdp_data:
                    deidentify_template = sdp_data['deidentify_template']
                    if deidentify_template and not deidentify_template.startswith('projects/'):
                        deidentify_template = f"projects/{project}/locations/{location}/deidentifyTemplates/{deidentify_template}"
                    advanced_config['deidentifyTemplate'] = deidentify_template
                sdp_config['advancedConfig'] = advanced_config
            else:
                sdp_config['basicConfig'] = {'filterEnforcement': 'DISABLED'}
                sdp_config['advancedConfig'] = None
                
            filter_config['sdpSettings'] = sdp_config
            update_mask.append('filterConfig.sdpSettings')

        if filter_config:
            payload['filterConfig'] = filter_config

        template_metadata = {}
        if 'modalities' in config_data:
            template_metadata['modalities'] = config_data['modalities']
            update_mask.append('templateMetadata.modalities')

        if 'filter_version' in config_data:
            version_value = config_data['filter_version']
            template_metadata['filterVersionSelector'] = (
                {'alias': version_value} if str(version_value).startswith('FILTER_VERSION_ALIAS_')
                else {'version': version_value}
            )
            update_mask.append('templateMetadata.filterVersionSelector')

        if 'logging_enabled' in config_data:
            logging_val = bool(config_data['logging_enabled'])
            template_metadata['logTemplateOperations'] = logging_val
            template_metadata['logSanitizeOperations'] = logging_val
            update_mask.append('templateMetadata.logTemplateOperations')
            update_mask.append('templateMetadata.logSanitizeOperations')
        if template_metadata:
            payload['templateMetadata'] = template_metadata

        # Make direct REST API call
        import google.auth
        from google.auth.transport.requests import Request
        import requests
        
        credentials, _ = google.auth.default()
        credentials.refresh(Request())
        token = credentials.token
        
        url = f"https://{endpoint_info['endpoint']}/v1/{name}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        if update_mask:
            url += f"?updateMask={','.join(update_mask)}"
            
        print(f"Patching template to {url}...", flush=True)
        print(f"Payload: {payload}", flush=True)
        
        resp = requests.patch(url, headers=headers, json=payload)
        print(f"Patch status: {resp.status_code}", flush=True)
        
        if resp.status_code != 200:
            print(f"Failed to update template: {resp.text}", flush=True)
            return jsonify({'error': f"Failed to update template: {resp.text}"}), resp.status_code
            
        # Clear cache for this location
        cache_key = f"templates_{location}"
        if cache_key in template_cache:
            del template_cache[cache_key]
            
        # Clear Model Armor cache for this template
        keys_to_delete = [k for k in model_armor_cache if k.endswith(f"_{template_name}_{location}")]
        for k in keys_to_delete:
            del model_armor_cache[k]
            
        return jsonify({'status': 'success', 'message': f'Template {template_name} updated successfully'})

    except Exception as e:
        print(f"ERROR in /update_template: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    file_data = None

    if 'file' in request.files:
        file = request.files['file']
        prompt_text = request.form.get('prompt', '')
        model_name_from_ui = request.form.get('model')
        location = request.form.get('location', 'us-central1')
        prompt_template = request.form.get('promptTemplate')
        response_template = request.form.get('responseTemplate')
        default_response = request.form.get('defaultResponse')
        use_default_response = request.form.get('useDefaultResponse') == 'true'
        system_instruction = request.form.get('systemInstruction', '')
        
        # Security Check: Enforce file extensions
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
            
        # Security Check: Sanitize filename
        file.filename = secure_filename(file.filename)
        
        file_data_base64, mime_type = get_file_data(file)
        
        file_data = {
            'base64_data': file_data_base64,
            'mime_type': mime_type,
            'filename': file.filename,
            'is_image': is_image_mime(mime_type)
        }
        if prompt_text:
            prompt = prompt_text
        elif file_data['is_image']:
            prompt = f"Please describe this image: {file.filename}"
        else:
            prompt = f"Please analyze this document: {file.filename}"

    else:
        data = request.get_json()
        prompt = data.get('prompt')
        model_name_from_ui = data.get('model')
        location = data.get('location', 'us-central1')
        prompt_template = data.get('promptTemplate')
        response_template = data.get('responseTemplate')
        default_response = data.get('defaultResponse')
        use_default_response = data.get('useDefaultResponse', True)
        system_instruction = data.get('systemInstruction', '')
        prompt_text = prompt

    model_info = next((m for m in foundation_models if m.get("display_name") == model_name_from_ui or m["name"] == model_name_from_ui), None)
    if not model_info: 
        return jsonify({'error': 'Invalid model selected'}), 400

    endpoint_info = next((e for e in model_armor_endpoints if e["location"] == location), None)
    if not endpoint_info: 
        return jsonify({'error': 'Invalid location provided'}), 400

    try:
        # Always use Model Armor mode
        print("Using corrected sequential processing approach with Model Armor")
        
        # Use the corrected sequential processing approach with Model Armor
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(process_chat_async(
            prompt, model_info, system_instruction, file_data,
            prompt_template, response_template, location, endpoint_info,
            use_default_response, default_response, prompt_text
        ))
        
        loop.close()
        
        response_text = result['response']
        prompt_analysis = result['prompt_analysis']
        response_analysis = result['response_analysis']
        
        return jsonify({
            'response': response_text,
            'source': model_info.get('provider'),
            'model_armor': {
                'prompt_analysis': prompt_analysis,
                'response_analysis': response_analysis
            }
        })
        
    except Exception as e:
        error_message = f"An unexpected error occurred during chat processing: {str(e)}"
        print(f"ERROR in /chat: {traceback.format_exc()}")
        return jsonify({
            'error': error_message,
            'response': error_message,
            'source': 'System',
            'model_armor': {
                'prompt_analysis': {'status': 'error', 'raw_output': traceback.format_exc()},
                'response_analysis': {'status': 'error', 'raw_output': traceback.format_exc()}
            }
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)