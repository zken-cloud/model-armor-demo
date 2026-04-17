from google import genai
from google.genai import types
import os
import re
import requests
import google.auth
from flask import Flask, render_template, request, jsonify
from google.cloud import modelarmor_v1
from dotenv import load_dotenv
from google.api_core import exceptions
import base64
from werkzeug.utils import secure_filename
import mimetypes
import json
import traceback
import ast
import asyncio
import concurrent.futures
import hashlib
import time
from functools import lru_cache
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
TOKEN_TTL = 3300  # 55 minutes (tokens typically last 60 minutes)

def get_cached_auth_token():
    """Get cached authentication token or refresh if expired"""
    current_time = time.time()
    
    if 'token_data' in auth_token_cache:
        token, timestamp = auth_token_cache['token_data']
        if current_time - timestamp < TOKEN_TTL:
            return token
    
    # Token expired or doesn't exist, refresh it
    credentials, _ = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    
    # Cache the new token
    auth_token_cache['token_data'] = (credentials.token, current_time)
    print(f"INFO: Authentication token refreshed and cached")
    
    return credentials.token

# Add allowed file extensions
ALLOWED_EXTENSIONS = {
    'pdf', 'docx', 'docm', 'dotx', 'dotm',
    'pptx', 'pptm', 'potx', 'pot',
    'xlsx', 'xlsm', 'xltx', 'xltm'
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_byte_data_type(mime_type):
    """Convert MIME type to Model Armor byteDataType"""
    mime_to_type = {
        'application/pdf': 'PDF',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOCX',
        'application/msword': 'DOC',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'PPTX',
        'application/vnd.ms-powerpoint': 'PPT',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'XLSX',
        'application/vnd.ms-excel': 'XLS'
    }
    return mime_to_type.get(mime_type, 'PDF')

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
file_cache = {}  # Cache for file base64 data
CACHE_TTL = 300  # 5 minutes

def get_file_data(file):
    """Reads file, base64 encodes it, and manages cache."""
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Seek back to start
    file_cache_key = f"{file.filename}_{file_size}"
    
    if file_cache_key in file_cache:
        cached_data, timestamp = file_cache[file_cache_key]
        if time.time() - timestamp < CACHE_TTL:
            print(f"INFO: Using cached file data for {file.filename}")
            return cached_data['base64_data'], cached_data['mime_type']
        else:
            del file_cache[file_cache_key]
            
    file_content = file.read()
    file_data_base64 = base64.b64encode(file_content).decode('utf-8')
    mime_type = mimetypes.guess_type(file.filename)[0]
    
    file_cache[file_cache_key] = ({'base64_data': file_data_base64, 'mime_type': mime_type}, time.time())
    
    # Clean old cache entries if cache is too large
    if len(file_cache) > 50:
        current_time = time.time()
        expired_keys = [k for k, (_, timestamp) in file_cache.items()
                       if current_time - timestamp > CACHE_TTL]
        for k in expired_keys:
            del file_cache[k]
            
    return file_data_base64, mime_type

def get_cache_key(data, template_name, location):
    """Generate cache key for Model Armor results"""
    if isinstance(data, dict):  # File data
        content_hash = hashlib.md5(data['base64_data'][:1000].encode()).hexdigest()
        return f"file_{content_hash}_{template_name}_{location}"
    else:  # Text data
        content_hash = hashlib.md5(data.encode()).hexdigest()
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


model_armor_endpoints = [
    {"location": "us-central1", "endpoint": "modelarmor.us-central1.rep.googleapis.com", "display_name": "us-central1"},
    {"location": "us-east1", "endpoint": "modelarmor.us-east1.rep.googleapis.com", "display_name": "us-east1"},
    {"location": "europe-west4", "endpoint": "modelarmor.europe-west4.rep.googleapis.com", "display_name": "europe-west4"},
    {"location": "asia-southeast1", "endpoint": "modelarmor.asia-southeast1.rep.googleapis.com", "display_name": "asia-southeast1"},
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
    {"name": "gemini-3.1-flash-lite-preview", "provider": "Google", "location": "global", "display_name": "gemini-3.1-flash-lite-preview"},
    {"name": "gemini-3.1-pro-preview", "provider": "Google", "location": "global", "display_name": "gemini-3.1-pro-preview"},
]

def ensure_demo_templates_exist():
    print("INFO: Ensuring demo templates exist...")
    for endpoint_info in model_armor_endpoints:
        location = endpoint_info['location']
        endpoint = endpoint_info['endpoint']
        try:
            client = get_model_armor_client(location, endpoint)
            parent = f"projects/{project}/locations/{location}"
            
            request = modelarmor_v1.ListTemplatesRequest(parent=parent)
            page_result = client.list_templates(request=request)
            existing_names = [t.name.split('/')[-1] for t in page_result]
            
            for template_id in ["modelarmor-demo-prompt", "modelarmor-demo-response"]:
                if template_id not in existing_names:
                    print(f"  - Creating {template_id} in {location}...")
                    template = modelarmor_v1.Template()
                    template.filter_config = modelarmor_v1.FilterConfig()
                    
                    create_request = modelarmor_v1.CreateTemplateRequest(
                        parent=parent,
                        template_id=template_id,
                        template=template
                    )
                    client.create_template(request=create_request)
                    print(f"  - Successfully created {template_id} in {location}")
                else:
                    print(f"  - {template_id} already exists in {location}")
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
    return response.text

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
                        confidence_val = get_field(rai_filter, 'confidenceLevel') or get_field(rai_filter, 'confidence_level', 0)
                        
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
                confidence_val = get_field(pi_jb, 'confidenceLevel') or get_field(pi_jb, 'confidence_level', 0)
                
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

def process_template_results(output_str):
    matches = re.finditer(r'key: "([^"]+)"[^}]*?match_state: MATCH_FOUND', output_str, re.DOTALL)
    return [match.group(1) for match in matches]

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

def check_sdp_transformation(output_str):
    """Extracts the transformed (redacted) text from a Model Armor sanitization result string."""
    try:
        match = re.search(r'deidentify_result\s*{[^}]*?text:\s*"((?:[^"\\]|\\.)*)"', output_str, re.DOTALL)
        if match:
            return ast.literal_eval(f'"{match.group(1)}"')
    except Exception as e:
        print(f"Error extracting SDP transformation from string: {e}")
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
        output_str = str(response_check)
        filter_results = process_template_results(output_str)

        sdp_text = check_sdp_transformation(output_str)
        has_sdp = 'sdp' in filter_results

        if sdp_text and has_sdp and not use_default_response:
            response_text = sdp_text

        details = "❌ Violations found:\n" + "\n".join(f"• {result}" for result in filter_results) if filter_results else "✅ No template violations found"
        return {'response_text': response_text, 'analysis': {'template': template_display_name, 'status': 'fail' if filter_results else 'pass', 'details': details, 'matches': bool(filter_results), 'filter_results': filter_results, 'raw_output': output_str}, 'has_violations': bool(filter_results), 'has_sdp': has_sdp}
    except Exception as e:
        print(f"Error in response analysis: {e}")
        return {'response_text': response_text, 'analysis': {'template': template_display_name, 'status': 'error', 'details': f'Error in Model Armor analysis: {e}', 'matches': False, 'raw_output': str(e)}, 'has_violations': False}

def get_template_path(template_name, location):
    return f"projects/{project}/locations/{location}/templates/{template_name}"

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
    
    try:
        access_token = get_cached_auth_token()
        
        url = f"https://{endpoint}/v1alpha/projects/{project}/locations/{location}/templates/{template_name}:sanitizeUserPrompt"
        
        payload = {
            "userPromptData": {
                "byteItem": {
                    "byteDataType": get_byte_data_type(mime_type),
                    "byteData": file_data_base64
                }
            }
        }
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
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
        result = str(prompt_check)
        
        # Cache the result
        cache_result(cache_key, result)
        return result
        
    except Exception as e:
        print(f"Error in optimized text sanitization: {e}")
        raise e

# --- Async Processing Functions ---
async def analyze_prompt_async(prompt, file_data, prompt_template, location, endpoint_info):
    """Async wrapper for prompt analysis"""
    loop = asyncio.get_event_loop()
    
    def run_analysis():
        if file_data:
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
                'sdp_transformed_text': sdp_transformed_text,
                'is_file': True
            }
        else:
            result = sanitize_text_prompt_optimized(prompt, prompt_template, location, endpoint_info)
            filter_results = process_template_results(result)
            sdp_transformed_text = check_sdp_transformation(result)
            return {
                'output_str': result,
                'filter_results': filter_results,
                'sdp_transformed_text': sdp_transformed_text,
                'is_file': False
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
            'filter_results': filter_results, 'raw_output': output_str
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
        output_str = ""

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

            result = sanitize_file_prompt_with_rest_api_optimized(
                file_data_base64, mime_type, prompt_template, location, endpoint_info['endpoint']
            )
            output_str = json.dumps(result, indent=2)
            filter_results = process_rest_api_results(result)

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
            
            output_str = sanitize_text_prompt_optimized(prompt, prompt_template, location, endpoint_info)
            filter_results = process_template_results(output_str)

        # Common response structure
        prompt_analysis = {
            'template': prompt_template,
            'status': 'fail' if filter_results else 'pass',
            'raw_output': output_str
        }
        
        return jsonify({'prompt_analysis': prompt_analysis})

    except Exception as e:
        error_message = f"An unexpected error occurred in prompt analysis: {str(e)}"
        print(f"ERROR in /analyze_prompt: {traceback.format_exc()}")
        return jsonify({
            'error': error_message,
            'prompt_analysis': {
                'template': request.form.get('promptTemplate') or request.get_json().get('promptTemplate'),
                'status': 'error',
                'raw_output': traceback.format_exc()
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
                conf_map = {
                    0: 'DETECTION_CONFIDENCE_LEVEL_UNSPECIFIED',
                    1: 'LOW_AND_ABOVE',
                    2: 'MEDIUM_AND_ABOVE',
                    3: 'HIGH',
                    '0': 'DETECTION_CONFIDENCE_LEVEL_UNSPECIFIED',
                    '1': 'LOW_AND_ABOVE',
                    '2': 'MEDIUM_AND_ABOVE',
                    '3': 'HIGH'
                }
                conf_val = config_data['pi_jb_confidence']
                pi_jb_settings['confidenceLevel'] = conf_map.get(conf_val, 'DETECTION_CONFIDENCE_LEVEL_UNSPECIFIED')
                
            filter_config['piAndJailbreakFilterSettings'] = pi_jb_settings
            update_mask.append('filterConfig.piAndJailbreakFilterSettings')
            
        if 'malicious_uris' in config_data:
            filter_config['maliciousUriFilterSettings'] = {
                'filterEnforcement': 'ENABLED' if config_data['malicious_uris'] == 'ENABLED' else 'DISABLED'
            }
            update_mask.append('filterConfig.maliciousUriFilterSettings')
            
        if 'rai_filters' in config_data:
            rai_filters = []
            # Map from int to string for RAI Filter Type
            int_to_str_type = {
                2: 'HATE_SPEECH',
                3: 'DANGEROUS',
                6: 'HARASSMENT',
                17: 'SEXUALLY_EXPLICIT'
            }
            # Map from int to string for Confidence Level
            int_to_str_conf = {
                0: 'DETECTION_CONFIDENCE_LEVEL_UNSPECIFIED',
                1: 'LOW_AND_ABOVE',
                2: 'MEDIUM_AND_ABOVE',
                3: 'HIGH'
            }
            
            for f in config_data['rai_filters']:
                f_type = f.get('filter_type')
                c_level = f.get('confidence_level')
                
                str_type = int_to_str_type.get(f_type)
                if not str_type and isinstance(f_type, str):
                    str_type = f_type # Use string directly if already string
                    
                str_conf = int_to_str_conf.get(c_level)
                if not str_conf and isinstance(c_level, str):
                    str_conf = c_level # Use string directly if already string
                    
                if str_type:
                    rai_filters.append({
                        'filterType': str_type,
                        'confidenceLevel': str_conf or 'DETECTION_CONFIDENCE_LEVEL_UNSPECIFIED'
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
    is_file_upload = False

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
            'filename': file.filename
        }
        prompt = prompt_text if prompt_text else f"Please analyze this document: {file.filename}"
        is_file_upload = True

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