const systemInstructionModal = new bootstrap.Modal(document.getElementById('systemInstructionModal'));
let currentSystemInstruction = '';
let selectedFile = null;
let requestStartTime = 0;

const converter = new showdown.Converter();

// For the few places that still build HTML from strings.
function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// Template cache for performance
const templateCache = new Map();

// Debounce function for performance
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Debounced template update
const debouncedUpdateTemplates = debounce(updateTemplatesForLocation, 300);

// Performance indicator functions
function showPerformanceIndicator(message) {
    const indicator = document.getElementById('performanceIndicator');
    indicator.textContent = message;
    indicator.style.display = 'block';
    requestStartTime = performance.now();
}

function hidePerformanceIndicator() {
    const indicator = document.getElementById('performanceIndicator');
    if (requestStartTime > 0) {
        const duration = Math.round(performance.now() - requestStartTime);
        indicator.textContent = `Completed in ${duration}ms`;
        setTimeout(() => {
            indicator.style.display = 'none';
        }, 2000);
    } else {
        indicator.style.display = 'none';
    }
}

// Preload templates for every offered location
async function preloadTemplates() {
    const promises = availableLocations().map(async location => {
        try {
            const response = await fetch(`/templates/${location}`);
            const data = await response.json();
            templateCache.set(location, data);
            console.log(`Preloaded templates for ${location}`);
        } catch (error) {
            console.warn(`Failed to preload templates for ${location}:`, error);
        }
    });
    await Promise.all(promises);
}

let currentTemplateData = null;   // last /templates/<location> payload

function populateTemplateSelects(data, selectedPromptName = null, selectedResponseName = null) {
    currentTemplateData = data;
    const promptSelect = document.getElementById('promptTemplate');
    const responseSelect = document.getElementById('responseTemplate');
 
    // Store full template data for hover details
    const promptTemplatesMap = new Map();
    const responseTemplatesMap = new Map();
 
    promptSelect.innerHTML = '<option value="">No template</option>';
    data.prompt_templates.forEach(t => {
        promptTemplatesMap.set(t.name, t);
        promptSelect.innerHTML += `<option value="${escapeHtml(t.name)}" data-updated="${escapeHtml(t.last_updated)}" data-supports-images="${t.supports_images ? 'true' : 'false'}" data-supports-text="${t.supports_text === false ? 'false' : 'true'}">${escapeHtml(t.display_name)}${t.supports_images ? ' 🖼' : ''}</option>`;
    });
 
    responseSelect.innerHTML = '<option value="">No template</option>';
    data.response_templates.forEach(t => {
        responseTemplatesMap.set(t.name, t);
        responseSelect.innerHTML += `<option value="${escapeHtml(t.name)}" data-updated="${escapeHtml(t.last_updated)}" data-supports-images="${t.supports_images ? 'true' : 'false'}">${escapeHtml(t.display_name)}</option>`;
    });
 
    // Explicitly set values to preserve selection or default to "No template"
    if (selectedPromptName && data.prompt_templates.some(t => t.name === selectedPromptName)) {
        promptSelect.value = selectedPromptName;
    } else {
        promptSelect.value = "";
    }

    if (selectedResponseName && data.response_templates.some(t => t.name === selectedResponseName)) {
        responseSelect.value = selectedResponseName;
    } else {
        responseSelect.value = "";
    }

    // Attach hover listeners
    attachHoverListeners(promptSelect, promptTemplatesMap);
    attachHoverListeners(responseSelect, responseTemplatesMap);
    updateTemplateInfoIcons();
}

function attachHoverListeners(selectElement, templateMap) {
    const popup = document.getElementById('template-details-popup');

    selectElement.addEventListener('mouseenter', () => {
        const selectedValue = selectElement.value;
        if (selectedValue && templateMap.has(selectedValue)) {
            showPopup(templateMap.get(selectedValue), selectElement);
        }
    });

    selectElement.addEventListener('mousemove', (e) => {
        const selectedValue = selectElement.value;
        if (selectedValue && templateMap.has(selectedValue)) {
            // Position popup near cursor but slightly offset
            popup.style.left = (e.pageX + 15) + 'px';
            popup.style.top = (e.pageY + 15) + 'px';
        }
    });

    selectElement.addEventListener('mouseleave', () => {
        popup.style.display = 'none';
    });

    // Also update on change if mouse is still over
    selectElement.addEventListener('change', () => {
        const selectedValue = selectElement.value;
        if (selectedValue && templateMap.has(selectedValue)) {
            showPopup(templateMap.get(selectedValue), selectElement);
        } else {
            popup.style.display = 'none';
        }
    });
}

function showPopup(templateData, targetElement) {
    const popup = document.getElementById('template-details-popup');

    let configHTML = '';
    if (templateData.config) {
        const config = templateData.config;
 
        // Display RAI Filters
        configHTML += '<div class="config-section"><strong>🛡️ Responsible AI Filters:</strong><ul>';
        if (config.rai_filters && config.rai_filters.length > 0) {
            config.rai_filters.forEach(filter => {
                configHTML += `<li>${escapeHtml(filter)}</li>`;
            });
        } else {
            configHTML += '<li>Not Configured</li>';
        }
        configHTML += '</ul></div>';
 
        // Display Detection Filters (PI, Jailbreak, Malicious URL)
        configHTML += '<div class="config-section"><strong>🔍 Detection Filters:</strong><ul>';
        
        // PI & Jailbreak
        const hasPiJb = config.detection_filters && config.detection_filters.some(f => f.includes('Prompt Injection & Jailbreak'));
        configHTML += `<li>Prompt Injection & Jailbreak: ${escapeHtml(hasPiJb ? 'Enabled' : 'Disabled')}</li>`;
        
        // Malicious URL
        const hasMalUrls = config.detection_filters && config.detection_filters.some(f => f.includes('Malicious URL'));
        configHTML += '</ul></div>';
 
        // Display SDP Settings
        configHTML += '<div class="config-section"><strong>🔒 Sensitive Data Protection:</strong><ul>';
        configHTML += `<li>Mode: ${escapeHtml(config.sdp_settings?.mode || 'Not Configured')}</li>`;
        configHTML += `<li>Inspect Template: ${escapeHtml(config.sdp_settings?.inspect_template || 'None')}</li>`;
        configHTML += `<li>Deidentify Template: ${escapeHtml(config.sdp_settings?.deidentify_template || 'None')}</li>`;
        if (config.sdp_settings?.info_types) {
            configHTML += `<li>Info Types: ${escapeHtml(config.sdp_settings.info_types.join(', '))}</li>`;
        }
        configHTML += '</ul></div>';
 
        // Display other settings
        if (config.other_settings && Object.keys(config.other_settings).length > 0) {
            configHTML += '<div class="config-section"><strong>⚙️ Other Settings:</strong><ul>';
            if (config.other_settings.logging_enabled !== undefined) {
                configHTML += `<li>Logging: ${config.other_settings.logging_enabled ? 'Enabled' : 'Disabled'}</li>`;
            }
            configHTML += '</ul></div>';
        }
    } else {
        configHTML = '<p class="no-config">No configuration details available.</p>';
    }

    popup.innerHTML = `
        <h4>${escapeHtml(templateData.display_name)}</h4>
        <p class="template-meta"><strong>Last Updated:</strong> ${escapeHtml(templateData.last_updated)}</p>
        ${configHTML}
    `;
    popup.style.display = 'block';
}

document.getElementById('sendButton').addEventListener('click', sendMessage);
document.getElementById('clearButton').addEventListener('click', clearChat);
document.getElementById('systemInstructionButton').addEventListener('click', () => { systemInstructionModal.show(); });
document.getElementById('saveSystemInstruction').addEventListener('click', () => { currentSystemInstruction = document.getElementById('systemInstruction').value.trim(); systemInstructionModal.hide(); });
document.getElementById('userInput').addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

// --- Sidebar Toggle Logic ---
document.getElementById('sidebar-toggle').addEventListener('click', () => {
    const sidebar = document.getElementById('sidebar');
    const isCollapsed = sidebar.classList.contains('collapsed');

    if (isCollapsed) {
        sidebar.classList.remove('collapsed');
    } else {
        sidebar.classList.add('collapsed');
    }
    updateLayout();
});

// Close sidebar on mobile
document.getElementById('sidebar-close').addEventListener('click', () => {
    document.getElementById('sidebar').classList.add('collapsed');
    updateLayout();
});

// --- Model Armor Toggle Logic ---
document.getElementById('model-armor-toggle').addEventListener('click', () => {
    const modelArmorSection = document.getElementById('model-armor-section');
    const isCollapsed = modelArmorSection.classList.contains('collapsed');

    if (isCollapsed) {
        modelArmorSection.classList.remove('collapsed');
    } else {
        modelArmorSection.classList.add('collapsed');
    }
    updateLayout();
});

function updateLayout() {
    const sidebar = document.getElementById('sidebar');
    const modelArmorSection = document.getElementById('model-armor-section');
    const mainContent = document.getElementById('main-content');
    const chatSection = document.querySelector('.chat-section');

    const isSidebarCollapsed = sidebar.classList.contains('collapsed');
    const isModelArmorCollapsed = modelArmorSection.classList.contains('collapsed');

    // Update main content width based on sidebar
    if (isSidebarCollapsed) {
        mainContent.style.width = '100%';
    } else {
        mainContent.style.width = '75%';
    }

    // Update chat section flex based on model armor section
    // If model armor is collapsed, chat section takes full available width of main-content
    // CSS handles the flex-grow, but we can be explicit if needed, 
    // but the CSS .model-armor-section.collapsed { flex: 0; ... } should handle it.
    // The chat-section has flex: 2, model-armor has flex: 1.
    // When model-armor is flex: 0, chat-section takes all space.
}

// Auto-collapse sidebar on mobile when page loads
if (window.innerWidth <= 768) {
    document.getElementById('sidebar').classList.add('collapsed');
    // On mobile, we might want to keep model armor visible or collapsed by default?
    // Let's keep it visible but stacked for now, or maybe collapsed to save space?
    // User request didn't specify, but "mobile friendly" implies saving space.
    // Let's leave it as is (stacked) but maybe collapse it if it's too much.
    // For now, just sidebar is collapsed.
}

// Handle window resize
let resizeTimer;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        const sidebar = document.getElementById('sidebar');
        if (window.innerWidth <= 768 && !sidebar.classList.contains('collapsed')) {
            sidebar.classList.add('collapsed');
        }
        // Reset styles when going back to desktop if needed
        if (window.innerWidth > 768) {
            // Ensure layout is correct
            updateLayout();
        }
    }, 250);
});

// --- SDP (DLP) template dropdowns ---
const dlpTemplateCache = new Map();

// Show a (?) icon whose hover/focus tooltip carries `text`; hide it when empty.
function setInfoIcon(iconId, text) {
    const icon = document.getElementById(iconId);
    if (!icon) return;
    if (!text) {
        icon.hidden = true;
        return;
    }
    icon.hidden = false;
    icon.setAttribute('title', text);            // plain fallback
    icon.setAttribute('data-bs-title', text);
    if (window.bootstrap && bootstrap.Tooltip) {
        const tip = bootstrap.Tooltip.getOrCreateInstance(icon, {
            trigger: 'hover focus', placement: 'right', container: 'body',
        });
        tip.setContent({ '.tooltip-inner': text });
    }
}

function fullInfoTypeList(infoTypes) {
    if (!infoTypes || !infoTypes.length) {
        return 'No infoTypes configured - Sensitive Data Protection defaults apply';
    }
    return infoTypes.join(', ');
}

function describeInfoTypes(infoTypes, limit = 12) {
    if (!infoTypes || !infoTypes.length) {
        return 'No infoTypes configured - Sensitive Data Protection defaults apply';
    }
    const shown = infoTypes.slice(0, limit).join(', ');
    const rest = infoTypes.length - limit;
    return rest > 0 ? `${shown} +${rest} more` : shown;
}

function updateDlpHover(selectId) {
    const select = document.getElementById(selectId);
    const option = select.options[select.selectedIndex];
    const detail = option ? (option.dataset.infoTypes || '') : '';
    // Hovering the control itself shows what will trigger the template.
    select.title = detail || 'Select a template to see the infoTypes it matches';
    const hint = document.getElementById(selectId + 'Hint');
    if (hint) hint.textContent = detail ? `Triggers on: ${detail}` : '';
    // The (?) icon carries the complete list, not the truncated sample.
    const full = option && option.value ? (option.dataset.infoTypesFull || detail) : '';
    setInfoIcon(selectId + 'Info', full ? `Triggers on: ${full}` : '');
}

function fillDlpSelect(selectId, templates, currentValue) {
    const select = document.getElementById(selectId);
    select.innerHTML = '';

    const none = document.createElement('option');
    none.value = '';
    none.textContent = templates.length ? '-- Select a template --' : '-- None available in this region --';
    select.appendChild(none);

    templates.forEach(t => {
        const option = document.createElement('option');
        option.value = t.id;
        const label = t.display_name === t.id ? t.id : `${t.display_name} (${t.id})`;
        option.textContent = t.kind ? `${label} [${t.kind}]` : label;
        if (t.kind) option.dataset.kind = t.kind;
        const detail = describeInfoTypes(t.info_types);
        option.dataset.infoTypes = detail;
        option.dataset.infoTypesFull = fullInfoTypeList(t.info_types);
        option.title = detail;   // hover on the option itself
        select.appendChild(option);
    });

    // Keep a value already on the template even if it lives elsewhere.
    const shortValue = currentValue ? currentValue.split('/').pop() : '';
    if (shortValue && !templates.some(t => t.id === shortValue)) {
        const option = document.createElement('option');
        option.value = shortValue;
        option.textContent = `${shortValue} (not in this region)`;
        option.dataset.infoTypes = 'This template is not in the selected region; its infoTypes could not be read';
        select.appendChild(option);
    }
    select.value = shortValue || '';
    updateDlpHover(selectId);
}

async function getDlpTemplates(location) {
    let data = dlpTemplateCache.get(location);
    if (!data) {
        try {
            const response = await fetch(`/dlp_templates/${location}`);
            data = await response.json();
            dlpTemplateCache.set(location, data);
        } catch (e) {
            data = { inspect_templates: [], deidentify_templates: [] };
        }
    }
    return data;
}

// Sidebar (?) icons: for a selected template that uses Advanced SDP, list the
// infoTypes its inspect / de-identify templates cover.
async function updateTemplateInfoIcons() {
    const location = document.getElementById('locationSelect').value;
    const pairs = [
        ['promptTemplate', 'promptTemplateInfo', 'prompt_templates'],
        ['responseTemplate', 'responseTemplateInfo', 'response_templates'],
    ];
    let dlp = null;
    for (const [selectId, iconId, listKey] of pairs) {
        const name = document.getElementById(selectId).value;
        const tpl = currentTemplateData && (currentTemplateData[listKey] || []).find(t => t.name === name);
        const sdp = tpl && tpl.config && tpl.config.sdp_settings;
        if (!sdp || !String(sdp.mode || '').startsWith('Advanced')) {
            setInfoIcon(iconId, '');
            continue;
        }
        dlp = dlp || await getDlpTemplates(location);
        const inspect = (dlp.inspect_templates || []).find(t => t.id === sdp.inspect_template);
        const deid = (dlp.deidentify_templates || []).find(t => t.id === sdp.deidentify_template);
        const lines = [];
        if (sdp.inspect_template) {
            lines.push(`Inspect (${sdp.inspect_template}): ` +
                (inspect ? fullInfoTypeList(inspect.info_types) : 'not found in this region'));
        }
        if (sdp.deidentify_template) {
            lines.push(`De-identify (${sdp.deidentify_template}): ` +
                (deid ? fullInfoTypeList(deid.info_types) : 'not found in this region'));
        }
        setInfoIcon(iconId, lines.join('\n') || 'Advanced SDP with no templates set');
    }
}

async function loadDlpTemplates(location, inspectValue, deidentifyValue) {
    const data = await getDlpTemplates(location);
    fillDlpSelect('customInspectTemplate', data.inspect_templates || [], inspectValue);
    fillDlpSelect('customDeidentifyTemplate', data.deidentify_templates || [], deidentifyValue);
    updateSdpImageWarning();

    // Explain regions where advanced SDP cannot be used at all.
    const note = document.getElementById('sdpUnavailableNote');
    if (note) {
        note.textContent = data.note || '';
        note.style.display = data.note ? 'block' : 'none';
    }
}

// File handling event listeners
document.getElementById('fileButton').addEventListener('click', () => {
    document.getElementById('fileInput').click();
});

// Model Armor screens images up to 4MB, and only in the us/eu multi-regions.
const MAX_IMAGE_BYTES = 4 * 1024 * 1024;
const IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp'];

function isImageFile(file) {
    return IMAGE_EXTENSIONS.includes(file.name.split('.').pop().toLowerCase());
}

function selectedPromptTemplateSupportsText() {
    const select = document.getElementById('promptTemplate');
    const option = select.options[select.selectedIndex];
    if (!option || !option.value) return true;
    return option.getAttribute('data-supports-text') !== 'false';
}

// Image capability is a property of the selected prompt template
// (its modalities), not of the region.
function selectedPromptTemplateSupportsImages() {
    const select = document.getElementById('promptTemplate');
    const option = select.options[select.selectedIndex];
    if (!option || !option.value) return true;   // no screening at all
    return option.getAttribute('data-supports-images') === 'true';
}

document.getElementById('fileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (isImageFile(file)) {
        if (file.size > MAX_IMAGE_BYTES) {
            alert(`"${file.name}" is ${(file.size / 1024 / 1024).toFixed(1)}MB. ` +
                  `Model Armor screens images up to 4MB.`);
            e.target.value = '';
            return;
        }
        if (!selectedPromptTemplateSupportsImages()) {
            alert('The selected prompt template is text-only, so Model Armor will not screen ' +
                  'this image and the request will be blocked. Pick a prompt template marked 🖼 ' +
                  'to send images.');
        }
    }

    selectedFile = file;
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('filePreview').style.display = 'block';
});

document.getElementById('removeFile').addEventListener('click', () => {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('filePreview').style.display = 'none';
});

function checkFileSupport() {
    const modelSelect = document.getElementById('modelSelect');
    const selectedOption = modelSelect.options[modelSelect.selectedIndex];
    const provider = selectedOption ? selectedOption.getAttribute('data-provider') : '';
    const warning = document.getElementById('fileWarning');

    if (provider === 'Anthropic') {
        warning.style.display = 'block';
    } else {
        warning.style.display = 'none';
    }
}

function addMessage(message, isUser, source = null, attachmentName = null) {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    if (isUser) {
        messageDiv.textContent = message;
        if (attachmentName) {
            if (message) messageDiv.appendChild(document.createElement('br'));
            const chip = document.createElement('span');
            chip.className = 'file-indicator';
            chip.textContent = `📎 ${attachmentName}`;
            messageDiv.appendChild(chip);
        }
        messageDiv.style.whiteSpace = 'pre-wrap';
    } else {
        messageDiv.innerHTML = '<div class="loading-spinner"></div>';
        messageDiv.id = 'loading-message';
    }
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function updateBotMessage(text, source, redactedImage = null) {
    const loadingMessage = document.getElementById('loading-message');
    if (loadingMessage) {
        const html = converter.makeHtml(text);
        const manipulatedHtml = html.replace(/\[([A-Z_]+)\]/g, '<span class="model-armor-transform">[$1]</span>');
        loadingMessage.innerHTML = DOMPurify.sanitize(manipulatedHtml);

        if (redactedImage) {
            // The picture Model Armor actually handed to the model, findings boxed out.
            const figure = document.createElement('figure');
            figure.className = 'redacted-figure';
            const img = document.createElement('img');
            img.src = redactedImage;            // data: URL, permitted by img-src
            img.alt = 'Image after Model Armor redaction';
            img.className = 'redacted-image';
            const caption = document.createElement('figcaption');
            caption.textContent = 'Image as redacted by Model Armor — this is what the model received';
            figure.appendChild(img);
            figure.appendChild(caption);
            loadingMessage.insertBefore(figure, loadingMessage.firstChild);
        }

        if (source) {
            const sourceDiv = document.createElement('div');
            sourceDiv.className = 'source-tag';
            sourceDiv.textContent = `Source: ${source}`;
            loadingMessage.appendChild(sourceDiv);
        }
        loadingMessage.id = '';
    }
}

async function updateTemplatesForLocation(newlyUpdatedTemplateName = null) {
    const locationSelect = document.getElementById('locationSelect');
    const location = locationSelect.value;
    if (!location) {
        return;   // nothing selected yet; fetching /templates/ would 404
    }
    const promptSelect = document.getElementById('promptTemplate');
    const responseSelect = document.getElementById('responseTemplate');

    // Remember current selections
    const currentPrompt = promptSelect.value;
    const currentResponse = responseSelect.value;

    let selectPrompt = currentPrompt;
    let selectResponse = currentResponse;

    if (newlyUpdatedTemplateName) {
        if (newlyUpdatedTemplateName.endsWith('-prompt')) {
            selectPrompt = newlyUpdatedTemplateName;
        } else if (newlyUpdatedTemplateName.endsWith('-response')) {
            selectResponse = newlyUpdatedTemplateName;
        }
    }

    // Check cache first
    if (templateCache.has(location)) {
        const data = templateCache.get(location);
        populateTemplateSelects(data, selectPrompt, selectResponse);
        console.log(`Used cached templates for ${location}`);
        return;
    }

    locationSelect.disabled = true;
    promptSelect.innerHTML = '<option>Loading...</option>';
    responseSelect.innerHTML = '<option>Loading...</option>';

    try {
        const response = await fetch(`/templates/${location}`);
        const data = await response.json();

        // Cache the result
        templateCache.set(location, data);

        populateTemplateSelects(data, selectPrompt, selectResponse);
    } catch (error) {
        console.error("Error updating templates:", error);
        promptSelect.innerHTML = '<option value="">Error</option>';
        responseSelect.innerHTML = '<option value="">Error</option>';
    } finally {
        locationSelect.disabled = false;
    }
}

// --- START: CORRECTED sendMessage FUNCTION ---
async function sendMessage() {
    const userInput = document.getElementById('userInput');
    const message = userInput.value.trim();
    const modelSelect = document.getElementById('modelSelect');

    if ((!message && !selectedFile) || !modelSelect.value) {
        alert('Please enter a message or select a file, and choose a model');
        return;
    }
    if (message && !selectedPromptTemplateSupportsText()) {
        alert('The selected prompt template screens images only and cannot process text, ' +
              'so a caption cannot be sent with the image. Clear the message and send the image on its own.');
        return;
    }

    const location = document.getElementById('locationSelect').value;
    const promptTemplate = document.getElementById('promptTemplate').value;

    // Display user message in chat
    addMessage(message, true, null, selectedFile ? selectedFile.name : null);
    addMessage('', false);

    userInput.value = '';
    document.getElementById('userInput').disabled = true;
    document.getElementById('sendButton').disabled = true;
    document.getElementById('fileButton').disabled = true;

    // --- Step 1: Perform Prompt Analysis First (for UI only) ---
    if (promptTemplate) {
        showPerformanceIndicator('Analyzing prompt...');
        try {
            let analysisBody;
            let analysisHeaders = {}; // Let browser set Content-Type for FormData

            if (selectedFile) {
                analysisBody = new FormData();
                analysisBody.append('file', selectedFile);
                analysisBody.append('promptTemplate', promptTemplate);
                analysisBody.append('location', location);
            } else {
                analysisBody = JSON.stringify({
                    prompt: message,
                    promptTemplate: promptTemplate,
                    location: location
                });
                analysisHeaders['Content-Type'] = 'application/json';
            }

            const analysisResponse = await fetch('/analyze_prompt', {
                method: 'POST',
                headers: analysisHeaders,
                body: analysisBody
            });

            const analysisData = await analysisResponse.json();
            if (!analysisResponse.ok) {
                throw new Error(analysisData.error || `HTTP error! status: ${analysisResponse.status}`);
            }

            if (analysisData.prompt_analysis) {
                updateVisualAnalysis('prompt', analysisData.prompt_analysis);
            }
        } catch (error) {
            const promptContent = document.getElementById('promptAnalysisContent');
            promptContent.innerHTML = `<div class="analysis-card fail"><span class="analysis-icon">❌</span><span class="analysis-text">Prompt analysis failed: ${escapeHtml(error.message)}</span></div>`;
        }
    }

    // --- Step 2: Make the main call to the /chat endpoint ---
    showPerformanceIndicator('Waiting for LLM response...');
    try {
        let chatBody;
        let chatHeaders = {}; // Let browser set Content-Type for FormData

        if (selectedFile) {
            chatBody = new FormData();
            chatBody.append('file', selectedFile);
            chatBody.append('prompt', message);
            chatBody.append('model', modelSelect.value);
            chatBody.append('location', location);
            // *** THE FIX: Always send the prompt template so the server can re-validate ***
            chatBody.append('promptTemplate', document.getElementById('promptTemplate').value);
            chatBody.append('responseTemplate', document.getElementById('responseTemplate').value);
            chatBody.append('defaultResponse', document.getElementById('defaultResponse').value.trim());
            chatBody.append('useDefaultResponse', document.getElementById('useDefaultResponse').checked);
            chatBody.append('systemInstruction', currentSystemInstruction);
        } else {
            chatBody = JSON.stringify({
                prompt: message,
                model: modelSelect.value,
                location: location,
                // *** THE FIX: Always send the prompt template so the server can re-validate ***
                promptTemplate: document.getElementById('promptTemplate').value,
                responseTemplate: document.getElementById('responseTemplate').value,
                defaultResponse: document.getElementById('defaultResponse').value.trim(),
                useDefaultResponse: document.getElementById('useDefaultResponse').checked,
                systemInstruction: currentSystemInstruction
            });
            chatHeaders['Content-Type'] = 'application/json';
        }

        const chatResponse = await fetch('/chat', {
            method: 'POST',
            headers: chatHeaders,
            body: chatBody
        });

        const chatData = await chatResponse.json();
        if (!chatResponse.ok) {
            throw new Error(chatData.error || `HTTP error! status: ${chatResponse.status}`);
        }

        updateBotMessage(chatData.response, chatData.source,
            chatData.model_armor.prompt_analysis && chatData.model_armor.prompt_analysis.redacted_image);
        // The prompt analysis from the /chat call is now the definitive one.
        if (chatData.model_armor.prompt_analysis) {
            updateVisualAnalysis('prompt', chatData.model_armor.prompt_analysis);
        }
        updateVisualAnalysis('response', chatData.model_armor.response_analysis);

    } catch (error) {
        updateBotMessage(`Error: ${error.message}`, 'system');
    } finally {
        // Final cleanup
        document.getElementById('userInput').disabled = false;
        document.getElementById('sendButton').disabled = false;
        document.getElementById('fileButton').disabled = false;
        userInput.focus();

        selectedFile = null;
        document.getElementById('fileInput').value = '';
        document.getElementById('filePreview').style.display = 'none';

        hidePerformanceIndicator();
    }
}
// --- END: CORRECTED sendMessage FUNCTION ---


function updateVisualAnalysis(type, analysis) {
    const contentContainer = document.getElementById(`${type}AnalysisContent`);
    const rawOutputContainer = document.getElementById(`${type}RawOutput`);
    const rawOutputToggle = rawOutputContainer.previousElementSibling;

    if (!analysis) {
        contentContainer.innerHTML = '<p>No analysis performed.</p>';
        rawOutputToggle.style.display = 'none';
        rawOutputContainer.style.display = 'none';
        return;
    }

    contentContainer.innerHTML = '';

    // The server normalises every Model Armor result (text, document and
    // image) into filter_details, so there is one shape to render here.
    const cards = analysis.filter_details || [];
    if (!cards.length) {
        contentContainer.innerHTML = '<p>No filter results returned.</p>';
    } else {
        const icons = { pass: '\u2714\ufe0f', fail: '\u274c', skipped: '\u26a0\ufe0f' };
        const fallbackTitle = {
            pass: 'No match found',
            fail: 'No specific details available',
            skipped: 'This filter did not run, so the content was not checked'
        };
        cards.forEach(card => {
            const div = document.createElement('div');
            div.className = `analysis-card ${card.status}`;
            // Details can echo attacker-controlled text (matched URLs, OCR
            // output), so set it as a property rather than building HTML.
            div.title = card.details || fallbackTitle[card.status] || '';

            const icon = document.createElement('span');
            icon.className = 'analysis-icon';
            icon.textContent = icons[card.status] || '\u2753';

            const text = document.createElement('span');
            text.className = 'analysis-text';
            const scope = card.scope ? ` (${card.scope})` : '';
            text.textContent = `${card.label}${scope} - ${card.status.toUpperCase()}`;

            div.appendChild(icon);
            div.appendChild(text);
            contentContainer.appendChild(div);
        });
    }

    if (analysis.filter_version) {
        const version = document.createElement('p');
        version.className = 'filter-version';
        version.textContent = `Filter model: ${analysis.filter_version}`;
        contentContainer.appendChild(version);
    }

    rawOutputContainer.textContent = analysis.raw_output;
    rawOutputToggle.style.display = 'block';
    rawOutputToggle.textContent = 'Show Raw Output';
    rawOutputContainer.style.display = 'none';
}

function toggleRawOutput(type) {
    const rawOutput = document.getElementById(`${type}RawOutput`);
    const toggle = rawOutput.previousElementSibling;
    if (rawOutput.style.display !== 'block') {
        rawOutput.style.display = 'block';
        toggle.textContent = 'Hide Raw Output';
    } else {
        rawOutput.style.display = 'none';
        toggle.textContent = 'Show Raw Output';
    }
}

function clearChat() {
    document.getElementById('chatContainer').innerHTML = '';
    const promptContent = document.getElementById('promptAnalysisContent');
    const responseContent = document.getElementById('responseAnalysisContent');
    promptContent.innerHTML = '<p>Select a prompt template to see analysis.</p>';
    responseContent.innerHTML = '<p>Select a response template to see analysis.</p>';
    document.querySelector('#promptAnalysis .raw-output-toggle').style.display = 'none';
    document.querySelector('#responseAnalysis .raw-output-toggle').style.display = 'none';
}

// Resizable panels functionality

const templateCustomizationModal = new bootstrap.Modal(document.getElementById('templateCustomizationModal'));

document.getElementById('editPromptTemplateBtn').addEventListener('click', () => {
    openCustomizationModal('prompt');
});

document.getElementById('editResponseTemplateBtn').addEventListener('click', () => {
    openCustomizationModal('response');
});

function toggleSdpFields() {
    const mode = document.getElementById('customSdpMode').value;
    const advancedFields = document.getElementById('sdpAdvancedFields');
    const imageWarning = document.getElementById('sdpImageWarning');

    advancedFields.style.display = mode === 'Advanced' ? 'block' : 'none';
    updateSdpImageWarning();
}

// Only a de-identify template breaks image screening; inspect alone is fine.
function updateSdpImageWarning() {
    const warning = document.getElementById('sdpImageWarning');
    if (!warning) return;
    const advanced = document.getElementById('customSdpMode').value === 'Advanced';
    const deidentify = document.getElementById('customDeidentifyTemplate');
    const option = deidentify && deidentify.options[deidentify.selectedIndex];
    const kind = option && option.value ? (option.dataset.kind || 'text') : '';
    const isPromptTemplate = (document.getElementById('customTemplateName').value || '')
        .endsWith('-prompt');
    // A DLP de-identify config is either text or image, never both. Say which
    // kind of input this template will therefore refuse.
    let text = '';
    if (advanced && kind === 'text' && isPromptTemplate) {
        text = 'This is a TEXT de-identify template (infoTypeTransformations). Text prompts are ' +
               'redacted; an uploaded image cannot be processed and the request is blocked. ' +
               'To screen images with this template, set SDP Mode to Basic (or clear De-identify). ' +
               'For image redaction use the US Image Redaction template.';
    } else if (advanced && kind === 'image') {
        text = 'This is an IMAGE de-identify template (imageTransformations). Images are redacted ' +
               'and the redacted copy is what reaches the model; text prompts and captions cannot ' +
               'be processed and are blocked, so this belongs on an image-only prompt template.' +
               (isPromptTemplate ? '' : ' A response template is always text, so it will fail here.');
    }
    warning.textContent = text;
    warning.style.display = text ? 'block' : 'none';
}

async function openCustomizationModal(type) {
    const selectElement = document.getElementById(`${type}Template`);
    const templateName = selectElement.value;
    const location = document.getElementById('locationSelect').value;

    if (!templateName) {
        alert('Please select a template to edit');
        return;
    }

    if (!templateName.startsWith('modelarmor-demo-')) {
        alert('Only dedicated demo templates can be modified');
        return;
    }

    document.getElementById('customTemplateName').value = templateName;
    document.getElementById('customLocation').value = location;

    // Try to find the current config from the template cache
    const cachedData = templateCache.get(location);
    if (cachedData) {
        const templates = type === 'prompt' ? cachedData.prompt_templates : cachedData.response_templates;
        const template = templates.find(t => t.name === templateName);
        if (template && template.config) {
            const config = template.config;
            
            const hasPiJb = config.detection_filters && config.detection_filters.some(f => f.includes('Prompt Injection & Jailbreak'));
            document.getElementById('customPiJailbreak').value = hasPiJb ? 'ENABLED' : 'DISABLED';
            
            const versionSelect = document.getElementById('customFilterVersion');
            const currentVersion = config.other_settings?.filter_version || 'FILTER_VERSION_ALIAS_LATEST';
            if (![...versionSelect.options].some(o => o.value === currentVersion)) {
                // keep whatever the template has, even if it is not one we offer
                const extra = document.createElement('option');
                extra.value = currentVersion; extra.textContent = currentVersion;
                versionSelect.appendChild(extra);
            }
            versionSelect.value = currentVersion;

            const piJbConfidence = config.other_settings?.pi_jb_confidence
                || 'DETECTION_CONFIDENCE_LEVEL_UNSPECIFIED';
            document.getElementById('customPiJailbreakConfidence').value = piJbConfidence;
            
            const hasMalUrls = config.detection_filters && config.detection_filters.some(f => f.includes('Malicious URL'));
            document.getElementById('customMaliciousUrls').value = hasMalUrls ? 'ENABLED' : 'DISABLED';

            // Populate RAI filters
            if (config.rai_filters_structured) {
                const raiFilters = document.querySelectorAll('.rai-filter');
                raiFilters.forEach(select => {
                    const type = select.getAttribute('data-type');
                    // filter_type and confidence_level are the API's enum names.
                    const filter = config.rai_filters_structured.find(
                        f => String(f.filter_type) === type);
                    select.value = filter ? String(filter.confidence_level)
                                          : 'DETECTION_CONFIDENCE_LEVEL_UNSPECIFIED';
                });
            }

            // Populate SDP settings
            let sdpMode = config.sdp_settings?.mode || 'None';
            if (sdpMode === 'Advanced (DLP)') sdpMode = 'Advanced';
            document.getElementById('customSdpMode').value = sdpMode;
            

            
            await loadDlpTemplates(
                document.getElementById('locationSelect').value,
                config.sdp_settings?.inspect_template || '',
                config.sdp_settings?.deidentify_template || ''
            );
            
            toggleSdpFields();
        }
    }

    templateCustomizationModal.show();
}

document.getElementById('saveTemplateCustomization').addEventListener('click', async () => {
    const templateName = document.getElementById('customTemplateName').value;
    const location = document.getElementById('customLocation').value;
    const piJailbreak = document.getElementById('customPiJailbreak').value;
    const maliciousUrls = document.getElementById('customMaliciousUrls').value;
    const piJbConfidence = document.getElementById('customPiJailbreakConfidence').value;

    const raiFilters = [];
    document.querySelectorAll('.rai-filter').forEach(select => {
        raiFilters.push({
            filter_type: select.getAttribute('data-type'),
            confidence_level: select.value
        });
    });

    const sdpMode = document.getElementById('customSdpMode').value;
    const sdpSettings = {
        mode: sdpMode
    };

    if (sdpMode === 'Advanced') {
        sdpSettings.inspect_template = document.getElementById('customInspectTemplate').value;
        sdpSettings.deidentify_template = document.getElementById('customDeidentifyTemplate').value;
    }

    const config = {
        filter_version: document.getElementById('customFilterVersion').value,
        pi_and_jailbreak: piJailbreak,
        malicious_uris: maliciousUrls,
        pi_jb_confidence: piJbConfidence,
        rai_filters: raiFilters,
        sdp_settings: sdpSettings
    };

    try {
        const response = await fetch('/update_template', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                templateName: templateName,
                location: location,
                config: config
            })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }

        alert(data.message);
        // templateCustomizationModal.hide(); // Keep modal open after save
        
        // Refresh templates to get updated config
        templateCache.delete(location);
        updateTemplatesForLocation(templateName);
        
    } catch (error) {
        alert(`Error updating template: ${error.message}`);
    }
});

// The location list is rendered by the server, so read the default from it
// rather than hardcoding a region that may no longer be offered.
function availableLocations() {
    return Array.from(document.getElementById('locationSelect').options)
        .map(option => option.value)
        .filter(Boolean);
}

// Event wiring that used to live in inline on* attributes. Kept here so the
// page can ship a Content-Security-Policy without 'unsafe-inline'.
function bindStaticHandlers() {
    document.getElementById('modelSelect').addEventListener('change', checkFileSupport);
    document.getElementById('locationSelect').addEventListener('change', () => debouncedUpdateTemplates());
    document.getElementById('customSdpMode').addEventListener('change', toggleSdpFields);
    document.getElementById('customInspectTemplate').addEventListener('change',
        () => updateDlpHover('customInspectTemplate'));
    document.getElementById('customDeidentifyTemplate').addEventListener('change', () => {
        updateDlpHover('customDeidentifyTemplate');
        updateSdpImageWarning();
    });
    document.getElementById('promptTemplate').addEventListener('change', updateTemplateInfoIcons);
    document.getElementById('responseTemplate').addEventListener('change', updateTemplateInfoIcons);
    document.getElementById('promptRawOutputToggle').addEventListener('click', () => toggleRawOutput('prompt'));
    document.getElementById('responseRawOutputToggle').addEventListener('click', () => toggleRawOutput('response'));
}

window.onload = () => {
    bindStaticHandlers();
    const locationSelect = document.getElementById('locationSelect');
    const locations = availableLocations();
    locationSelect.value = locations.includes('us') ? 'us' : (locations[0] || '');
    updateTemplatesForLocation();
    checkFileSupport();
    preloadTemplates();
};
