/**
 * Enhanced Editor JavaScript
 * Handles all the interactive functionality for the editor page
 */

// Global state
const state = {
    parameters: {},
    metadata: null,
    currentPreview: 'template',
    parameterValues: {},
    mode: 'basic',
    imageType: 'auto'
};

// DOM Elements
const elements = {};

// Initialize the editor
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    cacheElements();
    
    // Initialize the parameter UI
    initializeParameterUI();
    
    // Set up event handlers
    setupEventHandlers();
    
    // Add preview button
    addPreviewButton();
});

// Cache frequently used DOM elements
function cacheElements() {
    elements.parameterContainer = document.getElementById('parameter-container');
    elements.previewImage = document.getElementById('preview-image');
    elements.previewTabs = document.querySelectorAll('.preview-tab');
    elements.updatePreviewBtn = document.getElementById('update-preview-btn');
    elements.processBtn = document.getElementById('process-btn');
    elements.analyzeBtn = document.getElementById('analyze-btn');
    elements.loadPresetBtn = document.getElementById('load-preset-btn');
    elements.savePresetBtn = document.getElementById('save-preset-btn');
    elements.presetSelector = document.getElementById('preset-selector');
    elements.presetName = document.getElementById('preset-name');
    elements.presetDescription = document.getElementById('preset-description');
    elements.imageTypeSelector = document.getElementById('image-type');
    elements.modeToggle = document.querySelectorAll('.toggle-button');
    elements.loadingIndicator = document.getElementById('loading');
    elements.messageContainer = document.getElementById('message-container');
    elements.analysisPanel = document.getElementById('analysis-panel');
    elements.analysisContent = document.getElementById('analysis-content');
}

// Set up event handlers
function setupEventHandlers() {
    // Preview tab switching
    elements.previewTabs.forEach(tab => {
        tab.addEventListener('click', handlePreviewTabClick);
    });
    
    // Update preview button
    elements.updatePreviewBtn.addEventListener('click', handleUpdatePreview);
    
 
    
    // Analyze settings button
    elements.analyzeBtn.addEventListener('click', handleAnalyzeSettings);
    
    // Load preset button
    elements.loadPresetBtn.addEventListener('click', handleLoadPreset);
    
    // Save preset button
    elements.savePresetBtn.addEventListener('click', handleSavePreset);
    
    // Image type selector
    elements.imageTypeSelector.addEventListener('change', handleImageTypeChange);
    
    // Mode toggle
    elements.modeToggle.forEach(button => {
        button.addEventListener('click', handleModeToggle);
    });
}

// Initialize the parameter UI
function initializeParameterUI() {
    // Fetch parameter metadata from the page data
    const metadataElement = document.getElementById('parameter-metadata');
    if (metadataElement) {
        state.metadata = JSON.parse(metadataElement.textContent);
    } else {
        // If not embedded in page, use hardcoded metadata
        fetchParameterMetadata().then(metadata => {
            state.metadata = metadata;
            buildParameterControls();
        });
    }
    
    // Build controls if metadata is available
    if (state.metadata) {
        buildParameterControls();
    }
}

// Fetch parameter metadata from API
async function fetchParameterMetadata() {
    try {
        const response = await fetch('/enhanced/api/parameters');
        const data = await response.json();
        return data.metadata;
    } catch (error) {
        showMessage('Error loading parameter metadata', 'error');
        console.error('Error fetching parameter metadata:', error);
        return {};
    }
}

// Build parameter controls based on metadata
function buildParameterControls() {
    if (!state.metadata || !state.metadata.categories || !state.metadata.parameters) {
        console.error('Invalid parameter metadata');
        return;
    }
    
    // Clear container
    elements.parameterContainer.innerHTML = '';
    
    // Get categories and sort by order
    const categories = Object.entries(state.metadata.categories)
        .sort((a, b) => a[1].order - b[1].order)
        .map(entry => ({
            id: entry[0],
            ...entry[1]
        }));
    
    // Create a group for each category
    categories.forEach(category => {
        // Create parameter group from template
        const template = document.getElementById('parameter-group-template');
        const groupNode = document.importNode(template.content, true).firstElementChild;
        
        // Set group properties
        groupNode.dataset.category = category.id;
        groupNode.querySelector('h3').textContent = category.name;
        
        // Get parameters for this category
        const parameters = Object.entries(state.metadata.parameters)
            .filter(([_, param]) => param.category === category.id)
            .map(entry => ({
                id: entry[0],
                ...entry[1]
            }));
        
        // Add parameters to group
        const parametersContainer = groupNode.querySelector('.parameters');
        parameters.forEach(param => {
            const paramNode = createParameterControl(param);
            if (paramNode) {
                parametersContainer.appendChild(paramNode);
            }
        });
        
        // Add group to container
        elements.parameterContainer.appendChild(groupNode);
    });
    
    // Initialize parameter values
    updateControlVisibility();
}

// Create control for a parameter
function createParameterControl(parameter) {
    // Create parameter element from template
    const template = document.getElementById('parameter-template');
    const paramNode = document.importNode(template.content, true).firstElementChild;
    
    // Set parameter properties
    paramNode.dataset.paramName = parameter.id;
    paramNode.querySelector('.parameter-title').textContent = parameter.name;
    paramNode.querySelector('.parameter-description').textContent = parameter.description;
    
    // Create appropriate control based on UI control type
    const controlContainer = paramNode.querySelector('.parameter-control');
    const uiControl = parameter.ui_control || 'slider';
    
    // Get validation info
    const validation = state.metadata.validation ? state.metadata.validation[parameter.id] : null;
    
    // Default value
    const defaultValue = parameter.default;
    
    // Set parameter value in state
    if (defaultValue !== undefined) {
        state.parameterValues[parameter.id] = defaultValue;
    }
    
    // Create appropriate control
    switch (uiControl) {
        case 'slider':
            createSliderControl(controlContainer, parameter, validation);
            break;
            
        case 'dropdown':
            createDropdownControl(controlContainer, parameter);
            break;
            
        case 'checkbox':
            createCheckboxControl(controlContainer, parameter);
            break;
            
        case 'color':
            createColorControl(controlContainer, parameter);
            break;
            
        default:
            // Text input as fallback
            createTextInputControl(controlContainer, parameter, validation);
    }
    
    // Add complexity/impact indicators for advanced mode
    if (parameter.visual_impact || parameter.complexity_impact) {
        const impactDiv = document.createElement('div');
        impactDiv.className = 'parameter-impact';
        impactDiv.innerHTML = `<span class="impact-visual" title="Visual Impact: ${parameter.visual_impact || 0}/5">👁️ ${parameter.visual_impact || 0}</span>
                              <span class="impact-complexity" title="Complexity Impact: ${parameter.complexity_impact || 0}/5">🧩 ${parameter.complexity_impact || 0}</span>`;
        paramNode.appendChild(impactDiv);
        impactDiv.classList.add('advanced-only');
    }
    
    // Add tooltip if there's a long description
    if (parameter.long_description) {
        const tooltipTrigger = document.createElement('span');
        tooltipTrigger.className = 'tooltip-trigger';
        tooltipTrigger.textContent = '?';
        
        const tooltipContent = document.createElement('div');
        tooltipContent.className = 'tooltip-content';
        tooltipContent.textContent = parameter.long_description;
        
        paramNode.querySelector('.parameter-title').appendChild(tooltipTrigger);
        paramNode.querySelector('.parameter-title').appendChild(tooltipContent);
    }
    
    return paramNode;
}

// Create slider control
function createSliderControl(container, parameter, validation) {
    // Get range information
    let min = 0, max = 100, step = 1;
    
    if (validation) {
        if (validation.type === 'int' || validation.type === 'float') {
            min = validation.min !== undefined ? validation.min : min;
            max = validation.max !== undefined ? validation.max : max;
            step = validation.type === 'float' ? 0.1 : 1;
        }
    }
    
    // Default value
    let value = parameter.default !== undefined ? parameter.default : (min + (max - min) / 2);
    
    // Set value in state
    state.parameterValues[parameter.id] = value;
    
    // Create control
    const sliderHTML = `
        <div class="slider-container">
            <div class="slider-top">
                <input type="range" id="${parameter.id}" min="${min}" max="${max}" 
                       step="${step}" value="${value}">
                <span class="value-display">${value}</span>
            </div>
            <div class="range-values">
                <span class="min-value">${min}</span>
                <span class="max-value">${max}</span>
            </div>
        </div>
    `;
    
    container.innerHTML = sliderHTML;
    
    // Add event listener
    const slider = container.querySelector('input[type="range"]');
    const valueDisplay = container.querySelector('.value-display');
    
    slider.addEventListener('input', function() {
        const value = parseFloat(this.value);
        valueDisplay.textContent = value;
        state.parameterValues[parameter.id] = value;
    });
}

// Create dropdown control
function createDropdownControl(container, parameter) {
    // Get options
    let options = {};
    
    if (parameter.options) {
        options = parameter.options;
    } else if (parameter.values) {
        // Simple array of values
        parameter.values.forEach(val => {
            options[val] = val;
        });
    }
    
    // Default value
    let value = parameter.default !== undefined ? parameter.default : Object.keys(options)[0];
    
    // Set value in state
    state.parameterValues[parameter.id] = value;
    
    // Create options HTML
    let optionsHTML = '';
    for (const [key, label] of Object.entries(options)) {
        const selected = key === value ? 'selected' : '';
        optionsHTML += `<option value="${key}" ${selected}>${label}</option>`;
    }
    
    // Create control
    const dropdownHTML = `
        <select id="${parameter.id}">
            ${optionsHTML}
        </select>
    `;
    
    container.innerHTML = dropdownHTML;
    
    // Add event listener
    const dropdown = container.querySelector('select');
    dropdown.addEventListener('change', function() {
        state.parameterValues[parameter.id] = this.value;
        updateControlVisibility();
    });
}

// Create checkbox control
function createCheckboxControl(container, parameter) {
    // Default value
    let checked = parameter.default !== undefined ? parameter.default : false;
    
    // Set value in state
    state.parameterValues[parameter.id] = checked;
    
    // Create control
    const checkboxHTML = `
        <input type="checkbox" id="${parameter.id}" ${checked ? 'checked' : ''}>
    `;
    
    container.innerHTML = checkboxHTML;
    
    // Add event listener
    const checkbox = container.querySelector('input[type="checkbox"]');
    checkbox.addEventListener('change', function() {
        state.parameterValues[parameter.id] = this.checked;
        updateControlVisibility();
    });
}

// Create text input control
function createTextInputControl(container, parameter, validation) {
    // Default value
    let value = parameter.default !== undefined ? parameter.default : '';
    
    // Set value in state
    state.parameterValues[parameter.id] = value;
    
    // Input type
    let inputType = 'text';
    if (validation && validation.type === 'int' || validation.type === 'float') {
        inputType = 'number';
    }
    
    // Create control
    const inputHTML = `
        <input type="${inputType}" id="${parameter.id}" value="${value}">
    `;
    
    container.innerHTML = inputHTML;
    
    // Add event listener
    const input = container.querySelector('input');
    input.addEventListener('change', function() {
        let value = this.value;
        
        // Convert to number if needed
        if (inputType === 'number') {
            value = parseFloat(value);
        }
        
        state.parameterValues[parameter.id] = value;
    });
}

// Create color picker control
function createColorControl(container, parameter) {
    // Default value
    let value = parameter.default !== undefined ? parameter.default : '#000000';
    
    // Set value in state
    state.parameterValues[parameter.id] = value;
    
    // Create control
    const colorHTML = `
        <input type="color" id="${parameter.id}" value="${value}">
    `;
    
    container.innerHTML = colorHTML;
    
    // Add event listener
    const colorPicker = container.querySelector('input[type="color"]');
    colorPicker.addEventListener('change', function() {
        state.parameterValues[parameter.id] = this.value;
    });
}

// Update which controls are visible based on dependencies
function updateControlVisibility() {
    // Check each parameter for dependencies
    for (const [paramName, paramInfo] of Object.entries(state.metadata.parameters)) {
        // Find the parameter element
        const paramElement = document.querySelector(`.parameter[data-param-name="${paramName}"]`);
        if (!paramElement) continue;
        
        // Check if this parameter depends on another parameter
        if (paramInfo.depends_on) {
            const dependsOn = paramInfo.depends_on;
            const dependsOnValue = state.parameterValues[dependsOn];
            
            // If the dependency is a boolean (checkbox)
            if (typeof dependsOnValue === 'boolean') {
                paramElement.style.display = dependsOnValue ? 'block' : 'none';
            }
            // If the dependency is a specific value
            else if (paramInfo.depends_on_value) {
                paramElement.style.display = 
                    dependsOnValue === paramInfo.depends_on_value ? 'block' : 'none';
            }
        }
        
        // Handle mode visibility (basic vs advanced)
        const isAdvancedParam = paramInfo.visual_impact > 3 || 
                                paramInfo.complexity_impact > 3 || 
                                paramInfo.advanced === true;
        
        if (state.mode === 'basic' && isAdvancedParam) {
            paramElement.style.display = 'none';
        } else if (paramElement.style.display === 'none' && 
                  !paramInfo.depends_on &&
                  !paramInfo.depends_on_value) {
            paramElement.style.display = 'block';
        }
        
        // Show/hide advanced elements
        const advancedElements = paramElement.querySelectorAll('.advanced-only');
        advancedElements.forEach(el => {
            el.style.display = state.mode === 'advanced' ? 'block' : 'none';
        });
    }
}

// Handle clicking on preview tabs
function handlePreviewTabClick(event) {
    // Update active tab
    elements.previewTabs.forEach(tab => tab.classList.remove('active'));
    event.target.classList.add('active');
    
    // Get preview type
    const previewType = event.target.dataset.preview;
    state.currentPreview = previewType;
    
    // Update preview
    updatePreview();
}

// Handle Update Preview button click
async function handleUpdatePreview() {
    showLoading(true);
    try {
        await updatePreview();
    } finally {
        showLoading(false);
    }
}


// Handle Analyze Settings button click
async function handleAnalyzeSettings() {
    try {
        // Send analysis request
        const response = await fetch('/enhanced/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                settings: state.parameterValues
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Show analysis panel
            elements.analysisPanel.classList.remove('hidden');
            
            // Display analysis
            const analysis = data.analysis;
            const analysisHTML = `
                <div class="analysis-result">
                    <h4>Complexity Score: ${analysis.complexity_score}%</h4>
                    <div class="progress">
                        <div class="progress-bar" style="width: ${analysis.complexity_score}%"></div>
                    </div>
                    <p>Complexity Level: <strong>${analysis.complexity_level}</strong></p>
                    
                    <h4>Key Factors:</h4>
                    <ul>
                        ${analysis.key_factors.map(factor => `<li>${factor}</li>`).join('')}
                    </ul>
                </div>
            `;
            
            elements.analysisContent.innerHTML = analysisHTML;
        } else {
            showMessage(data.error || 'Failed to analyze settings', 'error');
        }
    } catch (error) {
        console.error('Error analyzing settings:', error);
        showMessage('Error analyzing settings: ' + error.message, 'error');
    }
}

// Handle Load Preset button click
async function handleLoadPreset() {
    const presetName = elements.presetSelector.value;
    if (!presetName) return;
    
    try {
        // Send load preset request
        const response = await fetch(`/enhanced/api/presets/load/${encodeURIComponent(presetName)}`);
        const data = await response.json();
        
        if (data.success) {
            // Update parameter values
            state.parameterValues = data.settings;
            
            // Update UI controls
            updateUIFromSettings();
            
            showMessage(`Preset "${presetName}" loaded successfully`, 'success');
        } else {
            showMessage(data.error || 'Failed to load preset', 'error');
        }
    } catch (error) {
        console.error('Error loading preset:', error);
        showMessage('Error loading preset: ' + error.message, 'error');
    }
}

// Handle Save Preset button click
async function handleSavePreset() {
    const presetName = elements.presetName.value.trim();
    const description = elements.presetDescription.value.trim();
    
    if (!presetName) {
        showMessage('Please enter a preset name', 'error');
        return;
    }
    
    try {
        // Send save preset request
        const response = await fetch('/enhanced/api/presets/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: presetName,
                description: description,
                settings: state.parameterValues
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showMessage(data.message, 'success');
            
            // Clear inputs
            elements.presetName.value = '';
            elements.presetDescription.value = '';
            
            // Add to preset dropdown
            const option = document.createElement('option');
            option.value = presetName;
            option.textContent = presetName;
            elements.presetSelector.appendChild(option);
            elements.presetSelector.value = presetName;
        } else {
            showMessage(data.error || 'Failed to save preset', 'error');
        }
    } catch (error) {
        console.error('Error saving preset:', error);
        showMessage('Error saving preset: ' + error.message, 'error');
    }
}

// Handle Image Type change
function handleImageTypeChange() {
    state.imageType = elements.imageTypeSelector.value;
}

// Handle Mode Toggle (Basic/Advanced)
function handleModeToggle(event) {
    // Update active toggle
    elements.modeToggle.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    // Set mode
    state.mode = event.target.dataset.mode;
    
    // Update control visibility
    updateControlVisibility();
}

// Update UI controls from settings
function updateUIFromSettings() {
    // Update each control with value from settings
    for (const [paramName, value] of Object.entries(state.parameterValues)) {
        const paramElement = document.querySelector(`.parameter[data-param-name="${paramName}"]`);
        if (!paramElement) continue;
        
        const control = paramElement.querySelector(`#${paramName}`);
        if (!control) continue;
        
        // Update based on control type
        if (control.type === 'checkbox') {
            control.checked = Boolean(value);
        } 
        else if (control.type === 'range') {
            control.value = value;
            const valueDisplay = paramElement.querySelector('.value-display');
            if (valueDisplay) valueDisplay.textContent = value;
        }
        else if (control.tagName === 'SELECT') {
            control.value = value;
        }
        else {
            control.value = value;
        }
    }
    
    // Update control visibility based on new values
    updateControlVisibility();
}

// Show/hide loading indicator
function showLoading(show) {
    elements.loadingIndicator.classList.toggle('hidden', !show);
}

// Show message
function showMessage(message, type = 'info') {
    // Create message element
    const messageElement = document.createElement('div');
    messageElement.className = `alert alert-${type}`;
    messageElement.textContent = message;
    
    // Add close button
    const closeButton = document.createElement('button');
    closeButton.className = 'close-btn';
    closeButton.innerHTML = '&times;';
    closeButton.addEventListener('click', () => messageElement.remove());
    messageElement.appendChild(closeButton);
    
    // Add to container
    elements.messageContainer.appendChild(messageElement);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (messageElement.parentNode === elements.messageContainer) {
            messageElement.remove();
        }
    }, 5000);
}

// Process image button click handler
document.addEventListener('DOMContentLoaded', function() {
    // Find the Process Image button
    const processBtn = document.getElementById('process-btn');
    if (processBtn) {
        processBtn.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Show loading indicator
            const loadingIndicator = document.getElementById('processing-indicator');
            if (loadingIndicator) {
                loadingIndicator.classList.remove('hidden');
            }
            
            // Collect form settings
            const form = document.querySelector('form');
            const formData = new FormData(form);
            const settings = {};
            
            // Convert form data to JSON
            formData.forEach((value, key) => {
                settings[key] = value;
            });
            
            // Add parameter to stay on page
            settings.stay_on_page = true;
            
            // Send request to backend
            fetch('/enhanced/api/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                if (loadingIndicator) {
                    loadingIndicator.classList.add('hidden');
                }
                
                if (data.success) {
                    // Show preview grid after processing
                    displayPreviewGrid();
                } else {
                    alert('Error processing image: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                if (loadingIndicator) {
                    loadingIndicator.classList.add('hidden');
                }
                console.error('Error:', error);
                alert('An error occurred during processing');
            });
        });
    }
});

// Preview grid display function
function displayPreviewGrid() {
    // Get timestamp from uploaded image
    const uploadedImg = document.getElementById('uploaded-image');
    if (!uploadedImg) return;
    
    const imgSrc = uploadedImg.src;
    const parts = imgSrc.split('_');
    if (parts.length < 2) return;
    
    const timestamp = parts[1].split('.')[0];
    
    // Create or clear grid container
    const container = document.getElementById('preview-grid-container');
    if (!container) return;
    
    container.innerHTML = '';
    container.classList.remove('hidden');
    
    // Add heading
    const heading = document.createElement('h3');
    heading.textContent = 'Processing Stages';
    container.appendChild(heading);
    
    // Create grid for previews
    const grid = document.createElement('div');
    grid.className = 'preview-items';
    
    // Add preview items
    const previewTypes = [
        { name: 'Original', url: imgSrc },
        { name: 'Preprocessed', url: `/static/previews/preview_${timestamp}_preprocessed.jpg?t=${Date.now()}` },
        { name: 'Segments', url: `/static/previews/preview_${timestamp}_segments.jpg?t=${Date.now()}` },
        { name: 'Edges', url: `/static/previews/preview_${timestamp}_edges.jpg?t=${Date.now()}` },
        { name: 'Template', url: `/static/previews/preview_${timestamp}_template.jpg?t=${Date.now()}` }
    ];
    
    previewTypes.forEach(item => {
        const previewItem = document.createElement('div');
        previewItem.className = 'preview-item';
        
        const title = document.createElement('h4');
        title.textContent = item.name;
        
        const img = document.createElement('img');
        img.src = item.url;
        img.alt = item.name;
        img.className = 'preview-image';
        
        previewItem.appendChild(title);
        previewItem.appendChild(img);
        grid.appendChild(previewItem);
    });
    
    container.appendChild(grid);
    
    // Add continue button
    const continueBtn = document.createElement('button');
    continueBtn.className = 'btn btn-primary continue-btn';
    continueBtn.textContent = 'Continue to Final Result';
    continueBtn.addEventListener('click', function() {
        window.location.href = '/enhanced/results';
    });
    
    container.appendChild(continueBtn);
    
    // Scroll to preview section
    container.scrollIntoView({behavior: 'smooth'});
}