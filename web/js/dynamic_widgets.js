/**
 * JiekouAI ComfyUI Plugin - Dynamic Widgets
 * Dynamically renders model-specific parameters when model selection changes
 * Also provides rich model dropdown with name display (while storing IDs)
 * 
 * IMPORTANT: Dynamic widget values are passed to Python via getExtraWidgetValues hook
 */

import { app } from "../../scripts/app.js";

// Node types that need dynamic parameters
const DYNAMIC_NODES = [
    "JieKouTextToImage",
    "JieKouImageToImage",
    "JieKouVideoGeneration",
    "JieKouTTS"
];

// Node type to model API endpoint mapping
const NODE_MODEL_ENDPOINTS = {
    "JieKouTextToImage": "/jiekou/models/image",
    "JieKouImageToImage": "/jiekou/models/image",
    "JieKouVideoGeneration": "/jiekou/models/video",
    "JieKouTTS": "/jiekou/models"
};

// Parameters that are already defined as static widgets (don't duplicate)
const STATIC_PARAMS = ["model", "prompt", "text", "image_url", "reference_video_urls", "_dynamic_params", "save_to_disk"];

// Models that support reference_video_urls parameter
const MODELS_WITH_REFERENCE_VIDEO = ["wan2.6_v2v"];

/**
 * Get extra widget values for a node (used for passing dynamic params to Python)
 * This function is called by our hook during graph serialization
 */
function getJiekouDynamicValues(node) {
    const values = {};
    if (!node.widgets) return values;
    
    for (const widget of node.widgets) {
        if (widget._jiekouDynamic && widget.name !== "doc_link") {
            values[widget.name] = widget.value;
        }
    }
    return values;
}

// Cache for model schemas
const schemaCache = new Map();

// Cache for model info: id -> { name, description }
const modelInfoCache = new Map();

// T027: Cache for model configs (from /jiekou/models with valid_combinations)
let fullModelConfigsCache = null;

/**
 * Fetch model list and populate info cache
 */
async function fetchModelList(nodeType) {
    const endpoint = NODE_MODEL_ENDPOINTS[nodeType];
    if (!endpoint) return [];
    
    try {
        const response = await fetch(endpoint);
        if (!response.ok) return [];
        
        const data = await response.json();
        const models = data.models || [];
        
        // Cache model info (including doc_url)
        for (const model of models) {
            modelInfoCache.set(model.id, {
                name: model.name || model.id,
                description: model.description || "",
                doc_url: model.doc_url || ""
            });
        }
        
        return models;
    } catch (error) {
        console.error(`[JieKou] Error fetching models:`, error);
        return [];
    }
}

/**
 * Setup rich model combo widget
 * 
 * Strategy:
 * - widget.options.values: replace with names (for dropdown display)
 * - widget.value: stores name (for display)
 * - Provide helper to get actual model ID
 * - onModelChange callback receives model ID
 */
function setupRichModelCombo(widget, node, onModelChange) {
    // Get original values from widget
    const originalValues = [...(widget.options.values || [])];
    
    // Filter to only keep actual model IDs (those in modelInfoCache)
    // This handles the case where INPUT_TYPES includes both IDs and names for validation
    const originalIds = originalValues.filter(v => modelInfoCache.has(v));
    
    // If no IDs found in cache, fall back to original values
    const idsToUse = originalIds.length > 0 ? originalIds : originalValues;
    
    // Build mappings
    const idToName = new Map();
    const nameToId = new Map();
    
    for (const id of idsToUse) {
        const info = modelInfoCache.get(id);
        const name = info ? info.name : id;
        idToName.set(id, name);
        nameToId.set(name, id);
    }
    
    // Store on widget for reference
    widget._jiekouOriginalIds = idsToUse;
    widget._jiekouIdToName = idToName;
    widget._jiekouNameToId = nameToId;
    
    // Replace options.values with names (for dropdown menu)
    widget.options.values = idsToUse.map(id => idToName.get(id) || id);
    
    // Convert current value (ID) to name for display
    if (widget.value && idToName.has(widget.value)) {
        widget.value = idToName.get(widget.value);
    }
    
    // Set callback that converts name to ID and calls onModelChange
    widget.callback = function(selectedName) {
        // selectedName is from dropdown (a name), convert to ID
        const modelId = nameToId.get(selectedName) || selectedName;
        console.log(`[JieKou] Model selected: ${selectedName} -> ${modelId}`);
        
        // Call the provided callback with the ID
        if (onModelChange) {
            onModelChange(modelId);
        }
    };
    
    // Helper to get the actual model ID from widget.value (which is a name)
    widget._jiekouGetModelId = function() {
        return nameToId.get(this.value) || this.value;
    };
}

/**
 * Fetch model parameter schema from backend
 */
async function fetchModelSchema(modelId) {
    if (schemaCache.has(modelId)) {
        return schemaCache.get(modelId);
    }
    
    try {
        const response = await fetch(`/jiekou/models/${modelId}/schema`);
        if (!response.ok) {
            console.warn(`[JieKou] Failed to fetch schema for ${modelId}`);
            return null;
        }
        const schema = await response.json();
        schemaCache.set(modelId, schema);
        return schema;
    } catch (error) {
        console.error(`[JieKou] Error fetching schema for ${modelId}:`, error);
        return null;
    }
}

/**
 * Create a widget from schema property
 */
function createWidgetFromSchema(node, name, schema) {
    const type = schema.type || "string";
    const defaultValue = schema.default;
    const description = schema.description || name;
    
    let widget = null;
    
    try {
        if (schema.enum && schema.enum.length > 0) {
            // Dropdown/combo for enum values
            widget = node.addWidget("combo", name, defaultValue || schema.enum[0], 
                () => {}, { values: schema.enum });
        } else if (type === "string") {
            // Text input
            widget = node.addWidget("text", name, defaultValue || "", () => {});
        } else if (type === "integer") {
            // Integer input
            const min = schema.minimum ?? -2147483647;
            const max = schema.maximum ?? 2147483647;
            widget = node.addWidget("number", name, defaultValue ?? 0, 
                () => {}, { min, max, step: 1, precision: 0 });
        } else if (type === "number") {
            // Float input
            const min = schema.minimum ?? -1000;
            const max = schema.maximum ?? 1000;
            widget = node.addWidget("number", name, defaultValue ?? 0, 
                () => {}, { min, max, step: 0.1 });
        } else if (type === "boolean") {
            // Toggle
            widget = node.addWidget("toggle", name, defaultValue ?? false, () => {});
        }
        
        if (widget) {
            widget._jiekouDynamic = true;
            widget._jiekouDescription = description;
        }
    } catch (e) {
        console.warn(`[JieKou] Failed to create widget for ${name}:`, e);
    }
    
    return widget;
}

/**
 * Clear all dynamic widgets from a node
 */
function clearDynamicWidgets(node) {
    if (!node.widgets) return;
    
    // Find and remove dynamic widgets
    const dynamicWidgets = node.widgets.filter(w => w._jiekouDynamic);
    for (const widget of dynamicWidgets) {
        const index = node.widgets.indexOf(widget);
        if (index > -1) {
            node.widgets.splice(index, 1);
        }
    }
}

/**
 * Render dynamic widgets based on model schema
 */
async function renderDynamicWidgets(node, modelId) {
    if (!modelId || modelId === "loading...") return;
    
    console.log(`[JieKou] Loading parameters for model: ${modelId}`);
    
    // Clear existing dynamic widgets
    clearDynamicWidgets(node);
    
    // Control visibility of reference_video_urls widget (only for wan2.6_v2v)
    const refVideoWidget = node.widgets?.find(w => w.name === "reference_video_urls");
    if (refVideoWidget) {
        const shouldShow = MODELS_WITH_REFERENCE_VIDEO.includes(modelId);
        refVideoWidget.hidden = !shouldShow;
        // Move to appropriate position
        if (shouldShow) {
            // Remove and re-add to put it in the right order (after prompt)
            const idx = node.widgets.indexOf(refVideoWidget);
            if (idx > -1) {
                node.widgets.splice(idx, 1);
                // Find prompt widget position
                const promptIdx = node.widgets.findIndex(w => w.name === "prompt");
                node.widgets.splice(promptIdx + 1, 0, refVideoWidget);
            }
        }
    }
    
    // Fetch schema
    const schema = await fetchModelSchema(modelId);
    if (!schema || !schema.properties) {
        console.log(`[JieKou] No schema found for ${modelId}`);
        return;
    }
    
    // Create widgets for each property
    let addedCount = 0;
    for (const [propName, propSchema] of Object.entries(schema.properties)) {
        // Skip static params and already existing widgets
        if (STATIC_PARAMS.includes(propName)) continue;
        if (node.widgets?.some(w => w.name === propName && !w._jiekouDynamic)) continue;
        
        const widget = createWidgetFromSchema(node, propName, propSchema);
        if (widget) {
            addedCount++;
        }
    }
    
    // Add documentation link at the bottom (custom drawn, clickable)
    const modelInfo = modelInfoCache.get(modelId);
    if (modelInfo?.doc_url) {
        const docUrl = modelInfo.doc_url;
        
        // Create a custom widget for the link
        const docWidget = {
            name: "doc_link",
            type: "jiekou_link",
            value: docUrl,
            _jiekouDynamic: true,
            options: {},
            
            // Custom draw function
            draw: function(ctx, node, width, y, H) {
                const labelText = "文档地址: ";
                const linkText = "查看文档 ↗";
                
                ctx.save();
                
                // Draw label
                ctx.font = "12px Arial";
                ctx.fillStyle = "#aaa";
                ctx.textAlign = "left";
                ctx.fillText(labelText, 15, y + H * 0.7);
                
                // Draw link text
                const labelWidth = ctx.measureText(labelText).width;
                ctx.fillStyle = "#5b7bd5";
                ctx.fillText(linkText, 15 + labelWidth, y + H * 0.7);
                
                // Store link bounds for click detection
                const linkWidth = ctx.measureText(linkText).width;
                this._linkBounds = {
                    x: 15 + labelWidth,
                    y: y,
                    width: linkWidth,
                    height: H
                };
                
                ctx.restore();
            },
            
            // Handle mouse events
            mouse: function(event, pos, node) {
                if (event.type === "pointerdown" && this._linkBounds) {
                    const bounds = this._linkBounds;
                    // Check if click is within link bounds
                    const relX = pos[0];
                    const relY = pos[1];
                    if (relX >= bounds.x && relX <= bounds.x + bounds.width) {
                        window.open(docUrl, "_blank");
                        return true;
                    }
                }
                return false;
            },
            
            computeSize: function() {
                return [200, 20];
            }
        };
        
        node.widgets.push(docWidget);
        addedCount++;
    }
    
    console.log(`[JieKou] Added ${addedCount} dynamic parameters for ${modelId}`);
    
    // Resize node to fit new widgets
    if (addedCount > 0) {
        node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
    }
}

/**
 * Initialize dynamic widgets for a node
 */
export async function initializeNode(node) {
    if (!node.comfyClass || !DYNAMIC_NODES.includes(node.comfyClass)) {
        return;
    }
    
    console.log(`[JieKou] Initializing dynamic widgets for ${node.comfyClass}`);
    
    // Hide _dynamic_params widget (it's only for passing data)
    const dynamicParamsWidget = node.widgets?.find(w => w.name === "_dynamic_params");
    if (dynamicParamsWidget) {
        dynamicParamsWidget.type = "hidden";
        // ComfyUI way to hide a widget - set size to 0
        dynamicParamsWidget.computeSize = () => [0, -4];
    }
    
    // Fetch model list to populate cache
    await fetchModelList(node.comfyClass);
    
    // Find the model widget
    const modelWidget = node.widgets?.find(w => w.name === "model");
    if (!modelWidget) {
        console.warn(`[JieKou] No model widget found for ${node.comfyClass}`);
        return;
    }
    
    // Remember original value before setupRichModelCombo modifies it
    const originalValue = modelWidget.value;
    
    // Setup rich model combo with callback that receives model ID
    setupRichModelCombo(modelWidget, node, async (modelId) => {
        await renderDynamicWidgets(node, modelId);
    });
    
    // Initial render for current model
    // originalValue might be ID or name, try to resolve to ID
    if (originalValue) {
        // Check if originalValue is in modelInfoCache (meaning it's an ID)
        // Or try to use the widget's helper to convert
        let modelIdForRender = originalValue;
        if (!modelInfoCache.has(originalValue) && modelWidget._jiekouGetModelId) {
            // It might be a name, convert to ID
            modelIdForRender = modelWidget._jiekouGetModelId();
        }
        await renderDynamicWidgets(node, modelIdForRender);
    }
    
    // Mark node for redraw
    node.setDirtyCanvas(true, true);
    
    // Handle workflow loading
    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function(info) {
        if (originalOnConfigure) {
            originalOnConfigure.call(this, info);
        }
        
        setTimeout(async () => {
            const modelWidget = this.widgets?.find(w => w.name === "model");
            if (modelWidget?.value) {
                // widget.value might be name or ID (from old workflow)
                // Convert to ID for schema lookup
                const modelId = modelWidget._jiekouGetModelId 
                    ? modelWidget._jiekouGetModelId() 
                    : modelWidget.value;
                    
                await renderDynamicWidgets(this, modelId);
                
                // Restore saved dynamic widget values
                if (info.widgets_values) {
                    for (const widget of this.widgets || []) {
                        if (widget._jiekouDynamic && info.widgets_values[widget.name] !== undefined) {
                            widget.value = info.widgets_values[widget.name];
                        }
                    }
                }
            }
        }, 100);
    };
    
    // Handle workflow saving - save model as ID, not name
    const originalSerialize = node.serialize;
    node.serialize = function() {
        const data = originalSerialize ? originalSerialize.call(this) : {};
        
        if (!data.widgets_values) {
            data.widgets_values = {};
        }
        
        // Save dynamic widget values
        for (const widget of this.widgets || []) {
            if (widget._jiekouDynamic) {
                data.widgets_values[widget.name] = widget.value;
            }
        }
        
        // Convert model name back to ID for saving
        const modelWidget = this.widgets?.find(w => w.name === "model");
        if (modelWidget?._jiekouGetModelId) {
            data.widgets_values["model"] = modelWidget._jiekouGetModelId();
        }
        
        return data;
    };
}

// Export for compatibility
export function refreshModels() {
    schemaCache.clear();
    modelInfoCache.clear();
    fullModelConfigsCache = null;
    console.log("[JieKou] Cache cleared");
}

// ===== T027: Parameter Dynamic Linkage =====

/**
 * Fetch full model configs from /jiekou/models
 * These include valid_combinations for dynamic parameter linkage
 */
async function fetchFullModelConfigs() {
    if (fullModelConfigsCache) return fullModelConfigsCache;
    
    try {
        const response = await fetch("/jiekou/models");
        const data = await response.json();
        
        fullModelConfigsCache = {};
        for (const model of data.models || []) {
            fullModelConfigsCache[model.id] = model;
        }
        
        console.log("[JieKou] Loaded full model configs:", Object.keys(fullModelConfigsCache).length);
        return fullModelConfigsCache;
    } catch (error) {
        console.error("[JieKou] Failed to load full model configs:", error);
        return {};
    }
}

/**
 * Get valid options for a parameter based on current selections and valid_combinations
 * @param {Object} modelConfig - Model configuration with valid_combinations
 * @param {string} paramName - Parameter to get options for
 * @param {Object} currentParams - Current parameter selections
 * @returns {string[]|null} Valid options, or null if no filtering needed
 */
function getValidOptionsForParam(modelConfig, paramName, currentParams) {
    const validCombinations = modelConfig?.valid_combinations || [];
    
    if (validCombinations.length === 0) {
        return null; // No filtering needed
    }
    
    // Find all valid values for this param given current selections
    const validValues = new Set();
    
    for (const combo of validCombinations) {
        const comboParams = combo.params || {};
        
        // Check if this combo is compatible with current selections (excluding paramName)
        let isCompatible = true;
        for (const [key, value] of Object.entries(currentParams)) {
            if (key === paramName) continue; // Skip the param we're checking
            if (comboParams[key] !== undefined && comboParams[key] !== value) {
                isCompatible = false;
                break;
            }
        }
        
        if (isCompatible && comboParams[paramName] !== undefined) {
            validValues.add(comboParams[paramName]);
        }
    }
    
    if (validValues.size === 0) {
        return null; // No filtering or no valid combinations
    }
    
    return Array.from(validValues);
}

/**
 * Apply dynamic linkage to widgets based on valid_combinations
 * @param {Object} node - ComfyUI node
 * @param {Object} modelConfig - Model configuration
 */
function applyDynamicLinkage(node, modelConfig) {
    if (!modelConfig?.valid_combinations?.length) return;
    
    // Find widgets that are part of valid_combinations
    const linkedParams = new Set();
    for (const combo of modelConfig.valid_combinations) {
        for (const key of Object.keys(combo.params || {})) {
            linkedParams.add(key);
        }
    }
    
    // Add change callbacks to update other widgets
    for (const widget of node.widgets || []) {
        if (!linkedParams.has(widget.name)) continue;
        
        const originalCallback = widget.callback;
        widget.callback = function(value, ...args) {
            // Call original
            if (originalCallback) {
                originalCallback.call(this, value, ...args);
            }
            
            // Get current params
            const currentParams = {};
            for (const w of node.widgets || []) {
                if (w.name && w.value !== undefined) {
                    currentParams[w.name] = w.value;
                }
            }
            
            // Update other linked widgets
            for (const otherWidget of node.widgets || []) {
                if (otherWidget.name === widget.name) continue;
                if (!linkedParams.has(otherWidget.name)) continue;
                if (!otherWidget.options?.values) continue;
                
                const validOptions = getValidOptionsForParam(
                    modelConfig, 
                    otherWidget.name, 
                    currentParams
                );
                
                if (validOptions && validOptions.length > 0) {
                    // Store original values for reference
                    if (!otherWidget._jiekouOriginalValues) {
                        otherWidget._jiekouOriginalValues = [...otherWidget.options.values];
                    }
                    
                    // Filter to valid options
                    otherWidget.options.values = validOptions;
                    
                    // If current value is not valid, reset to first valid
                    if (!validOptions.includes(otherWidget.value)) {
                        otherWidget.value = validOptions[0];
                    }
                }
            }
            
            // Redraw node
            node.setDirtyCanvas(true, true);
        };
    }
    
    console.log("[JieKou] Applied dynamic linkage for params:", Array.from(linkedParams));
}

// Export for external use
export { fetchFullModelConfigs, getValidOptionsForParam, applyDynamicLinkage };
