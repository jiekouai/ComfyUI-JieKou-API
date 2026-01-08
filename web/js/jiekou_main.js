/**
 * JiekouAI ComfyUI Plugin - Main Extension
 * Settings panel, API key management, and progress display
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "JieKou.Extension";

// ===== Styles =====
const styles = `
.jiekou-modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10000;
}

.jiekou-modal-content {
    background: #2a2a2a;
    padding: 24px;
    border-radius: 8px;
    min-width: 400px;
    max-width: 500px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}

.jiekou-modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.jiekou-modal-title {
    font-size: 18px;
    font-weight: 600;
    color: #fff;
    margin: 0;
}

.jiekou-modal-close {
    background: none;
    border: none;
    color: #888;
    font-size: 24px;
    cursor: pointer;
    padding: 0;
    line-height: 1;
}

.jiekou-modal-close:hover {
    color: #fff;
}

.jiekou-form-group {
    margin-bottom: 16px;
}

.jiekou-label {
    display: block;
    color: #ccc;
    margin-bottom: 6px;
    font-size: 14px;
}

.jiekou-input {
    width: 100%;
    padding: 10px 12px;
    background: #1a1a1a;
    border: 1px solid #444;
    border-radius: 4px;
    color: #fff;
    font-size: 14px;
    box-sizing: border-box;
}

.jiekou-input:focus {
    outline: none;
    border-color: #5b7bd5;
}

.jiekou-btn {
    padding: 10px 20px;
    border-radius: 4px;
    border: none;
    font-size: 14px;
    cursor: pointer;
    transition: background 0.2s;
}

.jiekou-btn-primary {
    background: #5b7bd5;
    color: #fff;
}

.jiekou-btn-primary:hover {
    background: #4a6bc4;
}

.jiekou-btn-secondary {
    background: #444;
    color: #fff;
}

.jiekou-btn-secondary:hover {
    background: #555;
}

.jiekou-btn-group {
    display: flex;
    gap: 10px;
    justify-content: flex-end;
    margin-top: 20px;
}

.jiekou-status {
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 13px;
    margin-top: 12px;
}

.jiekou-status-success {
    background: #1e4d2e;
    color: #4ade80;
}

.jiekou-status-error {
    background: #4d1e1e;
    color: #f87171;
}

.jiekou-status-loading {
    background: #1e3a4d;
    color: #60a5fa;
}

.jiekou-progress-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: 100;
}

.jiekou-progress-text {
    color: #fff;
    margin-top: 10px;
    font-size: 14px;
}

.jiekou-progress-bar {
    width: 80%;
    height: 8px;
    background: #444;
    border-radius: 4px;
    overflow: hidden;
}

.jiekou-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #5b7bd5, #9b59b6);
    transition: width 0.3s ease;
}

/* Model Combo Dropdown Styles */
.litemenu-entry.jiekou-model-item {
    padding: 8px 12px !important;
    min-height: 40px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    justify-content: center !important;
}

.jiekou-model-name {
    font-size: 13px;
    font-weight: 500;
    color: #fff;
    margin-bottom: 2px;
}

.jiekou-model-desc {
    font-size: 11px;
    color: #888;
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.litemenu-entry.jiekou-model-item:hover .jiekou-model-desc {
    color: #aaa;
}
`;

// ===== Inject Styles =====
function injectStyles() {
    if (document.getElementById("jiekou-styles")) return;
    
    const styleEl = document.createElement("style");
    styleEl.id = "jiekou-styles";
    styleEl.textContent = styles;
    document.head.appendChild(styleEl);
}

// ===== Settings Modal =====
class JiekouSettingsModal {
    constructor() {
        this.modal = null;
        this.statusEl = null;
    }
    
    show() {
        if (this.modal) {
            this.modal.remove();
        }
        
        this.modal = document.createElement("div");
        this.modal.className = "jiekou-modal";
        this.modal.innerHTML = `
            <div class="jiekou-modal-content">
                <div class="jiekou-modal-header">
                    <h2 class="jiekou-modal-title">接口 AI 设置</h2>
                    <button class="jiekou-modal-close" id="jiekou-close">&times;</button>
                </div>
                
                <div class="jiekou-form-group">
                    <label class="jiekou-label">API Key</label>
                    <input type="password" class="jiekou-input" id="jiekou-api-key" 
                           placeholder="请输入您的接口 AI API Key">
                </div>
                
                <div class="jiekou-form-group">
                    <label class="jiekou-label">
                        <input type="checkbox" id="jiekou-show-key"> 显示 API Key
                    </label>
                </div>
                
                <div id="jiekou-status"></div>
                
                <div class="jiekou-btn-group">
                    <button class="jiekou-btn jiekou-btn-secondary" id="jiekou-test">
                        测试连接
                    </button>
                    <button class="jiekou-btn jiekou-btn-primary" id="jiekou-save">
                        保存
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(this.modal);
        this.statusEl = this.modal.querySelector("#jiekou-status");
        
        // Event listeners
        this.modal.querySelector("#jiekou-close").onclick = () => this.close();
        this.modal.querySelector("#jiekou-show-key").onchange = (e) => {
            const input = this.modal.querySelector("#jiekou-api-key");
            input.type = e.target.checked ? "text" : "password";
        };
        this.modal.querySelector("#jiekou-test").onclick = () => this.testConnection();
        this.modal.querySelector("#jiekou-save").onclick = () => this.save();
        
        // Close on backdrop click
        this.modal.onclick = (e) => {
            if (e.target === this.modal) this.close();
        };
        
        // Check current status
        this.checkStatus();
    }
    
    close() {
        if (this.modal) {
            this.modal.remove();
            this.modal = null;
        }
    }
    
    setStatus(type, message) {
        if (!this.statusEl) return;
        
        this.statusEl.className = `jiekou-status jiekou-status-${type}`;
        this.statusEl.textContent = message;
    }
    
    async checkStatus() {
        try {
            const response = await fetch("/jiekou/config");
            const data = await response.json();
            
            if (data.configured) {
                this.setStatus("success", "✓ API Key 已配置");
            } else {
                this.setStatus("error", "API Key 未配置，请输入您的 API Key");
            }
        } catch (error) {
            console.error("[JieKou] Failed to check status:", error);
        }
    }
    
    async testConnection() {
        const apiKey = this.modal.querySelector("#jiekou-api-key").value;
        
        if (!apiKey) {
            this.setStatus("error", "请输入 API Key");
            return;
        }
        
        this.setStatus("loading", "正在验证 API Key...");
        
        try {
            // First save the key temporarily
            await this.saveKey(apiKey);
            
            // Then verify the key via API
            const response = await fetch("/jiekou/verify");
            const data = await response.json();
            
            if (data.success) {
                this.setStatus("success", "✓ API Key 验证成功！");
            } else {
                this.setStatus("error", data.error || "API Key 验证失败");
            }
        } catch (error) {
            this.setStatus("error", error.message || "连接测试失败");
        }
    }
    
    async saveKey(apiKey) {
        const response = await fetch("/jiekou/config", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ api_key: apiKey })
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || "Failed to save");
        }
        
        return data;
    }
    
    async save() {
        const apiKey = this.modal.querySelector("#jiekou-api-key").value;
        
        if (!apiKey) {
            this.setStatus("error", "请输入 API Key");
            return;
        }
        
        this.setStatus("loading", "正在保存...");
        
        try {
            await this.saveKey(apiKey);
            this.setStatus("success", "✓ API Key 保存成功");
            
            // Close modal after short delay
            setTimeout(() => this.close(), 1500);
        } catch (error) {
            this.setStatus("error", error.message || "保存失败");
        }
    }
}

// ===== Video Progress Handler =====
class VideoProgressHandler {
    constructor() {
        this.progressNodes = new Map();
    }
    
    showProgress(nodeId, progress, message) {
        const node = app.graph.getNodeById(nodeId);
        if (!node || !node.domElement) return;
        
        let overlay = this.progressNodes.get(nodeId);
        
        if (!overlay) {
            overlay = document.createElement("div");
            overlay.className = "jiekou-progress-overlay";
            overlay.innerHTML = `
                <div class="jiekou-progress-bar">
                    <div class="jiekou-progress-fill"></div>
                </div>
                <div class="jiekou-progress-text"></div>
            `;
            node.domElement.appendChild(overlay);
            this.progressNodes.set(nodeId, overlay);
        }
        
        overlay.querySelector(".jiekou-progress-fill").style.width = `${progress}%`;
        overlay.querySelector(".jiekou-progress-text").textContent = message || `${progress}%`;
    }
    
    hideProgress(nodeId) {
        const overlay = this.progressNodes.get(nodeId);
        if (overlay) {
            overlay.remove();
            this.progressNodes.delete(nodeId);
        }
    }
}

// ===== Global Instances =====
const settingsModal = new JiekouSettingsModal();
const progressHandler = new VideoProgressHandler();

// ===== Register Extension =====
app.registerExtension({
    name: EXTENSION_NAME,
    
    async setup() {
        injectStyles();
        
        // T021: Initialize price display module
        try {
            await import("./price_display.js");
            if (window.JieKouPriceDisplay) {
                window.JieKouPriceDisplay.init();
                console.log("[JieKou] Price display module initialized");
            }
        } catch (error) {
            console.warn("[JieKou] Price display module not loaded:", error);
        }
        
        // Add floating settings button to canvas container
        const addSettingsButton = () => {
            // Find canvas container (new ComfyUI)
            const canvasContainer = document.getElementById("graph-canvas-container");
            if (canvasContainer) {
                const floatingBtn = document.createElement("button");
                floatingBtn.id = "jiekou-settings-btn";
                floatingBtn.textContent = "⚙️ 接口 AI 设置";
                floatingBtn.onclick = () => settingsModal.show();
                floatingBtn.style.cssText = `
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    z-index: 9999;
                    background: linear-gradient(135deg, #5b7bd5, #9b59b6);
                    border: none;
                    padding: 10px 16px;
                    border-radius: 6px;
                    color: white;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 500;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                    transition: transform 0.2s, box-shadow 0.2s;
                `;
                floatingBtn.onmouseenter = () => {
                    floatingBtn.style.transform = "scale(1.05)";
                    floatingBtn.style.boxShadow = "0 4px 15px rgba(0,0,0,0.4)";
                };
                floatingBtn.onmouseleave = () => {
                    floatingBtn.style.transform = "scale(1)";
                    floatingBtn.style.boxShadow = "0 2px 10px rgba(0,0,0,0.3)";
                };
                canvasContainer.appendChild(floatingBtn);
                console.log("[JieKou] Settings button added to #graph-canvas-container");
                return true;
            }
            
            // Fallback: Try old ComfyUI menu (.comfy-menu)
            const oldMenu = document.querySelector(".comfy-menu");
            if (oldMenu) {
                const settingsBtn = document.createElement("button");
                settingsBtn.textContent = "接口 AI 设置";
                settingsBtn.onclick = () => settingsModal.show();
                settingsBtn.style.cssText = `
                    margin-top: 10px;
                    background: linear-gradient(135deg, #5b7bd5, #9b59b6);
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    color: white;
                    cursor: pointer;
                    width: 100%;
                `;
                oldMenu.appendChild(settingsBtn);
                console.log("[JieKou] Settings button added to .comfy-menu");
                return true;
            }
            
            return false;
        };
        
        // Try to add button now, retry after delay if not ready
        if (!addSettingsButton()) {
            setTimeout(() => {
                if (!addSettingsButton()) {
                    // Final fallback: Add to body with fixed position
                    const floatingBtn = document.createElement("button");
                    floatingBtn.id = "jiekou-settings-btn";
                    floatingBtn.textContent = "⚙️ 接口 AI 设置";
                    floatingBtn.onclick = () => settingsModal.show();
                    floatingBtn.style.cssText = `
                        position: fixed;
                        top: 10px;
                        right: 10px;
                        z-index: 99999;
                        background: linear-gradient(135deg, #5b7bd5, #9b59b6);
                        border: none;
                        padding: 10px 16px;
                        border-radius: 6px;
                        color: white;
                        cursor: pointer;
                        font-size: 14px;
                        font-weight: 500;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                    `;
                    document.body.appendChild(floatingBtn);
                    console.log("[JieKou] Settings button added to body (fallback)");
                }
            }, 2000);
        }
        
        // Before queueing prompt, serialize dynamic widget values to _dynamic_params
        // This hook ensures dynamic parameters are passed to Python via a JSON string
        const originalQueuePrompt = app.queuePrompt;
        app.queuePrompt = async function(number, batchCount) {
            // Update _dynamic_params widget for all JieKou nodes
            for (const node of app.graph._nodes || []) {
                if (!node.comfyClass || !node.comfyClass.startsWith("JieKou")) {
                    continue;
                }
                
                // Find _dynamic_params widget
                const dynamicParamsWidget = node.widgets?.find(w => w.name === "_dynamic_params");
                if (!dynamicParamsWidget) {
                    continue;
                }
                
                // Collect dynamic widget values
                const dynamicValues = {};
                for (const widget of node.widgets || []) {
                    if (widget._jiekouDynamic && widget.name !== "doc_link") {
                        dynamicValues[widget.name] = widget.value;
                    }
                }
                
                // Serialize to JSON and set widget value
                dynamicParamsWidget.value = JSON.stringify(dynamicValues);
                console.log(`[JieKou] Serialized dynamic params for ${node.comfyClass}:`, dynamicValues);
            }
            
            return originalQueuePrompt.apply(this, arguments);
        };
        
        console.log("[JieKou] queuePrompt hook installed for dynamic parameters");
        
        // Listen for progress messages
        api.addEventListener("jiekou-progress", (event) => {
            const { node_id, progress, message, status } = event.detail;
            
            if (status === "complete" || status === "error") {
                progressHandler.hideProgress(node_id);
            } else {
                progressHandler.showProgress(node_id, progress, message);
            }
        });
        
        console.log("[JieKou] Extension loaded");
    },
    
    async nodeCreated(node) {
        // Initialize dynamic widgets for JieKou nodes
        if (node.comfyClass && node.comfyClass.startsWith("JieKou")) {
            // Import and initialize dynamic widgets
            try {
                const dynamicWidgets = await import("./dynamic_widgets.js");
                dynamicWidgets.initializeNode(node);
            } catch (error) {
                console.warn("[JieKou] Dynamic widgets not loaded:", error);
            }
            
            // T023: Initialize price display for this node
            try {
                if (window.JieKouPriceDisplay) {
                    // Update price when node is created
                    setTimeout(() => {
                        window.JieKouPriceDisplay.updateNodePrice(node);
                    }, 100);
                    
                    // T024: Add widget change listeners for price updates
                    if (node.widgets) {
                        for (const widget of node.widgets) {
                            const originalCallback = widget.callback;
                            widget.callback = function(value, ...args) {
                                // Call original callback if exists
                                if (originalCallback) {
                                    originalCallback.call(this, value, ...args);
                                }
                                // Update price after parameter change
                                if (window.JieKouPriceDisplay) {
                                    window.JieKouPriceDisplay.updateNodePrice(node);
                                }
                            };
                        }
                    }
                }
            } catch (error) {
                console.warn("[JieKou] Price display not initialized:", error);
            }
        }
    }
});

export { settingsModal, progressHandler };

