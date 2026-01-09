/**
 * Price Display Module for JieKou ComfyUI Plugin
 * 
 * Provides real-time price display functionality for JieKou nodes.
 * Features:
 * - Fetches prices from /jiekou/prices API
 * - Caches prices in localStorage with 1-hour TTL
 * - Updates price display when parameters change
 * - Formats prices as "$X.XXX/次"
 */

// ===== T022: Price Cache with 1-hour TTL =====

const PRICE_CACHE_KEY = "jiekou_price_cache";
const PRICE_CACHE_TTL = 60 * 60 * 1000; // 1 hour in milliseconds

/**
 * Get cached price data
 * @returns {Object|null} Cached data or null if expired/missing
 */
function getPriceCache() {
    try {
        const cached = localStorage.getItem(PRICE_CACHE_KEY);
        if (!cached) return null;
        
        const data = JSON.parse(cached);
        const now = Date.now();
        
        // Check if cache is expired
        if (data.timestamp && (now - data.timestamp) < PRICE_CACHE_TTL) {
            return data.prices || {};
        }
        
        // Cache expired
        return null;
    } catch (e) {
        console.warn("[JieKou] Price cache read error:", e);
        return null;
    }
}

/**
 * Set price cache
 * @param {Object} prices - Map of product_id -> price_info
 */
function setPriceCache(prices) {
    try {
        const data = {
            timestamp: Date.now(),
            prices: prices
        };
        localStorage.setItem(PRICE_CACHE_KEY, JSON.stringify(data));
    } catch (e) {
        console.warn("[JieKou] Price cache write error:", e);
    }
}

/**
 * Merge new prices into cache
 * @param {Object} newPrices - New prices to merge
 */
function mergePriceCache(newPrices) {
    const existing = getPriceCache() || {};
    const merged = { ...existing, ...newPrices };
    setPriceCache(merged);
}


// ===== T26: Get Product ID for Current Parameters =====

/**
 * Map current parameter values to a Product ID using valid_combinations
 * @param {Object} modelConfig - Model configuration from /jiekou/models
 * @param {Object} params - Current parameter values
 * @returns {string|null} Product ID or null if not found
 */
function getProductIdForCurrentParams(modelConfig, params) {
    if (!modelConfig) return null;
    
    const validCombinations = modelConfig.valid_combinations || [];
    const productIds = modelConfig.product_ids || [];
    
    // If no valid_combinations, use first product_id
    if (validCombinations.length === 0) {
        return productIds[0] || null;
    }
    
    // Find matching combination
    for (const combo of validCombinations) {
        const comboParams = combo.params || {};
        const productId = combo.product_id;
        
        // Check if all combo params match
        let match = true;
        for (const [key, value] of Object.entries(comboParams)) {
            if (params[key] !== value) {
                match = false;
                break;
            }
        }
        
        if (match && productId) {
            return productId;
        }
    }
    
    // Fallback to first product_id
    return productIds[0] || null;
}


// ===== T28 & T29: Price Fetching and Formatting =====

/**
 * Fetch prices for product IDs
 * @param {string[]} productIds - List of product IDs
 * @returns {Promise<Object>} Map of product_id -> price_info
 */
async function fetchPrices(productIds) {
    if (!productIds || productIds.length === 0) {
        return {};
    }
    
    // Filter out IDs we already have cached
    const cached = getPriceCache() || {};
    const uncachedIds = productIds.filter(id => !cached[id]);
    
    if (uncachedIds.length === 0) {
        // All prices are cached
        const result = {};
        for (const id of productIds) {
            if (cached[id]) result[id] = cached[id];
        }
        return result;
    }
    
    try {
        const response = await fetch("/jiekou/prices", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ product_ids: uncachedIds })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        const prices = data.prices || [];
        
        // Convert to map
        const newPrices = {};
        for (const item of prices) {
            newPrices[item.product_id] = {
                price: item.price,
                original_price: item.original_price,
                currency: item.currency || "USD",
                unit: item.unit || "次"
            };
        }
        
        // Merge into cache
        mergePriceCache(newPrices);
        
        // Return all requested prices
        const result = { ...cached };
        for (const id of productIds) {
            if (newPrices[id]) result[id] = newPrices[id];
        }
        return result;
    } catch (e) {
        console.error("[JieKou] Price fetch error:", e);
        return cached;
    }
}

/**
 * Format price for display (T029)
 * @param {number} price - Price value
 * @param {string} currency - Currency code
 * @param {string} unit - Unit string
 * @returns {string} Formatted price string
 */
function formatPrice(price, currency = "USD", unit = "次") {
    if (price === null || price === undefined) {
        return "$--";
    }
    
    if (currency === "USD") {
        if (price >= 1) {
            return `$${price.toFixed(2)}/${unit}`;
        } else if (price >= 0.01) {
            return `$${price.toFixed(3)}/${unit}`;
        } else {
            return `$${price.toFixed(4)}/${unit}`;
        }
    }
    
    return `${price.toFixed(4)} ${currency}/${unit}`;
}


// ===== T25: Price Badge Styles =====

const PRICE_BADGE_STYLES = `
.jiekou-price-badge {
    position: absolute;
    top: 2px;
    right: 8px;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
    pointer-events: none;
    z-index: 100;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

.jiekou-price-badge.loading {
    background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
}

.jiekou-price-badge.error {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
}

.jiekou-price-badge.discounted {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
}

.jiekou-price-badge .original-price {
    text-decoration: line-through;
    opacity: 0.7;
    font-size: 9px;
    margin-right: 4px;
}
`;

// Inject styles
function injectPriceStyles() {
    if (document.getElementById("jiekou-price-styles")) return;
    
    const style = document.createElement("style");
    style.id = "jiekou-price-styles";
    style.textContent = PRICE_BADGE_STYLES;
    document.head.appendChild(style);
}


// ===== T23 & T24: Node Integration =====

// Store model configs for lookup
let modelConfigsCache = null;

/**
 * Load model configs from API
 */
async function loadModelConfigs() {
    if (modelConfigsCache) return modelConfigsCache;
    
    try {
        const response = await fetch("/jiekou/models");
        const data = await response.json();
        
        // Create lookup by model ID
        modelConfigsCache = {};
        for (const model of data.models || []) {
            modelConfigsCache[model.id] = model;
        }
        
        console.log("[JieKou] Loaded", Object.keys(modelConfigsCache).length, "model configs");
        return modelConfigsCache;
    } catch (e) {
        console.error("[JieKou] Failed to load model configs:", e);
        return {};
    }
}

/**
 * Create or update price badge for a node
 * @param {Object} node - ComfyUI node
 * @param {string} priceText - Price text to display
 * @param {string} status - Badge status (loading, error, discounted, or empty)
 */
function updatePriceBadge(node, priceText, status = "") {
    if (!node.domElement) return;
    
    let badge = node.domElement.querySelector(".jiekou-price-badge");
    
    if (!badge) {
        badge = document.createElement("div");
        badge.className = "jiekou-price-badge";
        node.domElement.appendChild(badge);
    }
    
    badge.textContent = priceText;
    badge.className = "jiekou-price-badge" + (status ? " " + status : "");
}

/**
 * Update price for a JieKou node
 * @param {Object} node - ComfyUI node
 */
async function updateNodePrice(node) {
    if (!node || !node.type) return;
    
    // Check if this is a JieKou node
    if (!node.type.startsWith("JieKou")) return;
    
    // Get model config
    const configs = await loadModelConfigs();
    
    // Find model config by node type
    // Node type format: JieKouModelName -> model_name
    let modelConfig = null;
    for (const [modelId, config] of Object.entries(configs)) {
        // Match by class name or model ID
        const className = "JieKou" + modelId.replace(/-/g, "_").replace(/\./g, "_")
            .split("_").map(s => s.charAt(0).toUpperCase() + s.slice(1)).join("");
        if (className === node.type || node.type.includes(modelId)) {
            modelConfig = config;
            break;
        }
    }
    
    if (!modelConfig) {
        // Legacy node or not found
        return;
    }
    
    // Get current parameter values from widgets
    const params = {};
    if (node.widgets) {
        for (const widget of node.widgets) {
            if (widget.name && widget.value !== undefined) {
                params[widget.name] = widget.value;
            }
        }
    }
    
    // Get product ID for current params
    const productId = getProductIdForCurrentParams(modelConfig, params);
    
    if (!productId) {
        updatePriceBadge(node, "$--", "error");
        return;
    }
    
    // Show loading state
    updatePriceBadge(node, "...", "loading");
    
    // Fetch price
    const prices = await fetchPrices([productId]);
    const priceInfo = prices[productId];
    
    if (!priceInfo) {
        updatePriceBadge(node, "$--", "error");
        return;
    }
    
    // Format and display
    const priceText = formatPrice(priceInfo.price, priceInfo.currency, priceInfo.unit);
    const hasDiscount = priceInfo.original_price && priceInfo.original_price > priceInfo.price;
    
    updatePriceBadge(node, priceText, hasDiscount ? "discounted" : "");
}


// ===== Module Initialization =====

/**
 * Initialize price display module
 */
function initPriceDisplay() {
    injectPriceStyles();
    loadModelConfigs();
    
    console.log("[JieKou] Price display module initialized");
}

// Export for use in jiekou_main.js
window.JieKouPriceDisplay = {
    init: initPriceDisplay,
    updateNodePrice: updateNodePrice,
    updatePriceBadge: updatePriceBadge,
    getProductIdForCurrentParams: getProductIdForCurrentParams,
    fetchPrices: fetchPrices,
    formatPrice: formatPrice,
    loadModelConfigs: loadModelConfigs
};

