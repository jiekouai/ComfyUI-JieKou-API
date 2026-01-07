"""
JiekouAI ComfyUI Plugin
Integrates JiekouAI platform's multimodal API capabilities into ComfyUI

API Documentation: https://docs.jiekou.ai/llms.txt
"""

import os
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[JieKou]")

# Plugin metadata
__version__ = "1.1.2"  # Updated for API compatibility fixes
__author__ = "JiekouAI"

# ===== Web Directory for JavaScript Extensions =====
WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web", "js")

# ===== Node Imports =====
from .nodes.config_nodes import JieKouTestConnection
from .nodes.image_nodes import (
    JieKouTextToImage, 
    JieKouImageToImage,
    JieKouImageUpscale,
    JieKouRemoveBackground
)
from .nodes.video_nodes import JieKouVideoGeneration
# Audio nodes - not included in this version
# from .nodes.audio_nodes import JieKouTTS

# ===== Node Mappings =====
NODE_CLASS_MAPPINGS = {
    # Config/Auth Nodes
    "JieKouTestConnection": JieKouTestConnection,
    
    # Image Nodes
    "JieKouTextToImage": JieKouTextToImage,
    "JieKouImageToImage": JieKouImageToImage,
    "JieKouImageUpscale": JieKouImageUpscale,
    "JieKouRemoveBackground": JieKouRemoveBackground,
    
    # Video Nodes
    "JieKouVideoGeneration": JieKouVideoGeneration,
    
    # Audio Nodes - not included in this version
    # "JieKouTTS": JieKouTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Config/Auth Nodes
    "JieKouTestConnection": "JieKou Test Connection",
    
    # Image Nodes
    "JieKouTextToImage": "JieKou Text to Image",
    "JieKouImageToImage": "JieKou Image to Image",
    "JieKouImageUpscale": "JieKou Image Upscale",
    "JieKouRemoveBackground": "JieKou Remove Background",
    
    # Video Nodes
    "JieKouVideoGeneration": "JieKou Video Generation",
    
    # Audio Nodes - not included in this version
    # "JieKouTTS": "JieKou TTS",
}

# ===== API Routes =====
# These routes are registered with ComfyUI's PromptServer

try:
    from server import PromptServer
    from aiohttp import web
    
    routes = PromptServer.instance.routes
    
    # ===== T022: /jiekou/config route =====
    @routes.post("/jiekou/config")
    async def save_config(request):
        """
        Save API Key to config file
        
        POST body: { "api_key": "..." }
        """
        try:
            data = await request.json()
            api_key = data.get("api_key")
            
            if not api_key:
                return web.json_response(
                    {"success": False, "error": "API Key is required"},
                    status=400
                )
            
            from .jiekou_config import get_config
            config = get_config()
            
            if config.save_api_key(api_key):
                return web.json_response({"success": True})
            else:
                return web.json_response(
                    {"success": False, "error": "Failed to save config"},
                    status=500
                )
        
        except Exception as e:
            logger.error(f"[JieKou] Config save error: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500
            )
    
    @routes.get("/jiekou/config")
    async def get_config_status(request):
        """
        Get configuration status (not the actual key)
        """
        try:
            from .jiekou_config import get_config
            config = get_config()
            
            return web.json_response({
                "configured": config.is_configured(),
                "base_url": config.base_url
            })
        
        except Exception as e:
            logger.error(f"[JieKou] Config get error: {e}")
            return web.json_response(
                {"configured": False, "error": str(e)},
                status=500
            )
    
    # ===== /jiekou/models routes (using local registry) =====
    @routes.get("/jiekou/models/video")
    async def get_video_models(request):
        """
        Get all video models with full details (id, name, description, doc_url)
        Used by frontend for rich dropdown display
        """
        try:
            from .utils.model_registry import get_model_registry
            
            registry = get_model_registry()
            all_video = registry.get_all_video_models()
            
            models = [
                {
                    "id": m.id, 
                    "name": m.name, 
                    "description": m.description or "",
                    "category": m.category,
                    "doc_url": m.doc_url or ""
                }
                for m in all_video
            ]
            
            return web.json_response({"models": models})
        
        except Exception as e:
            logger.error(f"[JieKou] Get video models error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    @routes.get("/jiekou/models/image")
    async def get_image_models(request):
        """
        Get all image models with full details (id, name, description, doc_url)
        Used by frontend for rich dropdown display
        """
        try:
            from .utils.model_registry import get_model_registry
            
            registry = get_model_registry()
            all_image = registry.get_all_image_models()
            
            models = [
                {
                    "id": m.id, 
                    "name": m.name, 
                    "description": m.description or "",
                    "category": m.category,
                    "doc_url": m.doc_url or ""
                }
                for m in all_image
            ]
            
            return web.json_response({"models": models})
        
        except Exception as e:
            logger.error(f"[JieKou] Get image models error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    @routes.get("/jiekou/models")
    async def get_models(request):
        """
        Get available models from local registry
        
        Note: JiekouAI doesn't have a unified /models API,
        so we use a hardcoded registry based on their documentation.
        
        Query params:
          - type: Filter by model type (image, video_t2v, video_i2v, audio_tts)
        """
        try:
            from .utils.model_registry import get_model_registry
            
            registry = get_model_registry()
            model_type = request.query.get("type")
            
            if model_type:
                models = registry.get_models_by_category(model_type)
            else:
                # Return all models organized by category
                models = {
                    "video_t2v": [{"id": m.id, "name": m.name, "description": m.description} 
                                 for m in registry.get_video_t2v_models()],
                    "video_i2v": [{"id": m.id, "name": m.name, "description": m.description} 
                                 for m in registry.get_video_i2v_models()],
                    "video_v2v": [{"id": m.id, "name": m.name, "description": m.description} 
                                 for m in registry.get_video_v2v_models()],
                    "image": [{"id": m.id, "name": m.name, "description": m.description} 
                             for m in registry.get_all_image_models()],
                    "audio": [{"id": m.id, "name": m.name, "description": m.description} 
                             for m in registry.get_audio_models()],
                }
                return web.json_response({"models": models})
            
            # Convert to dicts for JSON
            model_list = [
                {"id": m.id, "name": m.name, "description": m.description, "category": m.category}
                for m in models
            ]
            
            return web.json_response({"models": model_list})
        
        except Exception as e:
            logger.error(f"[JieKou] Get models error: {e}")
            return web.json_response(
                {"models": [], "error": str(e)},
                status=500
            )
    
    @routes.get("/jiekou/models/{model_id}/schema")
    async def get_model_schema(request):
        """
        Get parameter schema for a model from local registry
        Supports:
        - Image models (parameters in ModelConfig)
        - Video models using unified API (VIDEO_PARAMS/VIDEO_MODEL_PARAMS)
        - Wan 2.6 models using dedicated endpoints (WAN26_PARAMS/WAN26_MODEL_PARAMS)
        """
        try:
            from .utils.model_registry import get_model_registry
            
            model_id = request.match_info["model_id"]
            registry = get_model_registry()
            model = registry.get_model(model_id)
            
            if not model:
                return web.json_response(
                    {"error": f"Model not found: {model_id}"},
                    status=404
                )
            
            # Convert parameters to JSON Schema format
            properties = {}
            required = []
            
            # Check if this is a Wan 2.6 model (uses dedicated endpoint)
            if registry.is_wan26_model(model_id):
                # Get Wan 2.6 model parameters (from both input and parameters sections)
                param_structure = registry.get_wan26_model_params(model_id)
                all_param_names = set(param_structure.get("input", []) + param_structure.get("parameters", []))
                
                # Skip img_url - it's handled as IMAGE input slot (converted from image tensor)
                # Also skip prompt - it's a fixed required input
                skip_params = {"img_url", "prompt", "reference_video_urls"}
                
                for param_name in all_param_names:
                    if param_name in skip_params:
                        continue
                    
                    # Handle duration_v2v special case
                    actual_param_name = "duration" if param_name == "duration_v2v" else param_name
                    param = registry.get_wan26_param(param_name)
                    if not param:
                        continue
                    
                    prop = {
                        "type": param.type,
                        "description": param.description,
                    }
                    
                    if param.default is not None:
                        prop["default"] = param.default
                    if param.enum:
                        prop["enum"] = param.enum
                    if param.minimum is not None:
                        prop["minimum"] = param.minimum
                    if param.maximum is not None:
                        prop["maximum"] = param.maximum
                    
                    # Use actual_param_name for display (duration instead of duration_v2v)
                    properties[actual_param_name if param_name == "duration_v2v" else param_name] = prop
                    
                    if param.required:
                        required.append(param_name)
                
                schema = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "supports_image": "img_url" in all_param_names,  # Wan 2.6 I2V uses img_url
                    "supports_end_image": False,  # Wan 2.6 doesn't support end_image
                    "is_wan26": True,  # Flag for frontend
                }
                
                return web.json_response(schema)
            
            # Check if this is a video model (uses VIDEO_MODEL_PARAMS)
            elif registry.is_video_model(model_id):
                # Get video model parameters
                # Skip image/end_image as they are fixed IMAGE input slots, not dynamic widgets
                video_params = registry.get_video_model_params(model_id)
                for param in video_params:
                    # Skip 'image' - it's handled as fixed IMAGE INPUT slot
                    # 'end_image' stays as dynamic string parameter for URL input
                    if param.name == "image":
                        continue
                    
                    prop = {
                        "type": param.type,
                        "description": param.description,
                    }
                    
                    if param.default is not None:
                        prop["default"] = param.default
                    if param.enum:
                        prop["enum"] = param.enum
                    if param.minimum is not None:
                        prop["minimum"] = param.minimum
                    if param.maximum is not None:
                        prop["maximum"] = param.maximum
                    
                    properties[param.name] = prop
                    
                    if param.required:
                        required.append(param.name)
                
                schema = {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
                
                # For video models, indicate which image inputs are supported
                param_names = registry.get_video_model_param_names(model_id)
                schema["supports_image"] = "image" in param_names
                schema["supports_end_image"] = "end_image" in param_names
                
                return web.json_response(schema)
            
            else:
                # Regular model with parameters in ModelConfig
                for param in model.parameters:
                    prop = {
                        "type": param.type,
                        "description": param.description,
                    }
                    
                    if param.default is not None:
                        prop["default"] = param.default
                    if param.enum:
                        prop["enum"] = param.enum
                    if param.minimum is not None:
                        prop["minimum"] = param.minimum
                    if param.maximum is not None:
                        prop["maximum"] = param.maximum
                    
                    properties[param.name] = prop
                    
                    if param.required:
                        required.append(param.name)
                
                schema = {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
                
                return web.json_response(schema)
        
        except Exception as e:
            logger.error(f"[JieKou] Get schema error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    # ===== /jiekou/verify route =====
    @routes.get("/jiekou/verify")
    async def verify_api_key(request):
        """
        Verify API key is valid
        """
        try:
            from .utils.api_client import JiekouAPI, JiekouAPIError
            
            api = JiekouAPI()
            result = api.verify_key()
            
            return web.json_response({
                "success": True,
                "message": result.get("message", "API Key 验证成功")
            })
        
        except JiekouAPIError as e:
            return web.json_response({
                "success": False,
                "error": e.message,
                "code": e.code
            }, status=401 if e.code == "UNAUTHORIZED" else 500)
        
        except Exception as e:
            logger.error(f"[JieKou] Verify error: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    logger.info("[JieKou] API routes registered")

except ImportError:
    logger.warning("[JieKou] Could not import PromptServer - running outside ComfyUI?")

# ===== Startup Message =====
logger.info(f"[JieKou] Plugin v{__version__} loaded")
logger.info(f"[JieKou] Registered nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
