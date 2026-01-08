"""
Dynamic Node Factory for JieKou ComfyUI Plugin

This module provides factory functions to dynamically create ComfyUI node
classes from the static model_config.json. Each model gets its own
dedicated node class with model-specific parameters.

Features:
- Automatic node class generation from JSON configuration
- Parameter-to-ComfyUI-type conversion
- Category path assignment for menu organization
- Proper return type handling (IMAGE for images, frames for video)
"""

import torch
import logging
import os
import time
import json
import numpy as np
from PIL import Image
from typing import Any, Optional

logger = logging.getLogger("[JieKou]")


# ===== T009: Parameter Type Converters =====

def param_to_comfyui_type(param: dict) -> tuple[Any, dict]:
    """
    Convert model_config parameter definition to ComfyUI INPUT_TYPES format.
    
    Args:
        param: Parameter dict from model_config.json
        
    Returns:
        tuple: (type_spec, widget_options)
        - type_spec: Either a string like "STRING" or a tuple of enum values
        - widget_options: Dict of widget configuration options
    """
    param_type = param.get("type", "string")
    name = param.get("name", "")
    description = param.get("description", "")
    default = param.get("default")
    enum = param.get("enum")
    minimum = param.get("minimum")
    maximum = param.get("maximum")
    required = param.get("required", False)
    
    options = {}
    
    # Handle enum types - convert to tuple for dropdown
    if enum:
        # ComfyUI uses tuple for dropdown options
        type_spec = (enum,)
        if default is not None and default in enum:
            options["default"] = default
        elif enum:
            options["default"] = enum[0]
        return type_spec, options
    
    # Handle different parameter types
    if param_type == "string":
        type_spec = "STRING"
        if default is not None:
            options["default"] = str(default)
        else:
            options["default"] = ""
        
        # Check if this is a multiline text field
        if name in ("prompt", "negative_prompt", "text", "content"):
            options["multiline"] = True
            options["dynamicPrompts"] = True
        
        if description:
            options["tooltip"] = description[:200]  # Limit tooltip length
    
    elif param_type == "integer":
        type_spec = "INT"
        if default is not None:
            options["default"] = int(default)
        else:
            options["default"] = 0
        
        if minimum is not None:
            options["min"] = int(minimum)
        if maximum is not None:
            options["max"] = int(maximum)
        
        # Add step for better UX
        options["step"] = 1
    
    elif param_type == "number":
        type_spec = "FLOAT"
        if default is not None:
            options["default"] = float(default)
        else:
            options["default"] = 0.0
        
        if minimum is not None:
            options["min"] = float(minimum)
        if maximum is not None:
            options["max"] = float(maximum)
        
        # Add step based on range
        if maximum and minimum:
            range_val = maximum - minimum
            if range_val <= 1:
                options["step"] = 0.01
            elif range_val <= 10:
                options["step"] = 0.1
            else:
                options["step"] = 1.0
        else:
            options["step"] = 0.1
    
    elif param_type == "boolean":
        type_spec = "BOOLEAN"
        options["default"] = bool(default) if default is not None else False
    
    else:
        # Fallback to string
        type_spec = "STRING"
        options["default"] = str(default) if default is not None else ""
    
    return type_spec, options


def build_input_types(model_config: dict, category: str) -> dict:
    """
    Build ComfyUI INPUT_TYPES dict from model configuration.
    
    Args:
        model_config: Model configuration dict
        category: Model category (image_t2i, video_t2v, etc.)
        
    Returns:
        dict: INPUT_TYPES for ComfyUI node class
    """
    required = {}
    optional = {}
    
    # Standard inputs based on category
    is_image_model = category.startswith("image_")
    is_video_model = category.startswith("video_")
    is_audio_model = category.startswith("audio_")
    
    # For I2I/I2V models, add image_url input
    needs_image_input = category in ("image_edit", "video_i2v")
    if needs_image_input:
        required["image_url"] = ("STRING", {
            "default": "",
            "placeholder": "输入图片 URL 或 base64",
            "dynamicPrompts": True
        })
    
    # For V2V models, add video_url input
    needs_video_input = category == "video_v2v"
    if needs_video_input:
        required["video_url"] = ("STRING", {
            "default": "",
            "placeholder": "输入视频 URL",
            "dynamicPrompts": True
        })
    
    # Process model parameters
    for param_data in model_config.get("parameters", []):
        name = param_data.get("name", "")
        is_required = param_data.get("required", False)
        
        # Skip parameters that are handled separately
        skip_params = {"model", "image", "images", "img_url", "image_url", "video"}
        if name in skip_params:
            continue
        
        type_spec, options = param_to_comfyui_type(param_data)
        
        if is_required:
            required[name] = (type_spec, options) if isinstance(type_spec, str) else (type_spec[0], options)
        else:
            optional[name] = (type_spec, options) if isinstance(type_spec, str) else (type_spec[0], options)
    
    # Add save_to_disk option
    try:
        import folder_paths
        output_dir = folder_paths.get_output_directory()
    except ImportError:
        output_dir = "output"
    
    required["save_to_disk"] = ("BOOLEAN", {
        "default": True,
        "label_on": f"保存到 {output_dir}",
        "label_off": "不保存到本地"
    })
    
    return {
        "required": required,
        "optional": optional if optional else None,
    }


# ===== T008: Dynamic Node Class Factory =====

def create_image_node_class(model_config: dict) -> type:
    """
    Create a ComfyUI node class for an image model.
    
    Args:
        model_config: Model configuration dict from model_config.json
        
    Returns:
        type: A new ComfyUI node class
    """
    model_id = model_config.get("id", "")
    model_name = model_config.get("name", model_id)
    category = model_config.get("category", "image_t2i")
    endpoint = model_config.get("endpoint", "")
    is_async = model_config.get("is_async", True)
    response_type = model_config.get("response_type", "image_urls")
    
    # Build category path for menu
    category_map = {
        "image_t2i": "JieKou AI/Image/Text to Image",
        "image_edit": "JieKou AI/Image/Edit",
        "image_tool": "JieKou AI/Image/Tools",
    }
    node_category = category_map.get(category, "JieKou AI/Image")
    
    # Build INPUT_TYPES
    input_types = build_input_types(model_config, category)
    
    # Create the node class dynamically
    class_name = f"JieKou{model_id.replace('-', '_').replace('.', '_').title().replace('_', '')}"
    
    def generate(self, save_to_disk: bool = True, image_url: str = "", **kwargs):
        """Generate image using this model"""
        from ..utils.api_client import JiekouAPI, JiekouAPIError
        from ..utils.tensor_utils import url_to_tensor, base64_to_tensor
        
        logger.info(f"[JieKou] ========== {model_name} ==========")
        logger.info(f"[JieKou] Endpoint: {endpoint}")
        logger.info(f"[JieKou] Params: {list(kwargs.keys())}")
        
        try:
            api = JiekouAPI()
            
            # Build request data
            data = {}
            
            # Add image for edit models
            if image_url and image_url.strip():
                # Different models use different field names
                if "flux" in model_id.lower() or "seedream" in model_id.lower():
                    data["images"] = [image_url.strip()]
                elif "gemini" in model_id.lower():
                    # Gemini uses base64 without prefix
                    img_data = image_url.strip()
                    if "," in img_data:
                        img_data = img_data.split(",", 1)[1]
                    data["image_base64s"] = [img_data]
                else:
                    data["image"] = image_url.strip()
            
            # Add all other parameters
            for key, value in kwargs.items():
                if value is None:
                    continue
                if isinstance(value, str) and value.strip() == "":
                    continue
                data[key] = value
            
            # Log request
            log_data = {k: (v[:50] + "..." if isinstance(v, str) and len(v) > 100 else v) for k, v in data.items()}
            logger.info(f"[JieKou] Request: {log_data}")
            
            # Make API request
            result = api.call_model_and_wait(endpoint, data, is_async)
            
            # Handle response
            image_result_url = ""
            
            if is_async:
                image_result_url = api.get_image_result_url(result)
                if not image_result_url:
                    raise ValueError("任务完成但未返回图像 URL")
                image_tensor = url_to_tensor(image_result_url)
            
            elif response_type == "b64_json":
                images = result.get("data", [])
                if not images:
                    raise ValueError("API 未返回图像数据")
                first_image = images[0]
                if first_image.get("b64_json"):
                    image_tensor = base64_to_tensor(first_image["b64_json"])
                    image_result_url = "(base64 data)"
                elif first_image.get("url"):
                    image_result_url = first_image["url"]
                    image_tensor = url_to_tensor(image_result_url)
                else:
                    raise ValueError("API 返回的图像数据格式未知")
            
            elif response_type == "image_urls":
                urls = result.get("image_urls") or result.get("images", [])
                if not urls:
                    raise ValueError("API 未返回图像数据")
                image_result_url = urls[0] if isinstance(urls[0], str) else urls[0].get("url", "")
                image_tensor = url_to_tensor(image_result_url)
            
            else:
                raise ValueError(f"不支持的响应类型: {response_type}")
            
            logger.info(f"[JieKou] Generated image: shape={image_tensor.shape}")
            
            # Save to disk if requested
            output_path = ""
            if save_to_disk:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
                filename = f"jiekou_{model_id}_{int(time.time())}.png"
                output_path = os.path.join(output_dir, filename)
                
                img_np = image_tensor[0].cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)
                img.save(output_path, format="PNG")
                logger.info(f"[JieKou] Saved to: {output_path}")
            
            # Preview
            preview_images = _save_for_preview(image_tensor)
            
            ui_info = {"images": preview_images}
            if save_to_disk and output_path:
                ui_info["text"] = [f"✅ 图片已保存: {os.path.basename(output_path)}"]
            
            return {"ui": ui_info, "result": (image_tensor, image_result_url,)}
        
        except JiekouAPIError as e:
            logger.error(f"[JieKou] API Error: {e.message}")
            raise RuntimeError(e.message)
        except Exception as e:
            logger.error(f"[JieKou] Error: {e}")
            raise
    
    # Create the class
    node_class = type(class_name, (), {
        "CATEGORY": node_category,
        "FUNCTION": "generate",
        "RETURN_TYPES": ("IMAGE", "STRING",),
        "RETURN_NAMES": ("image", "image_url",),
        "OUTPUT_NODE": True,
        "INPUT_TYPES": classmethod(lambda cls: input_types),
        "generate": generate,
        "_model_config": model_config,
    })
    
    return node_class


def create_video_node_class(model_config: dict) -> type:
    """
    Create a ComfyUI node class for a video model.
    
    Args:
        model_config: Model configuration dict from model_config.json
        
    Returns:
        type: A new ComfyUI node class
    """
    model_id = model_config.get("id", "")
    model_name = model_config.get("name", model_id)
    category = model_config.get("category", "video_t2v")
    endpoint = model_config.get("endpoint", "")
    is_async = model_config.get("is_async", True)
    
    # Build category path for menu
    category_map = {
        "video_t2v": "JieKou AI/Video/Text to Video",
        "video_i2v": "JieKou AI/Video/Image to Video",
        "video_v2v": "JieKou AI/Video/Video to Video",
    }
    node_category = category_map.get(category, "JieKou AI/Video")
    
    # Build INPUT_TYPES
    input_types = build_input_types(model_config, category)
    
    # Create the node class dynamically
    class_name = f"JieKou{model_id.replace('-', '_').replace('.', '_').title().replace('_', '')}"
    
    def generate_video(self, save_to_disk: bool = True, image_url: str = "", video_url: str = "", **kwargs):
        """Generate video using this model"""
        from ..utils.api_client import JiekouAPI, JiekouAPIError
        from ..utils.tensor_utils import video_to_frames
        
        logger.info(f"[JieKou] ========== {model_name} ==========")
        logger.info(f"[JieKou] Endpoint: {endpoint}")
        logger.info(f"[JieKou] Params: {list(kwargs.keys())}")
        
        try:
            api = JiekouAPI()
            
            # Build request data
            data = {}
            
            # Add video for V2V models
            if video_url and video_url.strip():
                data["video"] = video_url.strip()
            
            # Add image for I2V models
            if image_url and image_url.strip():
                # Check if endpoint uses nested structure (wan2.6 style)
                if "wan2.6" in model_id.lower():
                    data["input"] = {"img_url": image_url.strip()}
                else:
                    data["image"] = image_url.strip()
            
            # Add all other parameters
            for key, value in kwargs.items():
                if value is None:
                    continue
                if isinstance(value, str) and value.strip() == "":
                    continue
                data[key] = value
            
            # Log request
            log_data = {k: (v[:50] + "..." if isinstance(v, str) and len(v) > 100 else v) for k, v in data.items()}
            logger.info(f"[JieKou] Request: {log_data}")
            
            # Make API request
            result = api.call_model_and_wait(endpoint, data, is_async)
            
            # Get video URL
            video_url = api.get_video_result_url(result)
            if not video_url:
                raise ValueError("任务完成但未返回视频 URL")
            
            # Download video
            video_bytes = api.download_file(video_url)
            
            # Save to disk if requested
            video_path = ""
            filename = ""
            if save_to_disk:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
                filename = f"jiekou_{model_id}_{int(time.time())}.mp4"
                video_path = os.path.join(output_dir, filename)
                
                with open(video_path, "wb") as f:
                    f.write(video_bytes)
                logger.info(f"[JieKou] Video saved to: {video_path}")
            
            # Decode to frames
            frames_tensor = video_to_frames(video_bytes)
            logger.info(f"[JieKou] Generated video: {frames_tensor.shape[0]} frames")
            
            ui_info = {}
            if save_to_disk and filename:
                ui_info["text"] = [f"✅ 视频已保存: {filename}"]
            
            return {"ui": ui_info, "result": (frames_tensor, video_url,)}
        
        except JiekouAPIError as e:
            logger.error(f"[JieKou] API Error: {e.message}")
            raise RuntimeError(e.message)
        except Exception as e:
            logger.error(f"[JieKou] Error: {e}")
            raise
    
    # Create the class
    node_class = type(class_name, (), {
        "CATEGORY": node_category,
        "FUNCTION": "generate_video",
        "RETURN_TYPES": ("IMAGE", "STRING",),
        "RETURN_NAMES": ("frames", "video_url",),
        "OUTPUT_NODE": True,
        "INPUT_TYPES": classmethod(lambda cls: input_types),
        "generate_video": generate_video,
        "_model_config": model_config,
    })
    
    return node_class


def create_audio_node_class(model_config: dict) -> type:
    """
    Create a ComfyUI node class for an audio model.
    
    Args:
        model_config: Model configuration dict from model_config.json
        
    Returns:
        type: A new ComfyUI node class
    """
    model_id = model_config.get("id", "")
    model_name = model_config.get("name", model_id)
    category = model_config.get("category", "audio_tts")
    endpoint = model_config.get("endpoint", "")
    is_async = model_config.get("is_async", True)
    
    # Build category path for menu
    category_map = {
        "audio_tts": "JieKou AI/Audio/Text to Speech",
        "audio_asr": "JieKou AI/Audio/Speech to Text",
    }
    node_category = category_map.get(category, "JieKou AI/Audio")
    
    # Build INPUT_TYPES
    input_types = build_input_types(model_config, category)
    
    # Create the node class dynamically
    class_name = f"JieKou{model_id.replace('-', '_').replace('.', '_').title().replace('_', '')}"
    
    def generate_audio(self, save_to_disk: bool = True, **kwargs):
        """Generate audio using this model"""
        from ..utils.api_client import JiekouAPI, JiekouAPIError
        
        logger.info(f"[JieKou] ========== {model_name} ==========")
        logger.info(f"[JieKou] Endpoint: {endpoint}")
        logger.info(f"[JieKou] Params: {list(kwargs.keys())}")
        
        try:
            api = JiekouAPI()
            
            # Build request data
            data = {}
            for key, value in kwargs.items():
                if value is None:
                    continue
                if isinstance(value, str) and value.strip() == "":
                    continue
                data[key] = value
            
            logger.info(f"[JieKou] Request: {data}")
            
            # Make API request
            result = api.call_model_and_wait(endpoint, data, is_async)
            
            # Get audio URL
            audio_url = api.get_audio_result_url(result)
            if not audio_url:
                # Some sync APIs return audio directly
                audio_url = result.get("audio_url", "")
            
            if not audio_url:
                raise ValueError("任务完成但未返回音频 URL")
            
            # Download audio
            audio_bytes = api.download_file(audio_url)
            
            # Save to disk if requested
            audio_path = ""
            filename = ""
            if save_to_disk:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
                filename = f"jiekou_{model_id}_{int(time.time())}.mp3"
                audio_path = os.path.join(output_dir, filename)
                
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
                logger.info(f"[JieKou] Audio saved to: {audio_path}")
            
            ui_info = {}
            if save_to_disk and filename:
                ui_info["text"] = [f"✅ 音频已保存: {filename}"]
            
            # Return path and URL
            return {"ui": ui_info, "result": (audio_path, audio_url,)}
        
        except JiekouAPIError as e:
            logger.error(f"[JieKou] API Error: {e.message}")
            raise RuntimeError(e.message)
        except Exception as e:
            logger.error(f"[JieKou] Error: {e}")
            raise
    
    # Create the class
    node_class = type(class_name, (), {
        "CATEGORY": node_category,
        "FUNCTION": "generate_audio",
        "RETURN_TYPES": ("STRING", "STRING",),
        "RETURN_NAMES": ("audio_path", "audio_url",),
        "OUTPUT_NODE": True,
        "INPUT_TYPES": classmethod(lambda cls: input_types),
        "generate_audio": generate_audio,
        "_model_config": model_config,
    })
    
    return node_class


def _save_for_preview(tensor: torch.Tensor, subfolder: str = "jiekou") -> list:
    """Save image tensor to temp folder for UI preview"""
    import folder_paths
    import uuid
    
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    output_dir = folder_paths.get_temp_directory()
    full_output_folder = os.path.join(output_dir, subfolder)
    os.makedirs(full_output_folder, exist_ok=True)
    
    results = []
    for i in range(tensor.shape[0]):
        img_np = tensor[i].cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        filename = f"preview_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(full_output_folder, filename)
        img.save(filepath, format="PNG")
        
        results.append({
            "filename": filename,
            "subfolder": subfolder,
            "type": "temp"
        })
    
    return results


# ===== Factory Function =====

def create_node_classes_from_config() -> tuple[dict, dict]:
    """
    Create all node classes from model_config.json.
    
    Returns:
        tuple: (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
    """
    from ..utils.model_config_loader import get_model_config_loader
    
    loader = get_model_config_loader()
    all_models = loader.get_all_models()
    
    logger.info(f"[JieKou] Creating node classes for {len(all_models)} models...")
    
    class_mappings = {}
    display_mappings = {}
    
    # Count by category for logging
    category_counts = {}
    
    for model in all_models:
        model_dict = {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "category": model.category,
            "endpoint": model.endpoint,
            "is_async": model.is_async,
            "response_type": model.response_type,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum,
                    "minimum": p.minimum,
                    "maximum": p.maximum,
                }
                for p in model.parameters
            ],
            "product_ids": model.product_ids,
            "valid_combinations": model.valid_combinations,
        }
        
        try:
            if model.category.startswith("image_"):
                node_class = create_image_node_class(model_dict)
            elif model.category.startswith("video_"):
                node_class = create_video_node_class(model_dict)
            elif model.category.startswith("audio_"):
                node_class = create_audio_node_class(model_dict)
            else:
                logger.warning(f"[JieKou] Unknown category for model {model.id}: {model.category}")
                continue
            
            class_name = node_class.__name__
            class_mappings[class_name] = node_class
            display_mappings[class_name] = model.name or model.id
            
            # Track category counts
            cat = model.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            logger.debug(f"[JieKou] Created node class: {class_name} -> {model.name}")
        
        except Exception as e:
            logger.error(f"[JieKou] Failed to create node for {model.id}: {e}")
            continue
    
    logger.info(f"[JieKou] Created {len(class_mappings)} model node classes")
    logger.info(f"[JieKou] By category: {category_counts}")
    return class_mappings, display_mappings

