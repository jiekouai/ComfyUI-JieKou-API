"""
JiekouAI ComfyUI Plugin - Image Generation Nodes
Text-to-Image, Image-to-Image, and Image Tools

Parameters are dynamically rendered based on selected model.
See model_registry.py for model-specific parameters.
"""

import torch
import logging
import os
import uuid
import numpy as np
from PIL import Image

logger = logging.getLogger("[JieKou]")


def save_tensor_for_preview(tensor: torch.Tensor, subfolder: str = "jiekou") -> list:
    """
    Save image tensor to ComfyUI's temp folder for preview.
    
    Args:
        tensor: Image tensor [B, H, W, C] or [H, W, C]
        subfolder: Subfolder name in temp directory
        
    Returns:
        List of file info dicts for UI preview
    """
    import folder_paths
    
    # Ensure tensor is 4D [B, H, W, C]
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    output_dir = folder_paths.get_temp_directory()
    full_output_folder = os.path.join(output_dir, subfolder)
    os.makedirs(full_output_folder, exist_ok=True)
    
    results = []
    for i in range(tensor.shape[0]):
        # Convert tensor to PIL Image
        img_np = tensor[i].cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        # Generate unique filename
        filename = f"preview_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(full_output_folder, filename)
        
        img.save(filepath, format="PNG")
        
        results.append({
            "filename": filename,
            "subfolder": subfolder,
            "type": "temp"
        })
    
    return results


# ===== 硬编码模型列表 =====
# 文生图模型 ID
IMAGE_T2I_MODELS = [
    "gpt-image-1",
    "flux-1-kontext-pro",
    "flux-1-kontext-dev",
    "flux-1-kontext-max",
    "flux-2-pro",
    "flux-2-dev",
    "flux-2-flex",
    "seedream-3-0-t2i-250415",
    "seedream-4-0",
    "seedream-4-5",
    "qwen-image-t2i",
    "gemini-2.5-flash-image-t2i",
    "gemini-3-pro-image-preview-t2i",
    "hunyuan-image-3",
    "midjourney-txt2img",
    "z-image-turbo",
    "z-image-turbo-lora",
    "nano-banana-pro-light-t2i",
]

# 图生图/编辑模型 ID
IMAGE_EDIT_MODELS = [
    "gpt-image-1-edit",           # GPT 图像编辑 (sync)
    "flux-1-kontext-pro",          # FLUX.1 Kontext Pro (async)
    "flux-1-kontext-dev",          # FLUX.1 Kontext Dev (async)
    "flux-1-kontext-max",          # FLUX.1 Kontext Max (async)
    "qwen-image-edit",             # Qwen 图像编辑 (async)
    "gemini-2.5-flash-image-edit", # Gemini 2.5 Flash 图片编辑 (sync)
    "gemini-3-pro-image-preview-edit",  # Gemini 3 Pro 图片编辑 (sync)
    "nano-banana-pro-light-i2i",   # Nano Banana 图生图 (sync)
    "seedream-4-0",                # Seedream 4.0 (支持图生图)
    "seedream-4-5",                # Seedream 4.5 (支持图生图)
]


def _get_models_for_validation(model_ids: list) -> list:
    """
    Get model list including both IDs and names for ComfyUI validation.
    Frontend displays names, but ComfyUI validates against this list.
    """
    try:
        from ..utils.model_registry import get_model_registry
        registry = get_model_registry()
        
        result = []
        for model_id in model_ids:
            result.append(model_id)
            # Also add the display name for validation
            model = registry.get_model(model_id)
            if model and model.name and model.name != model_id:
                result.append(model.name)
        return result
    except Exception:
        return model_ids


def _resolve_model_id(model_value: str, model_ids: list) -> str:
    """
    Resolve model value (could be ID or name) to model ID.
    Frontend may send name, we need to convert to ID for API calls.
    """
    # If it's already a valid ID, return as-is
    if model_value in model_ids:
        return model_value
    
    # Try to find by name
    try:
        from ..utils.model_registry import get_model_registry
        registry = get_model_registry()
        
        for model_id in model_ids:
            model = registry.get_model(model_id)
            if model and model.name == model_value:
                return model_id
    except Exception:
        pass
    
    # Fallback: return as-is (might fail later, but at least we tried)
    return model_value


class JieKouTextToImage:
    """
    Generate images from text prompts using JiekouAI models.
    
    Parameters are dynamically loaded based on selected model.
    """
    
    CATEGORY = "JieKou/Image"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "image_url",)
    OUTPUT_NODE = True  # 允许直接运行
    
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        
        # Get model list with both IDs and names for validation
        models = _get_models_for_validation(IMAGE_T2I_MODELS)
        
        # Only define required inputs
        # Optional parameters will be added dynamically by JavaScript
        return {
            "required": {
                "model": (models, {
                    "default": IMAGE_T2I_MODELS[0]
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "描述您想要生成的图像...",
                    "dynamicPrompts": True  # 允许右键转换为输入槽
                }),
                "save_to_disk": ("BOOLEAN", {
                    "default": True,
                    "label_on": f"保存到 {output_dir}",
                    "label_off": "不保存到本地"
                }),
            },
            "optional": {
                # Hidden input for dynamic parameters (JSON string)
                # JavaScript will serialize dynamic widget values here
                "_dynamic_params": ("STRING", {
                    "default": "{}",
                    "multiline": False,
                }),
            }
        }
    
    def generate(self, model: str, prompt: str, save_to_disk: bool = True, _dynamic_params: str = "{}", **kwargs):
        """
        Generate image from text prompt.
        
        Args:
            model: Model ID or name (frontend may send name)
            prompt: Text prompt
            save_to_disk: Whether to save image to output folder
            _dynamic_params: JSON string of dynamic parameters from JavaScript
            **kwargs: Model-specific parameters (dynamically provided)
        
        Returns:
            tuple: (IMAGE tensor, image_url/path string)
        """
        import json
        
        # Parse dynamic parameters from JSON
        try:
            dynamic_params = json.loads(_dynamic_params) if _dynamic_params else {}
        except json.JSONDecodeError:
            dynamic_params = {}
        
        # Merge dynamic params into kwargs
        kwargs.update(dynamic_params)
        
        # Resolve model name to ID (frontend displays names, but API needs IDs)
        model_id = _resolve_model_id(model, IMAGE_T2I_MODELS)
        
        logger.info(f"[JieKou] ========== Text-to-Image ==========")
        logger.info(f"[JieKou] Model: {model} -> {model_id}")
        logger.info(f"[JieKou] Prompt: {prompt[:100]}...")
        logger.info(f"[JieKou] Save to disk: {save_to_disk}")
        logger.info(f"[JieKou] Dynamic params JSON: {_dynamic_params[:200]}...")
        logger.info(f"[JieKou] Parsed params: {kwargs}")
        
        if not prompt.strip():
            raise ValueError("Prompt 不能为空")
        
        try:
            from ..utils.api_client import JiekouAPI, JiekouAPIError
            from ..utils.tensor_utils import url_to_tensor, base64_to_tensor
            from ..utils.model_registry import get_model_registry
            
            api = JiekouAPI()
            registry = get_model_registry()
            
            # Resolve model by ID or name (frontend may pass display name)
            model_config = registry.resolve_model(model)
            if not model_config:
                raise ValueError(f"未知模型: {model}")
            
            model_id = model_config.id
            if model_id != model:
                logger.info(f"[JieKou] Resolved model: {model} -> {model_id}")
            
            endpoint = model_config.endpoint
            is_async = model_config.is_async
            response_type = model_config.response_type
            
            # Build request data
            # For midjourney, use "text" instead of "prompt"
            if model_id == "midjourney-txt2img":
                data = {"text": prompt}
            else:
                data = {"prompt": prompt}
            
            # Add model ID if required
            if model_id in ["gpt-image-1", "seedream-3-0-t2i-250415"]:
                data["model"] = model_id
            
            # Add all dynamic parameters (filter out empty strings and None)
            for key, value in kwargs.items():
                if key == "extra_params":
                    continue
                # Skip None and empty strings (API may reject empty optional params)
                if value is None:
                    continue
                if isinstance(value, str) and value.strip() == "":
                    continue
                data[key] = value
            
            # Log request body (hide base64 data for readability)
            log_data = {k: (v[:50] + "..." if isinstance(v, str) and len(v) > 100 else v) for k, v in data.items()}
            logger.info(f"[JieKou] ===== API Request =====")
            logger.info(f"[JieKou] Endpoint: {endpoint}, async={is_async}")
            logger.info(f"[JieKou] Request body: {log_data}")
            
            # Make API request
            # Sync APIs may take longer, use 20 min timeout; async just submits task
            request_timeout = 60 if is_async else 1200
            response = api._request("POST", endpoint, data=data, timeout=request_timeout)
            
            # Handle response based on type
            image_url = ""  # Track URL for output
            
            if is_async:
                # Async API - poll for result
                task_id = response.get("task_id")
                if not task_id:
                    raise ValueError("API 未返回 task_id")
                
                logger.info(f"[JieKou] Task submitted: {task_id}")
                result = api.poll_task_until_complete(task_id)
                
                image_url = api.get_image_result_url(result)
                if not image_url:
                    raise ValueError("任务完成但未返回图像 URL")
                
                image_tensor = url_to_tensor(image_url)
            
            elif response_type == "b64_json":
                # Sync API with base64 response (data array)
                images = response.get("data", [])
                if not images:
                    raise ValueError("API 未返回图像数据")
                
                first_image = images[0]
                if "b64_json" in first_image and first_image["b64_json"]:
                    image_tensor = base64_to_tensor(first_image["b64_json"])
                    image_url = "(base64 data)"  # No URL for base64 response
                elif "url" in first_image and first_image["url"]:
                    image_url = first_image["url"]
                    image_tensor = url_to_tensor(image_url)
                else:
                    # Check for error messages in response (e.g., content blocked)
                    error_msg = ""
                    if isinstance(first_image, dict):
                        # Check revised_prompt for error info (some APIs embed errors here)
                        revised = first_image.get("revised_prompt", "")
                        if "blocked" in revised.lower() or "error" in revised.lower():
                            error_msg = revised
                        # Check for explicit error field
                        elif first_image.get("error"):
                            error_msg = first_image.get("error")
                    
                    if error_msg:
                        raise ValueError(f"API 请求被拒绝: {error_msg[:200]}")
                    else:
                        raise ValueError("API 返回的图像数据格式未知")
            
            elif response_type == "image_urls":
                # Sync API with URL response
                urls = response.get("image_urls") or response.get("images", [])
                if not urls:
                    raise ValueError("API 未返回图像数据")
                # Handle both string URL and dict with url field
                image_url = urls[0] if isinstance(urls[0], str) else urls[0].get("url", "")
                image_tensor = url_to_tensor(image_url)
            
            else:
                raise ValueError(f"不支持的响应类型: {response_type}")
            
            logger.info(f"[JieKou] Generated image: shape={image_tensor.shape}")
            
            # Save to output folder if requested
            output_path = ""
            if save_to_disk:
                import folder_paths
                import time
                output_dir = folder_paths.get_output_directory()
                filename = f"jiekou_image_{int(time.time())}.png"
                output_path = os.path.join(output_dir, filename)
                
                # Save tensor as image
                img_np = image_tensor[0].cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)
                img.save(output_path, format="PNG")
                
                logger.info(f"[JieKou] Image saved to: {output_path}")
            
            # Save for preview
            preview_images = save_tensor_for_preview(image_tensor)
            
            # Always return the API's image_url (not local path)
            # so downstream nodes can use it as input
            
            # Build UI info
            ui_info = {"images": preview_images}
            if save_to_disk:
                ui_info["text"] = [f"✅ 图片已保存: {os.path.basename(output_path)}"]
            
            return {"ui": ui_info, "result": (image_tensor, image_url,)}
        
        except JiekouAPIError as e:
            logger.error(f"[JieKou] API Error: {e.message}")
            raise RuntimeError(e.message)
        
        except Exception as e:
            logger.error(f"[JieKou] Error: {e}")
            raise


class JieKouImageToImage:
    """
    Transform images using JiekouAI models with text guidance.
    
    Parameters are dynamically loaded based on selected model.
    
    Supports two ways to provide input image:
    - image: Connect an IMAGE tensor from other nodes
    - image_url: Directly provide an image URL (higher priority if both provided)
    """
    
    CATEGORY = "JieKou/Image"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "image_url",)
    OUTPUT_NODE = True  # 允许直接运行
    
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        
        # Get model list with both IDs and names for validation
        models = _get_models_for_validation(IMAGE_EDIT_MODELS)
        
        return {
            "required": {
                "model": (models, {
                    "default": IMAGE_EDIT_MODELS[0]
                }),
                # Image input: URL or base64 string, positioned after model
                "image_url": ("STRING", {
                    "default": "",
                    "placeholder": "输入图片 URL 或 base64",
                    "dynamicPrompts": True  # 允许右键转换为输入槽
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "描述想要的变换效果...",
                    "dynamicPrompts": True  # 允许右键转换为输入槽
                }),
                "save_to_disk": ("BOOLEAN", {
                    "default": True,
                    "label_on": f"保存到 {output_dir}",
                    "label_off": "不保存到本地"
                }),
            },
            "optional": {
                # Hidden input for dynamic parameters (JSON string)
                "_dynamic_params": ("STRING", {
                    "default": "{}",
                    "multiline": False,
                }),
            }
        }
    
    def generate(self, model: str, image_url: str, prompt: str, save_to_disk: bool = True, _dynamic_params: str = "{}", **kwargs):
        """
        Transform input image based on prompt.
        
        Args:
            model: Model ID to use
            image_url: Input image (URL or base64 string)
            prompt: Text prompt
            save_to_disk: Whether to save image to output folder
            _dynamic_params: JSON string of dynamic parameters from JavaScript
            **kwargs: Model-specific parameters
        
        Returns:
            tuple: (IMAGE tensor, image_url/path string)
        """
        import json
        
        # Parse dynamic parameters from JSON
        try:
            dynamic_params = json.loads(_dynamic_params) if _dynamic_params else {}
        except json.JSONDecodeError:
            dynamic_params = {}
        
        # Merge dynamic params into kwargs
        kwargs.update(dynamic_params)
        
        logger.info(f"[JieKou] ========== Image-to-Image ==========")
        logger.info(f"[JieKou] Model: {model}")
        logger.info(f"[JieKou] Image URL: {image_url[:100] if image_url else 'None'}...")
        logger.info(f"[JieKou] Prompt: {prompt[:100]}...")
        logger.info(f"[JieKou] Save to disk: {save_to_disk}")
        logger.info(f"[JieKou] Dynamic params JSON: {_dynamic_params[:200]}...")
        logger.info(f"[JieKou] Parsed params: {kwargs}")
        
        if not prompt.strip():
            raise ValueError("Prompt 不能为空")
        
        if not image_url or not image_url.strip():
            raise ValueError("请提供输入图片 URL 或 base64")
        
        try:
            from ..utils.api_client import JiekouAPI, JiekouAPIError
            from ..utils.tensor_utils import url_to_tensor, base64_to_tensor
            from ..utils.model_registry import get_model_registry
            
            api = JiekouAPI()
            registry = get_model_registry()
            
            # Resolve model by ID or name (frontend may pass display name)
            model_config = registry.resolve_model(model)
            if not model_config:
                raise ValueError(f"未知模型: {model}")
            
            model_id = model_config.id
            if model_id != model:
                logger.info(f"[JieKou] Resolved model: {model} -> {model_id}")
            
            # Handle input: URL, base64, or local file path
            input_image_data = image_url.strip()
            
            # Check if it's a local file path and convert to base64
            from ..utils.tensor_utils import is_local_file_path, local_file_to_base64
            if is_local_file_path(input_image_data):
                logger.info(f"[JieKou] Detected local file path, converting to base64...")
                input_image_data = local_file_to_base64(input_image_data)
            
            # Check if it's a URL (not base64)
            is_url = input_image_data.startswith("http://") or input_image_data.startswith("https://")
            
            endpoint = model_config.endpoint
            is_async = model_config.is_async
            response_type = model_config.response_type
            
            # Build request data
            data = {"prompt": prompt}
            
            # Add image input (different field names for different models)
            # Use input_image_data which is either URL string or data:image/png;base64,xxx
            if model_id.startswith("flux-") or model_id == "seedream-4-0" or model_id == "nano-banana-pro-light-i2i":
                # These models use images array
                data["images"] = [input_image_data]
            elif model_id == "seedream-4-5":
                # Seedream 4.5 uses image array (singular, but array type)
                data["image"] = [input_image_data]
            elif model_id.startswith("gemini-"):
                # Gemini uses image_base64s array - needs raw base64 without prefix
                if is_url:
                    # For URL input, Gemini might not support it directly, try anyway
                    data["image_base64s"] = [input_image_data]
                else:
                    # Extract base64 without data:image prefix (e.g., "data:image/png;base64,xxx" -> "xxx")
                    raw_b64 = input_image_data
                    if "," in input_image_data:
                        raw_b64 = input_image_data.split(",", 1)[1]
                    data["image_base64s"] = [raw_b64]
            elif model_id == "gpt-image-1-edit":
                # GPT uses image array with model field
                data["image"] = [input_image_data]
                data["model"] = "gpt-image-1"
            else:
                # Default: single image field
                data["image"] = input_image_data
            
            # Add all dynamic parameters (filter out empty strings and None)
            for key, value in kwargs.items():
                if key == "extra_params":
                    continue
                # Skip None and empty strings (API may reject empty optional params)
                if value is None:
                    continue
                if isinstance(value, str) and value.strip() == "":
                    continue
                data[key] = value
            
            # Log request body (hide base64 data for readability)
            log_data = {}
            for k, v in data.items():
                if isinstance(v, str) and len(v) > 100:
                    log_data[k] = v[:50] + "..."
                elif isinstance(v, list) and v and isinstance(v[0], str) and len(v[0]) > 100:
                    log_data[k] = [s[:50] + "..." for s in v]
                else:
                    log_data[k] = v
            logger.info(f"[JieKou] ===== API Request =====")
            logger.info(f"[JieKou] Endpoint: {endpoint}, async={is_async}")
            logger.info(f"[JieKou] Request body: {log_data}")
            
            # Make API request
            # Sync APIs may take longer, use 20 min timeout; async just submits task
            request_timeout = 60 if is_async else 1200
            response = api._request("POST", endpoint, data=data, timeout=request_timeout)
            
            # Handle response
            image_url = ""  # Track URL for output
            
            if is_async:
                task_id = response.get("task_id")
                if not task_id:
                    raise ValueError("API 未返回 task_id")
                
                result = api.poll_task_until_complete(task_id)
                image_url = api.get_image_result_url(result)
                if not image_url:
                    raise ValueError("任务完成但未返回图像 URL")
                
                image_tensor = url_to_tensor(image_url)
            
            elif response_type == "b64_json":
                # Handle data array with b64_json or url
                images = response.get("data", [])
                if not images:
                    raise ValueError("API 未返回图像数据")
                
                first_image = images[0]
                if "b64_json" in first_image and first_image["b64_json"]:
                    image_tensor = base64_to_tensor(first_image["b64_json"])
                    image_url = "(base64 data)"
                elif "url" in first_image and first_image["url"]:
                    image_url = first_image["url"]
                    image_tensor = url_to_tensor(image_url)
                else:
                    # Check for error messages in response (e.g., content blocked)
                    error_msg = ""
                    if isinstance(first_image, dict):
                        # Check revised_prompt for error info (some APIs embed errors here)
                        revised = first_image.get("revised_prompt", "")
                        if "blocked" in revised.lower() or "error" in revised.lower():
                            error_msg = revised
                        # Check for explicit error field
                        elif first_image.get("error"):
                            error_msg = first_image.get("error")
                    
                    if error_msg:
                        raise ValueError(f"API 请求被拒绝: {error_msg[:200]}")
                    else:
                        raise ValueError("API 返回的图像数据格式未知")
            
            elif response_type == "image_urls":
                urls = response.get("image_urls") or response.get("images", [])
                if not urls:
                    raise ValueError("API 未返回图像数据")
                # Handle both string URL and dict with url field
                image_url = urls[0] if isinstance(urls[0], str) else urls[0].get("url", "")
                image_tensor = url_to_tensor(image_url)
            
            else:
                raise ValueError(f"不支持的响应类型: {response_type}")
            
            logger.info(f"[JieKou] Generated image: shape={image_tensor.shape}")
            
            # Save to output folder if requested
            output_path = ""
            if save_to_disk:
                import folder_paths
                import time
                output_dir = folder_paths.get_output_directory()
                filename = f"jiekou_i2i_{int(time.time())}.png"
                output_path = os.path.join(output_dir, filename)
                
                # Save tensor as image
                img_np = image_tensor[0].cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)
                img.save(output_path, format="PNG")
                
                logger.info(f"[JieKou] Image saved to: {output_path}")
            
            # Save for preview
            preview_images = save_tensor_for_preview(image_tensor)
            
            # Always return the API's image_url (not local path)
            # so downstream nodes can use it as input
            
            # Build UI info
            ui_info = {"images": preview_images}
            if save_to_disk:
                ui_info["text"] = [f"✅ 图片已保存: {os.path.basename(output_path)}"]
            
            return {"ui": ui_info, "result": (image_tensor, image_url,)}
        
        except JiekouAPIError as e:
            logger.error(f"[JieKou] API Error: {e.message}")
            raise RuntimeError(e.message)
        
        except Exception as e:
            logger.error(f"[JieKou] Error: {e}")
            raise


class JieKouImageUpscale:
    """
    Upscale images to higher resolution (2K/4K/8K)
    
    Supports two ways to provide input image:
    - image: Connect an IMAGE tensor from other nodes
    - image_url: Directly provide an image URL (higher priority if both provided)
    """
    
    CATEGORY = "JieKou/Image"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "image_path",)
    OUTPUT_NODE = True  # 允许直接运行
    
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        
        return {
            "required": {
                # Image input: URL or base64 string
                "image_url": ("STRING", {
                    "default": "",
                    "placeholder": "输入图片 URL 或 base64",
                    "dynamicPrompts": True  # 允许右键转换为输入槽
                }),
                "save_to_disk": ("BOOLEAN", {
                    "default": True,
                    "label_on": f"保存到 {output_dir}",
                    "label_off": "不保存到本地"
                }),
            },
            "optional": {
                "resolution": (["2k", "4k", "8k"], {
                    "default": "4k"
                }),
                "output_format": (["jpeg", "png", "webp"], {
                    "default": "jpeg"
                }),
            }
        }
    
    def upscale(
        self,
        image_url: str,
        save_to_disk: bool = True,
        resolution: str = "4k",
        output_format: str = "jpeg"
    ):
        """Upscale image to higher resolution"""
        logger.info(f"[JieKou] Image Upscale: resolution={resolution}, save_to_disk={save_to_disk}")
        logger.info(f"[JieKou] Image URL: {image_url[:100] if image_url else 'None'}...")
        
        if not image_url or not image_url.strip():
            raise ValueError("请提供输入图片 URL 或 base64")
        
        try:
            from ..utils.api_client import JiekouAPI, JiekouAPIError
            from ..utils.tensor_utils import url_to_tensor, is_local_file_path, local_file_to_base64
            
            api = JiekouAPI()
            
            input_image_data = image_url.strip()
            
            # Check if it's a local file path and convert to base64
            if is_local_file_path(input_image_data):
                logger.info(f"[JieKou] Detected local file path, converting to base64...")
                input_image_data = local_file_to_base64(input_image_data)
            
            data = {
                "image": input_image_data,
                "resolution": resolution,
                "output_format": output_format
            }
            
            # Async API, just submitting task
            response = api._request("POST", "/v1/images/upscaler", data=data, timeout=60)
            task_id = response.get("task_id")
            if not task_id:
                raise ValueError("API 未返回 task_id")
            
            result = api.poll_task_until_complete(task_id)
            image_url = api.get_image_result_url(result)
            if not image_url:
                raise ValueError("任务完成但未返回图像 URL")
            
            image_tensor = url_to_tensor(image_url)
            logger.info(f"[JieKou] Upscaled image: shape={image_tensor.shape}")
            
            # Save to output folder if requested
            output_path = ""
            if save_to_disk:
                import folder_paths
                import time
                output_dir = folder_paths.get_output_directory()
                ext = "png" if output_format == "png" else "jpg"
                filename = f"jiekou_upscale_{int(time.time())}.{ext}"
                output_path = os.path.join(output_dir, filename)
                
                img_np = image_tensor[0].cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)
                img.save(output_path)
                
                logger.info(f"[JieKou] Image saved to: {output_path}")
            
            # Save for preview
            preview_images = save_tensor_for_preview(image_tensor)
            
            # Always return the API's image_url (not local path)
            ui_info = {"images": preview_images}
            if save_to_disk:
                ui_info["text"] = [f"✅ 图片已保存: {os.path.basename(output_path)}"]
            
            return {"ui": ui_info, "result": (image_tensor, image_url,)}
        
        except JiekouAPIError as e:
            logger.error(f"[JieKou] API Error: {e.message}")
            raise RuntimeError(e.message)
        except Exception as e:
            logger.error(f"[JieKou] Error: {e}")
            raise


class JieKouRemoveBackground:
    """
    Remove background from images
    
    Supports two ways to provide input image:
    - image: Connect an IMAGE tensor from other nodes
    - image_url: Directly provide an image URL (higher priority if both provided)
    """
    
    CATEGORY = "JieKou/Image"
    FUNCTION = "remove_bg"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "image_path",)
    OUTPUT_NODE = True  # 允许直接运行
    
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        
        return {
            "required": {
                # Image input: URL or base64 string
                "image_url": ("STRING", {
                    "default": "",
                    "placeholder": "输入图片 URL 或 base64",
                    "dynamicPrompts": True  # 允许右键转换为输入槽
                }),
                "save_to_disk": ("BOOLEAN", {
                    "default": True,
                    "label_on": f"保存到 {output_dir}",
                    "label_off": "不保存到本地"
                }),
            },
        }
    
    def remove_bg(self, image_url: str, save_to_disk: bool = True):
        """Remove background from image"""
        logger.info(f"[JieKou] Removing background... save_to_disk={save_to_disk}")
        logger.info(f"[JieKou] Image URL: {image_url[:100] if image_url else 'None'}...")
        
        if not image_url or not image_url.strip():
            raise ValueError("请提供输入图片 URL 或 base64")
        
        try:
            from ..utils.api_client import JiekouAPI, JiekouAPIError
            from ..utils.tensor_utils import url_to_tensor, is_local_file_path, local_file_to_base64
            
            api = JiekouAPI()
            
            input_image_data = image_url.strip()
            
            # Check if it's a local file path and convert to base64
            if is_local_file_path(input_image_data):
                logger.info(f"[JieKou] Detected local file path, converting to base64...")
                input_image_data = local_file_to_base64(input_image_data)
            
            data = {
                "image": input_image_data,
            }
            
            # Async API, just submitting task
            response = api._request("POST", "/v1/images/remove-background", data=data, timeout=60)
            task_id = response.get("task_id")
            if not task_id:
                raise ValueError("API 未返回 task_id")
            
            result = api.poll_task_until_complete(task_id)
            image_url = api.get_image_result_url(result)
            if not image_url:
                raise ValueError("任务完成但未返回图像 URL")
            
            image_tensor = url_to_tensor(image_url)
            logger.info(f"[JieKou] Background removed: shape={image_tensor.shape}")
            
            # Save to output folder if requested
            output_path = ""
            if save_to_disk:
                import folder_paths
                import time
                output_dir = folder_paths.get_output_directory()
                filename = f"jiekou_nobg_{int(time.time())}.png"
                output_path = os.path.join(output_dir, filename)
                
                img_np = image_tensor[0].cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img_np)
                img.save(output_path, format="PNG")
                
                logger.info(f"[JieKou] Image saved to: {output_path}")
            
            # Save for preview
            preview_images = save_tensor_for_preview(image_tensor)
            
            # Always return the API's image_url (not local path)
            ui_info = {"images": preview_images}
            if save_to_disk:
                ui_info["text"] = [f"✅ 图片已保存: {os.path.basename(output_path)}"]
            
            return {"ui": ui_info, "result": (image_tensor, image_url,)}
        
        except JiekouAPIError as e:
            logger.error(f"[JieKou] API Error: {e.message}")
            raise RuntimeError(e.message)
        except Exception as e:
            logger.error(f"[JieKou] Error: {e}")
            raise
