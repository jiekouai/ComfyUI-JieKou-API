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
- Size presets for image generation models
"""

import torch
import logging
import os
import time
import json
import re
import numpy as np
from PIL import Image
from typing import Any, Optional

logger = logging.getLogger("[JieKou]")


# ===== Size Presets Configuration =====

# Universal size presets pool - all common sizes
# Each preset: (display_name, width, height)
ALL_SIZE_PRESETS = [
    # 1:1 Square
    ("1:1 方形 (512×512)", 512, 512),
    ("1:1 方形 (768×768)", 768, 768),
    ("1:1 方形 (1024×1024)", 1024, 1024),
    ("1:1 方形 (2048×2048)", 2048, 2048),
    ("1:1 方形 (3072×3072)", 3072, 3072),
    # 16:9 Landscape
    ("16:9 横向 (1280×720)", 1280, 720),
    ("16:9 横向 (1920×1080)", 1920, 1080),
    ("16:9 横向 (2560×1440)", 2560, 1440),
    ("16:9 横向 (3840×2160)", 3840, 2160),
    # 9:16 Portrait
    ("9:16 竖向 (720×1280)", 720, 1280),
    ("9:16 竖向 (1080×1920)", 1080, 1920),
    ("9:16 竖向 (1440×2560)", 1440, 2560),
    ("9:16 竖向 (2160×3840)", 2160, 3840),
    # 4:3 Landscape
    ("4:3 横向 (800×600)", 800, 600),
    ("4:3 横向 (1024×768)", 1024, 768),
    ("4:3 横向 (1600×1200)", 1600, 1200),
    ("4:3 横向 (2560×1920)", 2560, 1920),
    # 3:4 Portrait
    ("3:4 竖向 (600×800)", 600, 800),
    ("3:4 竖向 (768×1024)", 768, 1024),
    ("3:4 竖向 (1200×1600)", 1200, 1600),
    ("3:4 竖向 (1920×2560)", 1920, 2560),
    # 3:2 Landscape
    ("3:2 横向 (768×512)", 768, 512),
    ("3:2 横向 (1152×768)", 1152, 768),
    ("3:2 横向 (1920×1280)", 1920, 1280),
    ("3:2 横向 (2880×1920)", 2880, 1920),
    # 2:3 Portrait
    ("2:3 竖向 (512×768)", 512, 768),
    ("2:3 竖向 (768×1152)", 768, 1152),
    ("2:3 竖向 (1280×1920)", 1280, 1920),
    ("2:3 竖向 (1920×2880)", 1920, 2880),
]

# Model size constraints configuration
# Each model: (param_type, separator, min_dim, max_dim, min_pixels, max_pixels, min_ratio, max_ratio)
# - param_type: "size" for single string param, "width_height" for separate params
# - separator: "*" or "x" for size string format
# - min_dim/max_dim: min/max for each dimension (width or height)
# - min_pixels/max_pixels: min/max for total pixels (width * height), None if not applicable
# - min_ratio/max_ratio: min/max aspect ratio (width/height), None if not applicable
MODELS_SIZE_CONSTRAINTS = {
    # Models with 256~1536 range per dimension
    "flux_1_kontext_dev": ("size", "*", 256, 1536, None, None, None, None),
    "flux_2_dev": ("size", "*", 256, 1536, None, None, None, None),
    "flux_2_flex": ("size", "*", 256, 1536, None, None, None, None),
    "flux_2_pro": ("size", "*", 256, 1536, None, None, None, None),
    "hunyuan_image_3": ("size", "*", 256, 1536, None, None, None, None),
    "qwen_text_to_image": ("size", "*", 256, 1536, None, None, None, None),
    "z_image_turbo": ("size", "*", 256, 1536, None, None, None, None),
    "z_image_turbo_lora": ("size", "*", 256, 1536, None, None, None, None),
    # MJ Inpaint: 500~4096 per dimension
    "mj_inpaint": ("width_height", None, 500, 4096, None, None, None, None),
    # Seedream 4.5: total pixels 3686400~16777216, ratio 1/16~16
    "seedream_4_5": ("size", "x", None, None, 3686400, 16777216, 1/16, 16),
}


def filter_presets_by_constraints(
    min_dim: int = None, 
    max_dim: int = None,
    min_pixels: int = None, 
    max_pixels: int = None,
    min_ratio: float = None,
    max_ratio: float = None
) -> list:
    """
    Filter size presets based on model constraints.
    
    Args:
        min_dim: Minimum dimension (applies to both width and height)
        max_dim: Maximum dimension (applies to both width and height)
        min_pixels: Minimum total pixels (width * height)
        max_pixels: Maximum total pixels (width * height)
        min_ratio: Minimum aspect ratio (width / height)
        max_ratio: Maximum aspect ratio (width / height)
    
    Returns:
        List of valid presets + "自定义" option
    """
    valid_presets = []
    
    for name, width, height in ALL_SIZE_PRESETS:
        # Check dimension constraints
        if min_dim is not None and (width < min_dim or height < min_dim):
            continue
        if max_dim is not None and (width > max_dim or height > max_dim):
            continue
        
        # Check pixel constraints
        total_pixels = width * height
        if min_pixels is not None and total_pixels < min_pixels:
            continue
        if max_pixels is not None and total_pixels > max_pixels:
            continue
        
        # Check ratio constraints
        if height > 0:
            ratio = width / height
            if min_ratio is not None and ratio < min_ratio:
                continue
            if max_ratio is not None and ratio > max_ratio:
                continue
        
        valid_presets.append((name, width, height))
    
    # Always add custom option at the end
    valid_presets.append(("自定义", 0, 0))
    
    return valid_presets


def get_size_presets_for_model(model_id: str) -> tuple:
    """
    Get filtered size presets for a specific model based on its constraints.
    
    Args:
        model_id: Model identifier
    
    Returns:
        tuple: (presets_list, param_type, separator) or (None, None, None) if no presets
    """
    constraints = MODELS_SIZE_CONSTRAINTS.get(model_id)
    if not constraints:
        return None, None, None
    
    param_type, separator, min_dim, max_dim, min_pixels, max_pixels, min_ratio, max_ratio = constraints
    
    # Filter presets based on constraints
    presets = filter_presets_by_constraints(
        min_dim=min_dim,
        max_dim=max_dim,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        min_ratio=min_ratio,
        max_ratio=max_ratio
    )
    
    return presets, param_type, separator

# Alias for backward compatibility
get_size_presets = get_size_presets_for_model


def parse_size_preset(preset_value: str) -> tuple:
    """
    Parse a size preset string to get width and height.
    
    Args:
        preset_value: Preset string like "1:1 方形 (1024×1024)" or "自定义"
        
    Returns:
        tuple: (width, height) or (0, 0) for custom
    """
    if preset_value == "自定义":
        return 0, 0
    
    # Extract dimensions from preset string like "1:1 方形 (1024×1024)"
    match = re.search(r'\((\d+)[×x](\d+)\)', preset_value)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


# ===== T009: Parameter Type Converters =====

def parse_default_from_description(description: str, param_type: str) -> Any:
    """
    Parse default value from parameter description.
    Many API docs include default values in descriptions like "默认值为 1" or "默认值为 3.5".
    
    Args:
        description: Parameter description text
        param_type: Parameter type ("integer", "number", "string", "boolean")
        
    Returns:
        Parsed default value or None if not found
    """
    import re
    
    # Common patterns for default values in Chinese/English descriptions
    patterns = [
        r'默认值[为是：:]\s*([\-]?\d+\.?\d*)',  # 默认值为 1 or 默认值: 3.5
        r'[Dd]efault[s]?\s*(?:value|is|:)?\s*([\-]?\d+\.?\d*)',  # Default value is 1
        r'默认[为是：:]\s*([\-]?\d+\.?\d*)',  # 默认为 1
    ]
    
    for pattern in patterns:
        match = re.search(pattern, description)
        if match:
            val_str = match.group(1)
            try:
                if param_type == "integer":
                    return int(float(val_str))  # Handle "1.0" as integer
                elif param_type == "number":
                    return float(val_str)
            except (ValueError, TypeError):
                continue
    
    return None


def parse_range_from_description(description: str) -> tuple[Any, Any]:
    """
    Parse min/max range from parameter description.
    Common patterns: "[1, 4]", "[1 ~ 4]", "取值范围：[1, 4]"
    
    Args:
        description: Parameter description text
        
    Returns:
        tuple: (minimum, maximum) or (None, None) if not found
    """
    import re
    
    # Patterns for range values
    patterns = [
        r'[\[【]\s*([\-]?\d+\.?\d*)\s*[,，~～]\s*([\-]?\d+\.?\d*)\s*[\]】]',  # [1, 4] or [1 ~ 4]
        r'取值范围[：:]\s*[\[【]?\s*([\-]?\d+\.?\d*)\s*[,，~～]\s*([\-]?\d+\.?\d*)',  # 取值范围：[1, 4]
        r'[Rr]ange[：:]?\s*[\[（(]?\s*([\-]?\d+\.?\d*)\s*[-,~]\s*([\-]?\d+\.?\d*)',  # Range: [1, 4]
    ]
    
    for pattern in patterns:
        match = re.search(pattern, description)
        if match:
            try:
                min_val = float(match.group(1))
                max_val = float(match.group(2))
                return min_val, max_val
            except (ValueError, TypeError):
                continue
    
    return None, None


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
    
    # Try to parse default value from description if not explicitly set
    if default is None and description and param_type in ("integer", "number"):
        default = parse_default_from_description(description, param_type)
    
    # Try to parse min/max from description if not explicitly set
    if (minimum is None or maximum is None) and description:
        desc_min, desc_max = parse_range_from_description(description)
        if minimum is None and desc_min is not None:
            minimum = desc_min
        if maximum is None and desc_max is not None:
            maximum = desc_max
    
    options = {}
    
    # Handle enum types - convert to tuple for dropdown
    if enum:
        # Ensure enum values match the declared type
        # Some configs have integer type but string enum values like ["5", "10"]
        if param_type == "integer":
            enum = [int(v) if isinstance(v, str) and v.isdigit() else v for v in enum]
        elif param_type == "number":
            enum = [float(v) if isinstance(v, str) else v for v in enum]
        
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


def coerce_param_value(value: Any, param_type: str) -> Any:
    """
    Convert parameter value to the correct type based on parameter definition.
    
    This is needed because ComfyUI dropdowns may return strings even for integer enums.
    
    Args:
        value: The value to convert
        param_type: The declared type from model_config ("integer", "number", "string", "boolean")
        
    Returns:
        The value converted to the correct type
    """
    if value is None:
        return None
    
    try:
        if param_type == "integer":
            if isinstance(value, str) and value.lstrip('-').isdigit():
                return int(value)
            elif isinstance(value, (int, float)):
                return int(value)
        elif param_type == "number":
            if isinstance(value, str):
                return float(value)
            elif isinstance(value, (int, float)):
                return float(value)
        elif param_type == "boolean":
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        # For string type, just return as-is
        return value
    except (ValueError, TypeError):
        return value


def build_param_type_map(model_config: dict) -> dict:
    """
    Build a mapping of parameter names to their declared types.
    Handles nested children parameters.
    
    Args:
        model_config: Model configuration dict
        
    Returns:
        dict: {param_name: param_type}
    """
    type_map = {}
    for param in model_config.get("parameters", []):
        name = param.get("name", "")
        param_type = param.get("type", "string")
        
        # For object types with children, map the children instead
        if param_type == "object" and param.get("children"):
            for child in param.get("children", []):
                child_name = child.get("name", "")
                child_type = child.get("type", "string")
                if child_name:
                    type_map[child_name] = child_type
        elif name:
            type_map[name] = param_type
    return type_map


def build_nested_param_map(model_config: dict) -> dict:
    """
    Build a mapping of child parameter names to their parent object names.
    
    Args:
        model_config: Model configuration dict
        
    Returns:
        dict: {child_param_name: parent_object_name}
    """
    nested_map = {}
    for param in model_config.get("parameters", []):
        parent_name = param.get("name", "")
        param_type = param.get("type", "string")
        
        if param_type == "object" and param.get("children"):
            for child in param.get("children", []):
                child_name = child.get("name", "")
                if child_name:
                    nested_map[child_name] = parent_name
    return nested_map


def build_input_types(model_config: dict, categories: list) -> dict:
    """
    Build ComfyUI INPUT_TYPES dict from model configuration.
    
    Args:
        model_config: Model configuration dict
        categories: List of model categories (e.g., ["image_t2i", "image_edit"])
        
    Returns:
        dict: INPUT_TYPES for ComfyUI node class
    """
    required = {}
    optional = {}
    
    model_id = model_config.get("id", "")
    
    # Check if this model uses size presets
    size_presets, preset_param_type, _ = get_size_presets(model_id)
    uses_size_presets = size_presets is not None
    
    # Standard inputs based on categories
    is_image_model = any(cat.startswith("image_") for cat in categories)
    is_video_model = any(cat.startswith("video_") for cat in categories)
    is_audio_model = any(cat.startswith("audio_") for cat in categories)
    
    # Check if model uses task_id (like MJ remix/variation) - these don't need image upload
    # They reference previous task results via task_id instead
    param_names = {p.get("name", "") for p in model_config.get("parameters", [])}
    uses_task_id = "task_id" in param_names
    
    # Special case: mj_remove_background only accepts URL, not image tensor
    is_mj_remove_bg = model_id == "mj_remove_background"
    
    # Special case: GPT image edit doesn't support image_url parameter (only accepts file array)
    is_gpt_image_edit = model_id in ("gpt_image_edits", "gpt-image-1-edit")
    
    # For I2I/I2V models, add image inputs (both tensor and URL)
    # But skip for models that use task_id to reference previous tasks
    # mj_remove_background only needs URL input, not image tensor
    needs_image_input = any(cat in ("image_edit", "video_i2v") for cat in categories) and not uses_task_id
    if needs_image_input:
        # Image tensor input - can connect from Load Image or other image nodes
        optional["image"] = ("IMAGE", {})
        # Image URL or base64 string input - higher priority if both provided
        # But skip for GPT image edit which doesn't support image_url
        if not is_gpt_image_edit:
            optional["image_url"] = ("STRING", {
                "default": "",
                "placeholder": "输入图片 URL 或 base64 (优先使用)",
                "dynamicPrompts": True
            })
    
    # mj_remove_background only accepts URL input (no image tensor support)
    if is_mj_remove_bg:
        required["image_url"] = ("STRING", {
            "default": "",
            "placeholder": "输入图片 URL",
        })
    
    # For V2V models, add video_url input
    needs_video_input = "video_v2v" in categories
    if needs_video_input:
        required["video_url"] = ("STRING", {
            "default": "",
            "placeholder": "输入视频 URL",
            "dynamicPrompts": True
        })
    
    # Add size preset dropdown if this model uses presets
    if uses_size_presets:
        preset_names = [p[0] for p in size_presets]
        optional["size_preset"] = (preset_names, {"default": preset_names[0]})
        
        # Add custom width/height inputs (shown when "自定义" is selected)
        # These will be used by frontend JS to show/hide dynamically
        if preset_param_type == "width_height":
            # For models with separate width/height params
            optional["custom_width"] = ("INT", {
                "default": 1024,
                "min": 256,
                "max": 4096,
                "step": 64,
                "tooltip": "自定义宽度（选择'自定义'时生效）"
            })
            optional["custom_height"] = ("INT", {
                "default": 1024,
                "min": 256,
                "max": 4096,
                "step": 64,
                "tooltip": "自定义高度（选择'自定义'时生效）"
            })
        else:
            # For models with size string param
            optional["custom_size"] = ("STRING", {
                "default": "1024*1024",
                "tooltip": "自定义尺寸（选择'自定义'时生效），格式：宽*高 或 宽x高"
            })
    
    # Process model parameters
    for param_data in model_config.get("parameters", []):
        name = param_data.get("name", "")
        param_type = param_data.get("type", "string")
        is_required = param_data.get("required", False)
        
        # Skip parameters that are handled separately
        # But for GPT image edit, include "model" parameter as enum input
        skip_params = {"model", "image", "images", "img_url", "image_url", "video"}
        if name in skip_params:
            # Special case: GPT image edit needs model parameter as enum input
            if name == "model" and is_gpt_image_edit:
                # Process model parameter for GPT image edit
                type_spec, options = param_to_comfyui_type(param_data)
                if is_required:
                    required[name] = (type_spec, options) if isinstance(type_spec, str) else (type_spec[0], options)
                else:
                    optional[name] = (type_spec, options) if isinstance(type_spec, str) else (type_spec[0], options)
            continue
        
        # For mj_remove_background, skip "url" param as it's filled from image input
        if is_mj_remove_bg and name == "url":
            continue
        
        # Skip size/width/height params if using presets
        if uses_size_presets and name in ("size", "width", "height"):
            continue
        
        # Handle object type parameters with children - flatten the children as inputs
        if param_type == "object" and param_data.get("children"):
            for child in param_data.get("children", []):
                child_name = child.get("name", "")
                child_required = child.get("required", False)
                
                # Skip child params that are handled separately
                if child_name in skip_params:
                    continue
                
                # Skip size/width/height in children if using presets
                if uses_size_presets and child_name in ("size", "width", "height"):
                    continue
                
                type_spec, options = param_to_comfyui_type(child)
                
                if child_required:
                    required[child_name] = (type_spec, options) if isinstance(type_spec, str) else (type_spec[0], options)
                else:
                    optional[child_name] = (type_spec, options) if isinstance(type_spec, str) else (type_spec[0], options)
            continue
        
        # Skip object type parameters without children (structural containers)
        if param_type == "object":
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
    
    # Note: ComfyUI requires "optional" to be a dict (not None) if present
    # because node_info() calls value.keys() on all items
    result = {
        "required": required,
    }
    if optional:
        result["optional"] = optional
    return result


# ===== T008: Dynamic Node Class Factory =====

def create_image_node_class(model_config: dict, target_category: str = None) -> type:
    """
    Create a ComfyUI node class for an image model.
    
    Args:
        model_config: Model configuration dict from model_config.json
        target_category: Specific category to create this node for (for multi-category models)
        
    Returns:
        type: A new ComfyUI node class
    """
    model_id = model_config.get("id", "")
    model_name = model_config.get("name", model_id)
    # Support both old "category" (string) and new "categories" (list)
    categories = model_config.get("categories", [])
    if not categories and model_config.get("category"):
        categories = [model_config.get("category")]
    if not categories:
        categories = ["image_t2i"]
    
    endpoint = model_config.get("endpoint", "")
    is_async = model_config.get("is_async", True)
    response_type = model_config.get("response_type", "image_urls")
    
    # Check if this is a Midjourney model - they need task_id output
    is_mj_model = model_id.startswith("mj_")
    
    # Build category path for menu - use target_category if specified
    category_map = {
        "image_t2i": "JieKou AI/Image/Text to Image",
        "image_edit": "JieKou AI/Image/Image to Image",
        "image_tool": "JieKou AI/Image/Tools",
    }
    node_category = category_map.get(target_category or categories[0], "JieKou AI/Image")
    
    # Build INPUT_TYPES - pass all categories so image input is added if any category needs it
    input_types = build_input_types(model_config, categories)
    
    # Build param type map for type coercion
    param_type_map = build_param_type_map(model_config)
    
    # Create the node class dynamically
    class_name = f"JieKou{model_id.replace('-', '_').replace('.', '_').title().replace('_', '')}"
    
    def generate(self, save_to_disk: bool = True, image: torch.Tensor = None, image_url: str = "", 
                 size_preset: str = None, custom_width: int = None, custom_height: int = None, 
                 custom_size: str = None, **kwargs):
        """Generate image using this model"""
        from ..utils.api_client import JiekouAPI, JiekouAPIError
        from ..utils.tensor_utils import url_to_tensor, base64_to_tensor, tensor_to_base64
        
        logger.info(f"[JieKou] ========== {model_name} ==========")
        logger.info(f"[JieKou] Endpoint: {endpoint}")
        logger.info(f"[JieKou] Params: {list(kwargs.keys())}")
        
        try:
            api = JiekouAPI()
            
            # Determine input image source: image_url has higher priority (if valid)
            # But GPT image edit doesn't support image_url, only accepts image tensor
            is_gpt_image_edit = model_id in ("gpt_image_edits", "gpt-image-1-edit")
            
            input_image_data = ""
            image_source = ""
            
            def is_valid_image_url(url: str) -> bool:
                if not url or not isinstance(url, str):
                    return False
                url = url.strip()
                if url.startswith("http://") or url.startswith("https://"):
                    return True
                if url.startswith("data:image/"):
                    return True
                if len(url) > 100 and " " not in url[:100]:
                    return True
                return False
            
            # mj_remove_background only accepts URL, handle separately
            mj_remove_bg_url = None
            if is_mj_model and model_id == "mj_remove_background":
                if image_url and is_valid_image_url(image_url):
                    mj_remove_bg_url = image_url.strip()
                    image_source = "image_url"
                else:
                    raise ValueError("Midjourney 移除背景需要提供有效的图片 URL")
            # GPT image edit only accepts image tensor, not image_url
            # For GPT image edit, we need to use multipart/form-data with file upload
            elif is_gpt_image_edit:
                gpt_image_bytes = None
                if image is not None:
                    from ..utils.tensor_utils import tensor_to_bytes
                    # Convert tensor to PNG bytes for file upload
                    gpt_image_bytes = tensor_to_bytes(image, format="PNG")
                    image_source = "image_tensor"
                else:
                    raise ValueError("GPT 图像编辑需要提供 image 输入（不支持 image_url）")
            elif image_url and is_valid_image_url(image_url):
                input_image_data = image_url.strip()
                image_source = "image_url"
            elif image is not None:
                b64_data = tensor_to_base64(image, format="PNG")
                input_image_data = f"data:image/png;base64,{b64_data}"
                image_source = "image_tensor"
            else:
                input_image_data = ""
            
            logger.info(f"[JieKou] Image source: {image_source if image_source else 'None'}")
            
            # Build request data
            data = {}
            files = None
            
            # First, add prompt parameter if present in kwargs (critical for text-to-image models)
            # This ensures prompt is always included and not accidentally skipped
            if "prompt" in kwargs and kwargs["prompt"] is not None:
                prompt_value = kwargs["prompt"]
                if isinstance(prompt_value, str) and prompt_value.strip():
                    param_type = param_type_map.get("prompt", "string")
                    data["prompt"] = coerce_param_value(prompt_value, param_type)
            
            # Handle size presets
            size_presets_config, preset_param_type, separator = get_size_presets(model_id)
            if size_presets_config and size_preset:
                width, height = parse_size_preset(size_preset)
                
                if width == 0 and height == 0:
                    # Custom size selected
                    if preset_param_type == "width_height":
                        # Use custom_width and custom_height
                        if custom_width and custom_height:
                            data["width"] = custom_width
                            data["height"] = custom_height
                    else:
                        # Use custom_size string
                        if custom_size:
                            data["size"] = custom_size
                else:
                    # Preset selected
                    if preset_param_type == "width_height":
                        data["width"] = width
                        data["height"] = height
                    else:
                        # Format size string with correct separator
                        data["size"] = f"{width}{separator}{height}"
                
                size_info = data.get('size') or f"{data.get('width')}x{data.get('height')}"
                logger.info(f"[JieKou] Size preset: {size_preset} -> {size_info}")
            
            # Add image for edit models
            # mj_remove_background uses URL directly (no tensor support)
            if mj_remove_bg_url:
                data["url"] = mj_remove_bg_url
                logger.info(f"[JieKou] MJ remove background: using URL")
            elif is_gpt_image_edit and gpt_image_bytes:
                # GPT image edit uses multipart/form-data with file upload
                # API expects file[] type, but for single file, use tuple format directly
                # For multiple files, would need to use list of tuples with same field name
                files = {
                    "image": ("image.png", gpt_image_bytes, "image/png")
                }
                logger.info(f"[JieKou] GPT image edit: uploading {len(gpt_image_bytes)} bytes as file")
            elif input_image_data:
                img = input_image_data
                # Different models use different field names
                model_id_lower = model_id.lower()
                if "flux" in model_id_lower:
                    data["images"] = [img]
                elif any(x in model_id_lower for x in ["seedream4_0", "seedream4.0", "seedream-4.0"]):
                    # Seedream 4.0 uses "images" (plural, string[])
                    data["images"] = [img]
                elif any(x in model_id_lower for x in ["seedream_4_5", "seedream4.5", "seedream-4.5"]):
                    # Seedream 4.5 uses "image" (singular, but array type)
                    data["image"] = [img]
                elif "nano_banana" in model_id_lower:
                    # Nano Banana Pro Light uses "images" (array type)
                    data["images"] = [img]
                elif "gemini" in model_id_lower:
                    # Gemini supports both image_urls and image_base64s
                    if img.startswith("data:"):
                        # Base64 format - extract data without prefix
                        img_data = img.split(",", 1)[1] if "," in img else img
                        data["image_base64s"] = [img_data]
                    else:
                        # URL format
                        data["image_urls"] = [img]
                else:
                    data["image"] = img
            
            # Add all other parameters with type coercion (skip prompt as it's already added)
            for key, value in kwargs.items():
                # Skip prompt as it's already added at the beginning
                if key == "prompt":
                    continue
                if value is None:
                    continue
                if isinstance(value, str) and value.strip() == "":
                    continue
                # Coerce value to correct type based on param definition
                param_type = param_type_map.get(key, "string")
                data[key] = coerce_param_value(value, param_type)
            
            # Special handling for flux-1-kontext-dev to ensure prompt is properly handled
            if model_id == "flux_1_kontext_dev":
                # Ensure prompt exists and is not empty
                if "prompt" not in data or not data.get("prompt"):
                    raise ValueError("flux-1-kontext-dev requires a non-empty prompt parameter")
                
                # Ensure prompt is the first parameter in the request
                # Some APIs may be sensitive to parameter order
                prompt_value = data.pop("prompt")
                # Create new dict with prompt first, then all other params
                ordered_data = {"prompt": prompt_value}
                ordered_data.update(data)
                data = ordered_data
                
                # Ensure images parameter is not present if no image was provided
                # Empty images array might interfere with prompt processing
                if "images" in data and not input_image_data:
                    # Remove images if it's an empty array or contains empty strings
                    images_value = data.get("images", [])
                    if not images_value or (isinstance(images_value, list) and all(not img or (isinstance(img, str) and not img.strip()) for img in images_value)):
                        del data["images"]
            
            # Log request
            log_data = {k: (v[:50] + "..." if isinstance(v, str) and len(v) > 100 else v) for k, v in data.items()}
            logger.info(f"[JieKou] Request: {log_data}")
            if files:
                logger.info(f"[JieKou] Files: {list(files.keys())}")
            
            # Make API request
            # For GPT image edit, pass files parameter for multipart/form-data
            result = api.call_model_and_wait(endpoint, data, files=files, is_async=is_async)
            
            # Handle response
            image_result_url = ""
            
            if is_async:
                image_result_url = api.get_image_result_url(result)
                if not image_result_url:
                    raise ValueError("任务完成但未返回图像 URL")
                image_tensor = url_to_tensor(image_result_url)
            
            elif response_type == "b64_json":
                images = result.get("data", [])
                logger.debug(f"[JieKou] Response data type: {type(images)}, length: {len(images) if isinstance(images, list) else 'N/A'}")
                if not images:
                    logger.error(f"[JieKou] Response result: {result}")
                    raise ValueError("API 未返回图像数据")
                first_image = images[0]
                logger.debug(f"[JieKou] First image keys: {first_image.keys() if isinstance(first_image, dict) else type(first_image)}")
                if first_image.get("b64_json"):
                    image_tensor = base64_to_tensor(first_image["b64_json"])
                    image_result_url = "(base64 data)"
                elif first_image.get("url"):
                    image_result_url = first_image["url"]
                    image_tensor = url_to_tensor(image_result_url)
                else:
                    logger.error(f"[JieKou] First image data: {first_image}")
                    raise ValueError(f"API 返回的图像数据格式未知，期望 b64_json 或 url，实际: {list(first_image.keys()) if isinstance(first_image, dict) else type(first_image)}")
            
            elif response_type == "image_urls":
                # Try multiple possible response formats
                urls = result.get("image_urls") or result.get("images", [])
                
                # Some APIs return data in a "data" field (e.g., nano-banana-pro-light)
                if not urls and "data" in result:
                    data_field = result.get("data", [])
                    if isinstance(data_field, list) and data_field:
                        # Extract URLs from data array
                        urls = []
                        for item in data_field:
                            if isinstance(item, dict) and "url" in item:
                                urls.append(item["url"])
                            elif isinstance(item, str):
                                urls.append(item)
                        logger.debug(f"[JieKou] Extracted {len(urls)} URLs from data field")
                
                if not urls:
                    logger.error(f"[JieKou] Response result keys: {list(result.keys())}")
                    logger.error(f"[JieKou] Response result: {result}")
                    raise ValueError("API 未返回图像数据")
                
                # Extract URL from first item (handle both string and dict formats)
                image_result_url = urls[0] if isinstance(urls[0], str) else urls[0].get("url", "")
                if not image_result_url:
                    logger.error(f"[JieKou] Could not extract URL from: {urls[0]}")
                    raise ValueError("API 返回的 URL 格式无效")
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
            
            # Extract task_id for MJ models
            task_id_output = result.get("_task_id", "")
            
            # Return different tuple based on model type
            if is_mj_model:
                return {"ui": ui_info, "result": (image_tensor, image_result_url, task_id_output)}
            else:
                return {"ui": ui_info, "result": (image_tensor, image_result_url,)}
        
        except JiekouAPIError as e:
            logger.error(f"[JieKou] API Error: {e.message}")
            raise RuntimeError(e.message)
        except Exception as e:
            logger.error(f"[JieKou] Error: {e}")
            raise
    
    # Create the class with different return types for MJ models
    if is_mj_model:
        return_types = ("IMAGE", "STRING", "STRING")
        return_names = ("image", "image_url", "task_id")
    else:
        return_types = ("IMAGE", "STRING")
        return_names = ("image", "image_url")
    
    node_class = type(class_name, (), {
        "CATEGORY": node_category,
        "FUNCTION": "generate",
        "RETURN_TYPES": return_types,
        "RETURN_NAMES": return_names,
        "OUTPUT_NODE": True,
        "INPUT_TYPES": classmethod(lambda cls: input_types),
        "generate": generate,
        "_model_config": model_config,
    })
    
    return node_class


def create_video_node_class(model_config: dict, target_category: str = None) -> type:
    """
    Create a ComfyUI node class for a video model.
    
    Args:
        model_config: Model configuration dict from model_config.json
        target_category: Specific category to create this node for (for multi-category models)
        
    Returns:
        type: A new ComfyUI node class
    """
    model_id = model_config.get("id", "")
    model_name = model_config.get("name", model_id)
    # Support both old "category" (string) and new "categories" (list)
    categories = model_config.get("categories", [])
    if not categories and model_config.get("category"):
        categories = [model_config.get("category")]
    if not categories:
        categories = ["video_t2v"]
    
    endpoint = model_config.get("endpoint", "")
    is_async = model_config.get("is_async", True)
    
    # Build category path for menu - use target_category if specified
    category_map = {
        "video_t2v": "JieKou AI/Video/Text to Video",
        "video_i2v": "JieKou AI/Video/Image to Video",
        "video_v2v": "JieKou AI/Video/Video to Video",
    }
    node_category = category_map.get(target_category or categories[0], "JieKou AI/Video")
    
    # Build INPUT_TYPES - pass all categories so image input is added if any category needs it
    input_types = build_input_types(model_config, categories)
    
    # Build param type map for type coercion
    param_type_map = build_param_type_map(model_config)
    
    # Build nested param map for request body construction
    nested_param_map = build_nested_param_map(model_config)
    
    # Create the node class dynamically
    class_name = f"JieKou{model_id.replace('-', '_').replace('.', '_').title().replace('_', '')}"
    
    def generate_video(self, save_to_disk: bool = True, image: torch.Tensor = None, image_url: str = "", video_url: str = "", **kwargs):
        """Generate video using this model"""
        from ..utils.api_client import JiekouAPI, JiekouAPIError
        from ..utils.tensor_utils import video_to_frames, tensor_to_base64
        
        logger.info(f"[JieKou] ========== {model_name} ==========")
        logger.info(f"[JieKou] Endpoint: {endpoint}")
        logger.info(f"[JieKou] Params: {list(kwargs.keys())}")
        
        try:
            api = JiekouAPI()
            
            # Determine input image source: image_url has higher priority (if valid)
            input_image_data = ""
            image_source = ""
            
            def is_valid_image_url(url: str) -> bool:
                if not url or not isinstance(url, str):
                    return False
                url = url.strip()
                if url.startswith("http://") or url.startswith("https://"):
                    return True
                if url.startswith("data:image/"):
                    return True
                if len(url) > 100 and " " not in url[:100]:
                    return True
                return False
            
            if image_url and is_valid_image_url(image_url):
                input_image_data = image_url.strip()
                image_source = "image_url"
            elif image is not None:
                b64_data = tensor_to_base64(image, format="PNG")
                input_image_data = f"data:image/png;base64,{b64_data}"
                image_source = "image_tensor"
            
            logger.info(f"[JieKou] Image source: {image_source if image_source else 'None (T2V mode)'}")
            
            # Build request data
            data = {}
            
            # Check if this model uses nested structure based on config
            # If nested_param_map has entries, the model uses nested structure
            uses_nested_structure = bool(nested_param_map)
            
            if uses_nested_structure:
                # Build nested structure dynamically based on config
                nested_objects = {}  # {parent_name: {child_params}}
                flat_params = {}  # params not in any nested object
                
                # Add image for I2V models - check which parent object "img_url" belongs to
                if input_image_data:
                    img_parent = nested_param_map.get("img_url", "input")
                    if img_parent not in nested_objects:
                        nested_objects[img_parent] = {}
                    nested_objects[img_parent]["img_url"] = input_image_data
                
                # Add video for V2V models
                if video_url and video_url.strip():
                    video_parent = nested_param_map.get("reference_video_urls", "input")
                    if video_parent not in nested_objects:
                        nested_objects[video_parent] = {}
                    nested_objects[video_parent]["reference_video_urls"] = [video_url.strip()]
                
                # Distribute parameters based on nested_param_map
                for key, value in kwargs.items():
                    if value is None:
                        continue
                    if isinstance(value, str) and value.strip() == "":
                        continue
                    
                    # Coerce value to correct type
                    param_type = param_type_map.get(key, "string")
                    coerced_value = coerce_param_value(value, param_type)
                    
                    # Check if this param belongs to a nested object
                    parent = nested_param_map.get(key)
                    if parent:
                        if parent not in nested_objects:
                            nested_objects[parent] = {}
                        nested_objects[parent][key] = coerced_value
                    else:
                        flat_params[key] = coerced_value
                
                # Build the final request data
                for parent_name, parent_obj in nested_objects.items():
                    if parent_obj:
                        data[parent_name] = parent_obj
                
                # Add any flat params at root level
                data.update(flat_params)
            else:
                # Flat structure for other models
                
                # Add video for V2V models
                if video_url and video_url.strip():
                    data["video"] = video_url.strip()
                
                # Add image for I2V models
                if input_image_data:
                    # Different models use different field names
                    if "vidu" in model_id.lower():
                        # Vidu uses "images" (plural, string[])
                        data["images"] = [input_image_data]
                    elif any(x in model_id.lower() for x in ["wan2_1", "wan2.1", "wan-2.1", "wan-i2v"]):
                        # Wan 2.1 uses "image_url"
                        data["image_url"] = input_image_data
                    else:
                        # Most models use "image" (singular)
                        data["image"] = input_image_data
                
                # Add all other parameters with type coercion
                for key, value in kwargs.items():
                    if value is None:
                        continue
                    if isinstance(value, str) and value.strip() == "":
                        continue
                    # Coerce value to correct type
                    param_type = param_type_map.get(key, "string")
                    data[key] = coerce_param_value(value, param_type)
            
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


def create_audio_node_class(model_config: dict, target_category: str = None) -> type:
    """
    Create a ComfyUI node class for an audio model.
    
    Args:
        model_config: Model configuration dict from model_config.json
        target_category: Specific category to create this node for (for multi-category models)
        
    Returns:
        type: A new ComfyUI node class
    """
    model_id = model_config.get("id", "")
    model_name = model_config.get("name", model_id)
    # Support both old "category" (string) and new "categories" (list)
    categories = model_config.get("categories", [])
    if not categories and model_config.get("category"):
        categories = [model_config.get("category")]
    if not categories:
        categories = ["audio_tts"]
    
    endpoint = model_config.get("endpoint", "")
    is_async = model_config.get("is_async", True)
    
    # Build category path for menu - use target_category if specified
    category_map = {
        "audio_tts": "JieKou AI/Audio/Text to Speech",
        "audio_asr": "JieKou AI/Audio/Speech to Text",
    }
    node_category = category_map.get(target_category or categories[0], "JieKou AI/Audio")
    
    # Build INPUT_TYPES - pass all categories
    input_types = build_input_types(model_config, categories)
    
    # Build param type map for type coercion
    param_type_map = build_param_type_map(model_config)
    
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
            
            # Build request data with type coercion
            data = {}
            for key, value in kwargs.items():
                if value is None:
                    continue
                if isinstance(value, str) and value.strip() == "":
                    continue
                # Coerce value to correct type
                param_type = param_type_map.get(key, "string")
                data[key] = coerce_param_value(value, param_type)
            
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
    
    For models with multiple categories, creates a node in each category.
    
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
    
    # Track models we've already processed to avoid duplicates
    processed_models = set()
    
    def param_to_dict(p):
        """Convert ModelParameter to dict, including children recursively"""
        result = {
            "name": p.name,
            "type": p.type,
            "description": p.description,
            "required": p.required,
            "default": p.default,
            "enum": p.enum,
            "minimum": p.minimum,
            "maximum": p.maximum,
        }
        if p.children:
            result["children"] = [param_to_dict(child) for child in p.children]
        return result
    
    for model in all_models:
        # Skip if already processed (for multi-category models appearing multiple times)
        if model.id in processed_models:
            continue
        processed_models.add(model.id)
        
        model_dict = {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "categories": model.categories,  # Use categories list
            "endpoint": model.endpoint,
            "is_async": model.is_async,
            "response_type": model.response_type,
            "parameters": [param_to_dict(p) for p in model.parameters],
            "product_ids": model.product_ids,
            "valid_combinations": model.valid_combinations,
        }
        
        try:
            # Create a node for each category the model belongs to
            for idx, category in enumerate(model.categories):
                # For first category, use standard class name
                # For additional categories, append category suffix to class name (not display name)
                if idx == 0:
                    class_suffix = ""
                else:
                    # Add category-specific suffix for additional nodes (internal only)
                    cat_short = category.split("_")[-1].upper()  # e.g., "image_edit" -> "EDIT"
                    class_suffix = f"_{cat_short}"
                
                if category.startswith("image_"):
                    node_class = create_image_node_class(model_dict, target_category=category)
                elif category.startswith("video_"):
                    node_class = create_video_node_class(model_dict, target_category=category)
                elif category.startswith("audio_"):
                    node_class = create_audio_node_class(model_dict, target_category=category)
                else:
                    logger.warning(f"[JieKou] Unknown category for model {model.id}: {category}")
                    continue
                
                # Modify class name for additional categories
                base_class_name = node_class.__name__
                if class_suffix:
                    new_class_name = base_class_name + class_suffix
                    # Get the correct CATEGORY path for this category
                    category_map = {
                        "image_t2i": "JieKou AI/Image/Text to Image",
                        "image_edit": "JieKou AI/Image/Image to Image",
                        "image_tool": "JieKou AI/Image/Tools",
                        "video_t2v": "JieKou AI/Video/Text to Video",
                        "video_i2v": "JieKou AI/Video/Image to Video",
                        "video_v2v": "JieKou AI/Video/Video to Video",
                        "audio_tts": "JieKou AI/Audio/Text to Speech",
                        "audio_asr": "JieKou AI/Audio/Speech to Text",
                    }
                    new_category_path = category_map.get(category, "JieKou AI/Other")
                    # Create a new class with modified name and CATEGORY
                    node_class = type(new_class_name, (node_class,), {
                        "__name__": new_class_name,
                        "CATEGORY": new_category_path,  # Override CATEGORY for this node
                    })
                
                class_name = node_class.__name__
                class_mappings[class_name] = node_class
                # Display name is the same for all categories (no suffix)
                display_mappings[class_name] = model.name or model.id
                
                # Track category counts
                category_counts[category] = category_counts.get(category, 0) + 1
        
        except Exception as e:
            logger.error(f"[JieKou] Failed to create node for {model.id}: {e}")
            continue
    
    logger.info(f"[JieKou] Created {len(class_mappings)} model node classes")
    logger.info(f"[JieKou] By category: {category_counts}")
    return class_mappings, display_mappings

