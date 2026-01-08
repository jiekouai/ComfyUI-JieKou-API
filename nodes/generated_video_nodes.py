"""
Generated Video Model Nodes for JieKou ComfyUI Plugin

This module contains dynamically generated node classes for all video models
defined in model_config.json. Each model has its own dedicated node.

Categories:
- video_t2v: Text to Video (文生视频)
- video_i2v: Image to Video (图生视频)
- video_v2v: Video to Video (视频风格转换)

Note: This module is auto-populated at import time from model_config.json.
Do not manually add node classes here.
"""

import logging

logger = logging.getLogger("[JieKou]")

# Node mappings will be populated by factory
VIDEO_NODE_CLASS_MAPPINGS = {}
VIDEO_NODE_DISPLAY_NAME_MAPPINGS = {}


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


def register_video_nodes():
    """
    Register all video model nodes from configuration.
    For models with multiple categories, creates a node in each category.
    Called during module initialization.
    """
    global VIDEO_NODE_CLASS_MAPPINGS, VIDEO_NODE_DISPLAY_NAME_MAPPINGS
    
    try:
        from .model_node_factory import create_video_node_class
        from ..utils.model_config_loader import get_model_config_loader
        
        loader = get_model_config_loader()
        video_models = loader.get_video_models()
        
        # Track processed models to handle multi-category models only once
        processed_models = set()
        
        for model in video_models:
            # Skip if already processed (multi-category models appear in multiple categories)
            if model.id in processed_models:
                continue
            processed_models.add(model.id)
            
            # Get all video categories for this model
            video_categories = [c for c in model.categories if c.startswith("video_")]
            
            model_dict = {
                "id": model.id,
                "name": model.name,
                "description": model.description,
                "categories": model.categories,  # Pass full categories list
                "endpoint": model.endpoint,
                "is_async": model.is_async,
                "response_type": model.response_type,
                "parameters": [param_to_dict(p) for p in model.parameters],
                "product_ids": model.product_ids,
                "valid_combinations": model.valid_combinations,
            }
            
            # Create a node for each video category this model belongs to
            for idx, category in enumerate(video_categories):
                try:
                    node_class = create_video_node_class(model_dict, target_category=category)
                    base_class_name = node_class.__name__
                    
                    # For additional categories, create a derived class with new CATEGORY
                    if idx > 0:
                        cat_short = category.split("_")[-1].upper()  # e.g., "video_i2v" -> "I2V"
                        new_class_name = base_class_name + f"_{cat_short}"
                        
                        # Get the correct CATEGORY path for this category
                        category_map = {
                            "video_t2v": "JieKou AI/Video/Text to Video",
                            "video_i2v": "JieKou AI/Video/Image to Video",
                            "video_v2v": "JieKou AI/Video/Video to Video",
                        }
                        new_category_path = category_map.get(category, "JieKou AI/Video")
                        
                        # Create a derived class with new name and CATEGORY
                        node_class = type(new_class_name, (node_class,), {
                            "__name__": new_class_name,
                            "CATEGORY": new_category_path,
                        })
                        class_name = new_class_name
                    else:
                        class_name = base_class_name
                    
                    # Display name is the same for all categories (no suffix)
                    display_name = model.name or model.id
                    
                    VIDEO_NODE_CLASS_MAPPINGS[class_name] = node_class
                    VIDEO_NODE_DISPLAY_NAME_MAPPINGS[class_name] = display_name
                    
                except Exception as e:
                    logger.error(f"[JieKou] Failed to create video node for {model.id} in {category}: {e}")
        
        logger.info(f"[JieKou] Registered {len(VIDEO_NODE_CLASS_MAPPINGS)} video model nodes")
    
    except Exception as e:
        logger.error(f"[JieKou] Failed to register video nodes: {e}")


# Register nodes on module import
register_video_nodes()

