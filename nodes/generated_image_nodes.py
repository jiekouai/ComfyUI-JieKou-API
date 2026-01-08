"""
Generated Image Model Nodes for JieKou ComfyUI Plugin

This module contains dynamically generated node classes for all image models
defined in model_config.json. Each model has its own dedicated node.

Categories:
- image_t2i: Text to Image (文生图)
- image_edit: Image Editing (图像编辑)
- image_tool: Image Tools (图像工具)

Note: This module is auto-populated at import time from model_config.json.
Do not manually add node classes here.
"""

import logging

logger = logging.getLogger("[JieKou]")

# Node mappings will be populated by factory
IMAGE_NODE_CLASS_MAPPINGS = {}
IMAGE_NODE_DISPLAY_NAME_MAPPINGS = {}


def register_image_nodes():
    """
    Register all image model nodes from configuration.
    Called during module initialization.
    """
    global IMAGE_NODE_CLASS_MAPPINGS, IMAGE_NODE_DISPLAY_NAME_MAPPINGS
    
    try:
        from .model_node_factory import create_image_node_class
        from ..utils.model_config_loader import get_model_config_loader
        
        loader = get_model_config_loader()
        image_models = loader.get_image_models()
        
        for model in image_models:
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
                node_class = create_image_node_class(model_dict)
                class_name = node_class.__name__
                IMAGE_NODE_CLASS_MAPPINGS[class_name] = node_class
                IMAGE_NODE_DISPLAY_NAME_MAPPINGS[class_name] = model.name or model.id
            except Exception as e:
                logger.error(f"[JieKou] Failed to create image node for {model.id}: {e}")
        
        logger.info(f"[JieKou] Registered {len(IMAGE_NODE_CLASS_MAPPINGS)} image model nodes")
    
    except Exception as e:
        logger.error(f"[JieKou] Failed to register image nodes: {e}")


# Register nodes on module import
register_image_nodes()

