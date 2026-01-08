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


def register_video_nodes():
    """
    Register all video model nodes from configuration.
    Called during module initialization.
    """
    global VIDEO_NODE_CLASS_MAPPINGS, VIDEO_NODE_DISPLAY_NAME_MAPPINGS
    
    try:
        from .model_node_factory import create_video_node_class
        from ..utils.model_config_loader import get_model_config_loader
        
        loader = get_model_config_loader()
        video_models = loader.get_video_models()
        
        for model in video_models:
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
                node_class = create_video_node_class(model_dict)
                class_name = node_class.__name__
                VIDEO_NODE_CLASS_MAPPINGS[class_name] = node_class
                VIDEO_NODE_DISPLAY_NAME_MAPPINGS[class_name] = model.name or model.id
            except Exception as e:
                logger.error(f"[JieKou] Failed to create video node for {model.id}: {e}")
        
        logger.info(f"[JieKou] Registered {len(VIDEO_NODE_CLASS_MAPPINGS)} video model nodes")
    
    except Exception as e:
        logger.error(f"[JieKou] Failed to register video nodes: {e}")


# Register nodes on module import
register_video_nodes()

