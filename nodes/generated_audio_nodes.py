"""
Generated Audio Model Nodes for JieKou ComfyUI Plugin

This module contains dynamically generated node classes for all audio models
defined in model_config.json. Each model has its own dedicated node.

Categories:
- audio_tts: Text to Speech (文本转语音)
- audio_asr: Automatic Speech Recognition (语音识别)

Note: This module is auto-populated at import time from model_config.json.
Do not manually add node classes here.
"""

import logging

logger = logging.getLogger("[JieKou]")

# Node mappings will be populated by factory
AUDIO_NODE_CLASS_MAPPINGS = {}
AUDIO_NODE_DISPLAY_NAME_MAPPINGS = {}


def register_audio_nodes():
    """
    Register all audio model nodes from configuration.
    Called during module initialization.
    """
    global AUDIO_NODE_CLASS_MAPPINGS, AUDIO_NODE_DISPLAY_NAME_MAPPINGS
    
    try:
        from .model_node_factory import create_audio_node_class
        from ..utils.model_config_loader import get_model_config_loader
        
        loader = get_model_config_loader()
        audio_models = loader.get_audio_models()
        
        for model in audio_models:
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
                node_class = create_audio_node_class(model_dict)
                class_name = node_class.__name__
                AUDIO_NODE_CLASS_MAPPINGS[class_name] = node_class
                AUDIO_NODE_DISPLAY_NAME_MAPPINGS[class_name] = model.name or model.id
            except Exception as e:
                logger.error(f"[JieKou] Failed to create audio node for {model.id}: {e}")
        
        logger.info(f"[JieKou] Registered {len(AUDIO_NODE_CLASS_MAPPINGS)} audio model nodes")
    
    except Exception as e:
        logger.error(f"[JieKou] Failed to register audio nodes: {e}")


# Register nodes on module import
register_audio_nodes()

