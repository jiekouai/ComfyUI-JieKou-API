"""
Generated Image Model Nodes for JieKou ComfyUI Plugin

This module contains dynamically generated node classes for all image models
defined in model_config.json. Each model has its own dedicated node.

Categories:
- image_t2i: Text to Image (文生图)
- image_edit: Image to Image (图生图)
- image_tool: Image Tools (图像工具)

Note: This module is auto-populated at import time from model_config.json.
Do not manually add node classes here.
"""

import logging

logger = logging.getLogger("[JieKou]")

# Node mappings will be populated by factory
IMAGE_NODE_CLASS_MAPPINGS = {}
IMAGE_NODE_DISPLAY_NAME_MAPPINGS = {}


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


def register_image_nodes():
    """
    Register all image model nodes from configuration.
    For models with multiple categories, creates a node in each category.
    Called during module initialization.
    """
    global IMAGE_NODE_CLASS_MAPPINGS, IMAGE_NODE_DISPLAY_NAME_MAPPINGS
    
    try:
        from .model_node_factory import create_image_node_class
        from ..utils.model_config_loader import get_model_config_loader
        
        loader = get_model_config_loader()
        image_models = loader.get_image_models()
        
        # Debug: Log all image models found
        logger.info(f"[JieKou] Found {len(image_models)} image models to register")
        for m in image_models:
            logger.debug(f"[JieKou]   - {m.id}: categories={m.categories}")
        
        # Track processed models to handle multi-category models only once
        processed_models = set()
        
        for model in image_models:
            # Skip if already processed (multi-category models appear in multiple categories)
            if model.id in processed_models:
                continue
            processed_models.add(model.id)
            
            # Get all image categories for this model
            image_categories = [c for c in model.categories if c.startswith("image_")]
            
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
            
            # Create a node for each image category this model belongs to
            for idx, category in enumerate(image_categories):
                try:
                    node_class = create_image_node_class(model_dict, target_category=category)
                    base_class_name = node_class.__name__
                    
                    # For additional categories, create a derived class with new CATEGORY
                    if idx > 0:
                        cat_short = category.split("_")[-1].upper()  # e.g., "image_edit" -> "EDIT"
                        new_class_name = base_class_name + f"_{cat_short}"
                        
                        # Get the correct CATEGORY path for this category
                        category_map = {
                            "image_t2i": "JieKou AI/Image/Text to Image",
                            "image_edit": "JieKou AI/Image/Image to Image",
                            "image_tool": "JieKou AI/Image/Tools",
                        }
                        new_category_path = category_map.get(category, "JieKou AI/Image")
                        
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
                    
                    IMAGE_NODE_CLASS_MAPPINGS[class_name] = node_class
                    IMAGE_NODE_DISPLAY_NAME_MAPPINGS[class_name] = display_name
                    
                except Exception as e:
                    logger.error(f"[JieKou] Failed to create image node for {model.id} in {category}: {e}")
        
        logger.info(f"[JieKou] Registered {len(IMAGE_NODE_CLASS_MAPPINGS)} image model nodes")
    
    except Exception as e:
        logger.error(f"[JieKou] Failed to register image nodes: {e}")


# Register nodes on module import
register_image_nodes()
