"""
Model Configuration Loader for JieKou ComfyUI Plugin

This module provides utilities to load and query model configurations
from the static model_config.json file. It serves as the single source
of truth for all model-related information.

Features:
- Lazy loading of configuration on first access
- Caching for performance
- Query methods for filtering by category, ID, etc.
- Type-safe access to model parameters and product IDs
"""

import json
import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

logger = logging.getLogger("[JieKou]")


@dataclass
class ModelParameter:
    """Represents a single model parameter definition"""
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[list] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class ModelConfig:
    """Represents a complete model configuration"""
    id: str
    name: str
    description: str
    category: str
    endpoint: str
    is_async: bool
    response_type: str
    parameters: list[ModelParameter]
    product_ids: list[str]
    valid_combinations: list[dict]
    
    @property
    def display_name(self) -> str:
        """Get display name for ComfyUI menu"""
        return self.name or self.id
    
    @property
    def node_class_name(self) -> str:
        """Generate ComfyUI node class name from model ID"""
        # Convert "seedream-4-0" to "JieKouSeedream4_0"
        parts = self.id.replace("-", "_").replace(".", "_").split("_")
        camel = "".join(part.capitalize() for part in parts)
        return f"JieKou{camel}"
    
    @property
    def category_path(self) -> str:
        """Get ComfyUI category path for menu organization"""
        category_map = {
            "image_t2i": "JieKou AI/Image/Text to Image",
            "image_edit": "JieKou AI/Image/Edit",
            "image_tool": "JieKou AI/Image/Tools",
            "video_t2v": "JieKou AI/Video/Text to Video",
            "video_i2v": "JieKou AI/Video/Image to Video",
            "video_v2v": "JieKou AI/Video/Video to Video",
            "audio_tts": "JieKou AI/Audio/Text to Speech",
            "audio_asr": "JieKou AI/Audio/Speech to Text",
        }
        return category_map.get(self.category, "JieKou AI/Other")
    
    def get_product_id_for_params(self, params: dict) -> Optional[str]:
        """
        Find the product ID that matches the given parameter values.
        
        Uses valid_combinations to map parameter values to product IDs.
        Returns None if no matching combination found.
        """
        if not self.valid_combinations:
            # No combinations defined - use first product_id if available
            return self.product_ids[0] if self.product_ids else None
        
        for combo in self.valid_combinations:
            combo_params = combo.get("params", {})
            product_id = combo.get("product_id")
            
            # Check if all combo params match the given params
            match = True
            for key, value in combo_params.items():
                if params.get(key) != value:
                    match = False
                    break
            
            if match and product_id:
                return product_id
        
        # No exact match - return first product_id as fallback
        return self.product_ids[0] if self.product_ids else None


class ModelConfigLoader:
    """
    Singleton loader for model configurations.
    
    Loads configuration from model_config.json on first access
    and caches the result for subsequent queries.
    """
    
    _instance: Optional["ModelConfigLoader"] = None
    _config_data: Optional[dict] = None
    _models: Optional[dict[str, ModelConfig]] = None
    _models_by_category: Optional[dict[str, list[ModelConfig]]] = None
    
    def __new__(cls) -> "ModelConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _load_config(self) -> dict:
        """Load configuration from JSON file"""
        if self._config_data is not None:
            return self._config_data
        
        config_path = Path(__file__).parent.parent / "model_config.json"
        
        if not config_path.exists():
            logger.error(f"[JieKou] model_config.json not found at {config_path}")
            raise FileNotFoundError(f"model_config.json not found at {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self._config_data = json.load(f)
        
        logger.info(f"[JieKou] Loaded model config v{self._config_data.get('version', 'unknown')}")
        return self._config_data
    
    def _parse_models(self) -> dict[str, ModelConfig]:
        """Parse model configurations from JSON data"""
        if self._models is not None:
            return self._models
        
        config = self._load_config()
        self._models = {}
        self._models_by_category = {}
        
        for model_data in config.get("models", []):
            # Parse parameters
            params = []
            for param_data in model_data.get("parameters", []):
                params.append(ModelParameter(
                    name=param_data.get("name", ""),
                    type=param_data.get("type", "string"),
                    description=param_data.get("description", ""),
                    required=param_data.get("required", False),
                    default=param_data.get("default"),
                    enum=param_data.get("enum"),
                    minimum=param_data.get("minimum"),
                    maximum=param_data.get("maximum"),
                ))
            
            # Create model config
            model = ModelConfig(
                id=model_data.get("id", ""),
                name=model_data.get("name", ""),
                description=model_data.get("description", ""),
                category=model_data.get("category", ""),
                endpoint=model_data.get("endpoint", ""),
                is_async=model_data.get("is_async", True),
                response_type=model_data.get("response_type", "image_urls"),
                parameters=params,
                product_ids=model_data.get("product_ids", []),
                valid_combinations=model_data.get("valid_combinations", []),
            )
            
            self._models[model.id] = model
            
            # Group by category
            if model.category not in self._models_by_category:
                self._models_by_category[model.category] = []
            self._models_by_category[model.category].append(model)
        
        logger.info(f"[JieKou] Parsed {len(self._models)} models across {len(self._models_by_category)} categories")
        return self._models
    
    def get_all_models(self) -> list[ModelConfig]:
        """Get all model configurations"""
        models = self._parse_models()
        return list(models.values())
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get a specific model by ID"""
        models = self._parse_models()
        return models.get(model_id)
    
    def get_model_by_name(self, name: str) -> Optional[ModelConfig]:
        """Get a model by display name"""
        models = self._parse_models()
        for model in models.values():
            if model.name == name:
                return model
        return None
    
    def resolve_model(self, id_or_name: str) -> Optional[ModelConfig]:
        """Resolve model by ID or name"""
        model = self.get_model(id_or_name)
        if model:
            return model
        return self.get_model_by_name(id_or_name)
    
    def get_models_by_category(self, category: str) -> list[ModelConfig]:
        """Get all models in a specific category"""
        self._parse_models()
        return self._models_by_category.get(category, [])
    
    def get_categories(self) -> list[str]:
        """Get list of all available categories"""
        self._parse_models()
        return list(self._models_by_category.keys())
    
    def get_image_models(self) -> list[ModelConfig]:
        """Get all image generation models"""
        result = []
        for cat in ["image_t2i", "image_edit", "image_tool"]:
            result.extend(self.get_models_by_category(cat))
        return result
    
    def get_video_models(self) -> list[ModelConfig]:
        """Get all video generation models"""
        result = []
        for cat in ["video_t2v", "video_i2v", "video_v2v"]:
            result.extend(self.get_models_by_category(cat))
        return result
    
    def get_audio_models(self) -> list[ModelConfig]:
        """Get all audio models"""
        result = []
        for cat in ["audio_tts", "audio_asr"]:
            result.extend(self.get_models_by_category(cat))
        return result
    
    def get_version(self) -> str:
        """Get configuration version"""
        config = self._load_config()
        return config.get("version", "unknown")
    
    def reload(self) -> None:
        """Force reload configuration from file"""
        self._config_data = None
        self._models = None
        self._models_by_category = None
        self._load_config()
        self._parse_models()


# Singleton accessor
_loader: Optional[ModelConfigLoader] = None


def get_model_config_loader() -> ModelConfigLoader:
    """Get the singleton model config loader instance"""
    global _loader
    if _loader is None:
        _loader = ModelConfigLoader()
    return _loader


def get_model(model_id: str) -> Optional[ModelConfig]:
    """Convenience function to get a model by ID"""
    return get_model_config_loader().get_model(model_id)


def get_all_models() -> list[ModelConfig]:
    """Convenience function to get all models"""
    return get_model_config_loader().get_all_models()

