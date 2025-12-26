"""
JiekouAI ComfyUI Plugin - Cache Manager
Manages model list caching with TTL-based invalidation
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("[JieKou]")

# Cache directory
PLUGIN_DIR = Path(__file__).parent.parent
CACHE_DIR = PLUGIN_DIR / "cache"
MODELS_CACHE_FILE = CACHE_DIR / "models.json"


class CacheManager:
    """Manages cached data with TTL-based invalidation"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ensure_cache_dir()
        return cls._instance
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_ttl(self) -> int:
        """Get cache TTL from config"""
        try:
            from ..jiekou_config import get_config
            return get_config().cache_ttl
        except Exception:
            return 3600  # Default 1 hour
    
    def _read_cache_file(self, cache_file: Path) -> dict | None:
        """
        Read cache file and check expiration
        
        Returns:
            Cache data dict or None if expired/missing
        """
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            # Check expiration
            cached_at = cache_data.get("cached_at", 0)
            ttl = self._get_cache_ttl()
            
            if time.time() - cached_at > ttl:
                logger.info(f"[JieKou] Cache expired: {cache_file.name}")
                return None
            
            return cache_data.get("data")
        
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"[JieKou] Failed to read cache: {e}")
            return None
    
    def _write_cache_file(self, cache_file: Path, data: Any):
        """Write data to cache file with timestamp"""
        try:
            cache_data = {
                "cached_at": time.time(),
                "data": data
            }
            
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[JieKou] Cache written: {cache_file.name}")
        
        except IOError as e:
            logger.warning(f"[JieKou] Failed to write cache: {e}")
    
    # ===== Model List Cache =====
    
    def get_models(self, model_type: str = None) -> list[dict] | None:
        """
        Get cached model list
        
        Args:
            model_type: Optional filter by type
        
        Returns:
            List of model dicts or None if cache miss
        """
        cache_data = self._read_cache_file(MODELS_CACHE_FILE)
        
        if cache_data is None:
            return None
        
        models = cache_data.get("models", [])
        
        # Filter by type if specified
        if model_type:
            models = [m for m in models if m.get("type") == model_type]
        
        return models
    
    def set_models(self, models: list[dict]):
        """
        Cache model list
        
        Args:
            models: List of model dicts
        """
        self._write_cache_file(MODELS_CACHE_FILE, {"models": models})
    
    def invalidate_models(self):
        """Invalidate model cache"""
        if MODELS_CACHE_FILE.exists():
            MODELS_CACHE_FILE.unlink()
            logger.info("[JieKou] Model cache invalidated")
    
    # ===== T063: Cache refresh logic =====
    
    def refresh_models(self, force: bool = False) -> list[dict]:
        """
        Refresh model cache from API
        
        Args:
            force: Force refresh even if cache is valid
        
        Returns:
            List of model dicts
        """
        # Check cache first (unless force)
        if not force:
            cached = self.get_models()
            if cached is not None:
                logger.info("[JieKou] Using cached model list")
                return cached
        
        # Fetch from API
        try:
            from .api_client import JiekouAPI
            api = JiekouAPI()
            models = api.get_models()
            
            # Update cache
            self.set_models(models)
            
            return models
        
        except Exception as e:
            logger.warning(f"[JieKou] Failed to refresh models: {e}")
            
            # Fall back to cache even if expired
            if MODELS_CACHE_FILE.exists():
                try:
                    with open(MODELS_CACHE_FILE, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                    return cache_data.get("data", {}).get("models", [])
                except Exception:
                    pass
            
            return []
    
    def is_cache_valid(self) -> bool:
        """Check if model cache is valid (not expired)"""
        return self._read_cache_file(MODELS_CACHE_FILE) is not None


# Singleton instance
def get_cache_manager() -> CacheManager:
    """Get the singleton cache manager instance"""
    return CacheManager()

