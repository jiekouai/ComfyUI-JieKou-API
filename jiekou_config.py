"""
JiekouAI ComfyUI Plugin - Configuration Management
Handles reading API Key from config.ini or environment variables
"""

import os
import configparser
import logging
from pathlib import Path

logger = logging.getLogger("[JieKou]")

# Plugin root directory
PLUGIN_DIR = Path(__file__).parent
CONFIG_FILE = PLUGIN_DIR / "config.ini"
CONFIG_EXAMPLE = PLUGIN_DIR / "config.ini.example"

# Environment variable name
ENV_API_KEY = "JIEKOU_API_KEY"
ENV_BASE_URL = "JIEKOU_BASE_URL"

# Default values
DEFAULT_BASE_URL = "https://api.jiekou.ai"
DEFAULT_CACHE_TTL = 3600  # 1 hour


class JiekouConfig:
    """Configuration manager for JiekouAI plugin"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from file"""
        self._config = configparser.ConfigParser()
        if CONFIG_FILE.exists():
            self._config.read(CONFIG_FILE)
            logger.info(f"Loaded config from {CONFIG_FILE}")
        else:
            logger.warning(f"Config file not found: {CONFIG_FILE}")
            logger.info("Using environment variables or defaults")
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
    
    @property
    def api_key(self) -> str | None:
        """
        Get API Key with priority:
        1. Environment variable JIEKOU_API_KEY
        2. config.ini [AUTH] api_key
        """
        # Priority 1: Environment variable
        env_key = os.environ.get(ENV_API_KEY)
        if env_key:
            return env_key
        
        # Priority 2: Config file
        try:
            key = self._config.get("AUTH", "api_key", fallback=None)
            if key and key != "YOUR_API_KEY_HERE":
                return key
        except (configparser.NoSectionError, configparser.NoOptionError):
            pass
        
        return None
    
    @property
    def base_url(self) -> str:
        """
        Get API Base URL with priority:
        1. Environment variable JIEKOU_BASE_URL
        2. config.ini [API] base_url
        3. Default value
        """
        # Priority 1: Environment variable
        env_url = os.environ.get(ENV_BASE_URL)
        if env_url:
            return env_url.rstrip("/")
        
        # Priority 2: Config file
        try:
            url = self._config.get("API", "base_url", fallback=None)
            if url:
                return url.rstrip("/")
        except (configparser.NoSectionError, configparser.NoOptionError):
            pass
        
        # Priority 3: Default
        return DEFAULT_BASE_URL
    
    @property
    def cache_ttl(self) -> int:
        """Get cache TTL in seconds"""
        try:
            return self._config.getint("CACHE", "cache_ttl", fallback=DEFAULT_CACHE_TTL)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return DEFAULT_CACHE_TTL
    
    def save_api_key(self, api_key: str) -> bool:
        """
        Save API Key to config.ini
        Returns True if successful, False otherwise
        """
        try:
            # Ensure AUTH section exists
            if not self._config.has_section("AUTH"):
                self._config.add_section("AUTH")
            
            self._config.set("AUTH", "api_key", api_key)
            
            # Write to file
            with open(CONFIG_FILE, "w") as f:
                self._config.write(f)
            
            logger.info("API Key saved to config.ini")
            return True
        except Exception as e:
            logger.error(f"Failed to save API Key: {e}")
            return False
    
    def is_configured(self) -> bool:
        """Check if API Key is configured"""
        return self.api_key is not None


# Singleton instance
def get_config() -> JiekouConfig:
    """Get the singleton config instance"""
    return JiekouConfig()

