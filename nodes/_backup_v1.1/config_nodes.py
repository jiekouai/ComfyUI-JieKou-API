"""
JiekouAI ComfyUI Plugin - Configuration Nodes
Test connection and API key verification
"""

import logging

logger = logging.getLogger("[JieKou]")


class JieKouTestConnection:
    """
    Test JiekouAI API connection and verify API Key
    
    Simply outputs connection status message
    """
    
    CATEGORY = "JieKou/Config"
    FUNCTION = "test_connection"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("*", {"default": None}),  # Dummy input for manual trigger
            }
        }
    
    def test_connection(self, trigger=None):
        """
        Test API connection
        
        Returns:
            tuple: (status message,)
        """
        logger.info("[JieKou] Testing API connection...")
        
        try:
            from ..utils.api_client import JiekouAPI, JiekouAPIError
            
            api = JiekouAPI()
            
            # Check if configured
            if not api.api_key:
                logger.warning("[JieKou] API Key not configured")
                return ("❌ API Key 未配置。请在设置中添加您的 API Key。",)
            
            # Call verify endpoint
            result = api.verify_key()
            
            logger.info("[JieKou] Connection successful")
            return ("✅ Connection successful",)
        
        except JiekouAPIError as e:
            logger.error(f"[JieKou] Connection test failed: {e.message}")
            return (f"❌ {e.message}",)
        
        except Exception as e:
            logger.error(f"[JieKou] Unexpected error: {e}")
            return (f"❌ 发生未知错误: {str(e)}",)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute to test connection"""
        import time
        return time.time()
