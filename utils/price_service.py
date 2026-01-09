"""
Price Service for JieKou ComfyUI Plugin

This module provides price lookup functionality by proxying requests
to the JieKou AI batch-price API.

Features:
- Batch price queries for multiple product IDs
- Error handling with graceful fallbacks
- Price formatting utilities
"""

import logging
from typing import Optional

logger = logging.getLogger("[JieKou]")


class PriceService:
    """
    Service for fetching product prices from JieKou AI API.
    
    Uses the batch-price endpoint:
    POST https://api-server.jiekou.ai/v1/product/batch-price
    """
    
    BATCH_PRICE_ENDPOINT = "/v1/product/batch-price"
    
    def __init__(self):
        from .api_client import JiekouAPI
        self._api = JiekouAPI()
    
    def get_prices(self, product_ids: list[str]) -> dict[str, dict]:
        """
        Fetch prices for a list of product IDs.
        
        Args:
            product_ids: List of product ID strings (e.g., ["SEEDREAM_4_0", "FLUX_2_PRO"])
            
        Returns:
            dict: Mapping of product_id -> price_info
                  price_info contains: {price, original_price, currency, unit}
        """
        if not product_ids:
            return {}
        
        try:
            response = self._api._request(
                "POST",
                self.BATCH_PRICE_ENDPOINT,
                data={"product_ids": product_ids},
                timeout=10
            )
            
            # Parse response
            # Expected format: { prices: [{ product_id, price, original_price, currency, unit }] }
            prices_list = response.get("prices", [])
            
            result = {}
            for item in prices_list:
                product_id = item.get("product_id")
                if product_id:
                    result[product_id] = {
                        "price": item.get("price", 0),
                        "original_price": item.get("original_price"),
                        "currency": item.get("currency", "USD"),
                        "unit": item.get("unit", "次"),
                    }
            
            logger.info(f"[JieKou] Fetched prices for {len(result)} products")
            return result
        
        except Exception as e:
            logger.error(f"[JieKou] Failed to fetch prices: {e}")
            return {}
    
    def get_price(self, product_id: str) -> Optional[dict]:
        """
        Fetch price for a single product ID.
        
        Args:
            product_id: Product ID string
            
        Returns:
            dict: Price info or None if not found
        """
        prices = self.get_prices([product_id])
        return prices.get(product_id)
    
    def format_price(self, price: float, currency: str = "USD", unit: str = "次") -> str:
        """
        Format price for display.
        
        Args:
            price: Price value
            currency: Currency code (default: USD)
            unit: Unit string (default: 次)
            
        Returns:
            str: Formatted price string (e.g., "$0.004/次")
        """
        if currency == "USD":
            if price >= 1:
                return f"${price:.2f}/{unit}"
            elif price >= 0.01:
                return f"${price:.3f}/{unit}"
            else:
                return f"${price:.4f}/{unit}"
        else:
            return f"{price:.4f} {currency}/{unit}"


# Singleton instance
_service: Optional[PriceService] = None


def get_price_service() -> PriceService:
    """Get the singleton price service instance"""
    global _service
    if _service is None:
        _service = PriceService()
    return _service

