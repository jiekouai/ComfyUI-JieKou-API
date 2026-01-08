"""
Base Model Node Class for JieKou ComfyUI Plugin

This module provides a base class that serves as the foundation for
dynamically generated model-specific nodes. It contains common
execution logic shared across all JieKou model nodes.

The base class handles:
- API client initialization
- Common parameter processing
- Response handling (sync/async)
- Image tensor conversion
- Error handling and logging
- File saving to output directory
"""

import torch
import logging
import os
import time
import numpy as np
from PIL import Image
from typing import Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger("[JieKou]")


class BaseModelNode(ABC):
    """
    Abstract base class for all JieKou model nodes.
    
    Subclasses must implement:
    - get_model_config(): Returns the model configuration
    - process_response(): Handles model-specific response parsing
    """
    
    # Default ComfyUI node properties
    OUTPUT_NODE = True
    
    # Polling configuration for async models
    POLL_INTERVAL = 3  # seconds
    MAX_POLL_TIME = 1800  # 30 minutes max
    
    @classmethod
    @abstractmethod
    def get_model_config(cls) -> dict:
        """Return the model configuration from model_config.json"""
        pass
    
    def get_api_client(self):
        """Get initialized JiekouAPI client"""
        from ..utils.api_client import JiekouAPI
        return JiekouAPI()
    
    def build_request_data(self, model_config: dict, **kwargs) -> dict:
        """
        Build API request data from kwargs.
        
        Filters out None values, empty strings, and internal parameters.
        """
        data = {}
        skip_keys = {"_dynamic_params", "save_to_disk", "extra_params"}
        
        for key, value in kwargs.items():
            if key in skip_keys:
                continue
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            data[key] = value
        
        return data
    
    def execute_sync_request(self, endpoint: str, data: dict, timeout: int = 1200) -> dict:
        """Execute synchronous API request"""
        api = self.get_api_client()
        return api._request("POST", endpoint, data=data, timeout=timeout)
    
    def execute_async_request(self, endpoint: str, data: dict) -> dict:
        """
        Execute async API request with polling.
        
        Returns the final result after task completion.
        """
        api = self.get_api_client()
        
        # Submit task
        response = api._request("POST", endpoint, data=data, timeout=60)
        task_id = response.get("task_id")
        
        if not task_id:
            raise ValueError("API 未返回 task_id")
        
        logger.info(f"[JieKou] Task submitted: {task_id}")
        
        # Poll for completion
        return api.poll_task_until_complete(task_id)
    
    def handle_image_response(
        self,
        response: dict,
        response_type: str,
        is_async: bool
    ) -> tuple[torch.Tensor, str]:
        """
        Handle API response and return image tensor and URL.
        
        Args:
            response: API response dict
            response_type: Expected response type (b64_json, image_urls)
            is_async: Whether this was an async request
            
        Returns:
            tuple: (image_tensor [B,H,W,C], image_url)
        """
        from ..utils.tensor_utils import url_to_tensor, base64_to_tensor
        
        api = self.get_api_client()
        image_url = ""
        
        if is_async:
            # Async result - already polled, extract URL
            image_url = api.get_image_result_url(response)
            if not image_url:
                raise ValueError("任务完成但未返回图像 URL")
            image_tensor = url_to_tensor(image_url)
        
        elif response_type == "b64_json":
            images = response.get("data", [])
            if not images:
                raise ValueError("API 未返回图像数据")
            
            first_image = images[0]
            if "b64_json" in first_image and first_image["b64_json"]:
                image_tensor = base64_to_tensor(first_image["b64_json"])
                image_url = "(base64 data)"
            elif "url" in first_image and first_image["url"]:
                image_url = first_image["url"]
                image_tensor = url_to_tensor(image_url)
            else:
                self._check_response_error(first_image)
                raise ValueError("API 返回的图像数据格式未知")
        
        elif response_type == "image_urls":
            urls = response.get("image_urls") or response.get("images", [])
            if not urls:
                raise ValueError("API 未返回图像数据")
            image_url = urls[0] if isinstance(urls[0], str) else urls[0].get("url", "")
            image_tensor = url_to_tensor(image_url)
        
        else:
            raise ValueError(f"不支持的响应类型: {response_type}")
        
        return image_tensor, image_url
    
    def handle_video_response(self, response: dict) -> tuple[torch.Tensor, str]:
        """
        Handle video API response and return frame tensor and URL.
        
        Args:
            response: API response dict (already polled for async)
            
        Returns:
            tuple: (frames_tensor [B,H,W,C], video_url)
        """
        from ..utils.tensor_utils import video_to_frames
        
        api = self.get_api_client()
        
        video_url = api.get_video_result_url(response)
        if not video_url:
            raise ValueError("任务完成但未返回视频 URL")
        
        # Download video
        video_bytes = api.download_file(video_url)
        
        # Decode to frames
        frames_tensor = video_to_frames(video_bytes)
        
        return frames_tensor, video_url
    
    def save_image_to_disk(
        self,
        tensor: torch.Tensor,
        prefix: str = "jiekou_image"
    ) -> str:
        """
        Save image tensor to output folder.
        
        Args:
            tensor: Image tensor [B,H,W,C]
            prefix: Filename prefix
            
        Returns:
            str: Full path to saved file
        """
        import folder_paths
        
        output_dir = folder_paths.get_output_directory()
        filename = f"{prefix}_{int(time.time())}.png"
        output_path = os.path.join(output_dir, filename)
        
        img_np = tensor[0].cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        img.save(output_path, format="PNG")
        
        logger.info(f"[JieKou] Image saved to: {output_path}")
        return output_path
    
    def save_video_to_disk(
        self,
        video_bytes: bytes,
        prefix: str = "jiekou_video"
    ) -> str:
        """
        Save video bytes to output folder.
        
        Args:
            video_bytes: Raw video data
            prefix: Filename prefix
            
        Returns:
            str: Full path to saved file
        """
        import folder_paths
        
        output_dir = folder_paths.get_output_directory()
        filename = f"{prefix}_{int(time.time())}.mp4"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, "wb") as f:
            f.write(video_bytes)
        
        logger.info(f"[JieKou] Video saved to: {output_path}")
        return output_path
    
    def save_for_preview(
        self,
        tensor: torch.Tensor,
        subfolder: str = "jiekou"
    ) -> list:
        """
        Save image tensor to temp folder for UI preview.
        
        Args:
            tensor: Image tensor [B,H,W,C] or [H,W,C]
            subfolder: Subfolder in temp directory
            
        Returns:
            list: File info dicts for UI
        """
        import folder_paths
        import uuid
        
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        output_dir = folder_paths.get_temp_directory()
        full_output_folder = os.path.join(output_dir, subfolder)
        os.makedirs(full_output_folder, exist_ok=True)
        
        results = []
        for i in range(tensor.shape[0]):
            img_np = tensor[i].cpu().numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            
            filename = f"preview_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(full_output_folder, filename)
            img.save(filepath, format="PNG")
            
            results.append({
                "filename": filename,
                "subfolder": subfolder,
                "type": "temp"
            })
        
        return results
    
    def _check_response_error(self, response_item: dict) -> None:
        """Check for error messages in response item"""
        if not isinstance(response_item, dict):
            return
        
        revised = response_item.get("revised_prompt", "")
        if "blocked" in revised.lower() or "error" in revised.lower():
            raise ValueError(f"API 请求被拒绝: {revised[:200]}")
        
        if response_item.get("error"):
            raise ValueError(f"API 返回错误: {response_item['error']}")
    
    def log_request(
        self,
        endpoint: str,
        data: dict,
        is_async: bool = False
    ) -> None:
        """Log API request details (truncate long values)"""
        log_data = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 100:
                log_data[k] = v[:50] + "..."
            elif isinstance(v, list) and v and isinstance(v[0], str) and len(v[0]) > 100:
                log_data[k] = [s[:50] + "..." for s in v]
            else:
                log_data[k] = v
        
        logger.info(f"[JieKou] ===== API Request =====")
        logger.info(f"[JieKou] Endpoint: {endpoint}, async={is_async}")
        logger.info(f"[JieKou] Request body: {log_data}")

