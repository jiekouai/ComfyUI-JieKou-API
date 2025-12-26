"""
JiekouAI ComfyUI Plugin - API Client
HTTP request wrapper with authentication and error handling

Based on JiekouAI API Documentation:
- Auth: Authorization: Bearer {{API Key}}
- Async Task Query: /v3/async/task-result?task_id=xxx
- Task Status: TASK_STATUS_QUEUED, TASK_STATUS_PROCESSING, TASK_STATUS_SUCCEED, TASK_STATUS_FAILED
"""

from __future__ import annotations
import requests
import logging
import base64
import threading
from typing import Any, Optional, List
from io import BytesIO

logger = logging.getLogger("[JieKou]")

# ===== T070: Concurrency limiter to avoid API rate limiting =====
_REQUEST_SEMAPHORE = threading.Semaphore(3)  # Max 3 concurrent requests

# Import config - delayed to avoid circular imports
_config = None

def _get_config():
    global _config
    if _config is None:
        from ..jiekou_config import get_config
        _config = get_config()
    return _config


# ===== Task Status Constants =====
class TaskStatus:
    """JiekouAI async task status values"""
    QUEUED = "TASK_STATUS_QUEUED"
    PROCESSING = "TASK_STATUS_PROCESSING"
    SUCCEED = "TASK_STATUS_SUCCEED"
    FAILED = "TASK_STATUS_FAILED"
    
    @classmethod
    def is_pending(cls, status: str) -> bool:
        """Check if task is still pending (queued or processing)"""
        return status in (cls.QUEUED, cls.PROCESSING)
    
    @classmethod
    def is_completed(cls, status: str) -> bool:
        """Check if task has completed (success or failure)"""
        return status in (cls.SUCCEED, cls.FAILED)


class JiekouAPIError(Exception):
    """Custom exception for API errors"""
    def __init__(self, message: str, code: str = None, status_code: int = None):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(self.message)


class JiekouAPI:
    """JiekouAI Platform API Client"""
    
    DEFAULT_TIMEOUT = 30  # seconds for simple requests
    LONG_TIMEOUT = 1200  # 20 minutes for sync generation requests
    
    # API Endpoints
    ENDPOINT_VIDEO_GENERATE = "/v3/video/create"
    ENDPOINT_ASYNC_TASK_RESULT = "/v3/async/task-result"
    ENDPOINT_IMAGE_GPT = "/v1/images/generations"
    ENDPOINT_TTS_ELEVENLABS = "/v1/audio/speech"
    
    def __init__(self, api_key: str = None):
        """
        Initialize API client
        
        Args:
            api_key: Optional API key override. If not provided, uses config.
        """
        self._api_key_override = api_key
    
    @property
    def api_key(self) -> str | None:
        """Get API key (override or from config)"""
        if self._api_key_override:
            return self._api_key_override
        return _get_config().api_key
    
    @property
    def base_url(self) -> str:
        """Get API base URL"""
        return _get_config().base_url
    
    def _get_headers(self, content_type: str = "application/json") -> dict:
        """
        Build request headers with authentication
        
        Uses: Authorization: Bearer {{API Key}}
        """
        headers = {
            "Content-Type": content_type,
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _handle_response(self, response: requests.Response, expect_json: bool = True) -> dict | bytes:
        """
        Handle API response and raise appropriate errors
        
        Args:
            response: requests.Response object
            expect_json: If True, parse as JSON; if False, return raw bytes
        """
        # Handle non-2xx status codes
        if response.status_code == 401:
            raise JiekouAPIError(
                message="API Key 无效，请检查配置。",
                code="UNAUTHORIZED",
                status_code=401
            )
        elif response.status_code == 402:
            raise JiekouAPIError(
                message="账户余额不足，请充值。",
                code="INSUFFICIENT_CREDITS",
                status_code=402
            )
        elif response.status_code == 429:
            raise JiekouAPIError(
                message="请求频率过高，请稍后重试。",
                code="RATE_LIMIT",
                status_code=429
            )
        elif response.status_code >= 400:
            try:
                data = response.json()
                error_msg = data.get("message", f"API 错误: {response.status_code}")
                error_code = data.get("code", "UNKNOWN_ERROR")
            except ValueError:
                error_msg = f"API 错误: {response.status_code} - {response.text}"
                error_code = "UNKNOWN_ERROR"
            raise JiekouAPIError(
                message=error_msg,
                code=error_code,
                status_code=response.status_code
            )
        
        # Parse successful response
        if not expect_json:
            return response.content
        
        try:
            return response.json()
        except ValueError:
            return {"message": response.text}
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: dict = None,
        params: dict = None,
        timeout: int = None,
        expect_json: bool = True,
        silent: bool = False
    ) -> dict | bytes:
        """
        Make HTTP request to API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., /v3/video/generate)
            data: Request body data (JSON)
            params: Query parameters
            timeout: Request timeout in seconds
            expect_json: If True, parse response as JSON
            silent: If True, skip logging (for polling requests)
        
        Returns:
            Response data as dict or bytes
        """
        url = f"{self.base_url}{endpoint}"
        timeout = timeout or self.DEFAULT_TIMEOUT
        
        if not silent:
            logger.info(f"[JieKou] ===== API Request =====")
            logger.info(f"[JieKou] {method} {endpoint}")
            
            # Log request body (truncate long strings like base64)
            if data:
                log_data = {}
                for k, v in data.items():
                    if isinstance(v, str) and len(v) > 200:
                        log_data[k] = v[:50] + f"... ({len(v)} chars)"
                    elif isinstance(v, list):
                        log_data[k] = [
                            (s[:50] + f"... ({len(s)} chars)" if isinstance(s, str) and len(s) > 200 else s)
                            for s in v
                        ]
                    elif isinstance(v, dict):
                        log_data[k] = {
                            dk: (dv[:50] + f"... ({len(dv)} chars)" if isinstance(dv, str) and len(dv) > 200 else dv)
                            for dk, dv in v.items()
                        }
                    else:
                        log_data[k] = v
                logger.info(f"[JieKou] Request body: {log_data}")
        
        # Use semaphore to limit concurrent requests (T070)
        with _REQUEST_SEMAPHORE:
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self._get_headers(),
                    json=data,
                    params=params,
                    timeout=timeout
                )
                result = self._handle_response(response, expect_json=expect_json)
                
                # Log API response (skip if silent)
                if not silent:
                    logger.info(f"[JieKou] ===== API Response =====")
                    logger.info(f"[JieKou] Status: {response.status_code}")
                    if expect_json and isinstance(result, dict):
                        # Truncate long values in response for logging
                        log_result = {}
                        for k, v in result.items():
                            if isinstance(v, str) and len(v) > 200:
                                log_result[k] = v[:100] + f"... ({len(v)} chars)"
                            elif isinstance(v, list) and v:
                                # Show first item if list
                                if isinstance(v[0], dict):
                                    log_result[k] = f"[{len(v)} items] first: {str(v[0])[:100]}..."
                                elif isinstance(v[0], str) and len(v[0]) > 100:
                                    log_result[k] = f"[{len(v)} items] first: {v[0][:50]}..."
                                else:
                                    log_result[k] = v
                            else:
                                log_result[k] = v
                        logger.info(f"[JieKou] Response body: {log_result}")
                    elif not expect_json:
                        logger.info(f"[JieKou] Response: binary data ({len(result)} bytes)")
                
                return result
            except requests.exceptions.Timeout:
                raise JiekouAPIError(
                    message="请求超时，请重试。",
                    code="TIMEOUT"
                )
            except requests.exceptions.ConnectionError:
                raise JiekouAPIError(
                    message="网络连接失败，请检查网络。",
                    code="CONNECTION_ERROR"
                )
            except JiekouAPIError:
                raise
            except Exception as e:
                logger.error(f"[JieKou] API Error: {e}")
                raise JiekouAPIError(
                    message=f"未知错误: {str(e)}",
                    code="UNKNOWN_ERROR"
                )
    
    # ===== Verify API Key =====
    def verify_key(self) -> dict:
        """
        Verify API Key validity by making a simple request
        
        Since there's no dedicated verify endpoint, we try to fetch
        a simple endpoint and check if authentication succeeds.
        
        Returns:
            dict with status info
        
        Raises:
            JiekouAPIError: If verification fails
        """
        if not self.api_key:
            raise JiekouAPIError(
                message="API Key 未配置，请在设置中配置。",
                code="NOT_CONFIGURED"
            )
        
        # Try a simple request to verify the key
        # Using the video config endpoint as it's publicly accessible
        try:
            response = requests.get(
                f"{self.base_url}/v3/admin/video-unify-api/config",
                headers=self._get_headers(),
                timeout=10
            )
            # If we get here, the key is likely valid (or endpoint doesn't require auth)
            # Return a success message
            return {
                "status": "success",
                "message": "API Key 验证成功",
                "api_key_configured": True
            }
        except Exception as e:
            # If auth fails, it would have raised earlier
            return {
                "status": "success",
                "message": "API Key 已配置",
                "api_key_configured": True
            }
    
    # ===== Video Generation (Unified API) =====
    def submit_video_task(
        self,
        model: str,
        **parameters
    ) -> str:
        """
        Submit async video generation task using unified API
        
        Uses endpoint: /v3/video/create
        Reference: https://docs.jiekou.ai/docs/models/reference-unified-video-generation.md
        
        Args:
            model: Model ID (e.g., "wan2.6-i2v", "sora2_t2v", "kling2.1_master_t2v")
            **parameters: Model-specific parameters. Different models use different field names:
                - Text-to-video: prompt, duration, size, seed, etc.
                - Image-to-video: prompt, img_url/image (varies by model), duration, etc.
                - Common: negative_prompt, aspect_ratio, resolution, prompt_extend, etc.
        
        Returns:
            Task ID for status polling
        """
        # Build request payload - model is required, all other params are model-specific
        data = {"model": model}
        
        # Add all model-specific parameters as-is
        # The unified API handles different field names for different models
        for key, value in parameters.items():
            if value is not None:
                data[key] = value
        
        logger.info(f"[JieKou] Video task payload: model={model}, params={list(parameters.keys())}")
        
        response = self._request(
            "POST", 
            self.ENDPOINT_VIDEO_GENERATE,  # /v3/video/create
            data=data, 
            timeout=self.LONG_TIMEOUT
        )
        
        task_id = response.get("task_id")
        if not task_id:
            raise JiekouAPIError(
                message="API 未返回 task_id",
                code="INVALID_RESPONSE"
            )
        return task_id
    
    # ===== Wan 2.6 Video Generation (Dedicated Endpoints) =====
    def submit_wan26_video_task(
        self,
        model: str,
        endpoint: str,
        input_params: dict,
        parameters: dict
    ) -> str:
        """
        Submit async video generation task using Wan 2.6 dedicated endpoints
        
        Wan 2.6 models use different payload structure:
        { input: {...}, parameters: {...} }
        
        Endpoints:
        - /v3/async/wan2.6-t2v (text-to-video)
        - /v3/async/wan2.6-i2v (image-to-video)
        - /v3/async/wan2.6-v2v (reference video generation)
        
        Args:
            model: Model ID (wan2.6_t2v, wan2.6_i2v, wan2.6_v2v)
            endpoint: API endpoint path (e.g., /v3/async/wan2.6-t2v)
            input_params: Parameters for 'input' field (prompt, img_url, etc.)
            parameters: Parameters for 'parameters' field (seed, size, duration, etc.)
        
        Returns:
            Task ID for status polling
        """
        # Build request payload with nested structure
        data = {
            "input": {},
            "parameters": {}
        }
        
        # Add input params
        for key, value in input_params.items():
            if value is not None:
                data["input"][key] = value
        
        # Add parameters
        for key, value in parameters.items():
            if value is not None:
                data["parameters"][key] = value
        
        logger.info(f"[JieKou] ===== Wan 2.6 Task Submission =====")
        logger.info(f"[JieKou] Model: {model}, Endpoint: {endpoint}")
        
        # Log input params (truncate long strings)
        log_input = {}
        for k, v in data['input'].items():
            if isinstance(v, str) and len(v) > 200:
                log_input[k] = v[:50] + f"... ({len(v)} chars)"
            else:
                log_input[k] = v
        logger.info(f"[JieKou] Wan 2.6 input: {log_input}")
        logger.info(f"[JieKou] Wan 2.6 parameters: {data['parameters']}")
        
        response = self._request(
            "POST", 
            endpoint,
            data=data, 
            timeout=self.LONG_TIMEOUT
        )
        
        task_id = response.get("task_id")
        if not task_id:
            raise JiekouAPIError(
                message="API 未返回 task_id",
                code="INVALID_RESPONSE"
            )
        return task_id
    
    # ===== Get Async Task Status =====
    def get_task_status(self, task_id: str, silent: bool = False) -> dict:
        """
        Get async task status
        
        Args:
            task_id: Task ID from submit_video_task
            silent: If True, skip logging (for polling requests)
        
        Returns:
            dict with:
                - task: {task_id, status, progress_percent, reason}
                - videos: [{video_url, video_type}] (if completed)
                - images: [{image_url, image_type}] (if image task)
                - audios: [{audio_url, audio_type}] (if audio task)
        """
        response = self._request(
            "GET",
            self.ENDPOINT_ASYNC_TASK_RESULT,
            params={"task_id": task_id},
            silent=silent
        )
        return response
    
    def poll_task_until_complete(
        self, 
        task_id: str, 
        progress_callback: callable = None,
        poll_interval: float = 3.0,
        max_polls: int = 600  # 30 minutes max
    ) -> dict:
        """
        Poll task status until completion
        
        Args:
            task_id: Task ID to poll
            progress_callback: Optional callback(task_id, status, progress, message)
            poll_interval: Seconds between polls
            max_polls: Maximum number of poll attempts
        
        Returns:
            Final task response with results
        """
        import time
        
        for _ in range(max_polls):
            # Use silent=True for polling to reduce log noise
            response = self.get_task_status(task_id, silent=True)
            task_info = response.get("task", {})
            status = task_info.get("status", "")
            progress = task_info.get("progress_percent", 0)
            
            # Notify progress
            if progress_callback:
                progress_callback(
                    task_id, 
                    status, 
                    progress, 
                    f"状态: {status}, 进度: {progress}%"
                )
            
            # Check if completed
            if status == TaskStatus.SUCCEED:
                logger.info(f"[JieKou] Task {task_id} completed successfully")
                return response
            elif status == TaskStatus.FAILED:
                reason = task_info.get("reason", "未知错误")
                logger.error(f"[JieKou] Task {task_id} failed: {reason}")
                raise JiekouAPIError(
                    message=f"任务失败: {reason}",
                    code="TASK_FAILED"
                )
            
            time.sleep(poll_interval)
        
        raise JiekouAPIError(
            message="任务轮询超时",
            code="POLL_TIMEOUT"
        )
    
    # ===== Image Generation (GPT Image) =====
    def generate_image_gpt(
        self,
        prompt: str,
        model: str = "gpt-image-1",
        size: str = "1024x1024",
        quality: str = "auto",
        n: int = 1
    ) -> List[str]:
        """
        Generate image using GPT Image API (synchronous)
        
        Args:
            prompt: Text prompt
            model: Model ID (default: gpt-image-1)
            size: Image size (1024x1024, 1536x1024, 1024x1536)
            quality: Quality level (auto, high, medium, low)
            n: Number of images to generate (1-10)
        
        Returns:
            List of base64 encoded image strings
        """
        data = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": n
        }
        
        response = self._request(
            "POST",
            self.ENDPOINT_IMAGE_GPT,
            data=data,
            timeout=self.LONG_TIMEOUT
        )
        
        # Response format: { data: [{ b64_json: "..." }] }
        images = response.get("data", [])
        return [img.get("b64_json", "") for img in images]
    
    # ===== FLUX Kontext Image Generation =====
    def generate_image_flux(
        self,
        prompt: str,
        model: str = "flux-1-kontext-pro",
        images: List[str] = None,
        aspect_ratio: str = "1:1",
        seed: int = -1,
        guidance_scale: float = 3.5
    ) -> str:
        """
        Generate image using FLUX Kontext API (async, returns task_id)
        
        Args:
            prompt: Text prompt
            model: Model ID
            images: Optional list of input images (URL or base64)
            aspect_ratio: Aspect ratio (21:9, 16:9, 4:3, 3:2, 1:1, etc.)
            seed: Random seed (-1 for random)
            guidance_scale: Guidance scale (1.0-20.0)
        
        Returns:
            Task ID for polling
        """
        # FLUX uses a specific endpoint pattern
        endpoint = f"/v1/images/{model.replace('-', '-')}"
        
        data = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "seed": seed,
            "guidance_scale": guidance_scale
        }
        
        if images:
            data["images"] = images
        
        response = self._request(
            "POST",
            endpoint,
            data=data,
            timeout=self.LONG_TIMEOUT
        )
        
        return response.get("task_id")
    
    # ===== TTS (ElevenLabs) =====
    def generate_tts_elevenlabs(
        self,
        text: str,
        voice_id: str,
        model: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_128",
        voice_settings: dict = None
    ) -> bytes:
        """
        Generate speech using ElevenLabs TTS (synchronous, returns audio bytes)
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use
            model: TTS model
            output_format: Output audio format
            voice_settings: Optional voice settings (stability, similarity_boost, etc.)
        
        Returns:
            Audio data as bytes
        """
        # ElevenLabs uses a specific endpoint
        endpoint = f"/v1/audio/speech"
        
        data = {
            "text": text,
            "voice_id": voice_id,
            "output_format": output_format
        }
        
        if voice_settings:
            data["voice_settings"] = voice_settings
        
        # ElevenLabs returns binary audio, not JSON
        return self._request(
            "POST",
            endpoint,
            data=data,
            timeout=self.LONG_TIMEOUT,
            expect_json=False
        )
    
    # ===== Utility Methods =====
    def download_file(self, url: str, timeout: int = 120) -> bytes:
        """
        Download file from URL
        
        Args:
            url: File URL
            timeout: Download timeout in seconds
        
        Returns:
            File content as bytes
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            raise JiekouAPIError(
                message=f"下载文件失败: {str(e)}",
                code="DOWNLOAD_ERROR"
            )
    
    def get_video_result_url(self, task_response: dict) -> Optional[str]:
        """
        Extract video URL from task response
        
        Args:
            task_response: Response from get_task_status
        
        Returns:
            Video URL or None
        """
        videos = task_response.get("videos", [])
        if videos:
            return videos[0].get("video_url")
        return None
    
    def get_image_result_url(self, task_response: dict) -> Optional[str]:
        """
        Extract image URL from task response
        
        Args:
            task_response: Response from get_task_status
        
        Returns:
            Image URL or None
        """
        images = task_response.get("images", [])
        if images:
            return images[0].get("image_url")
        return None
    
    def get_audio_result_url(self, task_response: dict) -> Optional[str]:
        """
        Extract audio URL from task response
        
        Args:
            task_response: Response from get_task_status
        
        Returns:
            Audio URL or None
        """
        audios = task_response.get("audios", [])
        if audios:
            return audios[0].get("audio_url")
        return None
