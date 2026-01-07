"""
JiekouAI ComfyUI Plugin - Video Generation Nodes
Async video generation with polling and progress feedback

Uses JiekouAI Unified Video API:
- Endpoint: /v3/video/create
- Async task query: /v3/async/task-result?task_id=xxx
- Status: TASK_STATUS_QUEUED, TASK_STATUS_PROCESSING, TASK_STATUS_SUCCEED, TASK_STATUS_FAILED

Reference: https://docs.jiekou.ai/docs/models/reference-unified-video-generation.md
"""

import torch
import time
import logging
import os
import folder_paths

logger = logging.getLogger("[JieKou]")

# Try to import PromptServer for progress updates
try:
    from server import PromptServer
    HAS_PROMPT_SERVER = True
except ImportError:
    HAS_PROMPT_SERVER = False
    logger.warning("[JieKou] PromptServer not available - progress updates disabled")


class JieKouVideoGeneration:
    """
    Generate video from text prompts using JiekouAI video models
    
    Supports models: Wan, Sora, Veo, Kling, Minimax, Seedance, etc.
    Uses async task submission with polling for progress updates
    Outputs frame sequence as IMAGE tensor
    
    Note: Different models support different parameters.
    Only parameters supported by the selected model will be sent to the API.
    """
    
    CATEGORY = "JieKou/Video"
    FUNCTION = "generate_video"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("frames", "video_url",)
    OUTPUT_NODE = True  # 允许直接运行
    
    # Polling configuration
    POLL_INTERVAL = 3  # seconds
    MAX_POLL_TIME = 1800  # 30 minutes max for video generation
    
    @classmethod
    def INPUT_TYPES(cls):
        # Import model registry for available models
        try:
            from ..utils.model_registry import get_model_registry
            registry = get_model_registry()
            t2v_models = registry.get_video_t2v_models()
            i2v_models = registry.get_video_i2v_models()
            v2v_models = registry.get_video_v2v_models()
            all_model_configs = t2v_models + i2v_models + v2v_models
            
            # Include both IDs and names for validation (frontend displays names)
            all_models = []
            for m in all_model_configs:
                all_models.append(m.id)
                if m.name and m.name != m.id:
                    all_models.append(m.name)
            
            default_model = all_model_configs[0].id if all_model_configs else "wan2.2_t2v"
        except Exception as e:
            logger.warning(f"[JieKou] Failed to load model registry: {e}")
            all_models = ["wan2.2_t2v", "sora2_t2v", "veo3_t2v", "kling2.5_turbo_pro_t2v", "wan2.6_t2v", "wan2.6_i2v", "wan2.6_v2v"]
            default_model = "wan2.2_t2v"
        
        # Get output directory for display
        output_dir = folder_paths.get_output_directory()
        
        # Only define required inputs
        # Optional parameters will be added dynamically by JavaScript based on selected model
        return {
            "required": {
                "model": (all_models, {
                    "default": default_model
                }),
                # Image input for I2V models: URL or base64 (positioned after model)
                "image_url": ("STRING", {
                    "default": "",
                    "placeholder": "图生视频：输入图片 URL 或 base64（文生视频留空）",
                    "dynamicPrompts": True  # 允许右键转换为输入槽
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "描述您想要生成的视频...",
                    "dynamicPrompts": True  # 允许右键转换为输入槽
                }),
                "save_to_disk": ("BOOLEAN", {
                    "default": True,
                    "label_on": f"保存到 {output_dir}",
                    "label_off": "不保存到本地"
                }),
            },
            "optional": {
                # Reference video URLs for Wan 2.6 V2V model
                "reference_video_urls": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "参考视频URL (1-3条，每行一个)",
                    "dynamicPrompts": True  # 允许右键转换为输入槽
                }),
                # Hidden input for dynamic parameters (JSON string)
                # JavaScript will serialize dynamic widget values here
                "_dynamic_params": ("STRING", {
                    "default": "{}",
                    "multiline": False,
                }),
            }
        }
    
    def generate_video(
        self,
        model: str,
        image_url: str,
        prompt: str,
        save_to_disk: bool = True,
        reference_video_urls: str = "",
        _dynamic_params: str = "{}",
        **kwargs
    ):
        """
        Generate video and return frame sequence
        
        Args:
            model: Model ID to use
            image_url: Input image for I2V models (URL or base64, leave empty for T2V)
            prompt: Text prompt
            save_to_disk: Whether to save video to output folder
            reference_video_urls: Optional reference video URLs for Wan 2.6 V2V (1-3 URLs, one per line)
            _dynamic_params: JSON string of dynamic parameters from JavaScript
            **kwargs: Model-specific parameters (dynamically provided by JavaScript, including end_image as URL string)
        
        Returns:
            tuple: (IMAGE tensor with shape [B, H, W, C] where B = frame count, video_url string)
        """
        import json
        
        # Parse dynamic parameters from JSON
        try:
            dynamic_params = json.loads(_dynamic_params) if _dynamic_params else {}
        except json.JSONDecodeError:
            dynamic_params = {}
        
        # Merge dynamic params into kwargs
        kwargs.update(dynamic_params)
        
        logger.info(f"[JieKou] ========== Video Generation ==========")
        logger.info(f"[JieKou] Model: {model}")
        logger.info(f"[JieKou] Image URL: {image_url[:100] if image_url else 'None (T2V mode)'}...")
        logger.info(f"[JieKou] Prompt: {prompt[:50]}...")
        logger.info(f"[JieKou] Save to disk: {save_to_disk}")
        logger.info(f"[JieKou] Reference video URLs: {reference_video_urls[:100] if reference_video_urls else 'None'}...")
        logger.info(f"[JieKou] Dynamic params JSON: {_dynamic_params[:200]}...")
        logger.info(f"[JieKou] Parsed params: {kwargs}")
        
        if not prompt.strip():
            raise ValueError("Prompt 不能为空")
        
        try:
            from ..utils.api_client import JiekouAPI, JiekouAPIError, TaskStatus
            from ..utils.tensor_utils import tensor_to_base64, video_to_frames
            from ..utils.model_registry import get_model_registry
            
            api = JiekouAPI()
            registry = get_model_registry()
            
            # Resolve model by ID or name (frontend may pass display name)
            model_config = registry.resolve_model(model)
            if model_config:
                model_id = model_config.id
                if model_id != model:
                    logger.info(f"[JieKou] Resolved model: {model} -> {model_id}")
            else:
                model_id = model
                logger.warning(f"[JieKou] Model not found: {model}, using as-is")
            
            # Submit async task
            logger.info("[JieKou] Submitting video generation task...")
            self._send_progress(0, "提交任务中...")
            
            # Check if this is a Wan 2.6 model (uses dedicated endpoint)
            if registry.is_wan26_model(model_id):
                task_id = self._submit_wan26_task(api, registry, model_id, prompt, image_url, reference_video_urls, kwargs)
            else:
                task_id = self._submit_unified_task(api, registry, model_id, prompt, image_url, kwargs)
            
            logger.info(f"[JieKou] Task submitted: {task_id}")
            
            # Poll for completion
            task_result = self._poll_task_status(api, task_id)
            
            # Extract video URL from response
            video_url = api.get_video_result_url(task_result)
            if not video_url:
                raise RuntimeError("任务完成但未返回视频 URL")
            
            # Download and decode video
            logger.info(f"[JieKou] Downloading video from {video_url[:50]}...")
            self._send_progress(92, "下载视频中...")
            
            video_bytes = api.download_file(video_url)
            
            # Save video file to output directory if requested
            video_path = ""
            filename = ""
            if save_to_disk:
                logger.info("[JieKou] Saving video file...")
                self._send_progress(95, "保存视频文件...")
                
                output_dir = folder_paths.get_output_directory()
                filename = f"jiekou_video_{int(time.time())}.mp4"
                video_path = os.path.join(output_dir, filename)
                
                with open(video_path, "wb") as f:
                    f.write(video_bytes)
                
                logger.info(f"[JieKou] Video saved to: {video_path}")
            
            # Decode video to frames for ComfyUI pipeline
            logger.info("[JieKou] Decoding video to frames...")
            self._send_progress(98, "解码帧序列中...")
            
            frames_tensor = video_to_frames(video_bytes)
            
            self._send_progress(100, "完成!", status="complete")
            
            logger.info(f"[JieKou] Video generated: {frames_tensor.shape[0]} frames")
            
            # Build UI info
            ui_info = {}
            if save_to_disk:
                ui_info["text"] = [f"✅ 视频已保存: {filename}"]
                ui_info["video_info"] = [{
                    "filename": filename,
                    "subfolder": "",
                    "type": "output"
                }]
            
            # Always return the original video URL from API
            return {
                "ui": ui_info,
                "result": (frames_tensor, video_url,)
            }
        
        except JiekouAPIError as e:
            self._send_progress(0, f"错误: {e.message}", status="error")
            error_msg = self._format_error(e)
            logger.error(f"[JieKou] API Error: {error_msg}")
            raise RuntimeError(error_msg)
        
        except Exception as e:
            self._send_progress(0, f"错误: {str(e)}", status="error")
            logger.error(f"[JieKou] Error: {e}")
            raise
    
    def _submit_unified_task(self, api, registry, model: str, prompt: str, image_url: str, kwargs) -> str:
        """Submit task using unified API (/v3/video/create)"""
        from ..utils.tensor_utils import is_local_file_path, local_file_to_base64
        
        # Get supported parameters for this model
        supported_params = set(registry.get_video_model_param_names(model))
        logger.info(f"[JieKou] Model {model} supports params: {supported_params}")
        
        # Build parameters dict
        params = {"prompt": prompt}
        
        # Add all dynamic parameters from kwargs (only if supported)
        for key, value in kwargs.items():
            if key == "extra_params":
                continue  # Skip hidden param marker
            if value is not None and key in supported_params:
                params[key] = value
        
        # Handle image input (only if model supports it)
        if image_url and image_url.strip():
            if "image" in supported_params:
                input_image_data = image_url.strip()
                # Check if it's a local file path and convert to base64
                if is_local_file_path(input_image_data):
                    logger.info(f"[JieKou] Detected local file path, converting to base64...")
                    input_image_data = local_file_to_base64(input_image_data)
                logger.info(f"[JieKou] Using image_url: {input_image_data[:50]}...")
                params["image"] = input_image_data
            else:
                logger.warning(f"[JieKou] Model {model} does not support image input, ignoring...")
        
        # Log final params (truncate long strings)
        log_params = {}
        for k, v in params.items():
            if isinstance(v, str) and len(v) > 100:
                log_params[k] = v[:50] + f"... ({len(v)} chars)"
            else:
                log_params[k] = v
        logger.info(f"[JieKou] Final request params: {log_params}")
        
        return api.submit_video_task(model=model, **params)
    
    def _submit_wan26_task(self, api, registry, model: str, prompt: str, image_url: str, reference_video_urls: str, kwargs) -> str:
        """
        Submit task using Wan 2.6 dedicated endpoint
        
        Wan 2.6 models use different payload structure:
        { input: {prompt, img_url, ...}, parameters: {seed, size, ...} }
        """
        from ..utils.tensor_utils import is_local_file_path, local_file_to_base64
        
        model_config = registry.get_model(model)
        endpoint = model_config.endpoint
        
        # Get param structure for this Wan 2.6 model
        param_structure = registry.get_wan26_model_params(model)
        input_param_names = set(param_structure.get("input", []))
        parameters_param_names = set(param_structure.get("parameters", []))
        
        logger.info(f"[JieKou] Wan 2.6 model {model}, endpoint: {endpoint}")
        logger.info(f"[JieKou] Input params: {input_param_names}")
        logger.info(f"[JieKou] Parameters params: {parameters_param_names}")
        
        # Build input dict
        input_params = {}
        if "prompt" in input_param_names:
            input_params["prompt"] = prompt
        
        # Build parameters dict
        parameters = {}
        
        # Add kwargs to appropriate dict
        for key, value in kwargs.items():
            if key == "extra_params":
                continue
            if value is None:
                continue
            
            # Handle duration_v2v -> duration mapping
            actual_key = key if key != "duration_v2v" else "duration"
            
            if key in input_param_names or actual_key in input_param_names:
                input_params[actual_key] = value
            elif key in parameters_param_names or actual_key in parameters_param_names:
                parameters[actual_key] = value
        
        # Handle image input (Wan 2.6 uses img_url in input)
        if image_url and image_url.strip():
            if "img_url" in input_param_names:
                input_image_data = image_url.strip()
                # Check if it's a local file path and convert to base64
                if is_local_file_path(input_image_data):
                    logger.info(f"[JieKou] Detected local file path, converting to base64...")
                    input_image_data = local_file_to_base64(input_image_data)
                logger.info(f"[JieKou] Using image_url for Wan 2.6 (img_url): {input_image_data[:50]}...")
                input_params["img_url"] = input_image_data
            else:
                logger.warning(f"[JieKou] Model {model} does not support img_url input, ignoring...")
        
        # Handle reference_video_urls for V2V model
        if reference_video_urls and reference_video_urls.strip():
            if "reference_video_urls" in input_param_names:
                # Parse URLs (one per line) into array
                urls = [url.strip() for url in reference_video_urls.strip().split('\n') if url.strip()]
                if urls:
                    input_params["reference_video_urls"] = urls
                    logger.info(f"[JieKou] Added {len(urls)} reference video URLs")
        
        logger.info(f"[JieKou] Wan 2.6 input: {list(input_params.keys())}")
        logger.info(f"[JieKou] Wan 2.6 parameters: {list(parameters.keys())}")
        
        return api.submit_wan26_video_task(
            model=model,
            endpoint=endpoint,
            input_params=input_params,
            parameters=parameters
        )
    
    def _poll_task_status(self, api, task_id: str) -> dict:
        """
        Poll task status until completion
        
        Returns:
            dict: Full task response with results
        
        Raises:
            RuntimeError: If task fails or times out
        """
        from ..utils.api_client import TaskStatus
        
        start_time = time.time()
        
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.MAX_POLL_TIME:
                raise RuntimeError(f"视频生成超时 ({self.MAX_POLL_TIME}秒)")
            
            # Get status
            response = api.get_task_status(task_id)
            task_info = response.get("task", {})
            
            status = task_info.get("status", "")
            progress = task_info.get("progress_percent", 0)
            
            logger.info(f"[JieKou] Task status: {status}, progress: {progress}%")
            
            # Send progress update based on status
            if status == TaskStatus.PROCESSING:
                # Scale progress to 5-90 range
                scaled_progress = 5 + int(progress * 0.85)
                self._send_progress(scaled_progress, f"生成中... {progress}%")
            elif status == TaskStatus.QUEUED:
                self._send_progress(5, "排队中，等待开始...")
            
            # Check completion
            if status == TaskStatus.SUCCEED:
                return response
            
            elif status == TaskStatus.FAILED:
                reason = task_info.get("reason", "未知错误")
                raise RuntimeError(f"视频生成失败: {reason}")
            
            # Wait before next poll
            time.sleep(self.POLL_INTERVAL)
    
    def _send_progress(self, progress: int, message: str, status: str = "processing"):
        """Send progress update to frontend via PromptServer"""
        if not HAS_PROMPT_SERVER:
            return
        
        try:
            PromptServer.instance.send_sync(
                "jiekou-progress",
                {
                    "node_id": id(self),
                    "progress": progress,
                    "message": message,
                    "status": status
                }
            )
        except Exception as e:
            logger.warning(f"[JieKou] Failed to send progress: {e}")
    
    def _format_error(self, error) -> str:
        """Format API error for user display"""
        error_messages = {
            "UNAUTHORIZED": "API Key 无效，请检查设置",
            "INSUFFICIENT_CREDITS": "额度不足，请充值后重试",
            "RATE_LIMIT": "请求过于频繁，请稍后重试",
            "INVALID_MODEL": "模型不存在或暂时不可用",
            "INVALID_PROMPT": "Prompt 内容不合规",
            "TIMEOUT": "请求超时，请重试",
            "TASK_FAILED": "视频生成失败，请调整参数重试",
            "POLL_TIMEOUT": "任务轮询超时",
        }
        
        return error_messages.get(error.code, error.message)
