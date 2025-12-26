"""
JiekouAI ComfyUI Plugin - Model Registry
Hardcoded model configurations based on JiekouAI API documentation

Since JiekouAI doesn't provide a unified /models endpoint,
we maintain the model list and their parameters here.

Reference: https://docs.jiekou.ai/llms.txt
"""

from __future__ import annotations
from typing import Optional, Any, List
from dataclasses import dataclass, field


@dataclass
class ModelParameter:
    """Definition of a model parameter"""
    name: str
    type: str  # string, integer, number, boolean, array
    description: str
    required: bool = False
    default: Any = None
    enum: List = None
    minimum: float = None
    maximum: float = None


@dataclass
class ModelConfig:
    """Configuration for a single model"""
    id: str
    name: str
    description: str
    category: str  # image_t2i, image_i2i, image_edit, image_tool, video_t2v, video_i2v, audio_tts
    endpoint: str = ""  # API endpoint path (empty for unified API models like video)
    parameters: List[ModelParameter] = field(default_factory=list)
    is_async: bool = True  # Whether the API is async (requires polling)
    response_type: str = "task_id"  # task_id, b64_json, image_urls, binary
    doc_url: str = ""  # API documentation URL (without .md suffix)


# ===== Image Generation Models (Text-to-Image) =====
IMAGE_T2I_MODELS = [
    # GPT Image - Sync, returns b64_json
    ModelConfig(
        id="gpt-image-1",
        name="GPT 文生图",
        description="OpenAI GPT 图像生成",
        category="image_t2i",
        endpoint="/v1/images/generations",
        is_async=False,
        response_type="b64_json",
        doc_url="https://docs.jiekou.ai/docs/models/reference-gpt-image-generations",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
            ModelParameter("size", "string", "图片尺寸 (如 1024x1024, 1536x1024, auto)", default="1024x1024"),
            ModelParameter("quality", "string", "质量", default="auto",
                          enum=["auto", "high", "medium", "low"]),
            ModelParameter("n", "integer", "生成数量", default=1, minimum=1, maximum=10),
        ]
    ),
    # FLUX Kontext Pro - Async
    ModelConfig(
        id="flux-1-kontext-pro",
        name="FLUX.1 Kontext Pro",
        description="FLUX.1 Kontext Pro 图像生成，支持多图输入",
        category="image_t2i",
        endpoint="/v3/async/flux-1-kontext-pro",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-flux-1-kontext-pro",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
            ModelParameter("images", "array", "输入图片列表(最多4张)"),
            ModelParameter("aspect_ratio", "string", "宽高比", default="1:1",
                          enum=["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"]),
            ModelParameter("guidance_scale", "number", "引导系数", default=3.5, minimum=1.0, maximum=20.0),
            ModelParameter("seed", "integer", "随机种子", default=-1, minimum=-1, maximum=2147483647),
            ModelParameter("safety_tolerance", "string", "安全容忍度", default="2", enum=["1", "2", "3", "4", "5"]),
        ]
    ),
    # FLUX Kontext Dev - Async
    ModelConfig(
        id="flux-1-kontext-dev",
        name="FLUX.1 Kontext Dev",
        description="FLUX.1 Kontext Dev 图像生成，高效速度适合编辑需求",
        category="image_t2i",
        endpoint="/v3/async/flux-1-kontext-dev",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-flux-1-kontext-dev",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
            ModelParameter("images", "array", "输入图片列表(最多4张)"),
            ModelParameter("size", "string", "尺寸(宽*高)，每维度256-1536"),
            ModelParameter("fast_mode", "boolean", "极速模式(更快但质量略低)", default=False),
            ModelParameter("num_inference_steps", "integer", "推理步数", default=28, minimum=1, maximum=50),
            ModelParameter("guidance_scale", "number", "引导系数", default=2.5, minimum=1.0, maximum=20.0),
            ModelParameter("num_images", "integer", "生成数量", default=1, minimum=1, maximum=4),
            ModelParameter("seed", "integer", "随机种子", default=-1, minimum=-1, maximum=2147483647),
            ModelParameter("output_format", "string", "输出格式", default="jpeg",
                          enum=["jpeg", "png", "webp"]),
        ]
    ),
    # FLUX Kontext Max - Async
    ModelConfig(
        id="flux-1-kontext-max",
        name="FLUX.1 Kontext Max",
        description="FLUX.1 Kontext Max 图像生成，最高质量",
        category="image_t2i",
        endpoint="/v3/async/flux-1-kontext-max",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-flux-1-kontext-max",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
            ModelParameter("images", "array", "输入图片列表(最多4张)"),
            ModelParameter("aspect_ratio", "string", "宽高比", default="1:1",
                          enum=["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"]),
            ModelParameter("guidance_scale", "number", "引导系数", default=3.5, minimum=1.0, maximum=20.0),
            ModelParameter("seed", "integer", "随机种子", default=-1, minimum=-1, maximum=2147483647),
            ModelParameter("safety_tolerance", "string", "安全容忍度", default="2", enum=["1", "2", "3", "4", "5"]),
        ]
    ),
    # Flux 2 Pro - Async
    ModelConfig(
        id="flux-2-pro",
        name="Flux 2 Pro 生图",
        description="Flux 2 Pro 图像生成",
        category="image_t2i",
        endpoint="/v3/async/flux-2-pro",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-flux-2-pro",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
            ModelParameter("images", "array", "输入图片列表(最多3张)"),
            ModelParameter("size", "string", "尺寸(宽*高)", default="1024*1024"),
            ModelParameter("seed", "integer", "随机种子", default=-1, minimum=-1, maximum=2147483647),
        ]
    ),
    # Flux 2 Dev - Async
    ModelConfig(
        id="flux-2-dev",
        name="Flux 2 Dev 生图",
        description="Flux 2 Dev 图像生成",
        category="image_t2i",
        endpoint="/v1/images/flux-2-dev",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-flux-2-dev",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
            ModelParameter("images", "array", "输入图片列表(最多3张)"),
            ModelParameter("size", "string", "尺寸(宽*高)", default="1024*1024"),
            ModelParameter("seed", "integer", "随机种子", default=-1, minimum=-1, maximum=2147483647),
        ]
    ),
    # Flux 2 Flex - Async
    ModelConfig(
        id="flux-2-flex",
        name="Flux 2 Flex 生图",
        description="Flux 2 Flex 图像生成",
        category="image_t2i",
        endpoint="/v1/images/flux-2-flex",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-flux-2-flex",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
            ModelParameter("images", "array", "输入图片列表(最多3张)"),
            ModelParameter("size", "string", "尺寸(宽*高)", default="1024*1024"),
            ModelParameter("seed", "integer", "随机种子", default=-1, minimum=-1, maximum=2147483647),
        ]
    ),
    # Seedream 3.0 - Sync, returns image_urls
    ModelConfig(
        id="seedream-3-0-t2i-250415",
        name="Seedream 文生图 3.0",
        description="Seedream 3.0 文生图模型",
        category="image_t2i",
        endpoint="/v1/images/seedream-3.0",
        is_async=False,
        response_type="image_urls",
        doc_url="https://docs.jiekou.ai/docs/models/reference-seedream3.0-text-to-image",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
            ModelParameter("size", "string", "尺寸(宽x高)", default="1024x1024"),
            ModelParameter("seed", "integer", "随机种子", default=-1, minimum=-1, maximum=2147483647),
            ModelParameter("guidance_scale", "number", "引导系数", default=2.5, minimum=1, maximum=10),
            ModelParameter("watermark", "boolean", "添加水印", default=True),
        ]
    ),
    # Seedream 4.0 - Sync, returns images array
    ModelConfig(
        id="seedream-4-0",
        name="Seedream 图片生成 4.0",
        description="Seedream 4.0 图像生成，支持4K分辨率",
        category="image_t2i",
        endpoint="/v3/seedream-4.0",
        is_async=False,
        response_type="image_urls",
        doc_url="https://docs.jiekou.ai/docs/models/reference-seedream4.0",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
            ModelParameter("images", "array", "参考图片列表(最多10张)"),
            ModelParameter("size", "string", "尺寸 (1K/2K/4K 或 宽x高如2048x2048)", default="2048x2048"),
            ModelParameter("watermark", "boolean", "添加水印", default=True),
        ]
    ),
    # Seedream 4.5 - Sync
    ModelConfig(
        id="seedream-4-5",
        name="Seedream 图片生成 4.5",
        description="Seedream 4.5 图像生成，支持单图或组图生成，最多14张参考图",
        category="image_t2i",
        endpoint="/v3/seedream-4.5",
        is_async=False,
        response_type="image_urls",
        doc_url="https://docs.jiekou.ai/docs/models/reference-seedream-4.5",
        parameters=[
            ModelParameter("prompt", "string", "提示词(建议不超过300汉字或600英文单词)", required=True),
            ModelParameter("image", "array", "参考图片列表(URL或Base64，最多14张)"),
            ModelParameter("size", "string", "尺寸 (2K/4K 或 宽x高如2048x2048)", default="2048x2048"),
            ModelParameter("watermark", "boolean", "添加水印", default=True),
        ]
    ),
    # Qwen Image - Async
    ModelConfig(
        id="qwen-image-t2i",
        name="Qwen-Image 文生图",
        description="Qwen 20B MMDiT 文生图，擅长创建带文字的海报",
        category="image_t2i",
        endpoint="/v1/images/qwen-image",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-qwen-text-to-image",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
            ModelParameter("size", "string", "尺寸(宽*高)", default="1024*1024"),
        ]
    ),
    # Gemini 2.5 Flash Image - Sync
    ModelConfig(
        id="gemini-2.5-flash-image-t2i",
        name="Gemini 2.5 Flash Image 文生图",
        description="Google Gemini 2.5 Flash 文生图",
        category="image_t2i",
        endpoint="/v1/images/gemini-2.5-flash",
        is_async=False,
        response_type="image_urls",
        doc_url="https://docs.jiekou.ai/docs/models/reference-gemini-2.5-text-to-image",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
        ]
    ),
    # Gemini 3 Pro Image Preview - Sync
    ModelConfig(
        id="gemini-3-pro-image-preview-t2i",
        name="Gemini 3 Pro Image Preview 文生图",
        description="Google Gemini 3 Pro Image Preview 文生图",
        category="image_t2i",
        endpoint="/v1/images/gemini-3-pro-preview",
        is_async=False,
        response_type="image_urls",
        doc_url="https://docs.jiekou.ai/docs/models/reference-gemini-3-text-to-image",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
        ]
    ),
    # Hunyuan Image 3 - Async
    ModelConfig(
        id="hunyuan-image-3",
        name="Hunyuan Image 3",
        description="腾讯混元 Image 3 文生图",
        category="image_t2i",
        endpoint="/v1/images/hunyuan-image-3",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-hunyuan-image-3",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
            ModelParameter("size", "string", "尺寸(宽*高)", default="1024*1024"),
            ModelParameter("seed", "integer", "随机种子", default=-1, minimum=-1, maximum=2147483647),
        ]
    ),
    # Midjourney - Async
    ModelConfig(
        id="midjourney-txt2img",
        name="Midjourney 文生图",
        description="Midjourney 文生图",
        category="image_t2i",
        endpoint="/v1/images/midjourney/txt2img",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-mj-txt2img",
        parameters=[
            ModelParameter("text", "string", "提示词", required=True),
        ]
    ),
    # Z Image Turbo - Async
    ModelConfig(
        id="z-image-turbo",
        name="Z Image 文生图 Turbo",
        description="Z Image Turbo 快速文生图",
        category="image_t2i",
        endpoint="/v1/images/z-image-turbo",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-z-image-turbo",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
            ModelParameter("size", "string", "尺寸(宽*高)", default="1024*1024"),
            ModelParameter("seed", "integer", "随机种子", default=-1, minimum=-1, maximum=2147483647),
            ModelParameter("enable_base64_output", "boolean", "输出base64", default=False),
        ]
    ),
    # Z Image Turbo LoRA - Async
    ModelConfig(
        id="z-image-turbo-lora",
        name="Z Image 文生图 Turbo LoRA",
        description="Z Image Turbo LoRA 文生图",
        category="image_t2i",
        endpoint="/v1/images/z-image-turbo-lora",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-z-image-turbo-lora",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
            ModelParameter("size", "string", "尺寸(宽*高)", default="1024*1024"),
            ModelParameter("seed", "integer", "随机种子", default=-1, minimum=-1, maximum=2147483647),
        ]
    ),
    # Nano Banana Pro Light T2I - Async
    ModelConfig(
        id="nano-banana-pro-light-t2i",
        name="Nano Banana Pro Light 文生图 (reverse)",
        description="Nano Banana Pro Light 文生图 (reverse)",
        category="image_t2i",
        endpoint="/v1/images/nano-banana-pro-light-t2i",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-nano-banana-pro-light-t2i",
        parameters=[
            ModelParameter("prompt", "string", "提示词", required=True),
        ]
    ),
]

# ===== Image Editing Models =====
IMAGE_EDIT_MODELS = [
    # GPT Image Edit - Sync, returns b64_json
    ModelConfig(
        id="gpt-image-1-edit",
        name="GPT 图像编辑",
        description="GPT 图像编辑，通过文本描述修改图像，支持最多16张图片",
        category="image_edit",
        endpoint="/v1/images/edits",
        is_async=False,
        response_type="b64_json",
        doc_url="https://docs.jiekou.ai/docs/models/reference-gpt-image-edits",
        parameters=[
            ModelParameter("prompt", "string", "编辑描述，最大32000字符", required=True),
            ModelParameter("image", "array", "待编辑的图像(最多16张)", required=True),
            ModelParameter("quality", "string", "质量", default="auto",
                          enum=["auto", "high", "medium", "low"]),
        ]
    ),
    # Qwen Image Edit - Async
    ModelConfig(
        id="qwen-image-edit",
        name="Qwen-Image 图像编辑",
        description="Qwen 20B 图像编辑，支持中英文双语文本编辑，保留风格",
        category="image_edit",
        endpoint="/v3/async/qwen-image-edit",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-qwen-image-edit",
        parameters=[
            ModelParameter("prompt", "string", "编辑描述", required=True),
            ModelParameter("image", "string", "输入图像(URL或base64)", required=True),
            ModelParameter("seed", "integer", "随机种子", default=-1, minimum=-1, maximum=2147483647),
            ModelParameter("output_format", "string", "输出格式", default="jpeg",
                          enum=["jpeg", "png", "webp"]),
        ]
    ),
    # Gemini 2.5 Flash Image Edit - Sync
    ModelConfig(
        id="gemini-2.5-flash-image-edit",
        name="Gemini 2.5 Flash Image 图片编辑",
        description="Google Gemini 2.5 Flash 图片编辑，使用自然语言修改图像",
        category="image_edit",
        endpoint="/v1/images/gemini-2.5-flash-edit",
        is_async=False,
        response_type="image_urls",
        doc_url="https://docs.jiekou.ai/docs/models/reference-gemini-2.5-image-edit",
        parameters=[
            ModelParameter("prompt", "string", "编辑描述", required=True),
            ModelParameter("image_urls", "array", "输入图像URL列表"),
            ModelParameter("image_base64s", "array", "输入图像base64列表"),
        ]
    ),
    # Gemini 3 Pro Image Edit - Sync
    ModelConfig(
        id="gemini-3-pro-image-preview-edit",
        name="Gemini 3 Pro Image Preview 图片编辑",
        description="Google Gemini 3 Pro 图片编辑，支持复杂多轮编辑",
        category="image_edit",
        endpoint="/v1/images/gemini-3-pro-preview-edit",
        is_async=False,
        response_type="image_urls",
        doc_url="https://docs.jiekou.ai/docs/models/reference-gemini-3.0-image-edit",
        parameters=[
            ModelParameter("prompt", "string", "编辑描述", required=True),
            ModelParameter("image_urls", "array", "输入图像URL列表"),
            ModelParameter("image_base64s", "array", "输入图像base64列表"),
            ModelParameter("aspect_ratio", "string", "宽高比",
                          enum=["1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]),
            ModelParameter("size", "string", "输出尺寸 (1K/2K/4K)", default="1K"),
        ]
    ),
    # Nano Banana Pro Light I2I - Sync, returns data array
    ModelConfig(
        id="nano-banana-pro-light-i2i",
        name="Nano Banana Pro Light 图生图 (reverse)",
        description="根据输入图像和文本描述生成新图像",
        category="image_edit",
        endpoint="/v3/nano-banana-pro-light-i2i",
        is_async=False,
        response_type="b64_json",
        doc_url="https://docs.jiekou.ai/docs/models/reference-nano-banana-pro-light-i2i",
        parameters=[
            ModelParameter("prompt", "string", "文本描述", required=True),
            ModelParameter("images", "array", "输入图像列表(1-10张)", required=True),
            ModelParameter("n", "integer", "生成数量", default=1, minimum=1, maximum=10),
            ModelParameter("size", "string", "尺寸比例 (如 1x1, 16x9)", default="1x1"),
            ModelParameter("quality", "string", "图像质量", default="1k",
                          enum=["1k", "2k", "4k"]),
            ModelParameter("mask", "string", "遮罩图像(URL或base64)"),
            ModelParameter("response_format", "string", "返回格式", default="url",
                          enum=["url", "b64_json"]),
        ]
    ),
]

# ===== Midjourney Advanced Operations =====
MIDJOURNEY_MODELS = [
    ModelConfig(
        id="midjourney-variation",
        name="Midjourney 变化",
        description="对 Midjourney 生成的图像进行变体",
        category="image_edit",
        endpoint="/v1/images/midjourney/variation",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-mj-variation",
        parameters=[
            ModelParameter("task_id", "string", "原始任务ID", required=True),
            ModelParameter("index", "integer", "变体索引(1-4)", required=True, minimum=1, maximum=4),
        ]
    ),
    ModelConfig(
        id="midjourney-upscale",
        name="Midjourney 高清",
        description="对 Midjourney 生成的图像进行高清放大",
        category="image_edit",
        endpoint="/v1/images/midjourney/upscale",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-mj-upscale",
        parameters=[
            ModelParameter("task_id", "string", "原始任务ID", required=True),
            ModelParameter("index", "integer", "放大索引(1-4)", required=True, minimum=1, maximum=4),
        ]
    ),
    ModelConfig(
        id="midjourney-reroll",
        name="Midjourney 重新执行",
        description="重新执行 Midjourney 任务",
        category="image_edit",
        endpoint="/v1/images/midjourney/reroll",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-mj-reroll",
        parameters=[
            ModelParameter("task_id", "string", "原始任务ID", required=True),
        ]
    ),
    ModelConfig(
        id="midjourney-outpaint",
        name="Midjourney 扩图",
        description="Midjourney 图像扩展",
        category="image_edit",
        endpoint="/v1/images/midjourney/outpaint",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-mj-outpaint",
        parameters=[
            ModelParameter("task_id", "string", "原始任务ID", required=True),
            ModelParameter("direction", "string", "扩展方向", required=True, enum=["up", "down", "left", "right"]),
        ]
    ),
    ModelConfig(
        id="midjourney-inpaint",
        name="Midjourney 区域重绘",
        description="Midjourney 区域重绘",
        category="image_edit",
        endpoint="/v1/images/midjourney/inpaint",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-mj-inpaint",
        parameters=[
            ModelParameter("task_id", "string", "原始任务ID", required=True),
            ModelParameter("mask", "string", "遮罩图像", required=True),
            ModelParameter("prompt", "string", "重绘提示词", required=True),
        ]
    ),
    ModelConfig(
        id="midjourney-remix",
        name="Midjourney 重塑",
        description="Midjourney 图像重塑",
        category="image_edit",
        endpoint="/v1/images/midjourney/remix",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-mj-remix",
        parameters=[
            ModelParameter("task_id", "string", "原始任务ID", required=True),
            ModelParameter("prompt", "string", "新提示词", required=True),
        ]
    ),
    ModelConfig(
        id="midjourney-remove-background",
        name="Midjourney 移除背景",
        description="Midjourney 背景移除",
        category="image_tool",
        endpoint="/v1/images/midjourney/remove-background",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-mj-remove-background",
        parameters=[
            ModelParameter("task_id", "string", "原始任务ID", required=True),
        ]
    ),
]

# ===== Image Tool Models =====
IMAGE_TOOL_MODELS = [
    # Image Upscaler - Async
    ModelConfig(
        id="image-upscaler",
        name="图像高清化",
        description="将图像高清放大至2K/4K/8K",
        category="image_tool",
        endpoint="/v1/images/upscaler",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-image-upscaler",
        parameters=[
            ModelParameter("image", "string", "输入图像URL", required=True),
            ModelParameter("resolution", "string", "目标分辨率", default="4k", enum=["2k", "4k", "8k"]),
            ModelParameter("output_format", "string", "输出格式", default="jpeg", enum=["jpeg", "png", "webp"]),
        ]
    ),
    # Image Background Remove - Async
    ModelConfig(
        id="image-remove-background",
        name="图像背景移除",
        description="自动移除图像背景",
        category="image_tool",
        endpoint="/v1/images/remove-background",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-image-remove-background",
        parameters=[
            ModelParameter("image", "string", "输入图像URL", required=True),
        ]
    ),
    # Image Eraser - Async
    ModelConfig(
        id="image-eraser",
        name="图像擦除",
        description="擦除图像中的指定区域",
        category="image_tool",
        endpoint="/v1/images/eraser",
        is_async=True,
        response_type="task_id",
        doc_url="https://docs.jiekou.ai/docs/models/reference-image-eraser",
        parameters=[
            ModelParameter("image", "string", "输入图像URL", required=True),
            ModelParameter("mask", "string", "遮罩图像", required=True),
        ]
    ),
]

# ===== Video Generation Models =====
# NOTE: Most video models use the unified API via api_client.submit_video_task()
# Actual API endpoint: /v3/video/create
# Model IDs and parameters are synced from jiekou-docs/video_models_config.json
#
# EXCEPTION: Wan 2.6 models use dedicated endpoints with different payload structure:
# - /v3/async/wan2.6-t2v, /v3/async/wan2.6-i2v, /v3/async/wan2.6-v2v
# - Payload: { input: {...}, parameters: {...} } instead of flat structure
# - Use api_client.submit_wan26_video_task() for these models

# Wan 2.6 dedicated models that don't use unified API
WAN26_MODELS = {"wan2.6_t2v", "wan2.6_i2v", "wan2.6_v2v"}

# Wan 2.6 parameter definitions (separate from unified API models)
WAN26_PARAMS = {
    # Common input params
    "prompt": ModelParameter("prompt", "string", "提示词 (0-2000字符)", required=True),
    "negative_prompt": ModelParameter("negative_prompt", "string", "反向提示词 (0-500字符)"),
    "audio_url": ModelParameter("audio_url", "string", "音频文件URL (wav/mp3, 3-30秒)"),
    # I2V specific
    "img_url": ModelParameter("img_url", "string", "输入图片 (URL 或 Base64)", required=True),
    "template": ModelParameter("template", "string", "视频特效模板名称"),
    # V2V specific
    "reference_video_urls": ModelParameter("reference_video_urls", "array", "参考视频URL (1-3条, mp4/mov, 2-30秒)"),
    # Common parameters
    "seed": ModelParameter("seed", "integer", "随机种子", default=0, minimum=0, maximum=2147483647),
    "size": ModelParameter("size", "string", "视频分辨率 (如 1920*1080, 1280*720)", default="1920*1080"),
    "resolution": ModelParameter("resolution", "string", "分辨率档位", default="1080P", enum=["720P", "1080P"]),
    "audio": ModelParameter("audio", "boolean", "自动添加音频 (audio_url为空时生效)", default=True),
    "duration": ModelParameter("duration", "integer", "视频时长(秒)", default=5, enum=[5, 10, 15]),
    "duration_v2v": ModelParameter("duration", "integer", "视频时长(秒)", default=5, enum=[5, 10]),  # v2v only supports 5,10
    "shot_type": ModelParameter("shot_type", "string", "视频生成模式", default="multi", enum=["single", "multi"]),
    "watermark": ModelParameter("watermark", "boolean", "添加水印", default=False),
    "prompt_extend": ModelParameter("prompt_extend", "boolean", "Prompt智能改写", default=True),
}

# Wan 2.6 model -> supported parameter names mapping
WAN26_MODEL_PARAMS = {
    # T2V: no image input
    "wan2.6_t2v": {
        "input": ["prompt", "audio_url", "negative_prompt"],
        "parameters": ["seed", "size", "audio", "duration", "shot_type", "watermark", "prompt_extend"]
    },
    # I2V: uses img_url for image input
    "wan2.6_i2v": {
        "input": ["prompt", "img_url", "template", "audio_url", "negative_prompt"],
        "parameters": ["seed", "audio", "duration", "shot_type", "watermark", "resolution", "prompt_extend"]
    },
    # V2V: uses reference_video_urls
    "wan2.6_v2v": {
        "input": ["prompt", "audio_url", "negative_prompt", "reference_video_urls"],
        "parameters": ["seed", "size", "audio", "duration_v2v", "shot_type", "watermark", "prompt_extend"]
    },
}

# Video model parameter definitions (superset of all possible parameters for unified API)
VIDEO_PARAMS = {
    "prompt": ModelParameter("prompt", "string", "提示词", required=True),
    "negative_prompt": ModelParameter("negative_prompt", "string", "反向提示词"),
    "image": ModelParameter("image", "string", "输入图片 (URL 或 Base64)"),
    "end_image": ModelParameter("end_image", "string", "结束帧图片"),
    "duration": ModelParameter("duration", "string", "视频时长(秒)", default="5", enum=["4", "5", "6", "8", "10", "12", "15"]),
    "aspect_ratio": ModelParameter("aspect_ratio", "string", "宽高比", default="16:9", enum=["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"]),
    "resolution": ModelParameter("resolution", "string", "分辨率", default="720p", enum=["480p", "720p", "768P", "1080p", "1080P"]),
    "size": ModelParameter("size", "string", "画面尺寸 (如 720*1280, 1920*1080)", default="720*1280"),
    "prompt_extend": ModelParameter("prompt_extend", "boolean", "提示词优化", default=True),
    "add_audio": ModelParameter("add_audio", "boolean", "添加音频", default=False),
    "audio": ModelParameter("audio", "string", "音频URL"),
    "seed": ModelParameter("seed", "integer", "随机种子", default=0, minimum=0, maximum=4294967295),
    "n": ModelParameter("n", "integer", "生成视频个数", default=1, minimum=1, maximum=4),
    "guidance_scale": ModelParameter("guidance_scale", "number", "引导程度", default=0.5, minimum=0, maximum=1),
    "camera_fixed": ModelParameter("camera_fixed", "boolean", "相机固定", default=False),
    "person_generation": ModelParameter("person_generation", "string", "人物生成控制", default="allow_adult", enum=["allow_adult", "dont_allow"]),
    "watermark": ModelParameter("watermark", "boolean", "水印", default=False),
    "loras": ModelParameter("loras", "array", "LoRA 模型列表"),
}

# Mapping: model_id -> list of supported parameter names
VIDEO_MODEL_PARAMS = {
    # Text-to-Video
    "wan2.2_t2v": ["prompt", "negative_prompt", "size", "duration", "seed", "watermark", "loras", "prompt_extend"],
    "wan2.5_preview_t2v": ["prompt", "negative_prompt", "audio", "size", "duration", "prompt_extend", "add_audio", "seed"],
    "kling2.1_master_t2v": ["prompt", "negative_prompt", "aspect_ratio", "duration", "guidance_scale"],
    "kling2.5_turbo_pro_t2v": ["prompt", "negative_prompt", "aspect_ratio", "duration", "guidance_scale"],
    "minimax_hailuo2.3_t2v": ["prompt", "duration", "resolution", "prompt_extend"],
    "seedance_v1_lite_t2v": ["prompt", "resolution", "aspect_ratio", "duration", "camera_fixed", "seed"],
    "seedance_v1_pro_t2v": ["prompt", "resolution", "aspect_ratio", "duration", "camera_fixed", "seed"],
    "sora2_t2v": ["prompt", "size", "duration"],
    "sora2_t2v_pro": ["prompt", "size", "duration"],
    "veo3_t2v": ["prompt", "negative_prompt", "aspect_ratio", "duration", "prompt_extend", "add_audio", "person_generation", "resolution", "n", "seed"],
    "veo3_preview_t2v": ["prompt", "negative_prompt", "aspect_ratio", "duration", "prompt_extend", "add_audio", "person_generation", "resolution", "n", "seed"],
    "veo3.1_preview_t2v": ["prompt", "negative_prompt", "aspect_ratio", "duration", "prompt_extend", "add_audio", "person_generation", "resolution", "n", "seed"],
    "veo3.1_fast_preview_t2v": ["prompt", "negative_prompt", "aspect_ratio", "duration", "prompt_extend", "add_audio", "person_generation", "resolution", "n", "seed"],
    # Image-to-Video
    "wan2.2_i2v": ["prompt", "image", "negative_prompt", "resolution", "duration", "seed", "watermark", "loras", "prompt_extend"],
    "wan2.5_preview_i2v": ["prompt", "image", "negative_prompt", "audio", "resolution", "duration", "prompt_extend", "add_audio", "seed"],
    "kling2.1_i2v": ["prompt", "image", "negative_prompt", "duration", "guidance_scale"],
    "kling2.1_pro_i2v": ["prompt", "image", "negative_prompt", "duration", "guidance_scale"],
    "kling2.1_master_i2v": ["prompt", "image", "negative_prompt", "duration", "guidance_scale"],
    "kling2.5_turbo_pro_i2v": ["prompt", "image", "negative_prompt", "duration", "guidance_scale"],
    "minimax_hailuo_02": ["prompt", "image", "end_image", "duration", "resolution", "prompt_extend"],
    "minimax_hailuo2.3_i2v": ["prompt", "image", "duration", "resolution", "prompt_extend"],
    "minimax_hailuo2.3_fast_i2v": ["prompt", "image", "duration", "resolution", "prompt_extend"],
    "seedance_v1_lite_i2v": ["prompt", "image", "end_image", "resolution", "aspect_ratio", "duration", "camera_fixed", "seed"],
    "seedance_v1_pro_i2v": ["prompt", "image", "resolution", "aspect_ratio", "duration", "camera_fixed", "seed"],
    "sora2_i2v": ["prompt", "image", "resolution", "duration"],
    "sora2_i2v_pro": ["prompt", "image", "resolution", "duration"],
    "veo3_preview_i2v": ["prompt", "image", "negative_prompt", "aspect_ratio", "duration", "prompt_extend", "add_audio", "person_generation", "resolution", "n", "seed"],
    "veo3.1_preview_i2v": ["prompt", "image", "end_image", "negative_prompt", "aspect_ratio", "duration", "prompt_extend", "add_audio", "person_generation", "resolution", "n", "seed"],
    "veo3.1_fast_preview_i2v": ["prompt", "image", "end_image", "negative_prompt", "aspect_ratio", "duration", "prompt_extend", "add_audio", "person_generation", "resolution", "n", "seed"],
    # Other (can be T2V or I2V based on image input)
    "sora2_reverse": ["prompt", "image", "size", "duration", "watermark"],
}

# Text-to-Video Models (synced from video_models_config.json)
VIDEO_T2V_MODELS = [
    ModelConfig(id="wan2.2_t2v", name="Wan 2.2 文生视频", description="wan 2.2 文生视频", category="video_t2v", doc_url="https://docs.jiekou.ai/docs/models/reference-wan2.2-t2v"),
    ModelConfig(id="wan2.5_preview_t2v", name="Wan 2.5 Preview 文生视频", description="wan 2.5 preview 文生视频", category="video_t2v", doc_url="https://docs.jiekou.ai/docs/models/reference-wan2.5-t2v"),
    ModelConfig(id="kling2.1_master_t2v", name="Kling 2.1 Master 文生视频", description="kling 2.1 master 文生视频", category="video_t2v", doc_url="https://docs.jiekou.ai/docs/models/reference-kling-v2.1-t2v-master"),
    ModelConfig(id="kling2.5_turbo_pro_t2v", name="Kling 2.5 Turbo Pro 文生视频", description="kling 2.5 turbo pro 文生视频", category="video_t2v", doc_url="https://docs.jiekou.ai/docs/models/reference-kling-2.5-turbo-t2v"),
    ModelConfig(id="minimax_hailuo2.3_t2v", name="Minimax Hailuo 2.3 文生视频", description="minimax hailuo 2.3 文生视频", category="video_t2v", doc_url="https://docs.jiekou.ai/docs/models/reference-minimax-hailuo-2.3-t2v"),
    ModelConfig(id="seedance_v1_lite_t2v", name="Seedance V1 Lite 文生视频", description="seedance v1 lite 文生视频", category="video_t2v", doc_url="https://docs.jiekou.ai/docs/models/reference-seedance-v1-lite-t2v"),
    ModelConfig(id="seedance_v1_pro_t2v", name="Seedance V1 Pro 文生视频", description="seedance v1 pro 文生视频", category="video_t2v", doc_url="https://docs.jiekou.ai/docs/models/reference-seedance-v1-pro-t2v"),
    ModelConfig(id="sora2_t2v", name="Sora 2 文生视频", description="sora 2 文生视频", category="video_t2v", doc_url="https://docs.jiekou.ai/docs/models/reference-sora-2-t2v"),
    ModelConfig(id="sora2_t2v_pro", name="Sora 2 Pro 文生视频", description="sora 2 pro 文生视频", category="video_t2v", doc_url="https://docs.jiekou.ai/docs/models/reference-sora-2-t2v"),
    ModelConfig(id="veo3_t2v", name="Veo 3.0 文生视频", description="veo 3.0 文生视频", category="video_t2v", doc_url="https://docs.jiekou.ai/docs/models/reference-veo-3.0-generate-001-text2video"),
    ModelConfig(id="veo3_preview_t2v", name="Veo 3.0 Preview 文生视频", description="veo3.0 preview", category="video_t2v", doc_url="https://docs.jiekou.ai/docs/models/reference-veo-3.0-generate-preview-text2video"),
    ModelConfig(id="veo3.1_preview_t2v", name="Veo 3.1 Preview 文生视频", description="veo 3.1 preview 文生视频", category="video_t2v", doc_url="https://docs.jiekou.ai/docs/models/reference-veo-3.1-generate-text2video"),
    ModelConfig(id="veo3.1_fast_preview_t2v", name="Veo 3.1 Fast Preview 文生视频", description="veo 3.1 fast preview 文生视频", category="video_t2v", doc_url="https://docs.jiekou.ai/docs/models/reference-veo-3.1-fast-generate-text2video"),
]

# Image-to-Video Models (synced from video_models_config.json)
VIDEO_I2V_MODELS = [
    ModelConfig(id="wan2.2_i2v", name="Wan 2.2 图生视频", description="wan 2.2 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-wan2.2-i2v"),
    ModelConfig(id="wan2.5_preview_i2v", name="Wan 2.5 Preview 图生视频", description="wan 2.5 preview 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-wan2.5-i2v"),
    ModelConfig(id="kling2.1_i2v", name="Kling 2.1 图生视频", description="kling 2.1 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-kling-v2.1-i2v"),
    ModelConfig(id="kling2.1_pro_i2v", name="Kling 2.1 Pro 图生视频", description="kling 2.1 pro 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-kling-v2.1-i2v"),
    ModelConfig(id="kling2.1_master_i2v", name="Kling 2.1 Master 图生视频", description="kling 2.1 master 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-kling-v2.1-i2v-master"),
    ModelConfig(id="kling2.5_turbo_pro_i2v", name="Kling 2.5 Turbo Pro 图生视频", description="kling 2.5 turbo pro 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-kling-2.5-turbo-i2v"),
    ModelConfig(id="minimax_hailuo_02", name="Minimax Hailuo 02", description="minimax hailuo 02", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-minimax-hailuo-02"),
    ModelConfig(id="minimax_hailuo2.3_i2v", name="Minimax Hailuo 2.3 图生视频", description="minimax hailuo 2.3 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-minimax-hailuo-2.3-i2v"),
    ModelConfig(id="minimax_hailuo2.3_fast_i2v", name="Minimax Hailuo 2.3 Fast 图生视频", description="minimax hailuo 2.3 fast 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-minimax-hailuo-2.3-fast-i2v"),
    ModelConfig(id="seedance_v1_lite_i2v", name="Seedance V1 Lite 图生视频", description="seedance v1 lite 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-seedance-v1-lite-i2v"),
    ModelConfig(id="seedance_v1_pro_i2v", name="Seedance V1 Pro 图生视频", description="seedance v1 pro 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-seedance-v1-pro-i2v"),
    ModelConfig(id="sora2_i2v", name="Sora 2 图生视频", description="sora 2 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-sora-2-i2v"),
    ModelConfig(id="sora2_i2v_pro", name="Sora 2 Pro 图生视频", description="sora 2 pro 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-sora-2-i2v"),
    ModelConfig(id="veo3_preview_i2v", name="Veo 3.0 Preview 图生视频", description="veo 3.0 preview 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-veo-3.0-generate-preview-img2video"),
    ModelConfig(id="veo3.1_preview_i2v", name="Veo 3.1 Preview 图生视频", description="veo 3.1 preview 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-veo-3.1-generate-img2video"),
    ModelConfig(id="veo3.1_fast_preview_i2v", name="Veo 3.1 Fast Preview 图生视频", description="veo 3.1 fast preview 图生视频", category="video_i2v", doc_url="https://docs.jiekou.ai/docs/models/reference-veo-3.1-fast-generate-img2video"),
]

# ===== Video V2V / Other Models =====
# Video-to-Video, reference-based generation, and other special video models
VIDEO_V2V_MODELS = [
    ModelConfig(id="sora2_reverse", name="Sora 2 Reverse", description="sora 2 video gen (reverse)", category="video_v2v", doc_url="https://docs.jiekou.ai/docs/models/reference-sora-2-video-reverse"),
]

# ===== Wan 2.6 Models (use dedicated endpoints, not unified API) =====
WAN26_VIDEO_MODELS = [
    ModelConfig(
        id="wan2.6_t2v",
        name="Wan 2.6 文生视频",
        description="Wan 2.6 文生视频，支持音频生成、多镜头模式",
        category="video_t2v",
        endpoint="/v3/async/wan2.6-t2v",
        is_async=True,
        doc_url="https://docs.jiekou.ai/docs/models/reference-wan2.6-t2v",
    ),
    ModelConfig(
        id="wan2.6_i2v",
        name="Wan 2.6 图生视频",
        description="Wan 2.6 图生视频，支持视频特效模板",
        category="video_i2v",
        endpoint="/v3/async/wan2.6-i2v",
        is_async=True,
        doc_url="https://docs.jiekou.ai/docs/models/reference-wan2.6-i2v",
    ),
    ModelConfig(
        id="wan2.6_v2v",
        name="Wan 2.6 参考生视频",
        description="Wan 2.6 参考生视频，支持1-3条参考视频生成",
        category="video_v2v",
        endpoint="/v3/async/wan2.6-v2v",
        is_async=True,
        doc_url="https://docs.jiekou.ai/docs/models/reference-wan2.6-v2v",
    ),
]

# ===== Audio Generation Models =====
AUDIO_MODELS = [
    ModelConfig(
        id="elevenlabs-tts-v3",
        name="ElevenLabs TTS V3",
        description="ElevenLabs 文字转语音 V3",
        category="audio_tts",
        endpoint="/v1/audio/speech",
        is_async=False,
        response_type="binary",
        doc_url="https://docs.jiekou.ai/docs/models/reference-elevenlabs-tts-v3",
        parameters=[
            ModelParameter("text", "string", "转换文本", required=True),
            ModelParameter("voice_id", "string", "声音 ID", required=True),
            ModelParameter("output_format", "string", "输出格式", default="mp3_44100_128",
                          enum=["mp3_22050_32", "mp3_44100_64", "mp3_44100_128", "mp3_44100_192"]),
            ModelParameter("stability", "number", "稳定性", default=0.5, minimum=0, maximum=1),
            ModelParameter("similarity_boost", "number", "相似度", default=0.75, minimum=0, maximum=1),
        ]
    ),
]


# ===== Registry Class =====
class ModelRegistry:
    """Registry for all supported models"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_models()
        return cls._instance
    
    def _init_models(self):
        """Initialize model collections"""
        self._all_models = {}
        
        # Register all models
        all_models = (
            IMAGE_T2I_MODELS + 
            IMAGE_EDIT_MODELS + 
            MIDJOURNEY_MODELS +
            IMAGE_TOOL_MODELS +
            VIDEO_T2V_MODELS + 
            VIDEO_I2V_MODELS + 
            VIDEO_V2V_MODELS +
            WAN26_VIDEO_MODELS +  # Wan 2.6 dedicated endpoint models
            AUDIO_MODELS
        )
        
        for model in all_models:
            self._all_models[model.id] = model
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID"""
        return self._all_models.get(model_id)
    
    def get_model_by_name(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by display name"""
        for m in self._all_models.values():
            if m.name == name:
                return m
        return None
    
    def resolve_model(self, identifier: str) -> Optional[ModelConfig]:
        """
        Resolve model by ID or name.
        Frontend passes display name, so we need to look up by name.
        Falls back to ID lookup for backwards compatibility.
        """
        # Try ID first (faster, for backwards compatibility)
        model = self._all_models.get(identifier)
        if model:
            return model
        
        # Try name lookup (frontend sends display name)
        return self.get_model_by_name(identifier)
    
    def get_models_by_category(self, category: str) -> List[ModelConfig]:
        """Get all models in a category"""
        return [m for m in self._all_models.values() if m.category == category]
    
    def get_video_t2v_models(self) -> List[ModelConfig]:
        """Get all text-to-video models"""
        return self.get_models_by_category("video_t2v")
    
    def get_video_i2v_models(self) -> List[ModelConfig]:
        """Get all image-to-video models"""
        return self.get_models_by_category("video_i2v")
    
    def get_all_video_models(self) -> List[ModelConfig]:
        """Get all video generation models"""
        return self.get_video_t2v_models() + self.get_video_i2v_models() + self.get_video_v2v_models()
    
    def get_image_t2i_models(self) -> List[ModelConfig]:
        """Get all text-to-image models"""
        return self.get_models_by_category("image_t2i")
    
    def get_image_edit_models(self) -> List[ModelConfig]:
        """Get all image editing models"""
        return self.get_models_by_category("image_edit")
    
    def get_image_tool_models(self) -> List[ModelConfig]:
        """Get all image tool models (upscale, remove bg, etc.)"""
        return self.get_models_by_category("image_tool")
    
    def get_all_image_models(self) -> List[ModelConfig]:
        """Get all image-related models"""
        return (
            self.get_image_t2i_models() + 
            self.get_image_edit_models() + 
            self.get_image_tool_models()
        )
    
    def get_audio_models(self) -> List[ModelConfig]:
        """Get all audio generation models"""
        return [m for m in self._all_models.values() if m.category.startswith("audio_")]
    
    def get_model_ids(self, category: str = None) -> List[str]:
        """Get list of model IDs, optionally filtered by category"""
        if category:
            return [m.id for m in self.get_models_by_category(category)]
        return list(self._all_models.keys())
    
    def get_model_choices(self, category: str = None) -> List[tuple]:
        """Get list of (id, name) tuples for UI dropdowns"""
        models = self.get_models_by_category(category) if category else self._all_models.values()
        return [(m.id, m.name) for m in models]
    
    def get_video_model_params(self, model_id: str) -> List[ModelParameter]:
        """Get list of supported parameters for a video model"""
        param_names = VIDEO_MODEL_PARAMS.get(model_id, [])
        return [VIDEO_PARAMS[name] for name in param_names if name in VIDEO_PARAMS]
    
    def get_video_model_param_names(self, model_id: str) -> List[str]:
        """Get list of supported parameter names for a video model"""
        return VIDEO_MODEL_PARAMS.get(model_id, [])
    
    def is_video_model(self, model_id: str) -> bool:
        """Check if a model is a video model"""
        model = self.get_model(model_id)
        return model is not None and model.category.startswith("video_")
    
    def get_all_video_params(self) -> dict:
        """Get the complete video parameters dictionary"""
        return VIDEO_PARAMS
    
    def is_wan26_model(self, model_id: str) -> bool:
        """Check if a model is a Wan 2.6 dedicated endpoint model"""
        return model_id in WAN26_MODELS
    
    def get_wan26_model_params(self, model_id: str) -> dict:
        """
        Get Wan 2.6 model parameter structure
        
        Returns:
            dict with 'input' and 'parameters' lists of parameter names
        """
        return WAN26_MODEL_PARAMS.get(model_id, {"input": [], "parameters": []})
    
    def get_wan26_param(self, param_name: str) -> Optional[ModelParameter]:
        """Get a Wan 2.6 parameter definition by name"""
        return WAN26_PARAMS.get(param_name)
    
    def get_wan26_all_params(self) -> dict:
        """Get all Wan 2.6 parameter definitions"""
        return WAN26_PARAMS
    
    def get_video_v2v_models(self) -> List[ModelConfig]:
        """Get all video-to-video models (reference generation)"""
        return self.get_models_by_category("video_v2v")


# Singleton accessor
def get_model_registry() -> ModelRegistry:
    """Get the model registry singleton"""
    return ModelRegistry()
