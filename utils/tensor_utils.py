"""
JiekouAI ComfyUI Plugin - Tensor Utilities
Image, video, and audio format conversion between API and ComfyUI formats
"""

import torch
import numpy as np
import base64
import requests
import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger("[JieKou]")


# ===== Image Utilities =====

def url_to_tensor(url: str) -> torch.Tensor:
    """
    Download image from URL and convert to ComfyUI IMAGE tensor
    
    Args:
        url: Image URL
    
    Returns:
        torch.Tensor with shape [1, H, W, C], dtype=float32, range 0.0-1.0
    """
    logger.info(f"[JieKou] Downloading image from URL")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return bytes_to_tensor(response.content)


def base64_to_tensor(b64_string: str) -> torch.Tensor:
    """
    Convert base64 encoded image to ComfyUI IMAGE tensor
    
    Args:
        b64_string: Base64 encoded image data
    
    Returns:
        torch.Tensor with shape [1, H, W, C], dtype=float32, range 0.0-1.0
    """
    # Remove data URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    
    image_bytes = base64.b64decode(b64_string)
    return bytes_to_tensor(image_bytes)


def bytes_to_tensor(image_bytes: bytes) -> torch.Tensor:
    """
    Convert image bytes to ComfyUI IMAGE tensor
    
    Args:
        image_bytes: Raw image bytes (PNG, JPEG, etc.)
    
    Returns:
        torch.Tensor with shape [1, H, W, C], dtype=float32, range 0.0-1.0
    """
    image = Image.open(BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to numpy array and normalize to 0.0-1.0
    np_image = np.array(image).astype(np.float32) / 255.0
    
    # Add batch dimension: [H, W, C] -> [1, H, W, C]
    tensor = torch.from_numpy(np_image).unsqueeze(0)
    
    return tensor


# ===== T028: download_image() =====
def download_image(url: str) -> Image.Image:
    """
    Download image from URL and return as PIL Image
    
    Args:
        url: Image URL
    
    Returns:
        PIL.Image.Image
    """
    logger.info(f"[JieKou] Downloading image")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


# ===== T029: image_to_tensor() =====
def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI IMAGE tensor
    
    Args:
        image: PIL.Image.Image
    
    Returns:
        torch.Tensor with shape [1, H, W, C], dtype=float32, range 0.0-1.0
    """
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to numpy array and normalize
    np_image = np.array(image).astype(np.float32) / 255.0
    
    # Add batch dimension
    tensor = torch.from_numpy(np_image).unsqueeze(0)
    
    return tensor


# ===== T033: tensor_to_base64() =====
def tensor_to_base64(tensor: torch.Tensor, format: str = "PNG") -> str:
    """
    Convert ComfyUI IMAGE tensor to base64 encoded string
    
    Args:
        tensor: torch.Tensor with shape [B, H, W, C] or [H, W, C]
        format: Image format (PNG, JPEG, WEBP)
    
    Returns:
        Base64 encoded image string
    """
    # Handle batch dimension - take first image if batched
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Convert to numpy and denormalize
    np_image = tensor.cpu().numpy()
    np_image = (np_image * 255).clip(0, 255).astype(np.uint8)
    
    # Check if tensor has alpha channel (RGBA)
    if np_image.shape[-1] == 4:
        mode = "RGBA"
        # Force PNG for alpha channel images
        if format.upper() == "JPEG":
            format = "PNG"
    else:
        mode = "RGB"
    
    # Create PIL Image
    image = Image.fromarray(np_image, mode=mode)
    
    # Encode to base64
    buffer = BytesIO()
    image.save(buffer, format=format)
    b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return b64_string


def tensor_to_data_uri(tensor: torch.Tensor, format: str = "PNG") -> str:
    """
    Convert ComfyUI IMAGE tensor to data URI string
    
    Args:
        tensor: torch.Tensor with shape [B, H, W, C] or [H, W, C]
        format: Image format (PNG, JPEG, WEBP)
    
    Returns:
        Data URI string like "data:image/png;base64,..."
    """
    b64_string = tensor_to_base64(tensor, format=format)
    
    # Map format to MIME type
    mime_map = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "WEBP": "image/webp",
        "GIF": "image/gif",
    }
    mime_type = mime_map.get(format.upper(), "image/png")
    
    return f"data:{mime_type};base64,{b64_string}"


def get_image_format_from_url(url: str) -> str:
    """
    Detect image format from URL or data URI
    
    Args:
        url: Image URL or data URI
    
    Returns:
        Format string (png, jpeg, webp) or empty string if unknown
    """
    if not url:
        return ""
    
    url_lower = url.lower().strip()
    
    # Check data URI format
    if url_lower.startswith("data:image/"):
        # Extract MIME type: data:image/jpeg;base64,...
        mime_part = url_lower.split(";")[0]  # "data:image/jpeg"
        format_part = mime_part.split("/")[-1]  # "jpeg"
        return format_part if format_part in ("png", "jpeg", "jpg", "webp", "gif") else ""
    
    # Check URL extension
    # Remove query string first
    url_path = url_lower.split("?")[0]
    
    for ext in (".png", ".jpeg", ".jpg", ".webp", ".gif"):
        if url_path.endswith(ext):
            return ext[1:]  # Remove dot
    
    return ""


# ===== tensor_to_bytes() =====
def tensor_to_bytes(tensor: torch.Tensor, format: str = "PNG") -> bytes:
    """
    Convert ComfyUI IMAGE tensor to image bytes
    
    Args:
        tensor: torch.Tensor with shape [B, H, W, C] or [H, W, C]
        format: Image format (PNG, JPEG, WEBP)
    
    Returns:
        Image bytes
    """
    # Handle batch dimension - take first image if batched
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Convert to numpy and denormalize
    np_image = tensor.cpu().numpy()
    np_image = (np_image * 255).clip(0, 255).astype(np.uint8)
    
    # Create PIL Image
    image = Image.fromarray(np_image, mode="RGB")
    
    # Save to bytes
    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


# ===== T042: video_to_frames() =====
def video_to_frames(video_bytes: bytes) -> torch.Tensor:
    """
    Decode video bytes to frame sequence tensor using OpenCV
    
    Args:
        video_bytes: Raw video bytes (MP4, etc.)
    
    Returns:
        torch.Tensor with shape [B, H, W, C] where B is frame count,
        dtype=float32, range 0.0-1.0
    """
    import cv2
    import tempfile
    import os
    
    logger.info("[JieKou] Decoding video to frames")
    
    # Write video to temporary file (cv2 needs file path)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name
    
    try:
        # Open video
        cap = cv2.VideoCapture(temp_path)
        
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to 0.0-1.0
            frame_float = frame_rgb.astype(np.float32) / 255.0
            
            frames.append(frame_float)
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Stack frames: list of [H, W, C] -> [B, H, W, C]
        tensor = torch.from_numpy(np.stack(frames, axis=0))
        
        logger.info(f"[JieKou] Decoded {len(frames)} frames")
        
        return tensor
    
    finally:
        # Clean up temp file
        os.unlink(temp_path)


# ===== T048: audio_to_comfy() =====
def audio_to_comfy(audio_bytes: bytes) -> dict:
    """
    Convert audio bytes to ComfyUI AUDIO format
    
    Args:
        audio_bytes: Raw audio bytes (WAV, MP3, etc.)
    
    Returns:
        dict with "waveform" (tensor [B, C, S]) and "sample_rate" (int)
    """
    import torchaudio
    
    logger.info("[JieKou] Converting audio to ComfyUI format")
    
    # Load audio from bytes
    buffer = BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(buffer)
    
    # Add batch dimension if not present: [C, S] -> [1, C, S]
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    
    return {
        "waveform": waveform,
        "sample_rate": sample_rate
    }


# Alias for binary audio data (same function, clearer naming)
audio_bytes_to_comfy = audio_to_comfy


# ===== Local file to base64 =====
def local_file_to_base64(file_path: str) -> str:
    """
    Read a local image file and convert to base64 data URL
    
    Args:
        file_path: Local file path to image
    
    Returns:
        Base64 data URL string (e.g., "data:image/png;base64,...")
    """
    import os
    import mimetypes
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "image/png"  # Default to PNG
    
    # Read and encode
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    
    b64_string = base64.b64encode(image_bytes).decode("utf-8")
    
    logger.info(f"[JieKou] Loaded local image: {file_path} ({len(image_bytes)} bytes)")
    
    return f"data:{mime_type};base64,{b64_string}"


def is_local_file_path(path: str) -> bool:
    """
    Check if a string looks like a local file path
    
    Args:
        path: String to check
    
    Returns:
        True if it looks like a local file path
    """
    import os
    
    # Skip URLs
    if path.startswith("http://") or path.startswith("https://"):
        return False
    
    # Skip base64 data URLs
    if path.startswith("data:"):
        return False
    
    # Check if it's an absolute or relative path that exists
    # Or if it looks like a path (contains path separators)
    if os.path.exists(path):
        return True
    
    # Check for common path patterns
    if path.startswith("/") or path.startswith("~") or path.startswith("./") or path.startswith("..\\"):
        return True
    
    # Windows absolute paths
    if len(path) > 2 and path[1] == ":" and (path[2] == "/" or path[2] == "\\"):
        return True
    
    return False

