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
        format: Image format (PNG, JPEG)
    
    Returns:
        Base64 encoded image string
    """
    # Handle batch dimension - take first image if batched
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Convert to numpy and denormalize
    np_image = tensor.cpu().numpy()
    np_image = (np_image * 255).clip(0, 255).astype(np.uint8)
    
    # Create PIL Image
    image = Image.fromarray(np_image, mode="RGB")
    
    # Encode to base64
    buffer = BytesIO()
    image.save(buffer, format=format)
    b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return b64_string


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

