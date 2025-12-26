"""
JiekouAI ComfyUI Plugin - Audio Generation Nodes
TTS (Text-to-Speech) using ElevenLabs API

Note: ElevenLabs TTS returns binary audio directly (not JSON)
"""

import torch
import logging

logger = logging.getLogger("[JieKou]")


class JieKouTTS:
    """
    Generate speech audio from text using ElevenLabs TTS
    
    ElevenLabs TTS returns binary audio directly, not via async task
    """
    
    CATEGORY = "JieKou/Audio"
    FUNCTION = "generate_speech"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_NODE = True  # 允许直接运行
    
    # Common ElevenLabs voice IDs (users should replace with actual IDs)
    VOICE_IDS = [
        "21m00Tcm4TlvDq8ikWAM",  # Rachel
        "AZnzlk1XvdvUeBnXmlld",  # Domi
        "EXAVITQu4vr4xnSDxMaL",  # Bella
        "ErXwobaYiN019PkySvjV",  # Antoni
        "MF3mGyEYCl7XYWbV9V6O",  # Elli
        "TxGEqnHWrfWFTfGW9XjX",  # Josh
        "VR6AewLTigWG4xSOukaG",  # Arnold
        "pNInz6obpgDQGcFmaJgB",  # Adam
        "yoZ06aMxZJJ28mfd3POQ",  # Sam
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        # Import model registry for available models
        try:
            from ..utils.model_registry import get_model_registry
            registry = get_model_registry()
            audio_models = [m.id for m in registry.get_audio_models()]
            default_model = "elevenlabs-tts-v3" if "elevenlabs-tts-v3" in audio_models else audio_models[0]
        except Exception as e:
            logger.warning(f"[JieKou] Failed to load model registry: {e}")
            audio_models = ["elevenlabs-tts-v3", "glm-tts"]
            default_model = "elevenlabs-tts-v3"
        
        return {
            "required": {
                "model": (audio_models, {
                    "default": default_model
                }),
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "输入要转换为语音的文本..."
                }),
                "voice_id": ("STRING", {
                    "default": "21m00Tcm4TlvDq8ikWAM",
                    "placeholder": "ElevenLabs Voice ID"
                }),
            },
            "optional": {
                "output_format": (["mp3_44100_128", "mp3_44100_64", "mp3_22050_32"], {
                    "default": "mp3_44100_128"
                }),
                "stability": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "similarity_boost": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }
    
    def generate_speech(
        self,
        model: str,
        text: str,
        voice_id: str,
        output_format: str = "mp3_44100_128",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        speed: float = 1.0
    ):
        """
        Generate speech audio from text using ElevenLabs
        
        Returns:
            tuple: (AUDIO dict with waveform tensor and sample_rate)
        """
        logger.info(f"[JieKou] TTS: model={model}, text={text[:50]}...")
        
        if not text.strip():
            raise ValueError("文本不能为空")
        
        if not voice_id.strip():
            raise ValueError("Voice ID 不能为空")
        
        try:
            from ..utils.api_client import JiekouAPI, JiekouAPIError
            from ..utils.tensor_utils import audio_bytes_to_comfy
            
            api = JiekouAPI()
            
            # Build voice settings
            voice_settings = {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "speed": speed,
            }
            
            # ElevenLabs TTS returns binary audio directly
            if model == "elevenlabs-tts-v3":
                logger.info("[JieKou] Calling ElevenLabs TTS API...")
                audio_bytes = api.generate_tts_elevenlabs(
                    text=text,
                    voice_id=voice_id,
                    output_format=output_format,
                    voice_settings=voice_settings
                )
                
                # Convert binary audio to ComfyUI AUDIO format
                logger.info("[JieKou] Converting audio to ComfyUI format...")
                audio_data = audio_bytes_to_comfy(audio_bytes)
                
            elif model == "glm-tts":
                # GLM TTS is async, need to poll
                logger.info("[JieKou] Calling GLM TTS API (async)...")
                # For GLM, we'd need async handling similar to video
                # For now, raise not implemented
                raise NotImplementedError("GLM TTS 暂未实现，请使用 ElevenLabs TTS")
            
            else:
                raise ValueError(f"不支持的模型: {model}")
            
            logger.info(f"[JieKou] Generated audio: sample_rate={audio_data['sample_rate']}")
            
            return (audio_data,)
        
        except JiekouAPIError as e:
            error_msg = self._format_error(e)
            logger.error(f"[JieKou] API Error: {error_msg}")
            raise RuntimeError(error_msg)
        
        except Exception as e:
            logger.error(f"[JieKou] Error: {e}")
            raise
    
    def _format_error(self, error) -> str:
        """Format API error for user display"""
        error_messages = {
            "UNAUTHORIZED": "API Key 无效，请检查设置",
            "INSUFFICIENT_CREDITS": "额度不足，请充值后重试",
            "RATE_LIMIT": "请求过于频繁，请稍后重试",
            "INVALID_MODEL": "模型不存在或暂时不可用",
            "TEXT_TOO_LONG": "文本过长，请缩短后重试",
            "TIMEOUT": "请求超时，请重试",
        }
        
        return error_messages.get(error.code, error.message)
