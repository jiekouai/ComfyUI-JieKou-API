# ComfyUI-JieKou-API

[JieKou AI](https://jiekou.ai) Platform ComfyUI Plugin - All-in-one Multimodal AI Integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Plugin-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Version](https://img.shields.io/badge/version-1.2.0-green)](https://github.com/jiekouai/ComfyUI-JieKou-API)

[中文文档](README_zh.md)

## ✨ Features

- 🖼️ **Text-to-Image** - GPT-Image, FLUX, Seedream, Midjourney, and more
- 🎨 **Image-to-Image** - Image editing, style transfer, and enhancement
- 📹 **Video Generation** - Wan, Sora, Veo, Kling, Minimax, and more
- 🔍 **Image Upscale** - 2K/4K/8K super-resolution
- ✂️ **Remove Background** - Intelligent background removal
- 🔊 **Audio Generation** - Text-to-speech with ElevenLabs and more
- 💰 **Real-time Pricing** - See estimated cost before generation (v1.2+)

## 🆕 What's New in v1.2

- **Model-Specific Nodes**: Each model now has its own dedicated node (e.g., "JieKou AI > Image > Seedream 4.0")
- **Real-time Price Display**: See the estimated cost in the top-right corner of each node
- **Dynamic Parameter Linkage**: Parameters automatically filter based on valid combinations
- **112 Models Supported**: Comprehensive coverage across image, video, and audio

## 📦 Supported Models

### Image Models (23 models)

| Model | Text-to-Image | Image-to-Image |
|-------|---------------|----------------|
| GPT Image 1 | ✅ | ✅ |
| FLUX.1 Kontext Pro/Dev/Max | ✅ | ✅ |
| FLUX 2 Pro/Dev/Flex | ✅ | - |
| Seedream 4.0/4.5 | ✅ | ✅ |
| Gemini 2.5 Flash / 3 Pro | ✅ | ✅ |
| Qwen Image | ✅ | ✅ |
| Midjourney | ✅ | - |
| Hunyuan Image 3 | ✅ | - |
| Z Image Turbo | ✅ | - |

### Video Models (64 models)

| Model | Text-to-Video | Image-to-Video |
|-------|---------------|----------------|
| Wan 2.2/2.6 | ✅ | ✅ |
| Sora 2 | ✅ | ✅ |
| Veo 3 | ✅ | ✅ |
| Kling 1.6/2.5 | ✅ | ✅ |
| Minimax Hailuo 2.3 | ✅ | ✅ |
| Seedance 1.0 | ✅ | ✅ |
| Hunyuan Video | ✅ | ✅ |
| Luma Ray 2 | ✅ | ✅ |
| Pika 2.2 | ✅ | ✅ |

### Audio Models (25 models)

| Model | Type |
|-------|------|
| ElevenLabs V1/V2 | Text-to-Speech |
| ElevenLabs Scribe | Speech-to-Text |
| Fish Audio 1.5 | Text-to-Speech |
| GPT-4o Realtime | Voice Chat |
| MiniMax TTS | Text-to-Speech |

## 🚀 Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for `JieKou` or `接口`
3. Click Install

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jiekouai/ComfyUI-JieKou-API
cd ComfyUI-JieKou-API
pip install -r requirements.txt
```

### Method 3: ZIP Installation

1. Download the [latest release](https://github.com/jiekouai/ComfyUI-JieKou-API/releases)
2. Extract to `ComfyUI/custom_nodes/`
3. Run `pip install -r requirements.txt`

## ⚙️ Configuration

### Option 1: UI Configuration (Recommended)

1. Start ComfyUI
2. Click **"⚙️ JieKou AI Settings"** button in the top-right corner
3. Enter your API Key
4. Click "Save"

### Option 2: Configuration File

```bash
cd ComfyUI/custom_nodes/ComfyUI-JieKou-API
cp config.ini.example config.ini
```

Edit `config.ini` with your API Key:

```ini
[AUTH]
api_key = your-api-key-here
```

### Getting an API Key

Visit [JieKou AI Platform](https://jiekou.ai) to register and get your API Key.

## 📖 Usage Guide

### Finding Nodes

In ComfyUI, right-click on the canvas and navigate to:

- **JieKou AI > Image > Text to Image** - For text-to-image models
- **JieKou AI > Image > Edit** - For image editing models
- **JieKou AI > Image > Tools** - For upscaling, background removal
- **JieKou AI > Video > Text to Video** - For text-to-video models
- **JieKou AI > Video > Image to Video** - For image-to-video models
- **JieKou AI > Audio > Text to Speech** - For TTS models

### Price Display

Each node displays the estimated cost in the top-right corner (e.g., "$0.004/次"). The price updates automatically when you change parameters like resolution or duration.

### Basic Workflow Examples

#### Text-to-Image

```
[JieKou AI > Image > Seedream 4.0]
  ├─ prompt: "A cute cat in a garden"
  ├─ aspect_ratio: "1:1"
  └─ save_to_disk: true
       │
       ▼
   [Preview Image]
```

#### Image-to-Video

```
[Load Image] ──► [JieKou AI > Video > Wan 2.6 I2V]
                   ├─ image_url: (connected from upstream)
                   ├─ prompt: "Make the image come alive"
                   ├─ duration: "5"
                   └─ save_to_disk: true
                        │
                        ▼
                   [Video Combine]
```

### Chaining Nodes

Image nodes output `image_url` which can be directly connected to downstream nodes:

```
[Text to Image] ──► image_url ──► [Image Edit] ──► image_url ──► [Video Generation]
```

## ❓ FAQ

**Q: Why is the price showing "$--"?**
A: The price may be temporarily unavailable. Check your network connection or try again later.

**Q: Can I use this plugin offline?**
A: No, this plugin requires an internet connection to call JieKou AI APIs.

**Q: How do I update the plugin?**
A: Use ComfyUI Manager to update, or `git pull` if installed manually.

**Q: Video generation is slow?**
A: Video generation is an async task that may take 1-5 minutes depending on the model and parameters.

## 📄 License

[MIT License](LICENSE)

## 🔗 Links

- [JieKou AI Platform](https://jiekou.ai)
- [API Documentation](https://docs.jiekou.ai)
- [GitHub Repository](https://github.com/jiekouai/ComfyUI-JieKou-API)
