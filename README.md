# ComfyUI-JieKou-API

[接口 AI](https://jiekou.ai) 平台 ComfyUI 插件 - 一站式接入多模态 AI 能力

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Plugin-blue)](https://github.com/comfyanonymous/ComfyUI)

## ✨ 功能特性

- 🖼️ **文生图 (Text-to-Image)** - 支持 GPT-Image、FLUX、Seedream、Midjourney 等多种模型
- 🎨 **图生图 (Image-to-Image)** - 图像编辑、风格转换
- 📹 **视频生成 (Video Generation)** - 支持 Wan、Sora、Veo、Kling、Minimax 等模型
- 🔍 **图像放大 (Image Upscale)** - 2K/4K/8K 超分辨率
- ✂️ **背景移除 (Remove Background)** - 智能抠图

## 📦 支持的模型

### 图像模型
| 模型 | 文生图 | 图生图 |
|------|--------|--------|
| GPT Image | ✅ | ✅ |
| FLUX Kontext Pro/Dev/Max | ✅ | ✅ |
| Seedream 3.0/4.0/4.5 | ✅ | ✅ |
| Gemini 2.5/3.0 | ✅ | ✅ |
| Qwen Image | ✅ | ✅ |
| Midjourney | ✅ | - |
| Hunyuan Image | ✅ | - |

### 视频模型
| 模型 | 文生视频 | 图生视频 |
|------|----------|----------|
| Wan 2.2/2.6 | ✅ | ✅ |
| Sora 2 | ✅ | ✅ |
| Veo 3 | ✅ | ✅ |
| Kling 2.5 | ✅ | ✅ |
| Minimax | ✅ | ✅ |
| Seedance | ✅ | ✅ |

## 🚀 安装

### 方式一：ComfyUI Manager（推荐）

1. 打开 ComfyUI Manager
2. 搜索 `JieKou` 或 `接口`
3. 点击安装

### 方式二：手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jiekouai/ComfyUI-JieKou-API
cd ComfyUI-JieKou-API
pip install -r requirements.txt
```

### 方式三：ZIP 安装

1. 下载 [最新版本](https://github.com/jiekouai/ComfyUI-JieKou-API)
2. 解压到 `ComfyUI/custom_nodes/` 目录
3. 运行 `pip install -r requirements.txt`

## ⚙️ 配置

### 方式一：界面配置（推荐）

1. 启动 ComfyUI
2. 点击画布右上角的 **「⚙️ 接口 AI」**
3. 输入您的 API Key
4. 点击「保存」

### 方式二：配置文件

```bash
cd ComfyUI/custom_nodes/ComfyUI-JieKou-API
cp config.ini.example config.ini
```

编辑 `config.ini`，填入您的 API Key：

```ini
[AUTH]
api_key = your-api-key-here
```

### 获取 API Key

访问 [接口 AI 平台](https://jiekou.ai) 注册并获取 API Key。

## 📖 使用说明

### 节点列表

| 节点名称 | 功能 | 输入 | 输出 |
|----------|------|------|------|
| JieKou Text to Image | 文生图 | prompt | IMAGE, image_url |
| JieKou Image to Image | 图生图 | image_url, prompt | IMAGE, image_url |
| JieKou Image Upscale | 图像放大 | image_url | IMAGE, image_url |
| JieKou Remove Background | 背景移除 | image_url | IMAGE, image_url |
| JieKou Video Generation | 视频生成 | prompt, image_url(可选) | IMAGE(帧序列), video_url |
| JieKou Test Connection | 测试连接 | - | status |

### 基础工作流示例

#### 文生图

```
[JieKou Text to Image]
  ├─ model: gpt-image-1
  ├─ prompt: "一只可爱的猫咪"
  └─ save_to_disk: true
       │
       ▼
   [Preview Image]
```

#### 图生视频

```
[Load Image] ──► [JieKou Video Generation]
                   ├─ model: wan2.6_i2v
                   ├─ image_url: (从上游获取)
                   ├─ prompt: "让画面动起来"
                   └─ save_to_disk: true
                        │
                        ▼
                   [Video Combine]
```

## 🔗 链式调用

生图节点可输出 `image_url`，可以直接连接到下游节点的 `image_url` 输入：

```
[Text to Image] ──► image_url ──► [Image to Image] ──► image_url ──► [Video Generation]
```

## 📄 许可证

[MIT License](LICENSE)

## 🔗 相关链接

- [接口 AI 平台](https://jiekou.ai)
- [API 文档](https://docs.jiekou.ai/docs/models/reference-authentication)

