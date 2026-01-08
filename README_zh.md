# ComfyUI-JieKou-API

[接口 AI](https://jiekou.ai) 平台 ComfyUI 插件 - 一站式接入多模态 AI 能力

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Plugin-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Version](https://img.shields.io/badge/version-1.2.0-green)](https://github.com/jiekouai/ComfyUI-JieKou-API)

[English Documentation](README.md)

## ✨ 功能特性

- 🖼️ **文生图 (Text-to-Image)** - 支持 GPT-Image、FLUX、Seedream、Midjourney 等多种模型
- 🎨 **图生图 (Image-to-Image)** - 图像编辑、风格转换、图像增强
- 📹 **视频生成 (Video Generation)** - 支持 Wan、Sora、Veo、Kling、Minimax 等模型
- 🔍 **图像放大 (Image Upscale)** - 2K/4K/8K 超分辨率
- ✂️ **背景移除 (Remove Background)** - 智能抠图
- 🔊 **音频生成 (Audio Generation)** - 支持 ElevenLabs 等文字转语音模型
- 💰 **实时价格显示** - 生成前查看预估费用 (v1.2 新增)

## 🆕 v1.2 版本更新

- **模型独立节点**：每个模型拥有独立节点（如 "JieKou AI > Image > Seedream 4.0"）
- **实时价格显示**：在节点右上角显示预估费用
- **参数动态联动**：参数选项根据有效组合自动过滤
- **支持 112 个模型**：全面覆盖图像、视频、音频模型

## 📦 支持的模型

### 图像模型（23 个）

| 模型 | 文生图 | 图生图 |
|------|--------|--------|
| GPT Image 1 | ✅ | ✅ |
| FLUX.1 Kontext Pro/Dev/Max | ✅ | ✅ |
| FLUX 2 Pro/Dev/Flex | ✅ | - |
| Seedream 4.0/4.5 | ✅ | ✅ |
| Gemini 2.5 Flash / 3 Pro | ✅ | ✅ |
| Qwen Image | ✅ | ✅ |
| Midjourney | ✅ | - |
| Hunyuan Image 3 | ✅ | - |
| Z Image Turbo | ✅ | - |

### 视频模型（64 个）

| 模型 | 文生视频 | 图生视频 |
|------|----------|----------|
| Wan 2.2/2.6 | ✅ | ✅ |
| Sora 2 | ✅ | ✅ |
| Veo 3 | ✅ | ✅ |
| Kling 1.6/2.5 | ✅ | ✅ |
| Minimax Hailuo 2.3 | ✅ | ✅ |
| Seedance 1.0 | ✅ | ✅ |
| Hunyuan Video | ✅ | ✅ |
| Luma Ray 2 | ✅ | ✅ |
| Pika 2.2 | ✅ | ✅ |

### 音频模型（25 个）

| 模型 | 类型 |
|------|------|
| ElevenLabs V1/V2 | 文字转语音 |
| ElevenLabs Scribe | 语音识别 |
| Fish Audio 1.5 | 文字转语音 |
| GPT-4o Realtime | 语音对话 |
| MiniMax TTS | 文字转语音 |

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

1. 下载 [最新版本](https://github.com/jiekouai/ComfyUI-JieKou-API/releases)
2. 解压到 `ComfyUI/custom_nodes/` 目录
3. 运行 `pip install -r requirements.txt`

## ⚙️ 配置

### 方式一：界面配置（推荐）

1. 启动 ComfyUI
2. 点击画布右上角的 **「⚙️ 接口 AI 设置」** 按钮
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

### 查找节点

在 ComfyUI 中，右键点击画布并导航到：

- **JieKou AI > Image > Text to Image** - 文生图模型
- **JieKou AI > Image > Edit** - 图像编辑模型
- **JieKou AI > Image > Tools** - 图像放大、背景移除
- **JieKou AI > Video > Text to Video** - 文生视频模型
- **JieKou AI > Video > Image to Video** - 图生视频模型
- **JieKou AI > Audio > Text to Speech** - 文字转语音模型

### 价格显示

每个节点右上角会显示预估费用（如 "$0.004/次"）。当您修改分辨率、时长等参数时，价格会自动更新。

### 基础工作流示例

#### 文生图

```
[JieKou AI > Image > Seedream 4.0]
  ├─ prompt: "花园里的一只可爱猫咪"
  ├─ aspect_ratio: "1:1"
  └─ save_to_disk: true
       │
       ▼
   [Preview Image]
```

#### 图生视频

```
[Load Image] ──► [JieKou AI > Video > Wan 2.6 I2V]
                   ├─ image_url: (从上游连接)
                   ├─ prompt: "让画面动起来"
                   ├─ duration: "5"
                   └─ save_to_disk: true
                        │
                        ▼
                   [Video Combine]
```

### 链式调用

生图节点输出的 `image_url` 可以直接连接到下游节点：

```
[文生图] ──► image_url ──► [图像编辑] ──► image_url ──► [视频生成]
```

## ❓ 常见问题

**Q: 为什么价格显示 "$--"？**
A: 价格可能暂时不可用，请检查网络连接或稍后重试。

**Q: 可以离线使用吗？**
A: 不可以，本插件需要联网调用接口 AI 的 API。

**Q: 如何更新插件？**
A: 使用 ComfyUI Manager 更新，或手动安装时执行 `git pull`。

**Q: 视频生成很慢？**
A: 视频生成是异步任务，根据模型和参数不同可能需要 1-5 分钟。

## 📄 许可证

[MIT License](LICENSE)

## 🔗 相关链接

- [接口 AI 平台](https://jiekou.ai)
- [API 文档](https://docs.jiekou.ai)
- [GitHub 仓库](https://github.com/jiekouai/ComfyUI-JieKou-API)

