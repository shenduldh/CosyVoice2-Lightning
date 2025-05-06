# CosyVoice2 - Lightning

## Introduction

参考 [CosyVoice2-08312f4](https://github.com/FunAudioLLM/CosyVoice/tree/08312f4c4615b465d66ff55036be1cbd642904e6) 和 [async_cosyvoice](https://github.com/qi-hua/async_cosyvoice) 修改的 CosyVoice2 API，支持以下特性：

1. 使用 VLLM 加速推理；
2. 支持流式输入和流式输出；
3. 支持多格式音频（opus, pcm, wav, mp3, flac, aac, m4a, wav）和多采样率输出;
4. 简化的部署流程。

## Usage

请首先按照 [install.md](./install.md) 中的描述配置环境。

### Gradio Demo

> 启动演示 Demo，包括克隆和指令控制。

```bash
sh run_demo.sh
```

### API

> 启动 API 服务，由 Websocket 提供流式生成。具体接口参考 [API docs](api/docs.md)。

```bash
cd api
sh run_server.sh
python test.sh
```
