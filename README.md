# CosyVoice2 - Lightning

## Introduction

参考 [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) 修改的 CosyVoice2 Websocket API，支持以下特性：

1. 使用 `vLLM 0.7` 加速推理；
2. 支持流式输入和流式输出；
3. 支持多格式音频（opus、pcm、wav、mp3、flac、aac、m4a、wav，默认是 wav）和多采样率输出；
4. 支持克隆音色；
5. 支持使用指令文本控制语音生成；
6. 支持保存和加载已克隆音色。

## Updates

- [2025/09/03] 新增支持 `vLLM 0.8`、`vLLM 0.9` 和 `SGLang`；提高长文本生成质量；简化部署；同步官方代码。


## TTFF

使用 `SGLang` 作为后端测试 TTFF：

- 单句话

    <img src="./assets/single_sentence_mttff.png" width="50%">

- 短段落

    <img src="./assets/short_paragraph_mttff.png" width="50%">

## Usage

请首先按照 [install.md](./install.md) 中的描述配置环境。

### Gradio Demo

> 启动演示 Demo，包括克隆和指令控制。

```bash
cd demo
sh run_demo.sh
```

### API

> 启动 API 服务，由 Websocket 提供流式生成。具体接口参考 [API docs](api/README.md)。

```bash
cd api
sh run_server.sh
python test.sh
```
