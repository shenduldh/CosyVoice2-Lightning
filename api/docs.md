# 接口说明


## 查看已克隆音色 ID

地址：`GET http://0.0.0.0:12244/speakers`

响应：`["example_speaker_id1", "example_speaker_id2", ...]`


## 删除已克隆音色

地址：`POST http://0.0.0.0:12244/remove`

参数：`{"prompt_id": "example_speaker_id"}`

响应：`{"status": "ok"}`


## 克隆音色

地址：`POST http://0.0.0.0:12244/clone`

参数：

```json
{
    "prompt_audio": "base64, url or local file path",
    "prompt_text": "Example text",
    "prompt_id": "example_speaker_id"
}
```

响应：`{"existed": False, "prompt_id": "example_speaker_id"}`


## 流式 TTS

地址：`WEBSOCKET ws://0.0.0.0:12244/tts`

参数：

- 首次发送请求参数

    ```json
    {
        "req_params": {
            "prompt_id": "example_speaker_id",
            "audio_format": "opus, pcm, wav, mp3, flac, aac, m4a, or wav, default wav",
            "sample_rate": 24000,
            "instruct_text": "text or null"
        }
    }
    ```

- 后续发送文本流

    ```json
    {
        "text": "Example text segment1",
        "done": false
    }
    ```

- 末尾文本流

    ```json
    {
        "text": "Example text segment2",
        "done": true
    }
    ```

响应：

- 出错返回

    ```json
    {"error": true, "message": "Error example message"}
    ```

- 流式返回

    ```json
    {
        "id": "request uuid",
        "error": false,
        "is_end": false,
        "index": 0,
        "data": "base64",
        "audio_format": "opus, pcm, wav, mp3, flac, aac, m4a, or wav",
        "sample_rate": 24000
    }
    ```

- 流式结束

    ```json
    {
        "id": "request uuid",
        "error": false,
        "is_end": true,
        "index": 10,
        "data": null,
        "audio_format": null,
        "sample_rate": null
    }
    ```
