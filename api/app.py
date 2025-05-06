import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from logger import logger
from starlette.responses import JSONResponse
import http
import traceback
import requests
from tempfile import NamedTemporaryFile
import time
from datetime import datetime
import uuid
import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Union, Tuple
import yaml
import pysbd

from bases import *
from utils import *
from CosyVoice.model import Pipeline


pipeline = None
frontend_config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, frontend_config

    frontend_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "frontend.yaml"
    )
    with open(frontend_config_path, "r") as f:
        frontend_config = yaml.safe_load(f)
    logger.info(f"Frontend Config: {frontend_config}")

    tts_model_dir = os.environ["TTS_MODEL_DIR"]
    pipeline = Pipeline(model_dir=tts_model_dir, load_trt=True, fp16=True)

    speaker_cache_path = os.getenv(
        "DEFAULT_SPEAKER_CACHE_PATH", "../assets/default_speaker_cache.pt"
    )
    if os.path.exists(speaker_cache_path):
        speaker_ids = pipeline.load_cache(speaker_cache_path)
        logger.info(f"Successfully load speakers: {speaker_ids}")

    logger.info("Ready.")
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def general_exception_handler(request, e: Exception):
    logger.info("Error: %s" % e)
    return JSONResponse(str(e), http.HTTPStatus.INTERNAL_SERVER_ERROR)


@app.get("/")
async def index() -> str:
    return f"Hello."


@app.get("/alive")
async def alive() -> dict:
    return {"status": "alive"}


@app.get("/speakers")
async def get_speakers() -> list:
    return pipeline.get_speakers()


@app.post("/remove")
async def remove_speaker(req: RemoveSpeakerInput) -> dict:
    pipeline.remove_speaker(req.prompt_id)
    return {"status": "ok"}


@app.post("/cache/save")
async def save_cache(req: SaveCacheInput) -> dict:
    cache_path = pipeline.save_cache(req.cache_dir, req.prompt_ids)
    return {"cache_path": cache_path}


@app.post("/cache/load")
async def load_cache(req: LoadCacheInput) -> dict:
    loaded_speaker_ids = pipeline.load_cache(req.cache_path, req.prompt_ids)
    return {"loaded_speakers": loaded_speaker_ids}


@app.post("/clone")
async def clone(req: CloneInput) -> CloneOutput:
    logger.info("Request Params: %s" % truncate_long_str(req))

    prompt_id = req.prompt_id
    prompt_text = req.prompt_text
    prompt_audio = req.prompt_audio
    sample_rate = req.sample_rate

    if prompt_id is None:
        prompt_id = f"{uuid.uuid4().hex[:7]}_{datetime.now().strftime('%Y-%m-%d')}"
    if prompt_id in pipeline.get_speakers():
        return CloneOutput(existed=True, prompt_id=prompt_id)

    if len(prompt_text.strip()) == 0:
        prompt_text = None

    s = time.time()

    if os.path.exists(prompt_audio):
        async for _ in pipeline.async_generate(
            "hello", prompt_text, prompt_audio, None, prompt_id
        ):
            pass
    elif prompt_audio.startswith("http"):
        prompt_audio = requests.get(prompt_audio).content
        with NamedTemporaryFile() as f:
            f.write(prompt_audio)
            f.flush()
            async for _ in pipeline.async_generate(
                "hello", prompt_text, f.name, None, prompt_id
            ):
                pass
    else:
        prompt_audio = base64_no_header_to_ndarray(prompt_audio, sample_rate)
        with NamedTemporaryFile(suffix=".wav") as f:
            save_audio(prompt_audio, f.name, sample_rate)
            f.flush()
            async for _ in pipeline.async_generate(
                "hello", prompt_text, f.name, None, prompt_id
            ):
                pass

    e = time.time()
    logger.info(f"Clone time: {e - s}")

    return CloneOutput(existed=False, prompt_id=prompt_id)


@dataclass
class TTSRequestBufferContent:
    audio_format: Union[None | str] = None
    sample_rate: Union[None | int] = None
    prompt_id: Union[None | str] = None
    instruct_text: Union[None | str] = None
    content: str = ""
    accu_start_time: float = time.time()
    wait_start_time: Union[None | float] = None
    req_done: bool = False
    tts_count: int = 0

    _lock: Lock = Lock()
    _can_be_removed: bool = False

    @property
    def finished(self):
        if self.req_done and self.size() == 0:
            self._can_be_removed = True
            return True
        return False

    def size(self):
        return len(self.content)

    def __setattr__(self, name, value):
        with self._lock:
            super().__setattr__(name, value)


class TTSRequestBuffer(defaultdict[str, TTSRequestBufferContent]):
    def __init__(self):
        super().__init__(lambda: TTSRequestBufferContent())

    def keys(self):
        return list(super().keys())

    def items(self):
        return list(super().items())

    def values(self):
        return list(super().values())

    def remove_finished(self):
        for k in self.keys():
            if self[k]._can_be_removed:
                self.pop(k)


@dataclass
class TTSFrontendConfig:
    #### unistream config
    max_buffer_size: int = 20
    max_accu_seconds: float = 0.2
    wait_timeout: float = 2.0
    # long sentence detection
    segment_aux_flag: str = "####"
    # short sentence detection
    do_short_sent_detection: bool = False
    short_sent_puncts: str = "；：，;:,"

    #### tts
    do_removing_silence: bool = False
    first_left_retention_seconds: float = 0.7
    left_retention_seconds: float = 1.0
    right_retention_seconds: float = 1.0


class TTSFrontend:

    def __init__(self, request_buffer: TTSRequestBuffer, websocket: WebSocket):
        self.req_buffer = request_buffer
        self.websocket = websocket

        self.tts_channel = asyncio.Queue()
        self.config = TTSFrontendConfig(**frontend_config)
        self.tts_sample_rate = pipeline.sample_rate
        self.tts_index = 0
        self.sent_segmenter = pysbd.Segmenter(language="zh", clean=False)

    async def _send(
        self,
        req_id: str,
        req_buffer: TTSRequestBufferContent,
        tts_text: Union[str | AsyncGenerator | None],
        finished: bool,
    ):
        if isinstance(tts_text, str) and len(tts_text) == 0:
            tts_text = None
        await self.tts_channel.put((req_id, req_buffer, tts_text, finished))

    async def _recv(
        self,
    ) -> Tuple[str, TTSRequestBufferContent, Union[str | AsyncGenerator | None], bool]:
        req_id, req_buffer, tts_text, finished = await self.tts_channel.get()
        return req_id, req_buffer, tts_text, finished

    @staticmethod
    def except_in_loop(interval: float = 0):
        def outer(func):
            async def _func(self: "TTSFrontend", *args, **kwargs):
                while True:
                    try:
                        await func(self, *args, **kwargs)
                        if interval > 0:
                            await asyncio.sleep(interval)
                    except Exception as e:
                        logger.info("Error: %s" % traceback.format_exc())
                        await self.websocket.send_json(
                            {"error": True, "message": whats_wrong_with(e)}
                        )

            return _func

        return outer

    @except_in_loop(0.005)
    async def _handle_text_unistream(self):
        self.req_buffer.remove_finished()

        for req_id in self.req_buffer.keys():
            _buffer = self.req_buffer[req_id]

            async def send_tts(sent="", finished=False):
                await self._send(req_id, _buffer, sent, finished)

            if _buffer.finished:
                await send_tts(finished=True)

            orig_size = _buffer.size()
            if orig_size == 0:
                continue

            # 累积一定文本才执行分句处理
            current_time = time.time()
            accu_time = current_time - _buffer.accu_start_time
            if accu_time > self.config.max_accu_seconds:
                _buffer.accu_start_time = current_time

                # 长句检测
                sents = self.sent_segmenter.segment(
                    _buffer.content + self.config.segment_aux_flag
                )
                if len(sents) > 1:
                    tts_sent = "".join(sents[:-1])
                    _buffer.content = _buffer.content[len(tts_sent) :]
                    await send_tts(tts_sent)

                if self.config.do_short_sent_detection:  # 短句检测
                    possible = True
                    while _buffer.size() > self.config.max_buffer_size and possible:
                        sents = re.split(
                            rf"[{self.config.short_sent_puncts}]", _buffer.content
                        )
                        if len(sents) > 1:
                            tts_sent = sents[0]
                            _buffer.content = _buffer.content[len(tts_sent) + 1 :]
                            await send_tts(tts_sent)
                        else:
                            possible = False

            curr_size = _buffer.size()
            if orig_size > curr_size:  # 存在分句，重置等待时间
                _buffer.wait_start_time = time.time()

            if curr_size == 0:
                continue

            # 等待超时处理（只有在无分句无新块期间才累积等待时间）
            if _buffer.wait_start_time is not None:
                waiting_time = time.time() - _buffer.wait_start_time
                if waiting_time > self.config.wait_timeout:
                    _buffer.wait_start_time = None
                    tts_sent = _buffer.content
                    _buffer.content = ""
                    await send_tts(tts_sent)

    @except_in_loop()
    async def _handle_tts(self):
        tts_packet = await self._recv()
        logger.info(f"TTS Params: {tts_packet}")
        req_id, req_buffer, tts_text, finished = tts_packet
        prompt_id = req_buffer.prompt_id
        audio_format = req_buffer.audio_format
        resample_rate = req_buffer.sample_rate
        instruct_text = req_buffer.instruct_text

        if tts_text is not None:
            async for chunk in repack(
                pipeline.async_generate(
                    tts_text, None, None, instruct_text, prompt_id, stream=True
                ),
                self.tts_sample_rate * 1,
            ):
                if self.config.do_removing_silence:
                    chunk = remove_silence(
                        chunk,
                        self.tts_sample_rate,
                        (
                            self.config.first_left_retention_seconds
                            if req_buffer.tts_count == 0
                            else self.config.left_retention_seconds
                        ),
                        self.config.right_retention_seconds,
                    )
                chunk = format_ndarray_to_base64(
                    chunk, self.tts_sample_rate, resample_rate, audio_format
                )
                await self.websocket.send_json(
                    TTSStreamOutput(
                        id=req_id,
                        is_end=False,
                        index=req_buffer.tts_count,
                        data=chunk,
                        audio_format=audio_format,
                        sample_rate=resample_rate,
                    ).model_dump()
                )
                req_buffer.tts_count += 1

        if finished:
            await self.websocket.send_json(
                TTSStreamOutput(
                    id=req_id, is_end=True, index=req_buffer.tts_count
                ).model_dump()
            )

    def run(self):
        self.text_task = asyncio.create_task(self._handle_text_unistream())
        self.tts_task = asyncio.create_task(self._handle_tts())

    def close(self):
        self.tts_task.cancel()
        self.text_task.cancel()


@app.websocket("/tts")
async def tts(websocket: WebSocket):
    await websocket.accept()

    req_buffer = TTSRequestBuffer()
    req_id = None
    tts_frontend = TTSFrontend(req_buffer, websocket)
    tts_frontend.run()

    while True:
        try:
            req = await websocket.receive_json()
            logger.info("Request Params: %s" % req)

            if req_id is None and "req_params" in req:
                req = TTSStreamRequestInput(**req)
                _prompt_id = req.req_params.prompt_id
                _audio_format = req.req_params.audio_format
                _sample_rate = req.req_params.sample_rate
                _instruct_text = req.req_params.instruct_text

                if _prompt_id not in pipeline.get_speakers():
                    raise ValueError("No such speaker.")

                req_id = uuid.uuid4().hex

                req_buffer[req_id].prompt_id = _prompt_id
                req_buffer[req_id].audio_format = _audio_format
                req_buffer[req_id].sample_rate = _sample_rate

                if _instruct_text is not None and len(_instruct_text) > 0:
                    req_buffer[req_id].instruct_text = _instruct_text

            elif req_id is not None:
                req = TTSStreamTextInput(**req)
                text, req_done = req.text.strip(), req.done
                if len(text) > 0:
                    req_buffer[req_id].wait_start_time = time.time()
                    req_buffer[req_id].content += text
                if req_done:
                    req_buffer[req_id].req_done = True
                    req_id = None

            else:
                raise ValueError("Invalid request.")

        except Exception as e:
            if isinstance(e, WebSocketDisconnect):
                tts_frontend.close()
                break
            logger.info("Error: %s" % traceback.format_exc())
            await websocket.send_json({"error": True, "message": whats_wrong_with(e)})
