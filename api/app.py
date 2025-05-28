import sys
from utils import path_to_root

sys.path.insert(0, path_to_root())

import os
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
from dataclasses import dataclass, field
from typing import Union
import yaml
import torch.distributed as dist

from bases import *
from utils import *
from CosyVoice.model import Pipeline as TTSModel
from seg2stream import (
    SegSent2GeneratorPipeline,
    SegSent2StreamPipeline,
    get_sentence_segmenter,
    get_phrase_segmenter,
)


class Global:
    tts_model: Union[None | TTSModel] = None
    config: Union[None | dict] = None
    sentence_segmenter = get_sentence_segmenter("jionlp")
    # phrase_segmenter = get_phrase_segmenter("regex")
    phrase_segmenter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 加载 config
    config_path = os.environ["CONFIG_PATH"]
    with open(config_path, "r") as f:
        Global.config = yaml.safe_load(f)
    logger.info(f"Successfully load config: {Global.config}")

    # 加载 TTS 模型
    tts_model_dir = os.environ["TTS_MODEL_DIR"]
    Global.tts_model = TTSModel(model_dir=tts_model_dir, load_trt=True, fp16=True)
    logger.info("TTS model is loaded successfully.")

    # 加载音色缓存
    speaker_cache_path = os.getenv(
        "DEFAULT_SPEAKER_CACHE_PATH", path_to_root("assets", "default_speaker_cache.pt")
    )
    if os.path.exists(speaker_cache_path):
        speaker_ids = Global.tts_model.load_cache(speaker_cache_path)
        logger.info(f"Successfully load speakers: {speaker_ids}")

    yield

    dist.destroy_process_group()


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
    return Global.tts_model.get_speakers()


@app.post("/remove")
async def remove_speakers(req: RemoveSpeakersInput) -> dict:
    removed = Global.tts_model.remove_speakers(req.prompt_ids)
    return {"removed_speakers": removed}


@app.post("/cache/save")
async def save_cache(req: SaveCacheInput) -> dict:
    cache_path = Global.tts_model.save_cache(
        req.cache_dir, req.filename, req.prompt_ids
    )
    return {"cache_path": cache_path}


@app.post("/cache/load")
async def load_cache(req: LoadCacheInput) -> dict:
    loaded_speaker_ids = Global.tts_model.load_cache(req.cache_path, req.prompt_ids)
    return {"loaded_speakers": loaded_speaker_ids}


@app.post("/clone")
async def clone(req: CloneInput) -> CloneOutput:
    logger.info("Request Params: %s" % truncate_long_str(req))

    prompt_id = req.prompt_id
    prompt_text = req.prompt_text
    prompt_audio = req.prompt_audio
    loudness = float(req.loudness)
    sample_rate = req.sample_rate
    audio_format = req.audio_format

    if prompt_id is None:
        prompt_id = f"{uuid.uuid4().hex[:7]}_{datetime.now().strftime('%Y-%m-%d')}"
    if prompt_id in Global.tts_model.get_speakers():
        return CloneOutput(existed=True, prompt_id=prompt_id)

    if len(prompt_text.strip()) == 0:
        prompt_text = None

    s = time.time()
    test_text = "这是一段测试文本。"

    if os.path.exists(prompt_audio):
        async for _ in Global.tts_model.async_generate(
            test_text, prompt_text, prompt_audio, None, prompt_id, loudness
        ):
            pass
    elif prompt_audio.startswith("http"):
        prompt_audio = requests.get(prompt_audio).content
        with NamedTemporaryFile() as f:
            f.write(prompt_audio)
            f.flush()
            async for _ in Global.tts_model.async_generate(
                test_text, prompt_text, f.name, None, prompt_id, loudness
            ):
                pass
    else:
        prompt_audio = any_format_to_ndarray(prompt_audio, audio_format, sample_rate)
        with NamedTemporaryFile(suffix=".wav") as f:
            save_audio(prompt_audio, f.name, sample_rate)
            f.flush()
            async for _ in Global.tts_model.async_generate(
                test_text, prompt_text, f.name, None, prompt_id, loudness
            ):
                pass

    e = time.time()
    logger.info(f"Clone time: {e - s}")

    return CloneOutput(existed=False, prompt_id=prompt_id)


@dataclass
class TTSInfo:
    request: TTSStreamRequestParameters
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    date: datetime = field(default_factory=datetime.now)
    done: bool = False
    count: int = 0


class TTSPipeline:
    def __init__(self, websocket: WebSocket):
        match Global.config["segmentation"]["mode"]:
            case "bistream":
                bistream_config = Global.config["segmentation"]["bistream"]
                self.seg_pipeline = SegSent2GeneratorPipeline(
                    sentence_segmenter=Global.sentence_segmenter,
                    phrase_segmenter=Global.phrase_segmenter,
                    segment_aux_suffix=Global.config["segmentation"]["seg_suffix"],
                    ################
                    max_waiting_time=bistream_config["max_waiting_time"],
                    max_stream_time=bistream_config["max_stream_time"],
                    first_min_seg_size=bistream_config["first_min_seg_size"],
                    min_seg_size=bistream_config["min_seg_size"],
                )
            case "unistream":
                unistream_config = Global.config["segmentation"]["unistream"]
                self.seg_pipeline = SegSent2StreamPipeline(
                    sentence_segmenter=Global.sentence_segmenter,
                    phrase_segmenter=Global.phrase_segmenter,
                    segment_aux_suffix=Global.config["segmentation"]["seg_suffix"],
                    ################
                    first_max_accu_time=unistream_config["first_max_accu_time"],
                    max_accu_time=unistream_config["max_accu_time"],
                    first_max_buffer_size=unistream_config["first_max_buffer_size"],
                    max_buffer_size=unistream_config["max_buffer_size"],
                    max_waiting_time=unistream_config["max_waiting_time"],
                    max_stream_time=unistream_config["max_stream_time"],
                    first_min_seg_size=unistream_config["first_min_seg_size"],
                    min_seg_size=unistream_config["min_seg_size"],
                    max_seg_size=unistream_config["max_seg_size"],
                    loose_steps=unistream_config["loose_steps"],
                    loose_size=unistream_config["loose_size"],
                    fade_in_out_time=unistream_config["fade_in_out_time"],
                    seconds_per_word=unistream_config["seconds_per_word"],
                )
        self.websocket = websocket

    async def __recv__(self):
        try:
            while True:
                req = await self.websocket.receive_json()
                req = TTSStreamTextInput(**req)
                logger.info("TTS Input: %s" % req)
                self.seg_pipeline.add_text(req.text)
                if req.done:
                    self.seg_pipeline.add_text(None)
                    break
        finally:
            self.seg_pipeline.add_text(None)

    async def __tts__(self):
        async for tts_text in self.seg_pipeline.get_out_stream():
            logger.info(
                "TTS Result:\n"
                f"--> ID: {self.tts_info.id}\n"
                f"--> Done: {self.tts_info.done}\n"
                f"--> Text: {tts_text}\n"
                f"--> Req: {self.tts_info.request}"
            )

            async for chunk_ndarray in repack(
                Global.tts_model.async_generate(
                    tts_text,
                    None,
                    None,
                    self.tts_info.request.instruct_text,
                    self.tts_info.request.prompt_id,
                    do_split_text=False,
                    stream=True,
                ),
                Global.tts_model.sample_rate * 1,
            ):
                if Global.config["tts"]["do_removing_silence"]:
                    chunk_ndarray = remove_silence(
                        chunk_ndarray,
                        Global.tts_model.sample_rate,
                        (
                            Global.config["tts"]["first_left_retention_seconds"]
                            if self.tts_info.count == 0
                            else Global.config["tts"]["left_retention_seconds"]
                        ),
                        Global.config["tts"]["right_retention_seconds"],
                    )
                chunk_base64 = format_ndarray_to_base64(
                    chunk_ndarray,
                    Global.tts_model.sample_rate,
                    self.tts_info.request.sample_rate,
                    self.tts_info.request.audio_format,
                )
                await self.websocket.send_json(
                    TTSStreamOutput(
                        id=self.tts_info.id,
                        is_end=False,
                        index=self.tts_info.count,
                        data=chunk_base64,
                        audio_format=self.tts_info.request.audio_format,
                        sample_rate=self.tts_info.request.sample_rate,
                    ).model_dump()
                )
                self.tts_info.count += 1

        await self.websocket.send_json(
            TTSStreamOutput(
                id=self.tts_info.id, is_end=True, index=self.tts_info.count
            ).model_dump()
        )

    async def start(self, tts_info: TTSInfo):
        self.seg_pipeline.reset_status()
        self.tts_info = tts_info
        await asyncio.gather(
            self.__tts__(),
            self.seg_pipeline.segment(),
            self.__recv__(),
        )

    async def close(self):
        pass


@app.websocket("/tts")
async def tts(websocket: WebSocket):
    await websocket.accept()

    tts_pipeline = TTSPipeline(websocket)

    while True:
        try:
            req = await websocket.receive_json()
            if "req_params" in req:
                req = TTSStreamRequestInput(**req)
                if req.req_params.prompt_id not in Global.tts_model.get_speakers():
                    raise ValueError("No such speaker.")
                if (
                    req.req_params.instruct_text is not None
                    and len(req.req_params.instruct_text) == 0
                ):
                    req.req_params.instruct_text = None

                tts_info = TTSInfo(request=req.req_params)
                logger.info(f"TTS Request: {tts_info}")
                await tts_pipeline.start(tts_info)
                await tts_pipeline.close()
            else:
                raise ValueError("Invalid request.")

        except Exception as e:
            if isinstance(e, WebSocketDisconnect):
                break
            logger.info(f"TTS Error: \n{traceback.format_exc()}")
            await websocket.send_json({"error": True, "message": whats_wrong_with(e)})
