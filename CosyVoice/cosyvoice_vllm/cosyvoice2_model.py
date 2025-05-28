import os
from typing import Generator, AsyncGenerator
import torch
import numpy as np
import threading
import asyncio
import time
from torch.nn import functional as F
from concurrent.futures import ThreadPoolExecutor
import uuid

from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
from cosyvoice.hifigan.generator import HiFTGenerator
from cosyvoice.utils.common import fade_in_out

from .llm_model import CosyVoice2LLM
from .utils import set_flow_decoder, stream_context, convert_onnx_to_trt


class CosyVoice2Model:
    def __init__(
        self,
        llm_dir: str,
        flow: CausalMaskedDiffWithXvec,
        hift: HiFTGenerator,
        fp16: bool,
        device: torch.device = None,
    ):
        self.cuda_available = torch.cuda.is_available()
        if device is None:
            self.device = (
                torch.device("cuda") if self.cuda_available else torch.device("cpu")
            )
        else:
            self.device = device

        self.llm = CosyVoice2LLM(llm_dir)
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.flow.fp16 = fp16
        if self.fp16:
            self.flow.half()

        self.token_hop_len = 2 * self.flow.input_frame_rate
        # here we fix flow encoder/decoder decoding_chunk_size, in the future we will send it as arguments, or use cache
        self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = (
            2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        )
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)

        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

        self.cache_lock = threading.Lock()
        self.async_cache_lock = asyncio.Lock()
        self.thread_executor = ThreadPoolExecutor(max_workers=10)

    def load(self, flow_model, hift_model):
        self.flow.load_state_dict(
            torch.load(flow_model, weights_only=True, map_location=self.device),
            strict=True,
        )
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {
            k.replace("generator.", ""): v
            for k, v in torch.load(
                hift_model, weights_only=True, map_location=self.device
            ).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(
        self,
        flow_decoder_estimator_model,
        flow_decoder_onnx_model,
        fp16,
        max_workspace_size=8,
        estimator_count=1,
    ):
        assert self.cuda_available, "tensorrt only supports gpu!"
        if not os.path.exists(flow_decoder_estimator_model):
            convert_onnx_to_trt(
                flow_decoder_estimator_model,
                flow_decoder_onnx_model,
                fp16,
                max_workspace_size,
            )
        if os.path.getsize(flow_decoder_estimator_model) == 0:
            raise ValueError(
                "{} is empty file, delete it and export again!".format(
                    flow_decoder_estimator_model
                )
            )
        del self.flow.decoder.estimator
        import tensorrt as trt

        with open(flow_decoder_estimator_model, "rb") as f:
            self.flow.decoder.estimator_engine = trt.Runtime(
                trt.Logger(trt.Logger.INFO)
            ).deserialize_cuda_engine(f.read())
        if self.flow.decoder.estimator_engine is None:
            raise ValueError(
                "failed to load trt {}".format(flow_decoder_estimator_model)
            )
        set_flow_decoder(self.flow.decoder, estimator_count)

    @stream_context()
    def token2wav(
        self,
        token,
        prompt_token,
        prompt_feat,
        embedding,
        uuid,
        token_offset,
        finalize=False,
        speed=1.0,
    ):
        with torch.amp.autocast("cuda", torch.float16 if self.fp16 else torch.float):
            tts_mel, _ = self.flow.inference(
                token=token.to(self.device),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(
                    self.device
                ),
                prompt_token=prompt_token.to(self.device),
                prompt_token_len=torch.tensor(
                    [prompt_token.shape[1]], dtype=torch.int32
                ).to(self.device),
                prompt_feat=prompt_feat.to(self.device),
                prompt_feat_len=torch.tensor(
                    [prompt_feat.shape[1]], dtype=torch.int32
                ).to(self.device),
                embedding=embedding.to(self.device),
                finalize=finalize,
            )
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio :]

        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = (
                self.hift_cache_dict[uuid]["mel"],
                self.hift_cache_dict[uuid]["source"],
            )
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)

        # keep overlap mel and hift cache
        if not finalize:
            tts_speech, tts_source = self.hift.inference(
                speech_feat=tts_mel, cache_source=hift_cache_source
            )
            if self.hift_cache_dict[uuid] is not None:
                # 当前开头和缓存结尾用汉明窗重叠部分
                tts_speech = fade_in_out(
                    tts_speech, self.hift_cache_dict[uuid]["speech"], self.speech_window
                )
            self.hift_cache_dict[uuid] = {
                "mel": tts_mel[:, :, -self.mel_cache_len :],
                "source": tts_source[:, :, -self.source_cache_len :],
                "speech": tts_speech[:, -self.source_cache_len :],
            }
            tts_speech = tts_speech[:, : -self.source_cache_len]
        else:
            if speed != 1.0:
                assert (
                    self.hift_cache_dict[uuid] is None
                ), "speed change only support non-stream inference mode"
                tts_mel = F.interpolate(
                    tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear"
                )
            tts_speech, tts_source = self.hift.inference(
                speech_feat=tts_mel, cache_source=hift_cache_source
            )
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(
                    tts_speech, self.hift_cache_dict[uuid]["speech"], self.speech_window
                )
        return tts_speech

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, uuid):
        if isinstance(text, Generator):
            for i in self.llm.inference_bistream(
                text=text,
                prompt_text=prompt_text.to(self.device),
                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                request_id=uuid,
            ):
                self.tts_speech_token_dict[uuid].append(i)
        else:
            for i in self.llm.inference(
                text=text.to(self.device),
                prompt_text=prompt_text.to(self.device),
                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                request_id=uuid,
            ):
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def tts(
        self,
        text,
        flow_embedding,
        prompt_text=torch.zeros(1, 0, dtype=torch.int32),
        llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat=torch.zeros(1, 0, 80),
        stream=False,
        speed=1.0,
        **kwargs
    ):
        this_uuid = str(uuid.uuid1())

        with self.cache_lock:
            self.tts_speech_token_dict[this_uuid] = []
            self.llm_end_dict[this_uuid] = False
            self.hift_cache_dict[this_uuid] = None

        llm_task = threading.Thread(
            target=self.llm_job,
            args=(text, prompt_text, llm_prompt_speech_token, this_uuid),
        )
        llm_task.start()

        if stream:
            token_offset = 0
            while True:
                if (
                    len(self.tts_speech_token_dict[this_uuid]) - token_offset
                    >= self.token_hop_len + self.flow.pre_lookahead_len
                ):
                    this_tts_speech_token = torch.tensor(
                        self.tts_speech_token_dict[this_uuid][
                            : token_offset
                            + self.token_hop_len
                            + self.flow.pre_lookahead_len
                        ]
                    ).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(
                        token=this_tts_speech_token,
                        prompt_token=flow_prompt_speech_token,
                        prompt_feat=prompt_speech_feat,
                        embedding=flow_embedding,
                        uuid=this_uuid,
                        token_offset=token_offset,
                        finalize=False,
                        speed=speed,
                    )
                    token_offset += self.token_hop_len
                    yield this_tts_speech.cpu()
                if (
                    self.llm_end_dict[this_uuid] is True
                    and len(self.tts_speech_token_dict[this_uuid]) - token_offset
                    < self.token_hop_len + self.flow.pre_lookahead_len
                ):
                    break
                time.sleep(0.01)

            llm_task.join()
            # deal with remain tokens, make sure inference remain token len
            # equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(
                self.tts_speech_token_dict[this_uuid]
            ).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                uuid=this_uuid,
                token_offset=token_offset,
                finalize=True,
                speed=speed,
            )
            yield this_tts_speech.cpu()

        else:
            # deal with all tokens
            llm_task.join()
            this_tts_speech_token = torch.tensor(
                self.tts_speech_token_dict[this_uuid]
            ).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                uuid=this_uuid,
                token_offset=0,
                finalize=True,
                speed=speed,
            )
            yield this_tts_speech.cpu()

        with self.cache_lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)

        torch.cuda.empty_cache()

    async def async_llm_job(self, text, prompt_text, llm_prompt_speech_token, uuid):
        if isinstance(text, AsyncGenerator):
            async for i in self.llm.async_inference_bistream(
                text=text,
                prompt_text=prompt_text.to(self.device),
                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                request_id=uuid,
            ):
                self.tts_speech_token_dict[uuid].append(i)
        else:
            async for i in self.llm.async_inference(
                text=text.to(self.device),
                prompt_text=prompt_text.to(self.device),
                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                request_id=uuid,
            ):
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    async def async_tts(
        self,
        text,
        flow_embedding,
        prompt_text=torch.zeros(1, 0, dtype=torch.int32),
        llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat=torch.zeros(1, 0, 80),
        stream=False,
        speed=1.0,
        **kwargs
    ):
        this_uuid = str(uuid.uuid1())

        async with self.async_cache_lock:
            self.tts_speech_token_dict[this_uuid] = []
            self.llm_end_dict[this_uuid] = False
            self.hift_cache_dict[this_uuid] = None

        llm_task = asyncio.create_task(
            self.async_llm_job(text, prompt_text, llm_prompt_speech_token, this_uuid)
        )

        loop = asyncio.get_event_loop()

        if stream:
            token_offset = 0
            while True:
                if (
                    len(self.tts_speech_token_dict[this_uuid]) - token_offset
                    >= self.token_hop_len + self.flow.pre_lookahead_len
                ):
                    this_tts_speech_token = torch.tensor(
                        self.tts_speech_token_dict[this_uuid][
                            : token_offset
                            + self.token_hop_len
                            + self.flow.pre_lookahead_len
                        ]
                    ).unsqueeze(dim=0)
                    this_tts_speech = await loop.run_in_executor(
                        self.thread_executor,
                        self.token2wav,
                        this_tts_speech_token,
                        flow_prompt_speech_token,
                        prompt_speech_feat,
                        flow_embedding,
                        this_uuid,
                        token_offset,
                        False,
                        speed,
                    )
                    token_offset += self.token_hop_len
                    yield this_tts_speech.cpu()
                if (
                    self.llm_end_dict[this_uuid] is True
                    and len(self.tts_speech_token_dict[this_uuid]) - token_offset
                    < self.token_hop_len + self.flow.pre_lookahead_len
                ):
                    break
                await asyncio.sleep(0.01)

            await llm_task
            # deal with remain tokens, make sure inference remain token len
            # equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(
                self.tts_speech_token_dict[this_uuid]
            ).unsqueeze(dim=0)
            this_tts_speech = await loop.run_in_executor(
                self.thread_executor,
                self.token2wav,
                this_tts_speech_token,
                flow_prompt_speech_token,
                prompt_speech_feat,
                flow_embedding,
                this_uuid,
                token_offset,
                True,
                speed,
            )
            yield this_tts_speech.cpu()

        else:
            # deal with all tokens
            await llm_task
            this_tts_speech_token = torch.tensor(
                self.tts_speech_token_dict[this_uuid]
            ).unsqueeze(dim=0)
            this_tts_speech = await loop.run_in_executor(
                self.thread_executor,
                self.token2wav,
                this_tts_speech_token,
                flow_prompt_speech_token,
                prompt_speech_feat,
                flow_embedding,
                this_uuid,
                0,
                True,
                speed,
            )
            yield this_tts_speech.cpu()

        async with self.async_cache_lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)

        torch.cuda.empty_cache()
