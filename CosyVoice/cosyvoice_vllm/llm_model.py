import os

# use vllm V1 version
os.environ["VLLM_USE_V1"] = "1"

import time
import queue
import asyncio
import threading
from typing import List, Generator, AsyncGenerator
import torch

from vllm import AsyncLLMEngine, ModelRegistry
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from .vllm_registration import CosyVoice2VLLMEngine

from .config import ENGINE_ARGS, SAMPLING_PARAMS


ModelRegistry.register_model("CosyVoice2Model", CosyVoice2VLLMEngine)


def tensor_to_list(tensor: torch.tensor):
    return tensor.view(-1).cpu().numpy().tolist()


class CosyVoice2LLM:
    def __init__(
        self,
        model_dir,
        mix_ratio: List[int] = [5, 15],
    ):
        self.mix_ratio = mix_ratio

        engine_args = AsyncEngineArgs(model=model_dir, **ENGINE_ARGS)
        self.llm_engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(engine_args)

        self.speech_token_size = 6564  # 6561 + 3
        self.llm_token_size = 151936  # llm vocab_size
        self.sos_eos_token_id = self.speech_token_size + self.llm_token_size + 1
        self.task_token_id = self.sos_eos_token_id + 1
        self.zero_token_id = self.task_token_id + 1

        # vllm inference task needs to be in a fixed event loop
        # start a background thread dedicated to the inference task
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def async_vllm_generate(
        self, prompt_token_ids, request_id, stop_token_ids, max_tokens=None
    ):
        sampling_params = SamplingParams(**SAMPLING_PARAMS)
        sampling_params.stop_token_ids = stop_token_ids
        if max_tokens:
            sampling_params.max_tokens = max_tokens
        async for output in self.llm_engine.generate(
            {"prompt_token_ids": prompt_token_ids},
            sampling_params=sampling_params,
            request_id=request_id or f"{time.time()}",
        ):
            yield output.outputs[0]

    async def async_queue_vllm_generate(
        self, out_queue, prompt_token_ids, request_id, stop_token_ids, max_tokens=None
    ):
        sampling_params = SamplingParams(**SAMPLING_PARAMS)
        sampling_params.stop_token_ids = stop_token_ids
        if max_tokens:
            sampling_params.max_tokens = max_tokens
        async for output in self.llm_engine.generate(
            {"prompt_token_ids": prompt_token_ids},
            sampling_params=sampling_params,
            request_id=request_id or f"{time.time()}",
        ):
            out_queue.put((output.outputs[0], output.finished))

    def vllm_generate(
        self,
        prompt_token_ids: List[int],
        request_id: str = None,
        stop_token_ids=None,
        max_tokens=None,
    ):
        out_queue = queue.Queue()
        asyncio.run_coroutine_threadsafe(
            self.async_queue_vllm_generate(
                out_queue, prompt_token_ids, request_id, stop_token_ids, max_tokens
            ),
            self.loop,
        )
        finished = False
        while not finished:
            (output, finished) = (
                out_queue.get_nowait() if not out_queue.empty() else out_queue.get()
            )
            yield output

    def inference(
        self,
        text: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        request_id=None,
    ) -> Generator[int, None, None]:
        """Only streaming output"""
        prompt_text = tensor_to_list(prompt_text + torch.tensor(6564))
        prompt_speech_token = tensor_to_list(prompt_speech_token)

        text = tensor_to_list(text + torch.tensor(6564))
        prompt_token_ids = (
            [self.sos_eos_token_id]
            + prompt_text
            + text
            + [self.task_token_id]
            + prompt_speech_token
        )
        max_tokens = len(text) * 20
        for output in self.vllm_generate(
            prompt_token_ids,
            request_id=request_id,
            stop_token_ids=[6561],
            max_tokens=max_tokens,
        ):
            if output.token_ids[-1] == 6561:
                need_add_tokens = output.token_ids[:-1]
            else:
                need_add_tokens = output.token_ids
            for token in need_add_tokens:
                yield token

    def inference_bistream(
        self,
        text: Generator[torch.Tensor, None, None],
        prompt_text: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        request_id=None,
    ) -> Generator[int, None, None]:
        """Supports streaming input and streaming output"""
        prompt_text = tensor_to_list(prompt_text + torch.tensor(6564))
        prompt_speech_token = tensor_to_list(prompt_speech_token)

        last_tokens = []
        prompt_token_ids = [self.sos_eos_token_id]
        text_tokens_cache = prompt_text
        for this_text in text:
            this_text = tensor_to_list(this_text + torch.tensor(6564))
            text_tokens_cache += this_text
            while len(prompt_speech_token) != 0:
                if len(text_tokens_cache) >= self.mix_ratio[0]:
                    text_input_token = text_tokens_cache[: self.mix_ratio[0]]
                    speech_input_token = prompt_speech_token[: self.mix_ratio[1]]
                    prompt_token_ids += text_input_token + speech_input_token
                    # reset the last cache
                    text_tokens_cache = text_tokens_cache[self.mix_ratio[0] :]
                    prompt_speech_token = prompt_speech_token[self.mix_ratio[1] :]
                else:
                    break

            if len(prompt_speech_token) == 0:
                if (len(last_tokens) > 0 and last_tokens[-1] == 6563) or len(
                    prompt_token_ids
                ) == 1:
                    if len(text_tokens_cache) >= self.mix_ratio[0]:
                        text_tokens_temp = text_tokens_cache[: self.mix_ratio[0]]
                        prompt_token_ids += text_tokens_temp
                        text_tokens_cache = text_tokens_cache[self.mix_ratio[0] :]
                    else:
                        continue

                for output in self.vllm_generate(
                    prompt_token_ids, request_id=request_id, stop_token_ids=[6563]
                ):
                    last_tokens = output.token_ids
                    if last_tokens[-1] == 6563:
                        need_add_tokens = last_tokens[:-1]
                    else:
                        need_add_tokens = last_tokens
                    for token in need_add_tokens:
                        yield token
                    prompt_token_ids.extend(need_add_tokens)

        prompt_token_ids += text_tokens_cache + [self.task_token_id]
        for output in self.vllm_generate(
            prompt_token_ids, request_id=request_id, stop_token_ids=[6561]
        ):
            if output.token_ids[-1] == 6561:
                need_add_tokens = output.token_ids[:-1]
            else:
                need_add_tokens = output.token_ids
            for token in need_add_tokens:
                yield token

    async def async_inference(
        self,
        text: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        request_id=None,
    ) -> AsyncGenerator[int, None]:
        """Only streaming output"""
        prompt_text = tensor_to_list(prompt_text + torch.tensor(6564))
        prompt_speech_token = tensor_to_list(prompt_speech_token)

        text = tensor_to_list(text + torch.tensor(6564))
        prompt_token_ids = (
            [self.sos_eos_token_id]
            + prompt_text
            + text
            + [self.task_token_id]
            + prompt_speech_token
        )
        max_tokens = len(text) * 20
        async for output in self.async_vllm_generate(
            prompt_token_ids,
            request_id=request_id,
            stop_token_ids=[6561],
            max_tokens=max_tokens,
        ):
            if output.token_ids[-1] == 6561:
                need_add_tokens = output.token_ids[:-1]
            else:
                need_add_tokens = output.token_ids
            for token in need_add_tokens:
                yield token

    async def async_inference_bistream(
        self,
        text: AsyncGenerator[torch.Tensor, None],
        prompt_text: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        request_id=None,
    ) -> AsyncGenerator[int, None]:
        """Supports streaming input and streaming output"""
        prompt_text = tensor_to_list(prompt_text + torch.tensor(6564))
        prompt_speech_token = tensor_to_list(prompt_speech_token)

        last_tokens = []
        prompt_token_ids = [self.sos_eos_token_id]
        text_tokens_cache = prompt_text
        async for this_text in text:
            this_text = tensor_to_list(this_text + torch.tensor(6564))
            text_tokens_cache += this_text
            while len(prompt_speech_token) != 0:
                if len(text_tokens_cache) >= self.mix_ratio[0]:
                    text_input_token = text_tokens_cache[: self.mix_ratio[0]]
                    speech_input_token = prompt_speech_token[: self.mix_ratio[1]]
                    prompt_token_ids += text_input_token + speech_input_token
                    # reset the last cache
                    text_tokens_cache = text_tokens_cache[self.mix_ratio[0] :]
                    prompt_speech_token = prompt_speech_token[self.mix_ratio[1] :]
                else:
                    break

            if len(prompt_speech_token) == 0:
                if (len(last_tokens) > 0 and last_tokens[-1] == 6563) or len(
                    prompt_token_ids
                ) == 1:
                    if len(text_tokens_cache) >= self.mix_ratio[0]:
                        text_tokens_temp = text_tokens_cache[: self.mix_ratio[0]]
                        prompt_token_ids += text_tokens_temp
                        text_tokens_cache = text_tokens_cache[self.mix_ratio[0] :]
                    else:
                        continue

                async for output in self.async_vllm_generate(
                    prompt_token_ids, request_id=request_id, stop_token_ids=[6563]
                ):
                    last_tokens = output.token_ids
                    if last_tokens[-1] == 6563:
                        need_add_tokens = last_tokens[:-1]
                    else:
                        need_add_tokens = last_tokens
                    for token in need_add_tokens:
                        yield token
                    prompt_token_ids.extend(need_add_tokens)

        prompt_token_ids += text_tokens_cache + [self.task_token_id]
        async for output in self.async_vllm_generate(
            prompt_token_ids, request_id=request_id, stop_token_ids=[6561]
        ):
            if output.token_ids[-1] == 6561:
                need_add_tokens = output.token_ids[:-1]
            else:
                need_add_tokens = output.token_ids
            for token in need_add_tokens:
                yield token
