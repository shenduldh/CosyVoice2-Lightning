from typing import Iterable, List, Optional, Tuple, Union
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import maybe_prefix, AutoWeightsLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class CosyVoice2LLM(nn.Module, SupportsLoRA, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.lora_config = vllm_config.lora_config
        self.quant_config = vllm_config.quant_config

        self.hidden_size = 896
        self.num_speech_tokens = 6561 + 3
        self.num_text_tokens = 151936
        self.num_task_tokens = 2
        self.num_zero_tokens = 1

        vllm_config.model_config.hf_config.vocab_size = self.num_text_tokens
        self.model = Qwen2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.logits_processor = LogitsProcessor(self.num_speech_tokens)
        self.sampler = get_sampler()

        self.llm_decoder = ParallelLMHead(
            self.num_speech_tokens,
            self.hidden_size,
            bias=True,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "llm_decoder"),
        )
        self.speech_embedding = torch.nn.Embedding(
            self.num_speech_tokens, self.hidden_size
        )
        self.llm_embedding = torch.nn.Embedding(self.num_task_tokens, self.hidden_size)
        self.mixed_embeddings = torch.nn.Embedding(
            self.num_speech_tokens
            + self.num_text_tokens
            + self.num_task_tokens
            + self.num_zero_tokens,
            self.hidden_size,
        )

        self.inputs_embeds_buffer = torch.zeros(
            (vllm_config.scheduler_config.max_num_batched_tokens, self.hidden_size),
            dtype=self.speech_embedding.weight.dtype,
            device=self.speech_embedding.weight.device,
        )

    def forward(
        self, input_ids: torch.Tensor, *args, **kwargs
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if kwargs["inputs_embeds"] is None:
            orig_shape = input_ids.shape
            flattened_input_ids = input_ids.view(-1)
            inputs_embeds = self.inputs_embeds_buffer[: flattened_input_ids.shape[0]]
            inputs_embeds[:] = self.mixed_embeddings(flattened_input_ids)
            inputs_embeds = inputs_embeds.view(*orig_shape, self.hidden_size)
            kwargs["inputs_embeds"] = inputs_embeds
        return self.model(input_ids, *args, **kwargs)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(
            self.llm_decoder, hidden_states, sampling_metadata
        )
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def convert_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        for name, loaded_weight in weights:
            if name.startswith(
                (
                    "llm_embedding.",
                    "llm.model.model.",
                    # "llm.model.lm_head.",
                    "llm_decoder.",
                    "speech_embedding.",
                )
            ):
                if name.startswith("llm.model.model."):
                    name = name.replace("llm.model.model.", "model.")
                # if name.startswith("llm.model.lm_head."):
                #     name = name.replace("llm.model.lm_head.", "lm_head.")
                yield name, loaded_weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = self.convert_weights(weights)
        AutoWeightsLoader(self).load_weights(weights)

        device = self.speech_embedding.weight.device
        dtype = self.speech_embedding.weight.dtype
        total_text_token_ids = torch.arange(self.num_text_tokens, device=device)
        speech_embeds = self.speech_embedding.weight
        text_embeds = self.model.get_input_embeddings(total_text_token_ids)
        task_embeds = self.llm_embedding.weight
        zero_embeds = torch.zeros(
            (self.num_zero_tokens, self.hidden_size), dtype=dtype, device=device
        )
        concatenated = torch.cat([speech_embeds, text_embeds, task_embeds, zero_embeds])
        default_weight_loader(self.mixed_embeddings.weight, concatenated)
