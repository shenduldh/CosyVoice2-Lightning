"""Inference-only Qwen2 model compatible with HuggingFace weights."""

from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from typing_extensions import TypeVar
from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.multimodal.inputs import NestedTensors
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    maybe_prefix,
    merge_multimodal_embeddings,
)


T = TypeVar("T", default="NestedTensors")
logger = init_logger(__name__)
IGNORE_ID = -1


class CosyVoice2VLLMEngine(nn.Module, SupportsLoRA, SupportsPP):

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.llm_input_size = 896
        self.llm_output_size = 896

        self.speech_token_size = 6561 + 3
        self.llm_token_size = config.vocab_size

        # 1. build speech token
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2

        self.allow_patterns_overrides = ["llm.*"]
        self.llm_embedding = torch.nn.Embedding(2, self.llm_input_size)
        self.model = Qwen2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        # self.llm_decoder = nn.Linear(self.llm_output_size, self.speech_token_size)
        self.llm_decoder = ParallelLMHead(
            self.speech_token_size,
            self.llm_output_size,
            bias=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "llm_decoder"),
        )
        self.logits_processor = LogitsProcessor(self.speech_token_size)

        # length_normalized_loss: bool = True,
        # lsm_weight: float = 0.0,
        # self.criterion_ce = LabelSmoothingLoss(
        #     size=self.speech_token_size,
        #     padding_idx=IGNORE_ID,
        #     smoothing=lsm_weight,
        #     normalize_length=length_normalized_loss,
        # )

        # 2. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(
            self.speech_token_size, self.llm_input_size
        )

        # 3. sampling method: use vllm sampling method
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.mix_ratio: List[int] = [5, 15]

        # 4. define token constants
        self.llm_token_id_delta = torch.tensor(
            self.speech_token_size, dtype=torch.int32
        )
        self.sos_eos_token_id = torch.tensor(
            (self.llm_token_id_delta + self.llm_token_size + 1), dtype=torch.int32
        )  # 163840 + 6564 = 170404
        self.task_token_id = self.sos_eos_token_id + torch.tensor(
            1, dtype=torch.int32
        )  # 170405
        self.zero_token_id = self.task_token_id + torch.tensor(1, dtype=torch.int32)

        self.zero_embed_buffer = torch.zeros(
            (vllm_config.scheduler_config.max_num_seqs, self.llm_input_size),
            dtype=self.llm_embedding.weight.dtype,
            device=self.llm_embedding.weight.device,
        )
        self.inputs_embed_buffer = torch.zeros(
            (vllm_config.scheduler_config.max_num_batched_tokens, self.llm_input_size),
            dtype=self.llm_embedding.weight.dtype,
            device=self.llm_embedding.weight.device,
        )

    def get_sos_eos_emb(self):
        return self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)

    def get_task_id_emb(self):
        return self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[T] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> torch.Tensor:
        """
        Returns the input embeddings merged from the text embeddings from
        input_ids and the multimodal embeddings generated from multimodal
        kwargs.
        """
        # create mask indicating which token_id belongs to speech tokens
        mask = input_ids < self.speech_token_size

        input_shape = input_ids.shape
        flat_input_ids = input_ids.view(-1)
        flat_mask = mask.view(-1)

        inputs_embeds = self.inputs_embed_buffer[: flat_input_ids.shape[0]]
        inputs_embeds.zero_()

        # process speech tokens
        if flat_mask.any():
            speech_token_ids = flat_input_ids[flat_mask]
            inputs_embeds[flat_mask] = self.speech_embedding(speech_token_ids)

        # process token_id greater than `delta`
        if (~flat_mask).any():
            llm_token_ids = flat_input_ids[~flat_mask]
            llm_embeds = torch.zeros_like(inputs_embeds[~flat_mask])

            sos_eos_mask = llm_token_ids == self.sos_eos_token_id
            task_mask = llm_token_ids == self.task_token_id
            zero_mask = llm_token_ids == self.zero_token_id
            normal_mask = ~(sos_eos_mask | task_mask | zero_mask)

            # SOS/EOS token
            if sos_eos_mask.any():
                llm_embeds[sos_eos_mask] = self.llm_embedding.weight[
                    self.sos_eos
                ].unsqueeze(0)

            # task token
            if task_mask.any():
                llm_embeds[task_mask] = self.llm_embedding.weight[
                    self.task_id
                ].unsqueeze(0)

            # empty speech token
            if zero_mask.any():
                llm_embeds[zero_mask] = self.zero_embed_buffer[
                    : len(llm_embeds[zero_mask])
                ]

            # LLM token
            if normal_mask.any():
                original_ids = llm_token_ids[normal_mask] - self.llm_token_id_delta
                llm_embeds[normal_mask] = self.model.get_input_embeddings(original_ids)

            inputs_embeds[~flat_mask] = llm_embeds

        inputs_embeds = inputs_embeds.view(*input_shape, self.llm_input_size)

        # merge multimodal embedding (if any)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.audio_token_index,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(
                input_ids,
                attn_metadata=attn_metadata,
            )
        return self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds,
        )

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

    @staticmethod
    def convert_weights(
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        for name, param in weights:
            if name.startswith("llm."):
                if name.startswith("llm.model.model."):
                    name = name.replace("llm.model.model.", "model.")
                else:
                    continue
            yield name, param

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = self.convert_weights(weights)
        loader = AutoWeightsLoader(self)
        loader.load_weights(weights)
