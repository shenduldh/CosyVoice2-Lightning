from functools import partial
from typing import Generator, AsyncGenerator
import json
import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable
import torchaudio.compliance.kaldi as kaldi
import os
import re
import inflect

try:
    import ttsfrd

    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer

    use_ttsfrd = False
from cosyvoice.utils.frontend_utils import (
    contains_chinese,
    replace_blank,
    replace_corner_mark,
    remove_bracket,
    spell_out_number,
    split_paragraph,
    is_only_punctuation,
)

from .config import OVERWRITE_NORMALIZER_CACHE, TTSFRD_RESOURCE_PATH


class CosyVoiceFrontEnd:

    def __init__(
        self,
        get_tokenizer: Callable,
        feat_extractor: Callable,
        campplus_model: str,
        speech_tokenizer_model: str,
        spk2info: str = "",
        allowed_special: str = "all",
    ):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model,
            sess_options=option,
            providers=[
                (
                    "CUDAExecutionProvider"
                    if torch.cuda.is_available()
                    else "CPUExecutionProvider"
                )
            ],
        )
        if os.path.exists(spk2info):
            self.spk2info = torch.load(
                spk2info, map_location=self.device, weights_only=False
            )
        else:
            self.spk2info = {}
        self.allowed_special = allowed_special
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            assert (
                self.frd.initialize(TTSFRD_RESOURCE_PATH) is True
            ), "failed to initialize ttsfrd resource"
            self.frd.set_lang_type("pinyinvg")
        else:
            self.zh_tn_model = ZhNormalizer(
                remove_erhua=False,
                full_to_half=False,
                overwrite_cache=OVERWRITE_NORMALIZER_CACHE,
            )
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine()

    def _extract_text_token(self, text):
        if isinstance(text, Generator):
            # add a dummy text_token_len for compatibility
            return self._extract_text_token_generator(text)
        elif isinstance(text, AsyncGenerator):
            # add a dummy text_token_len for compatibility
            return self._extract_text_token_async_generator(text)
        else:
            text_token = self.tokenizer.encode(
                text, allowed_special=self.allowed_special
            )
            text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
            # text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
            return text_token

    def _extract_text_token_generator(self, text_generator: Generator):
        for text in text_generator:
            text_token = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i : i + 1]

    async def _extract_text_token_async_generator(self, text_generator: AsyncGenerator):
        async for text in text_generator:
            text_token = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i : i + 1]

    def _extract_speech_token(self, speech):
        assert (
            speech.shape[1] / 16000 <= 30
        ), "do not support extract speech token for audio longer than 30s"
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        _input_name0 = self.speech_tokenizer_session.get_inputs()[0].name
        _input_name1 = self.speech_tokenizer_session.get_inputs()[1].name
        speech_token = self.speech_tokenizer_session.run(
            None,
            {
                _input_name0: feat.detach().cpu().numpy(),
                _input_name1: np.array([feat.shape[2]], dtype=np.int32),
            },
        )
        speech_token = speech_token[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(
            self.device
        )
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        _input_name = self.campplus_session.get_inputs()[0].name
        embedding = self.campplus_session.run(
            None, {_input_name: feat.unsqueeze(dim=0).cpu().numpy()}
        )
        embedding = embedding[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = (
            self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        )
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(
            self.device
        )
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True, text_frontend=True):
        if isinstance(text, (Generator, AsyncGenerator)):
            return [text]

        text = text.strip()

        if text_frontend is False or text == "":
            return text if not split else [text]

        if self.use_ttsfrd:
            texts = [
                i["text"]
                for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]
            ]
            if not split:
                return "".join(texts)
        else:
            if contains_chinese(text):
                text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r"[，,、]+$", "。", text)
                if not split:
                    return text
                texts = split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "zh",
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                if not split:
                    return text
                texts = split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "en",
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )

        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts
