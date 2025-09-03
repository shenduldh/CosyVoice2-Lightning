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


from cosyvoice.utils.frontend_utils import (
    contains_chinese,
    replace_blank,
    replace_corner_mark,
    remove_bracket,
    spell_out_number,
    split_paragraph,
    is_only_punctuation,
)


FRONTEND_MODE = os.getenv("FRONTEND_MODE", "wetext")
if FRONTEND_MODE == "wetext":
    import wetext
elif FRONTEND_MODE == "ttsfrd":
    import ttsfrd

TTSFRD_RESOURCE_PATH = os.getenv(
    "TTSFRD_RESOURCE_PATH",
    os.path.join(
        os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3]),
        "assets",
        "CosyVoice-ttsfrd",
        "resource",
    ),
)


class CosyVoice2FrontEnd:

    def __init__(
        self,
        get_tokenizer: Callable,
        feat_extractor: Callable,
        campplus_model: str,
        speech_tokenizer_model: str,
        allowed_special: str = "all",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
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

        self.allowed_special = allowed_special
        self.use_ttsfrd = FRONTEND_MODE == "ttsfrd"

        if self.use_ttsfrd:
            self.ttsfrd = ttsfrd.TtsFrontendEngine()
            assert (
                self.ttsfrd.initialize(TTSFRD_RESOURCE_PATH) is True
            ), "Failed to initialize ttsfrd resource."
            self.ttsfrd.set_lang_type("pinyinvg")
        else:
            self.wetext = wetext.Normalizer()
            self.inflect_parser = inflect.engine()
            self.tokenizer_encode_fn = partial(
                self.tokenizer.encode, allowed_special=allowed_special
            )

    def extract_text_token(self, text):
        if isinstance(text, Generator):
            # add a dummy text_token_len for compatibility
            return self.__ett_generator(text)
        elif isinstance(text, AsyncGenerator):
            # add a dummy text_token_len for compatibility
            return self.__ett_async_generator(text)
        else:
            text_token = self.tokenizer.encode(
                text, allowed_special=self.allowed_special
            )
            text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
            # text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
            return text_token

    def __ett_generator(self, text_generator: Generator):
        for text in text_generator:
            text_token = self.extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i : i + 1]

    async def __ett_async_generator(self, text_generator: AsyncGenerator):
        async for text in text_generator:
            text_token = self.extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i : i + 1]

    def extract_speech_token(self, speech):
        assert (
            speech.shape[1] / 16000 <= 30
        ), "Don't support extract speech token for audio longer than 30s."
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

    def extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        _input_name = self.campplus_session.get_inputs()[0].name
        embedding = self.campplus_session.run(
            None, {_input_name: feat.unsqueeze(dim=0).cpu().numpy()}
        )
        embedding = embedding[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def extract_speech_feat(self, speech):
        speech_feat = (
            self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        )
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(
            self.device
        )
        return speech_feat, speech_feat_len

    def text_normalize(
        self, text, split=False, use_frontend_model=True
    ) -> AsyncGenerator | str | list[str] | None:
        if isinstance(text, AsyncGenerator):
            return text

        text = text.strip()

        if not use_frontend_model or text == "":
            return text if not split else [text]

        if self.use_ttsfrd:
            res = json.loads(self.ttsfrd.do_voicegen_frd(text))["sentences"]
            if res is None:
                return None
            text = [i["text"] for i in res]
            if not split:
                text = "".join(text)
        else:
            lang = "zh" if contains_chinese(text) else "en"

            text = self.wetext.normalize(text, lang)
            if lang == "zh":
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r"[，,、]+$", "。", text)
            else:
                text = spell_out_number(text, self.inflect_parser)

            if split:
                text = split_paragraph(
                    text,
                    self.tokenizer_encode_fn,
                    lang,
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )

        if split:
            text = [i for i in text if not is_only_punctuation(i)]
        return text
