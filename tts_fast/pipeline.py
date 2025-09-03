import os
import sys

CosyVoice_path = os.path.abspath(os.path.dirname(__file__))
Matcha_path = os.path.join(CosyVoice_path, "third_party/Matcha-TTS")
sys.path.insert(0, CosyVoice_path)
sys.path.insert(0, Matcha_path)

from typing import Optional, Union, List
from collections import defaultdict
import numpy as np
import torchaudio
import librosa
import torch
import soundfile
import uuid
import pyloudnorm as pyln
from typing import AsyncGenerator
import asyncio
import queue

from cosyvoice2_fast.model import CosyVoice2, CosyVoiceInputType


def async_to_sync_gen(async_generator: AsyncGenerator):
    def sync_generator():
        q = queue.Queue()

        async def do_async_generation():
            async for i in async_generator:
                q.put((i, False))
            q.put((None, True))

        asyncio.run(do_async_generation())

        while True:
            output, finished = q.get_nowait() if not q.empty() else q.get()
            if finished:
                break
            yield output

    return sync_generator


def load_and_normalize_audio(
    audio_path: str,
    sr=16000,
    # normalize loudness
    target_loudness=20.0,
    # normalize amplitude
    max_value=0.75,
    # remove silence
    top_db=60,
    hop_length=128,
    frame_length=512,
    # concat tail silence
    concat_sr=24000,
    tail_silence_length=0.2,
) -> torch.Tensor:
    # normalize loudness
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    meter = pyln.Meter(sr)
    orig_loudness = meter.integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, orig_loudness, target_loudness)

    # normalize amplitude
    if np.abs(audio).max() > max_value:
        audio = audio / np.abs(audio).max() * max_value

    # remove silence
    audio, _ = librosa.effects.trim(
        audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )

    # concat tail silence
    audio = torch.as_tensor(audio).view(1, -1).float()
    tail_silence = torch.zeros(1, int(concat_sr * tail_silence_length)).float()
    audio = torch.concat([audio, tail_silence], dim=1)

    return audio


def clone_to_cpu(d: dict):
    _d = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.clone().cpu()
        _d[k] = v
    return _d


class CosyVoice2Pipeline:
    def __init__(self, model_dir):
        self.cosyvoice = CosyVoice2(
            model_dir, use_jit=False, use_trt=True, do_compile=True, fp16=True
        )
        self.speaker_cache = defaultdict(dict)
        self.prompt_cache = defaultdict(dict)
        self.sample_rate = self.cosyvoice.sample_rate

    def save_cache(
        self,
        cache_dir: str,
        filename: Union[str, None] = None,
        speaker_ids: List[str] = [],
    ):
        if len(speaker_ids) > 0:
            saved = {k: v for k, v in self.speaker_cache.items() if k in speaker_ids}
        else:
            saved = {k: v for k, v in self.speaker_cache.items()}

        if len(saved) == 0:
            return None

        os.makedirs(cache_dir, exist_ok=True)
        if filename is None:
            filename = f"speaker_cache_{uuid.uuid4().hex}.pt"
        else:
            filename = f"{filename}.pt"
        cache_path = os.path.join(cache_dir, filename)
        if os.path.exists(cache_path):
            os.remove(cache_path)
        torch.save(saved, cache_path)
        return cache_path

    def load_cache(self, cache_path: str, speaker_ids: List[str] = []):
        if not os.path.exists(cache_path):
            return []
        speaker_cache = torch.load(
            cache_path, map_location=torch.device("cpu"), weights_only=False
        )
        if len(speaker_ids) > 0:
            filtered = {k: v for k, v in speaker_cache.items() if k in speaker_ids}
        else:
            filtered = speaker_cache
        self.speaker_cache.update(filtered)
        return list(filtered.keys())

    def get_speakers(self):
        return list(self.speaker_cache.keys())

    def remove_speakers(self, speaker_ids=[]):
        if len(speaker_ids) == 0:
            speaker_ids = list(self.speaker_cache.keys())

        removed = []
        for spk_id in speaker_ids:
            if spk_id in self.speaker_cache:
                self.speaker_cache.pop(spk_id)
                removed.append(spk_id)
        return removed

    def calc_speaker_feature(self, audio_path, loudness, resample_rate):
        prompt_audio_16k = load_and_normalize_audio(
            audio_path, sr=16000, target_loudness=loudness, concat_sr=resample_rate
        )
        prompt_audio_resampled = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=resample_rate
        )(prompt_audio_16k)

        speech_feat, speech_feat_len = self.cosyvoice.frontend.extract_speech_feat(
            prompt_audio_resampled
        )
        speech_token, speech_token_len = self.cosyvoice.frontend.extract_speech_token(
            prompt_audio_16k
        )

        if resample_rate == 24000:  # force speech_feat % speech_token == 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat = speech_feat[:, : 2 * token_len]
            speech_feat_len[:] = 2 * token_len
            speech_token = speech_token[:, :token_len]
            speech_token_len[:] = token_len

        spk_embedding = self.cosyvoice.frontend.extract_spk_embedding(prompt_audio_16k)

        return {
            "flow_embedding": spk_embedding,
            "llm_prompt_speech_token": speech_token,
            "flow_prompt_speech_token": speech_token,
            "prompt_speech_feat": speech_feat,
        }

    def splitnorm_text(self, text, split=False, use_frontend_model=True):
        res = self.cosyvoice.frontend.text_normalize(
            text, split=split, use_frontend_model=use_frontend_model
        )
        return res

    def calc_text_feature(self, text):
        text_token = self.cosyvoice.frontend.extract_text_token(text)
        return {"prompt_text": text_token}

    def preprocess(
        self,
        prompt_text: str,
        prompt_audio: str,
        instruct_text: str,
        resample_rate: int,
        speaker_id: Optional[str | int] = None,
        speaker_loudness: float = 20.0,
        use_frontend_model=True,
    ):
        model_input = {}

        if speaker_id is None or speaker_id not in self.speaker_cache:
            # calc tone feature
            speaker_dict = self.calc_speaker_feature(
                prompt_audio, speaker_loudness, resample_rate
            )
        else:
            # get tone feature from cache
            speaker_dict = self.speaker_cache[speaker_id]
        model_input.update(speaker_dict)

        # cache tone feature
        if speaker_id is not None and speaker_id not in self.speaker_cache:
            self.speaker_cache[speaker_id].update(clone_to_cpu(model_input))

        # calc prompt text feature
        if prompt_text is not None:
            if prompt_text not in self.prompt_cache:
                prompt_text = self.splitnorm_text(
                    prompt_text, split=False, use_frontend_model=use_frontend_model
                )
                prompt_info = self.calc_text_feature(prompt_text)
                self.prompt_cache[prompt_text].update(prompt_info)
            else:
                prompt_info = self.prompt_cache[prompt_text]

            model_input.update(prompt_info)
            if speaker_id is not None:
                self.speaker_cache[speaker_id].update(clone_to_cpu(prompt_info))

        # calc instruction text feature
        if instruct_text is not None:
            if instruct_text not in self.prompt_cache:
                instruct_text = (
                    self.splitnorm_text(
                        instruct_text,
                        split=False,
                        use_frontend_model=use_frontend_model,
                    )
                    + "<|endofprompt|>"
                )
                instruct_info = self.calc_text_feature(instruct_text)
                self.prompt_cache[instruct_text].update(instruct_info)
            else:
                instruct_info = self.prompt_cache[instruct_text]

            model_input.update(instruct_info)
            del model_input["llm_prompt_speech_token"]

        return model_input

    def async_generate(
        self,
        id: str | None,
        tts_text,
        prompt_text,
        prompt_audio,
        instruct_text,
        speaker_id=None,
        speaker_loudness=20.0,
        use_frontend_model=True,
        split_text=False,
        speed=1.0,
        stream=True,
        input_type=CosyVoiceInputType.SINGLE,
    ):
        model_input = self.preprocess(
            prompt_text,
            prompt_audio,
            instruct_text,
            self.sample_rate,
            speaker_id=speaker_id,
            speaker_loudness=speaker_loudness,
            use_frontend_model=use_frontend_model,
        )
        model_input.update({"text": tts_text})
        return self.cosyvoice.async_tts(
            id,
            **model_input,
            split_text=split_text,
            stream=stream,
            speed=speed,
            input_type=input_type,
        )

    def generate(self, *args, **kwargs):
        return async_to_sync_gen(self.async_generate(*args, **kwargs))

    def save_audio(self, saved_path, audio_ndarray):
        soundfile.write(saved_path, audio_ndarray, self.sample_rate)
