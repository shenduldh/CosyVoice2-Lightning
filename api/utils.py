import numpy as np
from pydub import AudioSegment
import librosa
from urllib.request import urlopen
from io import BytesIO
import soundfile as sf
import io
import base64
import struct
from typing import AsyncGenerator, Union, Literal
import traceback
from tempfile import NamedTemporaryFile


def truncate_long_str(obj, max_len=70, ellipsis="......"):
    if isinstance(obj, str):
        if len(obj) > max_len:
            return obj[:max_len] + ellipsis
        return obj
    elif isinstance(obj, (tuple, list)):
        return [truncate_long_str(i, max_len, ellipsis) for i in obj]
    elif isinstance(obj, dict):
        return {k: truncate_long_str(v, max_len, ellipsis) for k, v in obj.items()}
    else:
        return obj


def whats_wrong_with(e):
    return traceback.format_exception_only(e)[-1].strip()


async def repack(generator: AsyncGenerator[np.ndarray, None], min_size=0, max_num=100):
    buffer = []
    buffer_size = 0

    async for chunk in generator:
        buffer.append(chunk)
        buffer_size += chunk.shape[-1]

        while len(buffer) > max_num:
            if buffer:
                yield np.concatenate(buffer)
                buffer.clear()
                buffer_size = 0

        while buffer_size >= min_size:
            output = []
            output_size = 0
            while buffer and output_size < min_size:
                block = buffer.pop(0)
                output.append(block)
                output_size += len(block)
                buffer_size -= len(block)

            yield np.concatenate(output)

    if buffer:
        yield np.concatenate(buffer)


def add_wav_header(
    audio_bytes: bytes, sample_rate=16000, num_channels=1, bits_per_sample=16
):
    if audio_bytes.startswith("RIFF".encode()):
        return audio_bytes

    SIGNED_INT = struct.Struct("<i")
    UNSIGNED_SHORT = struct.Struct("<H")
    num_samples = len(audio_bytes)

    header = b"RIFF"
    header += SIGNED_INT.pack(num_samples + 36)
    header += b"WAVEfmt "
    header += b"\x10\x00\x00\x00"  # fmt chunk size
    header += b"\x01\x00"  # audio format (1 is PCM)
    header += UNSIGNED_SHORT.pack(num_channels)
    header += SIGNED_INT.pack(sample_rate)
    header += SIGNED_INT.pack(
        sample_rate * num_channels * bits_per_sample // 8
    )  # bytes per sample
    header += UNSIGNED_SHORT.pack(
        num_channels * bits_per_sample // 8
    )  # block alignment
    header += UNSIGNED_SHORT.pack(bits_per_sample)
    header += b"data"
    header += SIGNED_INT.pack(num_samples)

    return header + audio_bytes


short_info = np.iinfo(np.short)
min_short = short_info.min
max_short = short_info.max
abs_max_short: int = 2 ** (short_info.bits - 1)
offset = min_short + abs_max_short


def float32_to_int16(audio_ndarray: np.ndarray):
    return (
        (audio_ndarray * abs_max_short + offset)
        .clip(min_short, max_short)
        .astype(np.short)
    )


def int16_to_float32(audio_ndarray: np.ndarray):
    return (audio_ndarray.astype(np.float32) - offset) / abs_max_short


def to_mono(audio_ndarray: np.ndarray) -> np.ndarray:
    if len(audio_ndarray.shape) == 2:
        n1, n2 = audio_ndarray.shape
        channel_axis = 0 if n1 < n2 else 1
        return audio_ndarray.mean(axis=channel_axis)
    return audio_ndarray


def save_audio(audio_ndarray: np.ndarray, path: str, sample_rate=16000):
    audio_ndarray = float32_to_int16(audio_ndarray)
    sf.write(path, audio_ndarray, samplerate=sample_rate)


def ndarray_to_pydub(audio_ndarray: np.ndarray, sample_rate=16000) -> AudioSegment:
    audio_segment = AudioSegment.from_raw(
        io.BytesIO(float32_to_int16(audio_ndarray).tobytes()),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    return audio_segment


def load_audio_segment(audio_path: str, format: str, sample_rate=16000):
    if audio_path.startswith("http"):
        audio = BytesIO(urlopen(audio_path).read())
    else:
        audio = audio_path

    audio_seg: AudioSegment = AudioSegment.from_file(audio, format=format)
    audio_seg = audio_seg.set_frame_rate(sample_rate).set_channels(1)
    return audio_seg


def pydub_to_ndarray(audio_seg: AudioSegment):
    audio_ndarray = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
    scale = 1.0 / float(1 << ((8 * audio_seg.sample_width) - 1))
    audio_ndarray *= scale
    return audio_ndarray


def bytes_no_header_to_ndarray(audio_bytes: bytes, sample_rate=16000):
    audio_bytes = add_wav_header(audio_bytes, sample_rate)
    audio_ndarray, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
    return audio_ndarray


def ndarray_to_bytes_with_wav_header(audio_ndarray: np.ndarray, sample_rate=16000):
    audio_bytes = float32_to_int16(to_mono(audio_ndarray)).tobytes()
    audio_bytes = add_wav_header(audio_bytes, sample_rate)
    return audio_bytes


def ndarray_to_base64_no_header(audio_ndarray: np.ndarray):
    audio_bytes = float32_to_int16(to_mono(audio_ndarray)).tobytes()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    return audio_base64


def base64_no_header_to_ndarray(audio_base64: str, sample_rate=16000):
    audio_bytes = base64.b64decode(audio_base64.encode("utf-8"))
    audio_bytes = add_wav_header(audio_bytes, sample_rate)
    audio_ndarray, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
    return audio_ndarray


def ndarray_to_base64_with_wav_header(
    audio_ndarray: np.ndarray, sample_rate=16000, resample_rate=16000
):
    if resample_rate != sample_rate:
        audio_ndarray = librosa.resample(
            audio_ndarray, orig_sr=sample_rate, target_sr=resample_rate
        )
    audio_bytes = float32_to_int16(to_mono(audio_ndarray)).tobytes()
    audio_bytes = add_wav_header(audio_bytes, sample_rate)
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    return audio_base64


def base64_with_wav_header_to_ndarray(audio_base64: str, sample_rate=16000):
    audio_bytes = base64.b64decode(audio_base64.encode("utf-8"))
    audio_ndarray, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
    return audio_ndarray


AUDIO_EXPORT_OPTIONS = {
    "opus": {"format": "opus", "bitrate": "32k"},
    "pcm": {"format": "s16le"},
    "wav": {"format": "wav"},  # 码率固定为 (采样率*位深度*声道数/1000)
    "mp3": {"format": "mp3", "bitrate": "128k"},
    "flac": {"format": "flac"},
    "aac": {"format": "adts", "codec": "aac", "bitrate": "128k"},
    "m4a": {"format": "ipod", "codec": "aac", "bitrate": "128k"},
}


def ndarray_to_any_format(
    audio_ndarray: np.ndarray,
    sample_rate=16000,
    resample_rate=16000,
    format: Literal["opus", "pcm", "wav", "mp3", "flac", "aac", "m4a"] = "opus",
    format_options: dict = {},
    return_base64=False,
):
    audio_segment = ndarray_to_pydub(to_mono(audio_ndarray), sample_rate)
    if resample_rate != sample_rate:
        audio_segment = audio_segment.set_frame_rate(resample_rate)
    buffer = BytesIO()
    export_options = AUDIO_EXPORT_OPTIONS[format]
    export_options.update(**format_options)
    audio_segment.export(buffer, **export_options)
    audio_bytes = buffer.getvalue()
    if return_base64:
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return audio_base64
    return audio_bytes


def any_format_to_ndarray(
    audio_bytes_or_base64: Union[bytes, str],
    format="opus",
    sample_rate=16000,
    resample_rate=16000,
):
    if isinstance(audio_bytes_or_base64, str):
        audio_bytes = base64.b64decode(audio_bytes_or_base64.encode("utf-8"))
    else:
        audio_bytes = audio_bytes_or_base64
    if format == "pcm":
        audio_bytes = add_wav_header(audio_bytes, sample_rate)
    with NamedTemporaryFile(suffix=f".{format}") as f:
        f.write(audio_bytes)
        f.flush()
        audio_ndarray, _ = librosa.load(f.name, sr=resample_rate, mono=True)
    return audio_ndarray


def format_ndarray_to_base64(
    audio_ndarray: np.ndarray, sample_rate: int, resample_rate: int, format: str
):
    if format == "wav":
        return ndarray_to_base64_with_wav_header(
            audio_ndarray, sample_rate, resample_rate
        )
    return ndarray_to_any_format(
        audio_ndarray, sample_rate, resample_rate, format, return_base64=True
    )


def remove_silence(
    audio_ndarray: np.ndarray,
    sample_rate: int,
    left_retention_seconds=0,
    right_retention_seconds=0,
    top_db=60,
    frame_length=440,
    hop_length=220,
):
    _, (s, e) = librosa.effects.trim(
        audio_ndarray, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    s = int(s - left_retention_seconds * sample_rate)
    s = 0 if s < 0 else s
    e = int(e + right_retention_seconds * sample_rate)
    audio_ndarray = audio_ndarray[s:e]
    return audio_ndarray
