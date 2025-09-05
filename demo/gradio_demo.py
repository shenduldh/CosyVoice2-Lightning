import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_path)

from tts_fast.pipeline import CosyVoice2Pipeline
import gradio as gr
import numpy as np
from api.utils import repack, float32_to_int16


def generate(tts_text, prompt_text: str, prompt_audio, instruct_text: str, prompt_id):
    if len(prompt_text.strip()) == 0:
        prompt_text = None
    if len(instruct_text.strip()) == 0:
        instruct_text = None

    whole_audio = []
    for chunk in repack(
        pipeline.generate(
            None,
            tts_text,
            prompt_text,
            prompt_audio,
            instruct_text,
            prompt_id,
            use_frontend_model=True,
            stream=True,
        ),
        pipeline.sample_rate * 2,
    ):
        chunk = float32_to_int16(chunk)
        whole_audio.append(chunk)
        yield (
            (pipeline.sample_rate, chunk),
            (pipeline.sample_rate, np.concatenate(whole_audio)),
        )


with gr.Blocks(title="CosyVoice2 Demo") as demo:
    with gr.Tab("CosyVoice2 Fast"):
        with gr.Row():
            prompt_audio = gr.Audio(
                label="参考音频", sources=["upload", "microphone"], type="filepath"
            )
            with gr.Column():
                prompt_text = gr.TextArea(label="参考文本")
                speaker_id = gr.Textbox(label="参考音频 ID")
                instruct_text = gr.Textbox(label="指令文本")
                tts_text = gr.TextArea(label="用于生成语音的文本")

        with gr.Row():
            gen_button = gr.Button("生成语音")

        stream_out = gr.Audio(
            label="流式语音",
            type="numpy",
            streaming=True,
            autoplay=True,
            every=1,
        )
        whole_out = gr.Audio(label="完整语音", type="numpy", streaming=False)

    gen_button.click(
        generate,
        inputs=[tts_text, prompt_text, prompt_audio, instruct_text, speaker_id],
        outputs=[stream_out, whole_out],
    )

    speakers_dir = os.path.join(root_path, "assets/speakers")
    speaker_names = os.listdir(speakers_dir)

    def get_transcript(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    gr.Examples(
        examples=[
            [
                os.path.join(speakers_dir, name, "audio.mp3"),
                get_transcript(os.path.join(speakers_dir, name, "transcript.txt")),
                name,
                "你好，这是一段测试文本。",
            ]
            for name in speaker_names
        ],
        inputs=[prompt_audio, prompt_text, speaker_id, tts_text],
    )


if __name__ == "__main__":
    tts_model_dir = os.environ["TTS_MODEL_DIR"]
    pipeline = CosyVoice2Pipeline(tts_model_dir)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "14477")),
        share=False,
        allowed_paths=[os.path.join(root_path, "assets/speakers")],
    )
