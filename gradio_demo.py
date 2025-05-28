import os
from CosyVoice.model import Pipeline
import gradio as gr
import uuid
import numpy as np
from api.utils import repack


async def generate(
    tts_text, prompt_text: str, prompt_audio, instruct_text: str, prompt_id
):
    yield gr.update(value=None), gr.update(value=None)

    if len(prompt_text.strip()) == 0:
        prompt_text = None

    if len(instruct_text.strip()) == 0:
        instruct_text = None

    task_id = uuid.uuid4().hex
    STATE["id"] = task_id

    whole_audio = []
    async for chunk in repack(
        pipeline.async_generate(
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
        if STATE["id"] != task_id:
            return

        whole_audio.append(chunk)

        yield (
            (pipeline.sample_rate, chunk),
            (pipeline.sample_rate, np.concatenate(whole_audio)),
        )


with gr.Blocks() as demo:
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
            every=0.1,
        )
        whole_out = gr.Audio(label="完整语音", type="numpy", streaming=False)

    gen_button.click(
        generate,
        inputs=[tts_text, prompt_text, prompt_audio, instruct_text, speaker_id],
        outputs=[stream_out, whole_out],
    )

    speakers_dir = os.path.join(os.path.dirname(__file__), "assets", "speakers")
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
    STATE = {"id": None}
    tts_model_dir = os.environ["TTS_MODEL_DIR"]
    pipeline = Pipeline(tts_model_dir, load_trt=True, fp16=True)
    demo.queue(max_size=1, default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=14444, share=False)
