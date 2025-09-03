LLM_ENGINE_MODE=sglang \
    TORCH_CUDA_ARCH_LIST=8.9 \
    COMPILATION_CACHE_DIR=../assets/cache \
    CUDA_VISIBLE_DEVICES=0 \
    TTSFRD_RESOURCE_PATH=/data1/work/tts/assets/CosyVoice-ttsfrd/resource \
    TTS_MODEL_DIR=/data1/work/tts/assets/CosyVoice2-0.5B_0606 \
    python gradio_demo.py
