LLM_ENGINE_MODE=sglang \
    PORT=14477 \
    TORCH_CUDA_ARCH_LIST=8.9 \
    COMPILATION_CACHE_DIR=../assets/cache \
    CUDA_VISIBLE_DEVICES=0 \
    TTSFRD_RESOURCE_PATH=../assets/CosyVoice-ttsfrd/resource \
    TTS_MODEL_DIR=../assets/CosyVoice2-0.5B \
    python gradio_demo.py
