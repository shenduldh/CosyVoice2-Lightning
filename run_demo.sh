VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDA_ARCH_LIST=8.9 VLLM_USE_V1=1 \
    CUDA_VISIBLE_DEVICES=1 \
    TTSFRD_RESOURCE_PATH=../assets/CosyVoice-ttsfrd/resource \
    TTS_MODEL_DIR=../assets/CosyVoice2-0.5B \
    python gradio_demo.py
