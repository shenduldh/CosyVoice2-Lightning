VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_CUDA_ARCH_LIST=8.9 \
    CUDA_VISIBLE_DEVICES=1 \
    TTSFRD_RESOURCE_PATH=../assets/CosyVoice-ttsfrd/resource \
    TTS_MODEL_DIR=../assets/CosyVoice2-0.5B.old \
    python gradio_demo.py
