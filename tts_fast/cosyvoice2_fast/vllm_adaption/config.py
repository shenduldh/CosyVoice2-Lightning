import os
from vllm.sampling_params import RequestOutputKind
from vllm.config import CompilationConfig


compilation_cache_dir = os.getenv("COMPILATION_CACHE_DIR", None)
if compilation_cache_dir is not None:
    compilation_cache_dir = os.path.join(compilation_cache_dir, "vllm")
else:
    compilation_cache_dir = ""
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


ENGINE_ARGS = {
    "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", 0.5)),
    "block_size": int(os.getenv("VLLM_BLOCK_SIZE", 32)),
    "max_num_batched_tokens": int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", 8192)),
    "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", 2048)),
    "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", 16)),
    "hf_overrides": {"architectures": ["CosyVoice2LLM"]},
    "dtype": "bfloat16",
    "swap_space": 0,
    "disable_log_requests": True,
    "disable_log_stats": True,
    "skip_tokenizer_init": True,
    "task": "generate",
    "compilation_config": CompilationConfig(
        cache_dir=compilation_cache_dir,
        use_inductor=True,
        use_cudagraph=True,
    ),
    "load_format": "pt",
    "enforce_eager": False,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
}


SAMPLING_PARAMS = {
    "temperature": 1,  # 不能低于 0.8，否则无法生成正常语音
    "top_p": 1,  # 不能低于 0.8，否则无法生成正常语音
    "top_k": 25,
    "detokenize": False,
    "ignore_eos": False,
    "output_kind": RequestOutputKind.DELTA,
}
