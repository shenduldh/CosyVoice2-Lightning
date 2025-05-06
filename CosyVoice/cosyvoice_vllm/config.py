import os
from vllm.sampling_params import RequestOutputKind


ENGINE_ARGS = {
    "block_size": 16,
    "swap_space": 0,
    # "enforce_eager": True,
    "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", 0.25)),
    "max_num_batched_tokens": int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", 16)),
    "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", 1024)),
    "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", 4)),
    "disable_log_requests": True,
    "disable_log_stats": True,
    "dtype": "float16",
    # "enable_prefix_caching": False,
    # "chunked_prefill_enabled": False,
}


SAMPLING_PARAMS = {
    "temperature": 1,  # 不能低于 0.8，否则会生成非常多的空音频，或者无法正常生成语音 Token
    "top_p": 1,  # 不能低于 0.8，否则会生成非常多的空音频，或者无法正常生成语音 Token
    "top_k": 25,
    # "min_tokens": 80,  # 不支持设置
    # "presence_penalty": 1.0,  # 不支持设置
    # "frequency_penalty": 0.0,  # 不支持设置
    "max_tokens": 2048,
    "detokenize": False,  # vllm 0.7.3 v1 版本中设置无效，待后续版本更新后减少计算
    "ignore_eos": False,
    "output_kind": RequestOutputKind.DELTA,  # 设置为 DELTA，如调整该参数，请同时调整 llm_inference 的处理代码
}


# 用于设置 ZhNormalizer 的 overwrite_cache 参数
# 首次运行时，需设置为 True 来正确生成缓存，避免过滤掉儿化音
# 后续可以设置为 False 来避免重复生成缓存
OVERWRITE_NORMALIZER_CACHE = True
# 限制将 flow oonx 转换为 trt 时使用的 GPU 显存大小
ONNX2TRT_WORKSPACE_SIZE = int(os.getenv("ONNX2TRT_WORKSPACE_SIZE", 2))
# 根据 GPU 显存大小量及性能设置合适的 ESTIMATOR_COUNT
ESTIMATOR_COUNT = int(os.getenv("ESTIMATOR_COUNT", 1))


TTSFRD_RESOURCE_PATH = os.getenv(
    "TTSFRD_RESOURCE_PATH",
    f"{os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3])}/assets/CosyVoice-ttsfrd/resource",
)
