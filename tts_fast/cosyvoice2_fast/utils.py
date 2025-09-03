import queue
import torch
import types
from contextlib import nullcontext
import queue
import logging

from cosyvoice.flow.flow_matching import CausalConditionalCFM


class EstimatorQueue:
    def __init__(self, estimator_engine, estimator_count):
        self.estimator_queue = queue.Queue()
        self.engine = estimator_engine
        for _ in range(estimator_count):
            estimator = self.engine.create_execution_context()
            if estimator is not None:
                self.estimator_queue.put(estimator)

    def get(self):
        return self.estimator_queue.get(), self.engine

    def put(self, estimator):
        self.estimator_queue.put(estimator)


def forward_estimator(
    self: CausalConditionalCFM, x, mask, mu, t, spks, cond, streaming=False
):
    estimator, engine = self.estimator_queue.get()

    estimator.set_input_shape("x", (2, 80, x.size(2)))
    estimator.set_input_shape("mask", (2, 1, x.size(2)))
    estimator.set_input_shape("mu", (2, 80, x.size(2)))
    estimator.set_input_shape("t", (2,))
    estimator.set_input_shape("spks", (2, 80))
    estimator.set_input_shape("cond", (2, 80, x.size(2)))

    data_ptrs = [
        x.contiguous().data_ptr(),
        mask.contiguous().data_ptr(),
        mu.contiguous().data_ptr(),
        t.contiguous().data_ptr(),
        spks.contiguous().data_ptr(),
        cond.contiguous().data_ptr(),
        x.data_ptr(),
    ]

    for idx, data_ptr in enumerate(data_ptrs):
        estimator.set_tensor_address(engine.get_tensor_name(idx), data_ptr)

    estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.current_stream().synchronize()
    self.estimator_queue.put(estimator)

    return x


def set_flow_decoder(flow_decoder: CausalConditionalCFM, estimator_count=1):
    flow_decoder.estimator_queue = EstimatorQueue(
        flow_decoder.estimator_engine, estimator_count
    )
    flow_decoder.forward_estimator = types.MethodType(forward_estimator, flow_decoder)


def stream_context(is_enable=True, stream_count=10):
    def outer(func):
        if is_enable:
            is_cuda_available = torch.cuda.is_available()
            synchronize = (
                (lambda: torch.cuda.current_stream().synchronize())
                if is_cuda_available
                else (lambda: None)
            )

            def _func(self, *args, **kwargs):
                if not hasattr(self, "stream_queue"):
                    self.stream_queue = queue.Queue()
                    for _ in range(stream_count):
                        self.stream_queue.put(
                            torch.cuda.stream(torch.cuda.Stream(self.device))
                            if is_cuda_available
                            else nullcontext()
                        )

                synchronize()
                stream_context = self.stream_queue.get()
                with stream_context:
                    res = func(self, *args, **kwargs)
                    synchronize()
                    self.stream_queue.put(stream_context)
                    return res

            return _func
        return lambda *args, **kwargs: func(*args, **kwargs)

    return outer


def convert_onnx_to_trt(trt_model, onnx_model, fp16, max_workspace_size=8):
    import tensorrt as trt

    _min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2,), (2, 80), (2, 80, 4)]
    _opt_shape = [(2, 80, 193), (2, 1, 193), (2, 80, 193), (2,), (2, 80), (2, 80, 193)]
    _max_shape = [
        (2, 80, 6800),
        (2, 1, 6800),
        (2, 80, 6800),
        (2,),
        (2, 80),
        (2, 80, 6800),
    ]
    input_names = ["x", "mask", "mu", "t", "spks", "cond"]

    logging.info("Converting onnx to trt...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, max_workspace_size * 1 << 30
    )
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("failed to parse {}".format(onnx_model))
    # set input shapes
    for i in range(len(input_names)):
        profile.set_shape(input_names[i], _min_shape[i], _opt_shape[i], _max_shape[i])
    tensor_dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
