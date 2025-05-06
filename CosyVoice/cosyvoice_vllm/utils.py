import queue
import torch
import types
from contextlib import nullcontext
import queue

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


def forward_estimator(self: CausalConditionalCFM, x, mask, mu, t, spks, cond):
    with self.lock:
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
            synchronize = lambda: None
            if torch.cuda.is_available():
                synchronize = lambda: torch.cuda.current_stream().synchronize()

            def _func(self, *args, **kwargs):
                if not hasattr(self, "stream_queue"):
                    self.stream_queue = queue.Queue()
                    for _ in range(stream_count):
                        self.stream_queue.put(
                            torch.cuda.stream(torch.cuda.Stream(self.device))
                            if self.cuda_available
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
