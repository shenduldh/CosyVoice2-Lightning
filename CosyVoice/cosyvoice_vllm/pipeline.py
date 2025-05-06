import os
import torch
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download

from .frontend import CosyVoiceFrontEnd
from .cosyvoice2_model import CosyVoice2Model
from .config import ESTIMATOR_COUNT, ONNX2TRT_WORKSPACE_SIZE


class CosyVoice2:
    def __init__(
        self,
        model_dir,
        load_jit=False,
        load_trt=False,
        fp16=False,
    ):
        self.model_dir = model_dir
        self.fp16 = fp16
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)

        with open("{}/cosyvoice.yaml".format(model_dir), "r") as f:
            configs = load_hyperpyyaml(
                f,
                overrides={
                    "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")
                },
            )
            del configs["llm"]

        self.frontend = CosyVoiceFrontEnd(
            configs["get_tokenizer"],
            configs["feat_extractor"],
            "{}/campplus.onnx".format(model_dir),
            "{}/speech_tokenizer_v2.onnx".format(model_dir),
            "{}/spk2info.pt".format(model_dir),
            configs["allowed_special"],
        )

        self.sample_rate = configs["sample_rate"]

        self.model = CosyVoice2Model(
            model_dir, configs["flow"], configs["hift"], self.fp16, self.device
        )
        self.model.load("{}/flow.pt".format(model_dir), "{}/hift.pt".format(model_dir))

        if load_jit:
            self.model.load_jit(
                "{}/flow.encoder.{}.zip".format(
                    model_dir, "fp16" if self.fp16 else "fp32"
                )
            )

        if load_trt:
            self.model.load_trt(
                "{}/flow.decoder.estimator.{}.mygpu.plan".format(
                    model_dir, "fp16" if self.fp16 else "fp32"
                ),
                "{}/flow.decoder.estimator.fp32.onnx".format(model_dir),
                self.fp16,
                ONNX2TRT_WORKSPACE_SIZE,
                ESTIMATOR_COUNT,
            )

        del configs
