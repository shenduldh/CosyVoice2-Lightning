### 请按照下面的步骤安装环境。

1. 拉取 Matcha-TTS 仓库

    ```bash
    git submodule update --init --recursive
    ```

2. 安装环境和依赖

    ```bash
    conda create -n tts_api -y python=3.10
    conda activate tts_api
    pip install -r requirements.txt
    ```

3. 确保 sox 库的兼容性

    ```bash
    # On Ubuntu
    apt-get install sox libsox-dev
    # On CentOS
    yum install sox sox-devel
    ```

4. 安装 ttsfrd

    ```bash
    git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git
    cd ./CosyVoice-ttsfrd
    unzip resource.zip -d .
    pip install ttsfrd_dependency-0.1-py3-none-any.whl
    pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
    ```

    修改环境参数文件 `api/.env` 和 `run_demo.sh` 中的 `TTSFRD_RESOURCE_PATH` 为 `CosyVoice-ttsfrd/resource` 的路径。

5. 下载模型文件

    - 从 [CosyVoice2-0.5B - huggingface](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) 或 [CosyVoice2-0.5B - modelscope](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) 下载模型文件。**注意，需要使用未更新 `flow cache` 前的版本。**

    - 修改 `api/.env` 和 `run_demo.sh` 中的 `TTS_MODEL_DIR` 为模型文件目录的路径。

6. 注册 VLLM

    ```bash
    cd CosyVoice/cosyvoice_vllm
    python register_model_to_vllm.py
    ```

    将 `CosyVoice/cosyvoice_vllm/CosyVoice2_vllm` 目录下的文件拷贝到模型文件目录 `CosyVoice2-0.5B` 下。

7. 修改 CUDA 架构版本

    - 查看你的 CUDA 架构版本

        ```bash
        python -c "import torch; print(torch.cuda.get_device_capability())"
        ```

    - 修改 `api/.env` 和 `run_demo.sh` 中的 `TORCH_CUDA_ARCH_LIST` 为你的 CUDA 架构版本。比如上面命令的输出是 `(8, 9)`，则修改为 `TORCH_CUDA_ARCH_LIST=8.9`。
