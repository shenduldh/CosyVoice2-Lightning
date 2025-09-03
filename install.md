### 环境安装

1. 拉取 `Matcha-TTS` 仓库

   ```bash
   git submodule update --init --recursive
   ```

2. 安装 `Python` 环境和依赖

   ```bash
   conda create -n cosyvoice -y python=3.10
   conda activate cosyvoice
   # 若使用 vllm 0.7 版本，执行命令：
   pip install -r requirements.txt
   # 若使用 vllm 0.8 版本，执行命令：
   pip install -r requirements.vllm_0_8.txt
   # 若使用 vllm 0.9 版本，执行命令：
   pip install -r requirements.vllm_0_9.txt
   # 若使用 sglang 0.4.10.post2 版本，执行命令：
   pip install -r requirements.sglang.txt
   ```

3. 安装 `sox` 库

   ```bash
   # On Ubuntu
   apt-get install sox libsox-dev
   # On CentOS
   yum install sox sox-devel
   ```

4. 安装 `ttsfrd` 库（可选）

   > 如果使用 `wetext` 归一化文本，则可以跳过该步骤。

   ```bash
   git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git
   cd ./CosyVoice-ttsfrd
   unzip resource.zip -d .
   pip install ttsfrd_dependency-0.1-py3-none-any.whl
   pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
   ```
   编辑环境参数文件 `api/.env` 和 `demo/run_demo.sh` 中的 `TTSFRD_RESOURCE_PATH` 为 `CosyVoice-ttsfrd/resource` 的路径。

   或者安装其他版本的 ttsfrd：

   ```bash
   git clone https://www.modelscope.cn/speech_tts/speech_kantts_ttsfrd.git
   cd speech_kantts_ttsfrd/
   unzip resource.zip -d .
   pip install ttsfrd-0.3.9-cp310-cp310-linux_x86_64.whl
   ```

5. 下载模型文件

   - 从 [iic/CosyVoice2-0.5B - ModelScope](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B)（兼容 `20250819` 前的版本）下载模型文件。

   - 编辑 `api/.env` 和 `demo/run_demo.sh` 中 TTS 模型文件的路径。

6. 修改 CUDA 架构版本

    - 使用 `python -c "import torch; print(torch.cuda.get_device_capability())` 查看你的 CUDA 架构版本。

    - 修改 `api/.env` 和 `run_demo.sh` 中的 `TORCH_CUDA_ARCH_LIST` 为你的 CUDA 架构版本。比如上面命令的输出是 `(8, 9)`，则修改为 `TORCH_CUDA_ARCH_LIST=8.9`。

7. 其他问题处理

   - `AttributeError: 'ClassDef' object has no attribute 'type_params'`

     安装 modelscope-1.14.0：`pip install modelscope==1.14.0`
   
   - `OSError: /etc/miniconda3/envs/cosyvoice2/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /root/.cache/flashinfer/89/cached_ops/sampling/sampling.so)`

     删除 `/etc/miniconda3/envs/cosyvoice2/bin/../lib/libstdc++.so.6`。
