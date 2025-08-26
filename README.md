# dllm-rl




## 🚀 Quick Start

```bash
conda create --name dllm-rl python=3.10
source activate dllm-rl
pip install torch==2.6.0
pip install -r requirements.txt
pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

You can also install [FlashAttention](https://github.com/Dao-AILab/flash-attention) based on your version of PyTorch and CUDA.
