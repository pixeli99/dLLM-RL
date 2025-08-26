# dllm-rl


## Features 

- Model Support: SDAR, Dream, LLaDA, Diffu-Coder
- Inference: KV-cache, jetengine (based on nano-vllm), different sampling strategies, support multi-nodes
- RL: TraceRL (and it's proximal version), Coupled RL, random masking RL, accelerated sampling, support multi-nodes
- SFT: flash block SFT, semi-AR SFT, random masking SFT


## 🚀 Quick Start

```bash
conda create --name dllm-rl python=3.10
source activate dllm-rl
pip install torch==2.6.0
pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt
```
