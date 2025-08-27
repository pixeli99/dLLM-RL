# DLLM-RL


## 🌱 Features 

- Model Support: [SDAR](https://github.com/JetAstra/SDAR), [Dream](https://github.com/DreamLM/Dream), [LLaDA](https://github.com/ML-GSAI/LLaDA), [MMada](https://github.com/Gen-Verse/MMaDA), [Diffu-Coder](https://github.com/apple/ml-diffucoder)
- Inference: [KV-cache](https://github.com/NVlabs/Fast-dLLM/tree/main), [jetengine](https://github.com/Labman42/JetEngine/tree/0ddc55ad3fb712b6374515b78d656f420e1a7243) (based on nano-vllm), different sampling strategies, support multi-nodes
- RL: TraceRL (and it's PPO version), [Coupled RL](https://github.com/apple/ml-diffucoder), [random masking RL](https://github.com/Gen-Verse/MMaDA), accelerated sampling, support multi-nodes
- SFT: Flash block SFT, semi-AR SFT, random masking SFT, multi-nodes supported


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
