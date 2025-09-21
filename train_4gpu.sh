#!/bin/bash

# Activate virtual environment
source ../DreamOn/.venv/bin/activate

# Set CUDA devices (optional, accelerate will handle this)
export CUDA_VISIBLE_DEVICES=6,7

# Add the project root to PYTHONPATH
export PYTHONPATH=/zju_0038/pengxiang/dLLM-RL:$PYTHONPATH

accelerate launch \
  --config_file accelerate_configs/1_node_4_gpus_deepspeed_zero3.yaml \
  train/sft_dream.py \
  config=configs/sft_dream.yaml