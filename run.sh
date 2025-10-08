export HF_ENDPOINT=https://hf-mirror.com
python trado_rl_rollout.py \
  config=../configs/rl_trado.yaml \
  experiment.function=evaluation \
  evaluation.eval_dataset=MATH500 \
  experiment.current_epoch=1 \
  evaluation.num_response_per_task=2 \
  evaluation.tensor_parallel_size=1 \
  evaluation.max_active=1