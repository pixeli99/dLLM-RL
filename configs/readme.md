# Instructions

We introduce the recommended configs for different tasks and explain how to modify your own configs. Each config file governs one project/job. Make sure you fill in all the required parameters in the config file before using it; at the very least, this includes the dataset name, the model name, and the num of gpus you have. The details on how to modify the given example configs, along with definitions of their elements, are included within them.



## Which one to use?


### Evaluation:

For TraDo instruction and SDAR models, use `sdar_eval.yaml` or `sdar_multinode_eval.yaml`. 

For long-CoT model, TraDo-8B-Thinking, use `trado_longcot_eval.yaml` or `trado_longcot_multinode_eval.yaml`. 

For Dream series and diffu-coder, use `dream_eval.yaml` or `dream_multinode_eval.yaml`. 

For LLaDA series and MMaDA, use `llada_eval.yaml` or `llada_multinode_eval.yaml`.

### SFT:

For TraDo and SDAR models, use `sft_sdar.yaml`. 

For dream and diffu-coder, use `sft_dream.yaml`. 

For LLaDA and MMaDA, use `sft_llada.yaml`.

### RL:

For TraDo and SDAR models, use `rl_sdar.yaml` or `multinode_rl_sdar.yaml`. 

For dream and diffu-coder, use `sft_dream.yaml` or `multinode_rl_dream.yaml`. 

For LLaDA and MMaDA, use `sft_llada.yaml` or `multinode_rl_llada.yaml`. 

If use value model, use `rl_sdar_with_value.yaml` or `multinode_rl_sdar_with_value.yaml`. 

We also support coding rl, see an example script `rl_sdar_code.yaml`.



## Main required fields:

The model name



## Create your own configs:

Keep `experiment.project` same as the corresponding file name. You can first try some related example configs to get familiar with.

