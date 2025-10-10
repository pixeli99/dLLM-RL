import json
# assume you have downloaded this data
# Read JSONL (one JSON object per line)
with open("/zju_0038/pengxiang/data_gen/out.jsonl", 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]
print(len(data))

from jinja2 import Template

# different models use different prompts and eos token:

# for dream and diffucoder:
system_prompts = '''<|im_start|>user\n{{problem}}\nYou FIRST think about the reasoning process as an internal monologue and then summarize the reasoning process to get the final answer. The summary process MUST BE enclosed within <summary> </summary> tags.<|im_end|>\n<|im_start|>assistant\n'''
eos_token = "<|im_end|>"

# llada and mmada:
#system_prompts = """<|startoftext|><|start_header_id|>user<|end_header_id|>You need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
#eos_token = "<|eot_id|>"

# trado and sdar:
# non-cot prompt, we used this as demon example to compare sft methods
# system_prompts = '''<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant\n'''
# cot prompt, trado and sdar are not cot-default, needs cot prompt to activate reasoning ability.
#system_prompts = '''<|im_start|>user\n{{problem}}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'''
# eos_token = "<|im_end|>"


def get_prompt(data_i):
    return Template(system_prompts).render(problem = data_i["question"])
processed_data = []
for i in range(len(data)):
    processed_data.append(
        {
            "prompt": get_prompt(data[i]),
            "response": data[i]["packed"] + eos_token
        }
    )
len(processed_data)

#with open("./sft_openr1math_dream.json", "w", encoding="utf-8") as f:
#with open("./sft_openr1math_llada.json", "w", encoding="utf-8") as f:
#with open("./sft_openr1math_trado.json", "w", encoding="utf-8") as f:
with open("./sft_openr1.json", "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)
