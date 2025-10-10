from models import DreamTokenizer, DreamModel
from generate import block_diffusion_generate, block_diffusion_generate_multi_think
import torch

model_name = "/zju_0038/pengxiang/dLLM-RL/sft_dream/ckpt/optimized"
model = DreamModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = DreamTokenizer.from_pretrained(model_name, trust_remote_code=True)

problem = '''Charlie wants to sell beeswax candles. For every pound of beeswax, he can make 10 tapered candles. One pound of beeswax and the wicks cost $10.00 in supplies. If he sells each candle for $2.00 each, what is his net profit if he makes and sells 20 candles?'''

system_prompts = f'''<|im_start|>user\n{problem}\nYou FIRST think about the reasoning process as an internal monologue and then summarize the reasoning process to get the final answer. The summary process MUST BE enclosed within <summary> </summary> tags.<|im_end|>\n<|im_start|>assistant\n<think 1>'''

# Tokenize prompt and move tensors to the model device
tokens = tokenizer.batch_encode_plus(
    [system_prompts], return_tensors='pt', padding=True, truncation=True, max_length=1024
)
tokens = {k: v.to(model.device) for k, v in tokens.items()}

# Call diffusion_generate with explicit input_ids and attention_mask
output = model.diffusion_generate(
    tokens["input_ids"],
    attention_mask=tokens.get("attention_mask", None),
    max_new_tokens=768,
    output_history=True,
    return_dict_in_generate=True,
    steps=512,
    temperature=1.0,
    top_p=0.95,
    alg="entropy",
    alg_temp=0.0,
)

# Decode only the newly generated tokens (after the prompt length)
input_ids = tokens["input_ids"]
sequences = output.sequences

gens = []
tokenizer.eos_token = "</summary>"
for i in range(sequences.size(0)):
    prompt_len = input_ids[i].size(0)
    gen_ids = sequences[i, prompt_len:]
    text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=False)
    if tokenizer.eos_token:
        text = text.split(tokenizer.eos_token)[0]
    gens.append(text)

print(gens[0])
