from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import block_diffusion_generate, block_diffusion_generate_multi_think
import torch

model_name = "/zju_0038/pengxiang/dLLM-RL/sft_trado/ckpt/optimized"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="float16", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

problem = '''Charlie wants to sell beeswax candles. For every pound of beeswax, he can make 10 tapered candles. One pound of beeswax and the wicks cost $10.00 in supplies. If he sells each candle for $2.00 each, what is his net profit if he makes and sells 20 candles?'''

system_prompts = f'''<|im_start|>user\n{problem}\nYou FIRST think about the reasoning process as an internal monologue and then summarize the reasoning process to get the final answer. The summary process MUST BE enclosed within <summary> </summary> tags.<|im_end|>\n<|im_start|>assistant\n'''
tokens = tokenizer.batch_encode_plus([system_prompts], return_tensors='pt', padding=True, truncation=True, max_length=1024)
tokens = {k: v.to(model.device) for k, v in tokens.items()}
think_token_ids = []
for i in range(1, 4):
    token_id = tokenizer.convert_tokens_to_ids(f"<think {i}>")
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer lacks special token <think {i}>")
    think_token_ids.append(token_id)
multi_output = block_diffusion_generate_multi_think(
    model,
    prompt=tokens,
    mask_id=tokenizer.mask_token_id,
    think_token_ids=think_token_ids,
    think_block_length=128,
    num_thinks=3,
    gen_length=384,
    temperature=1.0,
    top_k=0,
    top_p=1.0
)

prompt_length = tokens["input_ids"].shape[1]
think_blocks = []
for i in range(3):
    block_start = prompt_length + i * 256
    block_end = block_start + 256
    block_tokens = multi_output[0, block_start:block_end]
    think_text = tokenizer.decode(block_tokens, skip_special_tokens=False)
    end_tag = f"</think {i+1}>"
    if end_tag in think_text:
        think_text = think_text.split(end_tag, 1)[0] + end_tag
    think_text = think_text.replace('<|MASK|>', '').replace('<|endoftext|>', '')
    think_text = think_text.strip()
    think_blocks.append(think_text)

assistant_body = '\n\n'.join(think_blocks).rstrip() + '\n\n'

summary_prompt = f"{system_prompts}{assistant_body}<summary>"
summary_tokens = tokenizer.batch_encode_plus(
    [summary_prompt],
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=1024
)
summary_tokens = {k: v.to(model.device) for k, v in summary_tokens.items()}

tokenizer.eos_token_id = 151645
summary_output = block_diffusion_generate(
    model,
    prompt=summary_tokens,
    mask_id=tokenizer.mask_token_id,
    gen_length=256,
    block_length=16,
    denoising_steps=256,
    temperature=0.1,
    top_k=0,
    top_p=1.0,
    remasking_strategy="low_confidence_static",
    confidence_threshold=0.9,
    stopping_criteria_idx=[151645]
)

summary_prompt_len = summary_tokens["input_ids"].shape[1]
summary_tail = summary_output[0, summary_prompt_len:]
stop_id = tokenizer.eos_token_id
if stop_id is not None:
    indices = (summary_tail == stop_id).nonzero(as_tuple=True)[0]
    if indices.numel() > 0:
        summary_tail = summary_tail[:indices[0]]

summary_text = tokenizer.decode(summary_tail, skip_special_tokens=False)
summary_text = summary_text.replace('<|MASK|>', '').replace('<|endoftext|>', '')
summary_section = f"<summary>{summary_text}"

full_text = f"{assistant_body}{summary_section}"
print(full_text)
