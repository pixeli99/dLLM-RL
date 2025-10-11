from models import DreamTokenizer, DreamModel
from generate import block_diffusion_generate, block_diffusion_generate_multi_think
import torch
import os
import html as html_lib

model_name = "/zju_0038/pengxiang/dLLM-RL/sft_dream/ckpt/optimized"
model = DreamModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = DreamTokenizer.from_pretrained(model_name, trust_remote_code=True)

problem = '''Charlie wants to sell beeswax candles. For every pound of beeswax, he can make 10 tapered candles. One pound of beeswax and the wicks cost $10.00 in supplies. If he sells each candle for $2.00 each, what is his net profit if he makes and sells 20 candles?'''

system_prompts = f'''<|im_start|>user\n{problem}\nYou FIRST think about the reasoning process as an internal monologue and then summarize the reasoning process to get the final answer. The summary process MUST BE enclosed within <summary> </summary> tags.<|im_end|>\n<|im_start|>assistant\n'''

# Tokenize prompt and move tensors to the model device
tokens = tokenizer.batch_encode_plus(
    [system_prompts], return_tensors='pt', padding=True, truncation=True, max_length=1024
)
tokens = {k: v.to(model.device) for k, v in tokens.items()}

# Call diffusion_generate with explicit input_ids and attention_mask
output = model.diffusion_generate(
    tokens["input_ids"],
    attention_mask=None,
    max_new_tokens=1024,
    output_history=True,
    return_dict_in_generate=True,
    steps=1024,
    temperature=1.0,
    top_p=0.95,
    alg="topk_margin",
    alg_temp=0.0,
)

# Decode only the newly generated tokens (after the prompt length)
input_ids = tokens["input_ids"]
sequences = output.sequences

gens = []
tokenizer.eos_token = "<|im_end|>"
for i in range(sequences.size(0)):
    prompt_len = input_ids[i].size(0)
    gen_ids = sequences[i, prompt_len:]
    text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=False)
    if tokenizer.eos_token:
        text = text.split(tokenizer.eos_token)[0]
    gens.append(text)

print(gens[0])

# ------------------------------
# HTML visualization of decode order per token
# ------------------------------

def escape_html(s: str) -> str:
    return html_lib.escape(s, quote=False)

def to_color(step: int, min_step: int, max_step: int) -> str:
    # Map step in [min_step, max_step] to hue 240 (blue) -> 0 (red)
    if step < 0:
        return "#cccccc"  # unfilled mask
    if max_step == min_step:
        hue = 0
    else:
        ratio = (step - min_step) / float(max_step - min_step)
        hue = int(240 - 240 * ratio)
    return f"hsl({hue}, 85%, 75%)"

def make_html_decode_view(token_ids, steps, tokenizer, title="Decode Order Visualization"):
    # token_ids, steps are lists of equal length (generated region only)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    filled_steps = [s for s in steps if s >= 0]
    min_s = min(filled_steps) if filled_steps else 0
    max_s = max(filled_steps) if filled_steps else 0

    spans = []
    for idx, (tid, st) in enumerate(zip(token_ids, steps)):
        tok = tokens[idx]
        # make spaces visible for BPE-style tokens
        tok_disp = tok.replace("▁", "␣").replace("Ġ", "␣")
        if tok_disp == "":
            tok_disp = "∅"
        color = to_color(st, min_s, max_s)
        tip = f"step {st}" if st >= 0 else "unfilled"
        spans.append(f"<span class=tok style=\"background:{color}\" title=\"{tip}\">{escape_html(tok_disp)}</span>")

    gradient = "".join(
        f"<span class=legend-step style=\"background:{to_color(s, min_s, max_s)}\"></span>" for s in range(min_s, max_s + 1)
    ) if filled_steps else ""

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <title>{escape_html(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }}
    .wrap {{ white-space: pre-wrap; word-wrap: break-word; }}
    .tok {{ display:inline-block; margin:2px 2px; padding:2px 4px; border-radius:4px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    .legend {{ margin: 10px 0; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
    .legend-step {{ width:12px; height:12px; display:inline-block; border-radius:2px; margin-right:1px; }}
    .muted {{ color:#555; font-size: 12px; }}
    .header {{ margin-bottom: 8px; }}
    .box {{ border:1px solid #ddd; border-radius:8px; padding:12px; }}
  </style>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <meta name=\"robots\" content=\"noindex\">
  <link rel=\"icon\" href=\"data:,\">
  </head>
<body>
  <div class=\"header\"><strong>{escape_html(title)}</strong></div>
  <div class=\"legend\">
    <span class=\"muted\">early</span>
    <div>{gradient}</div>
    <span class=\"muted\">late</span>
  </div>
  <div class=\"box wrap\">{''.join(spans)}</div>
  <div class=\"muted\" style=\"margin-top:8px\">Hover tokens to see step indices.</div>
</body>
</html>
"""
    return html

# Build per-position first-fill steps from history
history = output.history or []
seq = sequences[0].tolist()
prompt_len = tokens["input_ids"].shape[1]
mask_id = getattr(model.config, "mask_token_id", None)
fill_step = [-1] * len(seq)

if mask_id is None:
    # Try tokenizer special token
    mask_id = tokenizer.mask_token_id

if history and mask_id is not None:
    for s_idx, x in enumerate(history, start=1):
        xrow = x[0].tolist()
        for pos in range(prompt_len, len(seq)):
            if fill_step[pos] == -1 and xrow[pos] != mask_id:
                fill_step[pos] = s_idx

# Determine end position (stop at eos if present, else last filled token)
eos_id = tokenizer.eos_token_id
end_pos = len(seq)
if eos_id is not None:
    for j in range(prompt_len, len(seq)):
        if seq[j] == eos_id:
            end_pos = j
            break
else:
    filled_positions = [i for i in range(prompt_len, len(seq)) if fill_step[i] >= 0]
    if filled_positions:
        end_pos = max(filled_positions) + 1

gen_ids = seq[prompt_len:end_pos]
gen_steps = fill_step[prompt_len:end_pos]

html_out = make_html_decode_view(gen_ids, gen_steps, tokenizer)
out_dir = os.path.join(os.getcwd(), "outputs")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "decode_order.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_out)

print(f"[decode-order] Wrote HTML to: {out_path}")
