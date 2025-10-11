import torch
from transformers import AutoTokenizer
from models.llada.modeling_llada import LLaDAModelLM


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64)
    noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / noise


@torch.no_grad()
def llada_generate_with_prefix_cache(
    model: LLaDAModelLM,
    input_ids: torch.Tensor,
    *,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    target: str = "confidence",
    mask_id: int = 50256,
    further_horizon: int | None = 128,
    use_cache: bool = True,
    dynamic_threshold: float | None = 0.95,
):
    """
    Single-sample generation aligned with sample/llada_sample.py logic.
    Returns sequences (B, L0+gen_length) and history list for visualization.
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1
    device = input_ids.device

    def get_num_transfer_tokens(mask_index: torch.Tensor, cur_steps: int) -> torch.Tensor:
        # mask_index: (B, L_block)
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // cur_steps
        remainder = mask_num % cur_steps
        out = torch.zeros(mask_num.size(0), cur_steps, device=mask_index.device, dtype=torch.int64) + base
        for i in range(mask_num.size(0)):
            out[i, : int(remainder[i].item())] += 1
        return out

    def get_transfer_index(
        logits: torch.Tensor,
        temperature: float,
        target: str,
        mask_index: torch.Tensor,
        x_cur: torch.Tensor,
        num_transfer_tokens: torch.Tensor,
        threshold: float | None,
    ):
        logits_noised = add_gumbel_noise(logits, temperature)
        x0 = torch.argmax(logits_noised, dim=-1)

        if target == "confidence":
            p = torch.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif target == "margin_confidence":
            p = torch.softmax(logits.to(torch.float64), dim=-1)
            top2 = torch.topk(p, 2, dim=-1).values
            x0_p = top2[..., 0] - top2[..., 1]
        elif target == "neg_entropy":
            p = torch.softmax(logits.to(torch.float64), dim=-1)
            x0_p = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
        elif target == "random":
            x0_p = torch.rand_like(x0, dtype=torch.float64)
        else:
            raise NotImplementedError(target)

        x0 = torch.where(mask_index, x0, x_cur)

        if threshold is not None:
            selected = mask_index & (x0_p >= threshold)
            has_mask = mask_index.any(dim=-1)
            none_sel = (~selected.any(dim=-1)) & has_mask
            if none_sel.any():
                masked_scores = x0_p.masked_fill(~mask_index, float("-inf"))
                best_idx = masked_scores.argmax(dim=-1)
                rows = torch.nonzero(none_sel, as_tuple=False).squeeze(-1)
                selected[rows, best_idx[rows]] = True
            return x0, selected

        confidence = x0_p.masked_fill(~mask_index, float("-inf"))
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            k = int(num_transfer_tokens[j].item() if torch.is_tensor(num_transfer_tokens[j]) else num_transfer_tokens[j])
            if k <= 0:
                continue
            _, sel = torch.topk(confidence[j], k=k)
            transfer_index[j, sel] = True
        return x0, transfer_index

    B, L0 = input_ids.shape
    x = torch.full((B, L0 + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :L0] = input_ids

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (i < rem) for i in range(num_blocks)]

    history: list[torch.Tensor] = []

    for blk in range(num_blocks):
        s = L0 + blk * block_length
        e = L0 + (blk + 1) * block_length

        if further_horizon is not None:
            window_end = min(e + further_horizon, L0 + gen_length)
            window_slice = slice(s, window_end)
        else:
            window_slice = slice(s, x.shape[1])

        cur_steps = steps_per_block[blk]
        num_transfer = get_num_transfer_tokens((x[:, s:e] == mask_id), cur_steps)

        # First forward to build (optional) prefix cache
        if use_cache:
            out = model(x, use_cache=True)
            pkv = out.past_key_values
            # Chop prefix out of past_kv to keep cache small
            new_pkv = tuple(tuple(t[:, :, :s] for t in layer) for layer in pkv)
            pkv = new_pkv
        else:
            out = model(x, use_cache=False)
            pkv = None

        mask_all = (x == mask_id)
        mask_all[:, e:] = 0

        x0, tr_idx = get_transfer_index(
            out.logits, temperature, target, mask_all, x, num_transfer[:, 0], dynamic_threshold
        )
        x[tr_idx] = x0[tr_idx]
        history.append(x.clone().cpu())

        i = 1
        while (x[:, s:e] == mask_id).any() and i < cur_steps:
            if use_cache:
                logits = model(x[:, window_slice], past_key_values=pkv, use_cache=True).logits
                x0, tr_idx = get_transfer_index(
                    logits, temperature, target, (x[:, window_slice] == mask_id), x[:, window_slice], num_transfer[:, i], dynamic_threshold
                )
                x[:, window_slice][tr_idx] = x0[tr_idx]
            else:
                logits = model(x, use_cache=False).logits
                logits = logits[:, s:]
                x0, tr_idx = get_transfer_index(
                    logits, temperature, target, (x[:, s:] == mask_id), x[:, s:], num_transfer[:, i], dynamic_threshold
                )
                x[:, s:][tr_idx] = x0[tr_idx]
            history.append(x.clone().cpu())
            i += 1

    return x, history


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/zju_0038/pengxiang/dLLM-RL/sft_llada/ckpt/optimized"

    # Load model/tokenizer similar to training
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    problem = '''Charlie wants to sell beeswax candles. For every pound of beeswax, he can make 10 tapered candles. One pound of beeswax and the wicks cost $10.00 in supplies. If he sells each candle for $2.00 each, what is his net profit if he makes and sells 20 candles?'''

    system_prompts = f'''<|im_start|>user\n{problem}\nYou FIRST think about the reasoning process as an internal monologue and then summarize the reasoning process to get the final answer. The summary process MUST BE enclosed within <summary> </summary> tags.<|im_end|>\n<|im_start|>assistant<think 1>'''


    enc = tokenizer([system_prompts], padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    mask_id = tokenizer.encode("<|mdm_mask|>")[0]

    seqs, history = llada_generate_with_prefix_cache(
        model,
        enc["input_ids"],
        steps=512,
        gen_length=512,
        block_length=512,
        temperature=0.0,
        target="confidence",
        mask_id=mask_id,
        use_cache=True
    )

    attn = enc.get("attention_mask")
    if attn is not None:
        prompt_len = int(attn[0].sum().item())
    else:
        prompt_len = enc["input_ids"].shape[1]

    out_text = tokenizer.decode(seqs[0, prompt_len:].tolist(), skip_special_tokens=True)
    print(out_text)

