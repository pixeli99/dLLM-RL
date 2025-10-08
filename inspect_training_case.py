import os
import sys
import json
import argparse

import torch
from omegaconf import OmegaConf

# Ensure local imports work when running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train.prompting_utils import UniversalPrompting  # noqa: E402


def load_config(config_path: str):
    cfg = OmegaConf.load(config_path)
    return cfg


def pick_tokenizer_path(cfg) -> str:
    # Prefer the saved tokenizer in ckpt if available; otherwise fall back to pretrained
    project_dir = cfg.experiment.project
    save_dir = os.path.join(project_dir, "ckpt", cfg.model.optimized_name)
    if os.path.isdir(save_dir):
        return save_dir
    return cfg.model.pretrained_model


def compute_keep_indices(tokenizer, prompts, max_prompt_len):
    enc = tokenizer(prompts, padding=False, truncation=False, return_length=True)
    lengths = enc["length"]
    keep_indices = [i for i, L in enumerate(lengths) if L <= max_prompt_len]
    return keep_indices


def collect_training_data_semi_ar(input_ids: torch.Tensor,
                                  start_pos: int,
                                  mask_id: int,
                                  pad_id: int,
                                  lower: float,
                                  upper: float,
                                  post_num: int | None,
                                  seed: int | None = None):
    """
    Vectorized reproduction of the 'semi-ar' branch in train/sft_trado.py to build
    extended_input_ids, p_mask, tok_idx_ext, labels for a given batch (B x L).
    """
    if seed is not None:
        torch.manual_seed(seed)

    B, L = input_ids.shape
    L0 = start_pos
    L1 = L - L0

    extended_input_ids_list, pmask_list = [], []

    for b in range(B):
        # Sample per-position mask probabilities uniformly in [lower, upper]
        prob_ramp = torch.empty(L1).uniform_(lower, upper)
        # Sample Bernoulli decisions per position
        rand_tail = torch.rand(L1)
        pmask_tail = rand_tail <= prob_ramp  # [L1]

        pmask_b = torch.cat([
            torch.zeros(L0, dtype=torch.bool),
            pmask_tail
        ], dim=0)  # [L]

        noise_tail = input_ids[b, L0:].clone()
        noise_tail.masked_fill_(pmask_tail, mask_id)
        extended_b = torch.cat([input_ids[b], noise_tail], dim=0)  # [L + L1]

        extended_input_ids_list.append(extended_b)
        pmask_list.append(pmask_b)

    extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
    p_mask = torch.stack(pmask_list, dim=0).to(torch.bool)

    # Post-process: limit supervision on padded positions if configured
    pad_resp = (extended_input_ids[:, :L] == pad_id) & p_mask
    if post_num is not None:
        cum_pad = torch.cumsum(pad_resp.int(), dim=1)
        p_mask &= ~(pad_resp & (cum_pad > post_num))

    # Labels are the original sequence (not the appended tail)
    labels = extended_input_ids[:, :L].clone()

    # Build position ids exactly as in training
    idx = torch.arange(L).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
    valid = (idx >= start_pos) | extended_input_ids[:, :L].ne(pad_id)
    tok_idx = valid.long().cumsum(dim=-1) - 1
    tok_idx = tok_idx.masked_fill(~valid, 1)
    tok_idx_resp = tok_idx[:, start_pos:]
    tok_idx_ext = torch.cat([tok_idx, tok_idx_resp], dim=1)

    return extended_input_ids, p_mask, tok_idx_ext, labels


def main():
    parser = argparse.ArgumentParser(description="Inspect a single training case as constructed in sft_trado.py")
    parser.add_argument("--config", type=str, default="sft_trado/config.yaml", help="Path to training config.yaml")
    parser.add_argument("--index", type=int, default=0, help="Index within the kept subset (after prompt length filter)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for semi-ar masking for reproducibility")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve tokenizer path
    tokenizer_path = pick_tokenizer_path(cfg)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Load dataset
    dataset_path = os.path.join("data", cfg.dataset.optimization_data + ".json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    prompt_list = [x["prompt"] for x in dataset]
    response_list = [x["response"] for x in dataset]

    # Filter by max_prompt_len to match training
    keep_idx = compute_keep_indices(tokenizer, prompt_list, cfg.training.max_prompt_len)
    if len(keep_idx) == 0:
        raise RuntimeError("No samples kept after max_prompt_len filtering.")

    kept_prompts = [prompt_list[i] for i in keep_idx]
    kept_responses = [response_list[i] for i in keep_idx]

    sel = max(0, min(args.index, len(kept_prompts) - 1))

    # Reproduce UniversalPrompting to build the LM inputs with dataset-level padding and max_gen_length
    uni = UniversalPrompting(
        tokenizer,
        max_prompt_len=cfg.training.max_prompt_len,
        max_gen_length=cfg.training.max_gen_length,
        ignore_id=-100,
    )

    # Instead of calling __call__ (which re-filters), we emulate its internal steps to keep index mapping stable
    # 1) Tokenize with dataset-level padding
    tokenizer.padding_side = "left"
    prompt_ids = tokenizer(kept_prompts, padding=True, return_tensors="pt")["input_ids"]
    tokenizer.padding_side = "right"
    response_ids = tokenizer(kept_responses, padding=True, return_tensors="pt")["input_ids"]

    # 2) Build input_ids as in lm_prompt()
    input_ids_lm, labels_lm, start_pos = uni.lm_prompt((prompt_ids, response_ids))

    # Pick the selected sample row
    inp = input_ids_lm[sel].unsqueeze(0)  # [1, L]
    L = inp.shape[1]
    L0 = start_pos
    L1 = L - L0

    # Build step_map only if needed (method == trace). For semi-ar we can skip.
    # Reproduce training semi-ar masking and extended sequences for a single sample
    extended_input_ids, p_mask, tok_idx_ext, labels = collect_training_data_semi_ar(
        input_ids=inp,
        start_pos=L0,
        mask_id=tokenizer.mask_token_id,
        pad_id=tokenizer.pad_token_id,
        lower=cfg.training.lower_p,
        upper=cfg.training.upper_p,
        post_num=cfg.training.get("post_num", None),
        seed=args.seed,
    )

    # Decode and print informative views
    def decode(toks):
        return tokenizer.decode(toks, skip_special_tokens=False)

    raw_prompt = kept_prompts[sel]
    raw_response = kept_responses[sel]
    model_prompt_text = decode(inp[0, :L0])
    model_response_text = decode(inp[0, L0:L0 + L1])
    appended_tail_text = decode(extended_input_ids[0, L0 + L1:])

    print("=== Training Case Inspection ===")
    print(f"Config: {args.config}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Kept size: {len(keep_idx)}; Selected kept index: {sel}")
    print(f"Shapes: L={L}, L0(start_pos)={L0}, L1={L1}")
    print("")
    print("-- Raw sample (from JSON) --")
    print("[prompt]\n" + raw_prompt)
    print("[response]\n" + raw_response)
    print("")
    print("-- Tokenized (fed to model before extension) --")
    print("[prompt tokens decoded]\n" + model_prompt_text)
    print("[response tokens decoded]\n" + model_response_text)
    print("")
    print("-- Extended tail (noisy; appended during training) --")
    print(appended_tail_text)
    print("")
    print("-- Supervision mask stats --")
    pm = p_mask[0]
    num_supervised = int(pm.sum().item())
    print(f"p_mask true count: {num_supervised} (of L={L})")
    tail_supervised = int(pm[L0:].sum().item())
    print(f"tail supervised positions: {tail_supervised} (of L1={L1})")
    true_indices = torch.nonzero(pm[L0:], as_tuple=False).squeeze(-1).tolist()
    if isinstance(true_indices, int):
        true_indices = [true_indices]
    print(f"first 32 tail supervised indices: {true_indices[:32]}")
    print("")
    print("-- Position ids (tok_idx_ext) shapes --")
    print(f"tok_idx_ext shape: {tuple(tok_idx_ext.shape)} (should be [1, L + L1])")


if __name__ == "__main__":
    main()




