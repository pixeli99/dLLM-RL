import os as _os

# Ensure CUDA devices follow ascending PCI order for reproducibility
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Consolidate caches onto fast local storage when available
_cache_root = "/dev/shm/torch_cache"
_os.makedirs(_cache_root, exist_ok=True)
_os.environ.setdefault("TORCH_EXTENSIONS_DIR", _os.path.join(_cache_root, "torch_extensions"))
_os.environ.setdefault("TRITON_CACHE_DIR", _os.path.join(_cache_root, "triton"))
_os.environ.setdefault("XDG_CACHE_HOME", _cache_root)
_os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

_os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
_os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
_os.environ.pop("NCCL_BLOCKING_WAIT", None)
_os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

import json
import math
import multiprocessing as mp
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from jinja2 import Template
from termcolor import cprint
from transformers import AutoModelForCausalLM, AutoTokenizer

from omegaconf import OmegaConf

from generate import (
    block_diffusion_generate,
    block_diffusion_generate_multi_think,
)


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def render_prompt(template: str, question: str) -> str:
    return Template(template).render(problem=question)


def extract_final_boxed_answer(s: str) -> str:
    tag = r"\\boxed{"  # last \boxed{ ... }
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1
    buf: List[str] = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1

    return "".join(buf) if depth == 0 else "Can not extract the answer!"


def get_data_chunk(data, num_nodes: int, node_idx: int):
    total = len(data)
    start = (total * node_idx) // num_nodes
    end = (total * (node_idx + 1)) // num_nodes
    return data[start:end]


def ensure_think_tokens(tokenizer, num_thinks: int) -> List[int]:
    ids: List[int] = []
    for i in range(1, num_thinks + 1):
        token = f"<think {i}>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer lacks required special token {token}.")
        ids.append(int(token_id))
    return ids


def decode_block(
    tokenizer,
    token_tensor: torch.Tensor,
    start_tag: str,
    end_tag: str,
) -> str:
    text = tokenizer.decode(token_tensor, skip_special_tokens=False)
    if end_tag in text:
        text = text.split(end_tag, 1)[0] + end_tag
    text = text.replace("<|MASK|>", "").replace("<|endoftext|>", "").strip()
    if not text.startswith(start_tag):
        text = start_tag + text
    return text


def truncate_at_token(sequence: torch.Tensor, stop_id: Optional[int]) -> torch.Tensor:
    if stop_id is None or sequence.numel() == 0:
        return sequence
    indices = (sequence == stop_id).nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return sequence
    return sequence[: indices[0]]


def chunk_evenly(lst: List, n_chunks: int) -> List[List]:
    if n_chunks <= 1 or len(lst) == 0:
        return [lst]
    chunk_size = math.ceil(len(lst) / n_chunks)
    chunks = []
    for start in range(0, len(lst), chunk_size):
        chunks.append(lst[start : start + chunk_size])
    while len(chunks) < n_chunks:
        chunks.append([])
    return chunks


def _worker_entry(args):
    (
        worker_id,
        device_id,
        prompts,
        global_indices,
        config_dict,
        model_path,
        seed,
    ) = args

    if not prompts:
        return []

    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if device_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(0)

    load_kwargs: Dict = {"trust_remote_code": True}
    if device_id is not None and torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "cuda"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    model_device = next(model.parameters()).device
    if not hasattr(model, "device"):
        model.device = model_device

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer is missing a mask token; required for block diffusion generation.")

    think_token_ids = ensure_think_tokens(tokenizer, config_dict["num_thinks"])
    eos_token_id = tokenizer.eos_token_id

    results: List[Tuple[int, str, int]] = []
    think_block_length = config_dict["think_block_length"]
    think_total_length = think_block_length * config_dict["num_thinks"]

    for prompt_text, global_idx in zip(prompts, global_indices):
        run_seed = seed + global_idx + worker_id * 17
        random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        try:
            tokens = tokenizer.batch_encode_plus(
                [prompt_text],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=config_dict["max_sequence_length"],
            )
            tokens = {k: v.to(model_device) for k, v in tokens.items()}

            multi_output = block_diffusion_generate_multi_think(
                model,
                prompt=tokens,
                mask_id=tokenizer.mask_token_id,
                think_token_ids=think_token_ids,
                think_block_length=think_block_length,
                num_thinks=config_dict["num_thinks"],
                gen_length=think_total_length,
                temperature=config_dict["temperature"],
                top_k=config_dict["top_k"],
                top_p=config_dict["top_p"],
            )

            prompt_len = tokens["input_ids"].shape[1]
            think_blocks: List[str] = []
            for think_idx in range(config_dict["num_thinks"]):
                block_start = prompt_len + think_idx * think_block_length
                block_end = block_start + think_block_length
                block_tokens = multi_output[0, block_start:block_end]
                start_tag = f"<think {think_idx + 1}>"
                end_tag = f"</think {think_idx + 1}>"
                think_text = decode_block(tokenizer, block_tokens, start_tag, end_tag)
                think_blocks.append(think_text)

            assistant_body = "\n\n".join(think_blocks).rstrip() + "\n\n"

            summary_prompt = f"{prompt_text}{assistant_body}<summary>"
            summary_tokens = tokenizer.batch_encode_plus(
                [summary_prompt],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=config_dict["max_sequence_length"],
            )
            summary_tokens = {k: v.to(model_device) for k, v in summary_tokens.items()}

            summary_output = block_diffusion_generate(
                model,
                prompt=summary_tokens,
                mask_id=tokenizer.mask_token_id,
                gen_length=config_dict["summary_max_tokens"],
                block_length=think_block_length,
                denoising_steps=think_block_length,
                temperature=config_dict["temperature"],
                top_k=config_dict["top_k"],
                top_p=config_dict["top_p"],
                remasking_strategy="low_confidence_static",
                confidence_threshold=config_dict["dynamic_threshold"],
                stopping_criteria_idx=[eos_token_id] if eos_token_id is not None else None,
            )

            summary_prompt_len = summary_tokens["input_ids"].shape[1]
            summary_tail = summary_output[0, summary_prompt_len:]
            summary_tail = truncate_at_token(summary_tail, eos_token_id)
            summary_text = tokenizer.decode(summary_tail, skip_special_tokens=False)
            summary_text = summary_text.replace("<|MASK|>", "").replace("<|endoftext|>", "").strip()
            if "</summary>" not in summary_text:
                summary_text = summary_text.rstrip() + "</summary>"
            summary_section = f"<summary>{summary_text}"

            full_text = f"{assistant_body}{summary_section}"
            response_len = len(tokenizer.encode(full_text, add_special_tokens=False))

        except Exception as exc:  # noqa: BLE001
            full_text = f"Generation failed: {exc}"
            response_len = 0

        results.append((global_idx, full_text, response_len))

    return results


if __name__ == "__main__":
    config = get_config()

    seed = OmegaConf.select(config, "experiment.seed", default=42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_name = config.dataset.eval_dataset
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    dataset_path = os.path.join(dataset_dir, dataset_name + ".json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_node = int(config.experiment.num_node)
    node_index = int(config.experiment.node_index)
    if num_node > 1:
        data = get_data_chunk(data, num_node, node_index)

    num_samples = len(data)
    k_sample = int(config.rollout.num_response_per_task)
    temperature = float(config.rollout.temperature)
    top_k = int(config.rollout.top_k)
    top_p = float(config.rollout.top_p)

    num_thinks = OmegaConf.select(config, "rollout.num_thinks", default=3)
    think_block_length = OmegaConf.select(config, "rollout.think_block_length", default=256)
    summary_max_tokens = OmegaConf.select(config, "rollout.summary_max_token", default=256)
    max_sequence_length = OmegaConf.select(config, "rollout.max_sequence_length", default=2048)
    dynamic_threshold = OmegaConf.select(config, "rollout.dynamic_threshold", default=0.9)

    think_total_length = think_block_length * num_thinks
    if summary_max_tokens <= 0:
        summary_max_tokens = think_block_length
    if think_total_length <= 0:
        raise ValueError("Invalid think block configuration: total generated length must be positive.")

    system_prompts = (
        "<|im_start|>user\n"
        "{{problem}}\n"
        "You FIRST write three independent reasoning traces labelled <think 1>, <think 2>, and <think 3>. "
        "Each trace should end with its corresponding </think i> tag. After the three think segments, summarize the reasoning "
        "inside <summary> </summary> and present the final numeric answer inside \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    model_path = os.path.expanduser(config.model)

    cprint(f"Loaded {num_samples} examples from {dataset_name}. Generating {k_sample} responses per example...", "green")

    for entry in data:
        entry["full_output"] = []
        entry["step_map"] = []
        entry["extracted_output"] = []
        entry["response_length"] = []
        entry["prompt"] = render_prompt(system_prompts, entry["question"])

    generation_prompts: List[str] = []
    index_list: List[int] = []
    for i, item in enumerate(data):
        for _ in range(k_sample):
            generation_prompts.append(item["prompt"])
            index_list.append(i)

    total_prompts = len(generation_prompts)
    global_indices = list(range(total_prompts))

    shuffled_order = list(range(total_prompts))
    random.shuffle(shuffled_order)
    shuffled_prompts = [generation_prompts[i] for i in shuffled_order]
    shuffled_global = [global_indices[i] for i in shuffled_order]

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        visible_gpus = [x.strip() for x in cvd.split(",") if x.strip() != ""]
        device_ids = [int(x) for x in visible_gpus]
    else:
        device_ids = list(range(torch.cuda.device_count()))

    if not device_ids:
        device_ids = [None]

    num_workers = min(len(device_ids), len(shuffled_prompts)) or 1
    device_ids = device_ids[:num_workers]

    prompt_chunks = chunk_evenly(shuffled_prompts, num_workers)
    global_chunks = chunk_evenly(shuffled_global, num_workers)

    worker_config = {
        "num_thinks": num_thinks,
        "think_block_length": think_block_length,
        "summary_max_tokens": summary_max_tokens,
        "max_sequence_length": max_sequence_length,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "dynamic_threshold": dynamic_threshold,
    }

    ctx = mp.get_context("spawn")
    worker_args = []
    for wid in range(num_workers):
        worker_args.append(
            (
                wid,
                device_ids[wid],
                prompt_chunks[wid],
                global_chunks[wid],
                worker_config,
                model_path,
                seed,
            )
        )

    with ctx.Pool(processes=num_workers) as pool:
        worker_results = pool.map(_worker_entry, worker_args)

    restored_outputs: List[Optional[str]] = [None] * total_prompts
    restored_lengths: List[Optional[int]] = [None] * total_prompts

    for chunk_results in worker_results:
        for global_idx, text, length in chunk_results:
            restored_outputs[global_idx] = text
            restored_lengths[global_idx] = length

    missing = [i for i, val in enumerate(restored_outputs) if val is None]
    if missing:
        raise RuntimeError(f"Missing outputs for indices: {missing[:10]}")

    for global_idx in range(total_prompts):
        entry_idx = index_list[global_idx]
        full_text = restored_outputs[global_idx] or ""
        response_len = restored_lengths[global_idx] or 0
        data[entry_idx]["full_output"].append(full_text)
        data[entry_idx]["step_map"].append([])
        data[entry_idx]["extracted_output"].append(extract_final_boxed_answer(full_text))
        data[entry_idx]["response_length"].append(response_len)

        if (global_idx + 1) % max(1, k_sample) == 0:
            processed = (global_idx + 1) // max(1, k_sample)
            if processed % 10 == 0 or processed == num_samples:
                cprint(f"Processed {processed}/{num_samples} problems", "cyan")

    outputs_name = "eval-" + config.model.replace("/", ".") + "-" + dataset_name
    if num_node > 1:
        output_file_name = os.path.join(
            "..",
            config.experiment.project,
            "temp_data",
            f"outputs-{node_index}-" + outputs_name + ".json",
        )
    else:
        output_file_name = os.path.join(
            "..",
            config.experiment.project,
            "temp_data",
            "outputs-" + outputs_name + ".json",
        )

    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), output_file_name))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    cprint(f"Results saved to {output_path}", "green")
