import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PATH_TEMPLATE = "<think {idx}>\n{content}\n</think {idx}>"
SUMMARY_TEMPLATE = "<summary>\n{content}\n</summary>"


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {line_num} of {path}") from exc


def _maybe_pad_delete(
    content: str,
    rng: random.Random,
    prob: float,
    token: str,
    min_n: int,
    max_n: int,
) -> str:
    if prob <= 0 or min_n <= 0 or max_n <= 0 or min_n > max_n:
        return content
    if rng.random() >= prob:
        return content
    k = rng.randint(min_n, max_n)
    pad = "".join([token] * k)
    # Append inside the path block (before closing tag) for structural stability
    if content and not content.endswith("\n"):
        content += "\n"
    return content + pad


def format_paths(
    paths: Dict[str, str],
    *,
    rng: random.Random,
    pad_prob: float,
    pad_token: str,
    pad_min: int,
    pad_max: int,
) -> List[str]:
    ordered_items = sorted(paths.items(), key=lambda kv: int(kv[0]))
    formatted_blocks = []
    for key, content in ordered_items:
        content = (content or "").strip()
        if not content:
            continue
        content = _maybe_pad_delete(content, rng, pad_prob, pad_token, pad_min, pad_max)
        formatted_blocks.append(PATH_TEMPLATE.format(idx=key, content=content))
    return formatted_blocks


def format_summary(summaries: Sequence[str]) -> str:
    clean_chunks = [chunk.strip() for chunk in summaries if chunk and chunk.strip()]
    if not clean_chunks:
        return ""
    summary_body = "\n\n".join(clean_chunks)
    return SUMMARY_TEMPLATE.format(content=summary_body)


def build_record(
    entry: Dict[str, Any],
    prompt_template: str,
    *,
    rng: random.Random,
    pad_prob: float,
    pad_token: str,
    pad_min: int,
    pad_max: int,
) -> Dict[str, str]:
    question = (entry.get("question") or "").strip()
    if not question:
        raise ValueError(f"Entry {entry.get('id', '<unknown>')} is missing a question")

    paths_raw = entry.get("paths")
    if not isinstance(paths_raw, dict) or not paths_raw:
        raise ValueError(f"Entry {entry.get('id', '<unknown>')} must contain a non-empty 'paths' dict")

    path_blocks = format_paths(
        paths_raw,
        rng=rng,
        pad_prob=pad_prob,
        pad_token=pad_token,
        pad_min=pad_min,
        pad_max=pad_max,
    )
    if not path_blocks:
        raise ValueError(f"Entry {entry.get('id', '<unknown>')} has no usable path content")

    summary_raw = entry.get("summary")
    if summary_raw is None:
        summary_raw = entry.get("summaries", [])
    if isinstance(summary_raw, str):
        summary_seq: Sequence[str] = [summary_raw]
    elif summary_raw is None:
        summary_seq = []
    else:
        summary_seq = list(summary_raw)
    summary_block = format_summary(summary_seq)

    response_sections = path_blocks + ([summary_block] if summary_block else [])
    response = "\n\n".join(response_sections)

    prompt = prompt_template.format(question=question)
    return {"prompt": prompt, "response": response}


def convert_dataset(
    input_path: Path,
    output_path: Path,
    prompt_template: str,
    *,
    seed: int,
    pad_prob: float,
    pad_token: str,
    pad_min: int,
    pad_max: int,
) -> None:
    rng = random.Random(seed)
    records = []
    for entry in load_jsonl(input_path):
        record = build_record(
            entry,
            prompt_template,
            rng=rng,
            pad_prob=pad_prob,
            pad_token=pad_token,
            pad_min=pad_min,
            pad_max=pad_max,
        )
        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert multi-path think dataset into SFT JSON format.")
    parser.add_argument("input", type=Path, help="Path to the source JSONL file")
    parser.add_argument("output", type=Path, help="Where to write the processed JSON file")
    parser.add_argument(
        "--prompt-template",
        default="{question}",
        help="Template for the prompt text. Use {question} as placeholder.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for padding decisions.")
    parser.add_argument("--pad-delete-prob", type=float, default=0.0, help="Probability to append <delete> tokens at the end of each path block.")
    parser.add_argument("--pad-delete-min", type=int, default=0, help="Minimum number of <delete> tokens to append when padding triggers.")
    parser.add_argument("--pad-delete-max", type=int, default=0, help="Maximum number of <delete> tokens to append when padding triggers.")
    parser.add_argument("--pad-delete-token", type=str, default="<delete>", help="Token string to use for padding.")
    args = parser.parse_args()
    convert_dataset(
        args.input,
        args.output,
        args.prompt_template,
        seed=args.seed,
        pad_prob=args.pad_delete_prob,
        pad_token=args.pad_delete_token,
        pad_min=args.pad_delete_min,
        pad_max=args.pad_delete_max,
    )
