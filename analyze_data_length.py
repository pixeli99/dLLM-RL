#!/usr/bin/env python
"""Analyze token lengths in JSONL data files"""

import json
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

def load_jsonl(path):
    """Load JSONL file"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_json(path):
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_raw_jsonl(file_path, tokenizer_name="Dream-org/Dream-v0-Instruct-7B"):
    """Analyze raw JSONL data with question, paths, and summaries"""
    print(f"Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except:
        try:
            print(f"Failed to load {tokenizer_name}, trying GPT2 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except:
            print(f"Using basic tokenizer with approximate counting")
            # Simple word-based approximation
            tokenizer = None

    print(f"Loading data from: {file_path}")
    data = load_jsonl(file_path)

    prompt_lengths = []
    response_lengths = []
    total_lengths = []

    print(f"Analyzing {len(data)} samples...")
    for item in tqdm(data):
        # Prompt is the question
        prompt = item.get("question", "")

        # Response includes all paths and summaries
        response_parts = []

        # Add paths
        if "paths" in item:
            for idx, content in item["paths"].items():
                response_parts.append(f"<think {idx}>\n{content}\n</think {idx}>")

        # Add summaries
        if "summary" in item:
            if isinstance(item["summary"], str):
                response_parts.append(f"<summary>\n{item['summary']}\n</summary>")
            elif isinstance(item["summary"], list):
                summary_text = "\n\n".join(item["summary"])
                response_parts.append(f"<summary>\n{summary_text}\n</summary>")
        elif "summaries" in item:
            if isinstance(item["summaries"], list):
                summary_text = "\n\n".join(item["summaries"])
                response_parts.append(f"<summary>\n{summary_text}\n</summary>")

        response = "\n\n".join(response_parts)

        # Tokenize
        if tokenizer:
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
        else:
            # Approximate: 1 token ≈ 0.75 words, 1 word ≈ 5 characters
            prompt_tokens = list(range(len(prompt) // 4))
            response_tokens = list(range(len(response) // 4))

        prompt_lengths.append(len(prompt_tokens))
        response_lengths.append(len(response_tokens))
        total_lengths.append(len(prompt_tokens) + len(response_tokens))

    return prompt_lengths, response_lengths, total_lengths

def analyze_processed_json(file_path, tokenizer_name="Dream-org/Dream-v0-Instruct-7B"):
    """Analyze processed JSON data with prompt and response fields"""
    print(f"Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except:
        try:
            print(f"Failed to load {tokenizer_name}, trying GPT2 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except:
            print(f"Using basic tokenizer with approximate counting")
            tokenizer = None

    print(f"Loading data from: {file_path}")
    data = load_json(file_path)

    prompt_lengths = []
    response_lengths = []
    total_lengths = []

    print(f"Analyzing {len(data)} samples...")
    for item in tqdm(data):
        prompt = item.get("prompt", "")
        response = item.get("response", "")

        # Tokenize
        if tokenizer:
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
        else:
            # Approximate: 1 token ≈ 0.75 words, 1 word ≈ 5 characters
            prompt_tokens = list(range(len(prompt) // 4))
            response_tokens = list(range(len(response) // 4))

        prompt_lengths.append(len(prompt_tokens))
        response_lengths.append(len(response_tokens))
        total_lengths.append(len(prompt_tokens) + len(response_tokens))

    return prompt_lengths, response_lengths, total_lengths

def print_statistics(prompt_lengths, response_lengths, total_lengths):
    """Print detailed statistics"""
    print("\n" + "="*60)
    print("TOKEN LENGTH STATISTICS")
    print("="*60)

    print("\n📊 Prompt Lengths:")
    print(f"  Mean:   {np.mean(prompt_lengths):.1f} tokens")
    print(f"  Median: {np.median(prompt_lengths):.1f} tokens")
    print(f"  Min:    {np.min(prompt_lengths)} tokens")
    print(f"  Max:    {np.max(prompt_lengths)} tokens")
    print(f"  Std:    {np.std(prompt_lengths):.1f} tokens")

    print("\n📊 Response Lengths:")
    print(f"  Mean:   {np.mean(response_lengths):.1f} tokens")
    print(f"  Median: {np.median(response_lengths):.1f} tokens")
    print(f"  Min:    {np.min(response_lengths)} tokens")
    print(f"  Max:    {np.max(response_lengths)} tokens")
    print(f"  Std:    {np.std(response_lengths):.1f} tokens")

    print("\n📊 Total Lengths (Prompt + Response):")
    print(f"  Mean:   {np.mean(total_lengths):.1f} tokens")
    print(f"  Median: {np.median(total_lengths):.1f} tokens")
    print(f"  Min:    {np.min(total_lengths)} tokens")
    print(f"  Max:    {np.max(total_lengths)} tokens")
    print(f"  Std:    {np.std(total_lengths):.1f} tokens")

    # Percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print("\n📊 Total Length Percentiles:")
    for p in percentiles:
        val = np.percentile(total_lengths, p)
        print(f"  {p:2d}th percentile: {val:.0f} tokens")

    # Distribution
    print("\n📊 Length Distribution:")
    ranges = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000), (16000, float('inf'))]
    for low, high in ranges:
        if high == float('inf'):
            count = sum(1 for l in total_lengths if l >= low)
            pct = 100 * count / len(total_lengths)
            print(f"  ≥{low:5d} tokens: {count:4d} samples ({pct:5.1f}%)")
        else:
            count = sum(1 for l in total_lengths if low <= l < high)
            pct = 100 * count / len(total_lengths)
            print(f"  {low:5d}-{high:5d} tokens: {count:4d} samples ({pct:5.1f}%)")

    # Recommendations
    print("\n💡 Training Recommendations:")
    p95 = np.percentile(total_lengths, 95)
    p99 = np.percentile(total_lengths, 99)
    print(f"  - For 95% coverage, set max_gen_length to at least {int(p95)}")
    print(f"  - For 99% coverage, set max_gen_length to at least {int(p99)}")

    if np.mean(total_lengths) > 4096:
        print("  - ⚠️  Average length > 4096, consider gradient_checkpointing_enable: True")
    if np.max(total_lengths) > 8192:
        print("  - ⚠️  Max length > 8192, some samples may be truncated")

def main():
    parser = argparse.ArgumentParser(description="Analyze token lengths in data files")
    parser.add_argument("file_path", type=Path, help="Path to JSONL or JSON data file")
    parser.add_argument("--tokenizer", default="Dream-org/Dream-v0-Instruct-7B",
                       help="Tokenizer to use (default: Dream-org/Dream-v0-Instruct-7B)")
    parser.add_argument("--format", choices=["auto", "raw", "processed"], default="auto",
                       help="Data format: 'raw' for JSONL with paths/summaries, 'processed' for JSON with prompt/response")

    args = parser.parse_args()

    if not args.file_path.exists():
        print(f"Error: File not found: {args.file_path}")
        return

    # Auto-detect format
    if args.format == "auto":
        if args.file_path.suffix == ".jsonl":
            args.format = "raw"
        elif args.file_path.suffix == ".json":
            args.format = "processed"
        else:
            print(f"Cannot auto-detect format for {args.file_path.suffix}, please specify --format")
            return

    # Analyze based on format
    if args.format == "raw":
        prompt_lengths, response_lengths, total_lengths = analyze_raw_jsonl(
            args.file_path, args.tokenizer
        )
    else:
        prompt_lengths, response_lengths, total_lengths = analyze_processed_json(
            args.file_path, args.tokenizer
        )

    # Print statistics
    print_statistics(prompt_lengths, response_lengths, total_lengths)

if __name__ == "__main__":
    main()