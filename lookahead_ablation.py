#!/usr/bin/env python3
"""
Lookahead Decoding Ablation Experiment for Split Inference Paper

Tests how n-gram size and prompt type affect acceptance rate and throughput.

Runs on the 3090 in split mode (cloud via tunnel on localhost:6000).

Usage:
    python lookahead_ablation.py --cloud http://localhost:6000
    python lookahead_ablation.py  # local-only mode for comparison
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import argparse
import json
import time
import sys
import os

# Add parent path for jacobi_server imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "models/mistral-7b-instruct"
DEVICE = "cuda"
MAX_NEW_TOKENS = 200

# Test prompts by category
PROMPTS = {
    "code": [
        "Write a Python function to implement binary search on a sorted array. Include docstring and type hints.",
        "Write a Python class for a linked list with insert, delete, and search methods.",
    ],
    "structured": [
        "List the top 10 largest countries by area with their capitals and populations in a formatted table.",
        "Explain the HTTP request lifecycle step by step, from DNS resolution to response rendering.",
    ],
    "creative": [
        "Write a short story about a robot that discovers it can dream. Make it emotional and surprising.",
        "Compose a poem about the ocean at midnight. Use vivid imagery and unexpected metaphors.",
    ],
    "conversational": [
        "What are the main differences between Python and Rust? When should I choose one over the other?",
        "Explain quantum computing to a 10-year-old. Use simple analogies they would understand.",
    ],
}

# N-gram sizes to test
NGRAM_SIZES = [3, 4, 5, 6, 7]


def run_lookahead_test(decoder, prompt, ngram_size, max_candidates=5):
    """Run a single lookahead generation and collect metrics."""
    results = {
        "ngram_size": ngram_size,
        "prompt_preview": prompt[:60] + "...",
        "tokens_generated": 0,
        "total_steps": 0,
        "total_accepted": 0,
        "time_ms": 0,
        "tok_per_sec": 0,
        "avg_accepted_per_step": 0,
        "max_accepted_in_step": 0,
        "steps_with_match": 0,
        "steps_without_match": 0,
    }

    gen_start = time.time()
    steps = 0
    accepted_counts = []

    for event in decoder.generate_stream_lookahead(
        prompt, max_new_tokens=MAX_NEW_TOKENS,
        ngram_size=ngram_size, max_candidates=max_candidates
    ):
        if "done" in event:
            stats = event
            results["tokens_generated"] = stats.get("tokens_generated", 0)
            results["time_ms"] = stats.get("total_time_ms", 0)
            results["tok_per_sec"] = stats.get("tokens_per_second", 0)

            jacobi = stats.get("jacobi_stats", {})
            results["total_steps"] = jacobi.get("blocks", 0)
            results["total_accepted"] = results["tokens_generated"]

            tpb = jacobi.get("tokens_per_block", [])
            if tpb:
                results["avg_accepted_per_step"] = round(sum(tpb) / len(tpb), 2)
                results["max_accepted_in_step"] = max(tpb)
                results["steps_with_match"] = sum(1 for t in tpb if t > 1)
                results["steps_without_match"] = sum(1 for t in tpb if t == 1)

    return results


def run_sequential_test(decoder, prompt):
    """Run sequential generation for baseline comparison."""
    results = {
        "prompt_preview": prompt[:60] + "...",
        "tokens_generated": 0,
        "time_ms": 0,
        "tok_per_sec": 0,
    }

    for event in decoder.generate_stream(prompt, max_new_tokens=MAX_NEW_TOKENS, block_size=1):
        if "done" in event:
            results["tokens_generated"] = event.get("tokens_generated", 0)
            results["time_ms"] = event.get("total_time_ms", 0)
            results["tok_per_sec"] = event.get("tokens_per_second", 0)

    return results


def main():
    global MAX_NEW_TOKENS

    parser = argparse.ArgumentParser(description="Lookahead Ablation Experiment")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--cloud", default=None, help="Cloud URL for split mode")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--split-after", type=int, default=1, help="Last local layer before cloud")
    parser.add_argument("--resume-at", type=int, default=None, help="First local layer after cloud")
    parser.add_argument("--output", default="experiment_data/lookahead_ablation.json")
    args = parser.parse_args()

    MAX_NEW_TOKENS = args.max_tokens

    # Import JacobiDecoder
    # We need to set up the module path
    from jacobi_server import JacobiDecoder, BLOCK_SIZE, MAX_JACOBI_ITERS
    import jacobi_server

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.cloud:
        # Split mode: load to CPU, move only local layers to GPU
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map="cpu"
        )
        model.eval()
        num_layers = len(model.model.layers)
        split_after = args.split_after
        resume_at = args.resume_at if args.resume_at is not None else num_layers - 2

        model.model.embed_tokens = model.model.embed_tokens.to(DEVICE)
        model.model.rotary_emb = model.model.rotary_emb.to(DEVICE)
        for i in range(split_after + 1):
            model.model.layers[i] = model.model.layers[i].to(DEVICE)
        for i in range(resume_at, num_layers):
            model.model.layers[i] = model.model.layers[i].to(DEVICE)
        model.model.norm = model.model.norm.to(DEVICE)
        model.lm_head = model.lm_head.to(DEVICE)

        # Update module-level constants used by JacobiDecoder
        jacobi_server.NUM_LAYERS = num_layers
        jacobi_server.SPLIT_AFTER = split_after
        jacobi_server.RESUME_AT = resume_at

        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Split mode: {num_layers} layers, local 0-{split_after} + {resume_at}-{num_layers-1}, "
              f"VRAM: {mem_gb:.1f}GB")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map=DEVICE
        )
        model.eval()
        print(f"Local-only mode loaded.")

    decoder = JacobiDecoder(
        model, tokenizer, DEVICE, cloud_url=args.cloud,
        block_size=BLOCK_SIZE, max_iters=MAX_JACOBI_ITERS
    )

    all_results = {
        "config": {
            "model": args.model,
            "cloud_url": args.cloud,
            "max_new_tokens": MAX_NEW_TOKENS,
            "ngram_sizes_tested": NGRAM_SIZES,
            "prompt_categories": list(PROMPTS.keys()),
        },
        "sequential_baseline": [],
        "lookahead_by_ngram": {},
        "summary_by_category": {},
        "summary_by_ngram": {},
    }

    # Run sequential baseline first
    print("\n" + "=" * 60)
    print("SEQUENTIAL BASELINE")
    print("=" * 60)
    for category, prompts in PROMPTS.items():
        for prompt in prompts:
            print(f"\n[Sequential] {category}: {prompt[:50]}...")
            result = run_sequential_test(decoder, prompt)
            result["category"] = category
            all_results["sequential_baseline"].append(result)
            print(f"  → {result['tok_per_sec']:.1f} tok/s, {result['tokens_generated']} tokens")

    # Run lookahead with different n-gram sizes
    for ngram_size in NGRAM_SIZES:
        print(f"\n{'=' * 60}")
        print(f"LOOKAHEAD N-GRAM SIZE = {ngram_size}")
        print(f"{'=' * 60}")

        ngram_results = []
        for category, prompts in PROMPTS.items():
            for prompt in prompts:
                print(f"\n[Lookahead n={ngram_size}] {category}: {prompt[:50]}...")
                result = run_lookahead_test(decoder, prompt, ngram_size)
                result["category"] = category
                ngram_results.append(result)
                print(f"  → {result['tok_per_sec']:.1f} tok/s, "
                      f"avg {result['avg_accepted_per_step']} tok/step, "
                      f"max {result['max_accepted_in_step']} tok/step, "
                      f"{result['steps_with_match']}/{result['total_steps']} steps matched")

        all_results["lookahead_by_ngram"][str(ngram_size)] = ngram_results

    # Compute summaries
    print("\n" + "=" * 60)
    print("SUMMARY TABLES")
    print("=" * 60)

    # By category (averaging across n-gram sizes)
    for category in PROMPTS.keys():
        cat_summary = {"sequential_tok_s": 0, "sequential_count": 0}
        seq_results = [r for r in all_results["sequential_baseline"] if r["category"] == category]
        if seq_results:
            cat_summary["sequential_tok_s"] = round(
                sum(r["tok_per_sec"] for r in seq_results) / len(seq_results), 1
            )

        for ngram_size in NGRAM_SIZES:
            key = str(ngram_size)
            la_results = [r for r in all_results["lookahead_by_ngram"].get(key, []) if r["category"] == category]
            if la_results:
                cat_summary[f"n{ngram_size}_tok_s"] = round(
                    sum(r["tok_per_sec"] for r in la_results) / len(la_results), 1
                )
                cat_summary[f"n{ngram_size}_acceptance"] = round(
                    sum(r["avg_accepted_per_step"] for r in la_results) / len(la_results), 2
                )

        all_results["summary_by_category"][category] = cat_summary

    # By n-gram size (averaging across categories)
    for ngram_size in NGRAM_SIZES:
        key = str(ngram_size)
        la_results = all_results["lookahead_by_ngram"].get(key, [])
        if la_results:
            all_results["summary_by_ngram"][key] = {
                "avg_tok_s": round(sum(r["tok_per_sec"] for r in la_results) / len(la_results), 1),
                "avg_acceptance": round(sum(r["avg_accepted_per_step"] for r in la_results) / len(la_results), 2),
                "avg_match_rate": round(
                    sum(r["steps_with_match"] / max(1, r["total_steps"]) for r in la_results) / len(la_results) * 100, 1
                ),
            }

    # Print summary tables
    print("\n--- By Category ---")
    print(f"{'Category':<15} {'Sequential':>10} ", end="")
    for n in NGRAM_SIZES:
        print(f"{'n='+str(n):>10} ", end="")
    print()
    for cat in PROMPTS.keys():
        s = all_results["summary_by_category"][cat]
        print(f"{cat:<15} {s['sequential_tok_s']:>8.1f}  ", end="")
        for n in NGRAM_SIZES:
            val = s.get(f"n{n}_tok_s", 0)
            print(f"{val:>8.1f}  ", end="")
        print()

    print("\n--- By N-gram Size ---")
    print(f"{'N-gram':>8} {'tok/s':>8} {'Accept':>8} {'Match%':>8}")
    for n in NGRAM_SIZES:
        s = all_results["summary_by_ngram"].get(str(n), {})
        print(f"{n:>8} {s.get('avg_tok_s', 0):>8.1f} {s.get('avg_acceptance', 0):>8.2f} {s.get('avg_match_rate', 0):>7.1f}%")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
