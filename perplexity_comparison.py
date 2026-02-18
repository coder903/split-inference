#!/usr/bin/env python3
"""
Perplexity Comparison: Sequential vs Lookahead Decoding

Proves that both decoding methods produce identical token sequences
(greedy argmax guarantee) and measures per-token log-probability /
perplexity from both paths.

Runs on the 3090 in split mode (cloud via tunnel on localhost:6000)
or in local-only mode.

Usage:
    python perplexity_comparison.py --cloud http://localhost:6000
    python perplexity_comparison.py  # local-only mode
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import math
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "models/mistral-7b-instruct"
DEVICE = "cuda"
MAX_NEW_TOKENS = 200

PROMPTS = {
    "code": [
        "Write a Python function to implement binary search on a sorted array. Include docstring and type hints.",
    ],
    "structured": [
        "List the top 10 largest countries by area with their capitals and populations in a formatted table.",
    ],
    "creative": [
        "Write a short story about a robot that discovers it can dream. Make it emotional and surprising.",
    ],
    "conversational": [
        "What are the main differences between Python and Rust? When should I choose one over the other?",
    ],
}


def run_comparison(decoder, prompt, max_new_tokens=200):
    """Run both sequential and lookahead, collecting logits for each token."""

    # Sequential (block_size=1)
    print("  Running sequential (block_size=1)...")
    decoder._step_timings = []
    seq_tokens = []
    seq_logits = []
    seq_stats = None
    for event in decoder.generate_stream(prompt, max_new_tokens, block_size=1, return_logits=True):
        if 'token_id' in event:
            seq_tokens.append(event['token_id'])
            seq_logits.append(event['logits'])
        elif 'done' in event:
            seq_stats = event

    # Lookahead
    print("  Running lookahead...")
    decoder._step_timings = []
    la_tokens = []
    la_logits = []
    la_stats = None
    for event in decoder.generate_stream_lookahead(prompt, max_new_tokens, return_logits=True):
        if 'token_id' in event:
            la_tokens.append(event['token_id'])
            la_logits.append(event['logits'])
        elif 'done' in event:
            la_stats = event

    # Check token identity
    tokens_match = seq_tokens == la_tokens
    mismatch_pos = None
    if not tokens_match:
        for i in range(min(len(seq_tokens), len(la_tokens))):
            if seq_tokens[i] != la_tokens[i]:
                mismatch_pos = i
                break
        if mismatch_pos is None:
            mismatch_pos = min(len(seq_tokens), len(la_tokens))

    # Compute per-token log-probability (using min length if mismatch)
    compare_len = min(len(seq_tokens), len(la_tokens))
    seq_log_probs = []
    la_log_probs = []
    max_logit_diff = 0.0

    for i in range(compare_len):
        tid = seq_tokens[i]
        seq_lp = F.log_softmax(seq_logits[i].float(), dim=-1)[tid].item()
        la_lp = F.log_softmax(la_logits[i].float(), dim=-1)[tid].item()
        seq_log_probs.append(seq_lp)
        la_log_probs.append(la_lp)

        logit_diff = (seq_logits[i].float() - la_logits[i].float()).abs().max().item()
        max_logit_diff = max(max_logit_diff, logit_diff)

    # Perplexity = exp(-mean(log_prob))
    seq_ppl = math.exp(-sum(seq_log_probs) / len(seq_log_probs)) if seq_log_probs else float('inf')
    la_ppl = math.exp(-sum(la_log_probs) / len(la_log_probs)) if la_log_probs else float('inf')

    return {
        "prompt_preview": prompt[:60] + "...",
        "tokens_identical": tokens_match,
        "mismatch_position": mismatch_pos,
        "seq_num_tokens": len(seq_tokens),
        "la_num_tokens": len(la_tokens),
        "sequential_perplexity": round(seq_ppl, 4),
        "lookahead_perplexity": round(la_ppl, 4),
        "perplexity_diff": round(abs(seq_ppl - la_ppl), 6),
        "max_logit_diff": round(max_logit_diff, 6),
        "sequential_tok_s": seq_stats.get("tokens_per_second", 0) if seq_stats else 0,
        "lookahead_tok_s": la_stats.get("tokens_per_second", 0) if la_stats else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Perplexity Comparison Experiment")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--cloud", default=None, help="Cloud URL for split mode")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--split-after", type=int, default=1, help="Last local layer before cloud")
    parser.add_argument("--resume-at", type=int, default=None, help="First local layer after cloud")
    parser.add_argument("--output", default="experiment_data/perplexity_comparison.json")
    args = parser.parse_args()

    from jacobi_server import JacobiDecoder, BLOCK_SIZE, MAX_JACOBI_ITERS
    import jacobi_server

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.cloud:
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
            "max_new_tokens": args.max_tokens,
        },
        "comparisons": [],
        "summary": {},
    }

    print("\n" + "=" * 60)
    print("PERPLEXITY COMPARISON: Sequential vs Lookahead")
    print("=" * 60)

    all_match = True
    for category, prompts in PROMPTS.items():
        for prompt in prompts:
            print(f"\n[{category}] {prompt[:50]}...")
            result = run_comparison(decoder, prompt, args.max_tokens)
            result["category"] = category
            all_results["comparisons"].append(result)

            status = "MATCH" if result["tokens_identical"] else f"MISMATCH at pos {result['mismatch_position']}"
            print(f"  Tokens: {status}")
            print(f"  Perplexity: seq={result['sequential_perplexity']:.4f}, "
                  f"la={result['lookahead_perplexity']:.4f}, "
                  f"diff={result['perplexity_diff']:.6f}")
            print(f"  Max logit diff: {result['max_logit_diff']:.6f}")
            print(f"  Throughput: seq={result['sequential_tok_s']:.1f}, "
                  f"la={result['lookahead_tok_s']:.1f} tok/s")

            if not result["tokens_identical"]:
                all_match = False

    # Summary
    comparisons = all_results["comparisons"]
    all_results["summary"] = {
        "all_tokens_identical": all_match,
        "num_prompts": len(comparisons),
        "avg_sequential_perplexity": round(
            sum(r["sequential_perplexity"] for r in comparisons) / len(comparisons), 4
        ),
        "avg_lookahead_perplexity": round(
            sum(r["lookahead_perplexity"] for r in comparisons) / len(comparisons), 4
        ),
        "avg_perplexity_diff": round(
            sum(r["perplexity_diff"] for r in comparisons) / len(comparisons), 6
        ),
        "max_logit_diff_overall": round(
            max(r["max_logit_diff"] for r in comparisons), 6
        ),
        "avg_sequential_tok_s": round(
            sum(r["sequential_tok_s"] for r in comparisons) / len(comparisons), 1
        ),
        "avg_lookahead_tok_s": round(
            sum(r["lookahead_tok_s"] for r in comparisons) / len(comparisons), 1
        ),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    s = all_results["summary"]
    print(f"  All tokens identical: {s['all_tokens_identical']}")
    print(f"  Avg perplexity: seq={s['avg_sequential_perplexity']:.4f}, "
          f"la={s['avg_lookahead_perplexity']:.4f}")
    print(f"  Avg perplexity diff: {s['avg_perplexity_diff']:.6f}")
    print(f"  Max logit diff: {s['max_logit_diff_overall']:.6f}")
    print(f"  Avg throughput: seq={s['avg_sequential_tok_s']:.1f}, "
          f"la={s['avg_lookahead_tok_s']:.1f} tok/s")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
