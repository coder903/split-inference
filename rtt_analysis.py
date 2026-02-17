#!/usr/bin/env python3
"""
RTT Compensation Analysis for Split Inference

Decomposes per-step wall time into:
    per_step_time = network_RTT + cloud_GPU_compute + local_overhead

Computes RTT-independent acceptance rate and projects tok/s at
arbitrary target RTTs for fair cross-provider comparison.

Usage:
    python rtt_analysis.py --cloud http://localhost:6000
    python rtt_analysis.py --cloud http://localhost:6000 --output results.json
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean, median
import argparse
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "/home/mike/D/models/mistral-7b-instruct"
DEVICE = "cuda"
MAX_NEW_TOKENS = 200

PROMPTS = [
    ("code", "Write a Python function to implement binary search on a sorted array. Include docstring and type hints."),
    ("structured", "List the top 10 largest countries by area with their capitals and populations in a formatted table."),
    ("creative", "Write a short story about a robot that discovers it can dream. Make it emotional and surprising."),
    ("conversational", "What are the main differences between Python and Rust? When should I choose one over the other?"),
]

TARGET_RTTS = [20, 40, 60, 80, 100, 120, 150, 200]


def analyze_run(decoder, prompt, max_new_tokens, mode="sequential"):
    """Run a generation and collect detailed per-step timings."""
    decoder._step_timings = []

    stats = None
    if mode == "lookahead":
        for event in decoder.generate_stream_lookahead(prompt, max_new_tokens):
            if 'done' in event:
                stats = event
    else:
        for event in decoder.generate_stream(prompt, max_new_tokens, block_size=1):
            if 'done' in event:
                stats = event

    timings = decoder._step_timings
    if not timings:
        return None

    # Skip first timing entry (prefill processes full prompt, not representative)
    # All subsequent entries are decode steps regardless of seq_len
    decode_timings = timings[1:] if len(timings) > 1 else timings

    rtts = [t['network_rtt_ms'] for t in decode_timings]
    cloud_computes = [t['cloud_gpu_ms'] for t in decode_timings]
    cloud_call_totals = [t['total_ms'] for t in decode_timings]
    ser_overheads = [t['serialize_ms'] + t['deserialize_ms'] for t in decode_timings]

    jacobi_stats = stats.get("jacobi_stats", {})
    tokens_per_block = jacobi_stats.get("tokens_per_block", [])
    acceptance_rate = mean(tokens_per_block) if tokens_per_block else 1.0
    tokens_generated = stats.get("tokens_generated", 0)
    total_time_ms = stats.get("total_time_ms", 0)
    num_steps = len(decode_timings)

    # Per-step wall time includes local compute (layers 0-1, 30-31, norm, lm_head)
    # which is NOT captured in _send_to_cloud() timings
    per_step_wall_ms = total_time_ms / num_steps if num_steps > 0 else 0
    cloud_call_mean_ms = mean(cloud_call_totals)
    local_compute_ms = max(0, per_step_wall_ms - cloud_call_mean_ms)

    # Fixed overhead = everything except network RTT (RTT-independent)
    mean_rtt = mean(rtts)
    fixed_overhead_ms = per_step_wall_ms - mean_rtt

    return {
        "mode": mode,
        "tokens_generated": tokens_generated,
        "measured_tok_s": stats.get("tokens_per_second", 0),
        "num_steps": num_steps,
        "acceptance_rate": round(acceptance_rate, 3),
        "timing_decomposition": {
            "per_step_wall_ms": round(per_step_wall_ms, 1),
            "network_rtt_mean_ms": round(mean_rtt, 1),
            "network_rtt_median_ms": round(median(rtts), 1),
            "network_rtt_min_ms": round(min(rtts), 1),
            "network_rtt_max_ms": round(max(rtts), 1),
            "cloud_compute_mean_ms": round(mean(cloud_computes), 1),
            "local_compute_ms": round(local_compute_ms, 1),
            "serialize_overhead_ms": round(mean(ser_overheads), 1),
            "fixed_overhead_ms": round(fixed_overhead_ms, 1),
        },
        "num_cloud_calls": len(decode_timings),
    }


def project_throughput(analysis, target_rtts=None):
    """Project tok/s at different RTTs using fixed overhead (RTT-independent)."""
    if target_rtts is None:
        target_rtts = TARGET_RTTS

    td = analysis["timing_decomposition"]
    fixed_overhead = td["fixed_overhead_ms"]
    acceptance = analysis["acceptance_rate"]

    projections = {}
    for rtt in target_rtts:
        per_step_s = (rtt + fixed_overhead) / 1000
        projected = acceptance / per_step_s if per_step_s > 0 else 0
        projections[f"{rtt}ms"] = round(projected, 1)

    return projections


def main():
    parser = argparse.ArgumentParser(description="RTT Compensation Analysis")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--cloud", required=True, help="Cloud URL (required for RTT analysis)")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--split-after", type=int, default=1, help="Last local layer before cloud")
    parser.add_argument("--resume-at", type=int, default=None, help="First local layer after cloud")
    parser.add_argument("--output", default="/home/mike/D/coding/python_3/my_projects/split-inference-3090/experiment_data/rtt_analysis.json")
    args = parser.parse_args()

    from jacobi_server import JacobiDecoder, BLOCK_SIZE, MAX_JACOBI_ITERS
    import jacobi_server

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

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

    jacobi_server.NUM_LAYERS = num_layers
    jacobi_server.SPLIT_AFTER = split_after
    jacobi_server.RESUME_AT = resume_at

    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Split mode: {num_layers} layers, local 0-{split_after} + {resume_at}-{num_layers-1}, "
          f"VRAM: {mem_gb:.1f}GB")

    decoder = JacobiDecoder(
        model, tokenizer, DEVICE, cloud_url=args.cloud,
        block_size=BLOCK_SIZE, max_iters=MAX_JACOBI_ITERS
    )

    all_results = {
        "config": {
            "model": args.model,
            "cloud_url": args.cloud,
            "max_new_tokens": args.max_tokens,
            "target_rtts_ms": TARGET_RTTS,
        },
        "runs": [],
        "projections": {},
        "summary": {},
    }

    print("\n" + "=" * 60)
    print("RTT COMPENSATION ANALYSIS")
    print("=" * 60)

    for category, prompt in PROMPTS:
        for mode in ["sequential", "lookahead"]:
            print(f"\n[{mode}] {category}: {prompt[:50]}...")
            analysis = analyze_run(decoder, prompt, args.max_tokens, mode)
            if analysis is None:
                print("  SKIPPED (no cloud calls)")
                continue

            analysis["category"] = category
            analysis["prompt_preview"] = prompt[:60] + "..."
            all_results["runs"].append(analysis)

            td = analysis["timing_decomposition"]
            print(f"  {analysis['tokens_generated']} tokens, {analysis['measured_tok_s']:.1f} tok/s")
            print(f"  Per-step wall: {td['per_step_wall_ms']:.1f}ms")
            print(f"  RTT: {td['network_rtt_mean_ms']:.0f}ms mean, "
                  f"{td['network_rtt_median_ms']:.0f}ms median "
                  f"({td['network_rtt_min_ms']:.0f}-{td['network_rtt_max_ms']:.0f}ms)")
            print(f"  Cloud GPU: {td['cloud_compute_mean_ms']:.1f}ms, "
                  f"Local GPU: {td['local_compute_ms']:.1f}ms, "
                  f"Serialize: {td['serialize_overhead_ms']:.1f}ms")
            print(f"  Fixed overhead (RTT-independent): {td['fixed_overhead_ms']:.1f}ms")
            print(f"  Acceptance rate: {analysis['acceptance_rate']:.3f} tok/step")

            # Project throughput
            projections = project_throughput(analysis)
            key = f"{category}_{mode}"
            all_results["projections"][key] = projections
            proj_str = ", ".join(f"{rtt}: {tok_s}" for rtt, tok_s in projections.items())
            print(f"  Projected tok/s: {proj_str}")

    # Aggregate summary by mode
    for mode in ["sequential", "lookahead"]:
        mode_runs = [r for r in all_results["runs"] if r["mode"] == mode]
        if not mode_runs:
            continue

        avg_rtt = mean(r["timing_decomposition"]["network_rtt_mean_ms"] for r in mode_runs)
        avg_fixed = mean(r["timing_decomposition"]["fixed_overhead_ms"] for r in mode_runs)
        avg_cloud = mean(r["timing_decomposition"]["cloud_compute_mean_ms"] for r in mode_runs)
        avg_local = mean(r["timing_decomposition"]["local_compute_ms"] for r in mode_runs)
        avg_acceptance = mean(r["acceptance_rate"] for r in mode_runs)
        avg_tok_s = mean(r["measured_tok_s"] for r in mode_runs)
        avg_wall = mean(r["timing_decomposition"]["per_step_wall_ms"] for r in mode_runs)

        summary = {
            "avg_per_step_wall_ms": round(avg_wall, 1),
            "avg_measured_rtt_ms": round(avg_rtt, 1),
            "avg_cloud_compute_ms": round(avg_cloud, 1),
            "avg_local_compute_ms": round(avg_local, 1),
            "avg_fixed_overhead_ms": round(avg_fixed, 1),
            "avg_acceptance_rate": round(avg_acceptance, 3),
            "avg_measured_tok_s": round(avg_tok_s, 1),
        }

        # Project aggregate using fixed overhead
        agg_projections = {}
        for rtt in TARGET_RTTS:
            per_step_s = (rtt + avg_fixed) / 1000
            agg_projections[f"{rtt}ms"] = round(avg_acceptance / per_step_s, 1) if per_step_s > 0 else 0

        summary["projected_tok_s"] = agg_projections
        all_results["summary"][mode] = summary

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for mode in ["sequential", "lookahead"]:
        s = all_results["summary"].get(mode)
        if not s:
            continue
        print(f"\n  {mode.upper()}")
        print(f"    Per-step wall time: {s['avg_per_step_wall_ms']:.1f}ms")
        print(f"    Network RTT: {s['avg_measured_rtt_ms']:.0f}ms")
        print(f"    Cloud compute: {s['avg_cloud_compute_ms']:.1f}ms")
        print(f"    Local compute: {s['avg_local_compute_ms']:.1f}ms")
        print(f"    Fixed overhead (RTT-independent): {s['avg_fixed_overhead_ms']:.1f}ms")
        print(f"    Acceptance rate: {s['avg_acceptance_rate']:.3f} tok/step")
        print(f"    Measured tok/s: {s['avg_measured_tok_s']:.1f}")
        print(f"    Projected tok/s at target RTTs:")
        for rtt, tok_s in s["projected_tok_s"].items():
            marker = " <-- measured" if abs(float(rtt.replace("ms", "")) - s["avg_measured_rtt_ms"]) < 15 else ""
            print(f"      {rtt}: {tok_s}{marker}")

    # Cross-validation note
    print("\n  CROSS-VALIDATION:")
    for mode in ["sequential", "lookahead"]:
        s = all_results["summary"].get(mode)
        if not s:
            continue
        measured_rtt = s["avg_measured_rtt_ms"]
        # Find closest projected RTT
        closest_rtt = min(TARGET_RTTS, key=lambda r: abs(r - measured_rtt))
        projected = s["projected_tok_s"].get(f"{closest_rtt}ms", 0)
        measured = s["avg_measured_tok_s"]
        error = abs(projected - measured) / measured * 100 if measured > 0 else 0
        print(f"    {mode}: projected at {closest_rtt}ms = {projected} tok/s vs "
              f"measured = {measured} tok/s (error: {error:.1f}%)")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
