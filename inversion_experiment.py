#!/usr/bin/env python3
"""
Privacy Inversion Experiment for Split Inference Paper

Tests whether an attacker can recover input tokens from layer-2 hidden states.

Experiment:
1. Collect (layer-2 activation, token ID) pairs from diverse text
2. Train a small MLP to predict token IDs from activations
3. Report top-1 and top-5 accuracy (baseline = random ~0.003% for 32K vocab)
4. Apply LoRA to layers 0-1 + embedding, re-collect activations, re-test

Usage:
    python inversion_experiment.py --model /path/to/mistral-7b --phase collect
    python inversion_experiment.py --model /path/to/mistral-7b --phase train
    python inversion_experiment.py --model /path/to/mistral-7b --phase lora
    python inversion_experiment.py --model /path/to/mistral-7b --phase all
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import time
import os
import numpy as np

MODEL_PATH = "/home/mike/D/models/mistral-7b-instruct"
DEVICE = "cuda"
HIDDEN_DIM = 4096
SPLIT_AFTER_LAYER = 1  # Collect activations after layer 1 (input to layer 2)
DATA_DIR = "/home/mike/D/coding/python_3/my_projects/split-inference-3090/experiment_data"

# Diverse text samples for collecting activation-token pairs
TEXT_SAMPLES = [
    # General knowledge
    "The process of photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight energy. This occurs primarily in the chloroplasts of plant cells, where chlorophyll molecules absorb light energy.",
    "The French Revolution began in 1789 with the storming of the Bastille and ended with Napoleon Bonaparte's rise to power. It fundamentally transformed French society, abolishing feudalism and establishing principles of citizenship and inalienable rights.",
    "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic level. Unlike classical physics, it introduces concepts like wave-particle duality, superposition, and quantum entanglement.",
    # Code
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nfor i in range(20):\n    print(f'F({i}) = {fibonacci(i)}')",
    "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model, nhead):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(d_model, nhead)\n        self.norm1 = nn.LayerNorm(d_model)\n        self.ffn = nn.Sequential(\n            nn.Linear(d_model, 4 * d_model),\n            nn.GELU(),\n            nn.Linear(4 * d_model, d_model)\n        )\n        self.norm2 = nn.LayerNorm(d_model)",
    # Medical/legal (sensitive content types)
    "The patient presents with elevated blood pressure of 160/95 mmHg, persistent headaches, and occasional dizziness. Laboratory results show elevated creatinine levels suggesting possible renal involvement. Recommend starting ACE inhibitor therapy and scheduling follow-up renal ultrasound.",
    "Pursuant to Section 7 of the Employment Agreement dated March 15, 2024, the non-compete clause restricts the employee from engaging in competitive business activities within a 50-mile radius for a period of 24 months following termination of employment.",
    # Conversational
    "Hey, I was wondering if you could help me understand how neural networks work? I've been trying to learn about machine learning but the math is really confusing. Like, what exactly is backpropagation and why is it important?",
    "I need to plan a surprise birthday party for my wife next Saturday. She loves Italian food and jazz music. We have about 30 guests coming. Can you help me figure out the menu and find a good jazz band in the Dallas area?",
    # Technical/structured
    "HTTP/1.1 200 OK\nContent-Type: application/json\nX-Request-ID: a1b2c3d4\n\n{\"status\": \"success\", \"data\": {\"user_id\": 12345, \"name\": \"John Doe\", \"email\": \"john@example.com\", \"roles\": [\"admin\", \"editor\"]}}",
    # Financial
    "Q3 2025 revenue reached $4.2 billion, up 18% year-over-year, driven by strong cloud services growth of 32%. Operating margin expanded 200 basis points to 28.5%. Free cash flow of $1.1 billion supported the share repurchase program.",
    # More diverse text for training data volume
    "The mitochondria is often called the powerhouse of the cell because it generates most of the cell's supply of adenosine triphosphate, used as a source of chemical energy. Mitochondria are found in nearly all eukaryotic organisms.",
    "SELECT u.name, u.email, COUNT(o.id) as order_count, SUM(o.total) as total_spent FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE u.created_at > '2025-01-01' GROUP BY u.id HAVING order_count > 5 ORDER BY total_spent DESC LIMIT 100;",
    "In a groundbreaking study published in Nature, researchers demonstrated that transformer-based language models exhibit emergent capabilities at scale that are not present in smaller models. These include chain-of-thought reasoning, few-shot learning, and the ability to follow complex instructions.",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump. The five boxing wizards jump quickly at dawn.",
    "Climate change poses significant risks to global food security. Rising temperatures, changing precipitation patterns, and increased frequency of extreme weather events are expected to reduce crop yields in many regions, particularly in tropical and subtropical areas.",
]


def collect_activations(model, tokenizer, texts, device, split_layer=None):
    """Run texts through the model and collect (activation, token ID) pairs after a given layer."""
    if split_layer is None:
        split_layer = SPLIT_AFTER_LAYER
    model.eval()
    all_activations = []
    all_token_ids = []

    captured = {}

    def hook_fn(module, input, output):
        hidden = output[0]
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        captured['hidden'] = hidden.detach()

    handle = model.model.layers[split_layer].register_forward_hook(hook_fn)

    with torch.no_grad():
        for text in texts:
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
            tokens = input_ids[0]

            # Full forward pass to trigger hooks
            _ = model(input_ids)

            # captured['hidden'] has shape [1, seq_len, hidden_dim]
            hidden = captured['hidden'][0]  # [seq_len, hidden_dim]

            # Each position i's hidden state corresponds to token i
            for i in range(len(tokens)):
                all_activations.append(hidden[i].cpu())
                all_token_ids.append(tokens[i].item())

    handle.remove()

    activations = torch.stack(all_activations)  # [N, hidden_dim]
    token_ids = torch.tensor(all_token_ids, dtype=torch.long)  # [N]

    return activations, token_ids


class InversionDecoder(nn.Module):
    """Small MLP that tries to predict token IDs from layer-2 activations."""

    def __init__(self, hidden_dim, vocab_size, intermediate_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, vocab_size),
        )

    def forward(self, x):
        return self.net(x)


def evaluate_decoder(decoder, activations, token_ids, device):
    """Evaluate a trained decoder on given activations. Returns accuracy metrics."""
    decoder.eval()
    act = activations.float().to(device)
    tok = token_ids.to(device)

    with torch.no_grad():
        logits = decoder(act)
        preds = logits.argmax(dim=-1)
        top1 = (preds == tok).float().mean().item() * 100
        top5_preds = logits.topk(5, dim=-1).indices
        top5 = (top5_preds == tok.unsqueeze(1)).any(dim=1).float().mean().item() * 100
        top10_preds = logits.topk(10, dim=-1).indices
        top10 = (top10_preds == tok.unsqueeze(1)).any(dim=1).float().mean().item() * 100

    return round(top1, 2), round(top5, 2), round(top10, 2)


def train_decoder(activations, token_ids, vocab_size, epochs=50, batch_size=256, lr=1e-3):
    """Train the inversion decoder and return (metrics_dict, trained_decoder)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = activations.shape[0]

    # Split 80/20 train/test
    perm = torch.randperm(N)
    split = int(0.8 * N)
    train_idx, test_idx = perm[:split], perm[split:]

    train_act = activations[train_idx].float().to(device)
    train_tok = token_ids[train_idx].to(device)
    test_act = activations[test_idx].float().to(device)
    test_tok = token_ids[test_idx].to(device)

    train_ds = TensorDataset(train_act, train_tok)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    decoder = InversionDecoder(HIDDEN_DIM, vocab_size).to(device)
    optimizer = optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"Training inversion decoder: {N} samples ({split} train, {N-split} test)")
    print(f"  Vocab size: {vocab_size}, Random baseline: {100/vocab_size:.4f}%")

    for epoch in range(epochs):
        decoder.train()
        total_loss = 0
        for batch_act, batch_tok in train_dl:
            logits = decoder(batch_act)
            loss = criterion(logits, batch_tok)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            top1, top5, top10 = evaluate_decoder(decoder, test_act.cpu(), test_tok.cpu(), device)
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_dl):.4f}, "
                  f"top-1={top1:.2f}%, top-5={top5:.2f}%, top-10={top10:.2f}%")

    # Final eval
    top1, top5, top10 = evaluate_decoder(decoder, test_act.cpu(), test_tok.cpu(), device)

    return {
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "top10_accuracy": top10,
        "train_samples": split,
        "test_samples": N - split,
        "random_baseline": round(100 / vocab_size, 4),
    }, decoder


def apply_lora_and_collect(model, tokenizer, texts, device, lora_rank=16):
    """Apply random LoRA perturbation to local layers, then collect activations.

    This simulates a production deployment where local layers have been
    fine-tuned with organization-specific LoRA adapters.
    """
    print(f"\nApplying LoRA (rank={lora_rank}) to embedding + layers 0-1...")

    # Save original weights for restoration
    original_weights = {}

    # Apply LoRA-style perturbation to embedding
    emb = model.model.embed_tokens
    original_weights['embed'] = emb.weight.data.clone()
    A = torch.randn(emb.weight.shape[1], lora_rank, device=device, dtype=torch.float16) * 0.01
    B = torch.randn(lora_rank, emb.weight.shape[0], device=device, dtype=torch.float16) * 0.01
    emb.weight.data += (A @ B).T

    # Apply LoRA-style perturbation to layers 0 and 1
    for layer_idx in range(SPLIT_AFTER_LAYER + 1):
        layer = model.model.layers[layer_idx]

        # Perturb attention weights
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            proj = getattr(layer.self_attn, proj_name)
            key = f'layer{layer_idx}_{proj_name}'
            original_weights[key] = proj.weight.data.clone()
            A = torch.randn(proj.weight.shape[1], lora_rank, device=device, dtype=torch.float16) * 0.01
            B = torch.randn(lora_rank, proj.weight.shape[0], device=device, dtype=torch.float16) * 0.01
            proj.weight.data += (A @ B).T

        # Perturb MLP weights
        for mlp_name in ['gate_proj', 'up_proj', 'down_proj']:
            mlp = getattr(layer.mlp, mlp_name)
            key = f'layer{layer_idx}_{mlp_name}'
            original_weights[key] = mlp.weight.data.clone()
            A = torch.randn(mlp.weight.shape[1], lora_rank, device=device, dtype=torch.float16) * 0.01
            B = torch.randn(lora_rank, mlp.weight.shape[0], device=device, dtype=torch.float16) * 0.01
            mlp.weight.data += (A @ B).T

    print("  LoRA applied. Collecting activations with perturbed weights...")
    activations, token_ids = collect_activations(model, tokenizer, texts, device)

    # Restore original weights
    print("  Restoring original weights...")
    emb.weight.data = original_weights['embed']
    for layer_idx in range(SPLIT_AFTER_LAYER + 1):
        layer = model.model.layers[layer_idx]
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            proj = getattr(layer.self_attn, proj_name)
            proj.weight.data = original_weights[f'layer{layer_idx}_{proj_name}']
        for mlp_name in ['gate_proj', 'up_proj', 'down_proj']:
            mlp = getattr(layer.mlp, mlp_name)
            mlp.weight.data = original_weights[f'layer{layer_idx}_{mlp_name}']

    return activations, token_ids


def main():
    parser = argparse.ArgumentParser(description="Privacy Inversion Experiment")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--phase", default="all", choices=["collect", "train", "lora", "all"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    results = {}

    if args.phase in ("collect", "all"):
        print("=" * 60)
        print("PHASE 1: Loading model and collecting activations")
        print("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map=DEVICE
        )
        model.eval()
        vocab_size = model.config.vocab_size
        print(f"Model loaded. Vocab size: {vocab_size}")

        # Collect activations from public (unmodified) model
        print("\nCollecting activations from PUBLIC model weights...")
        activations, token_ids = collect_activations(model, tokenizer, TEXT_SAMPLES, DEVICE)
        print(f"Collected {activations.shape[0]} activation-token pairs")

        torch.save(activations, os.path.join(DATA_DIR, "activations_public.pt"))
        torch.save(token_ids, os.path.join(DATA_DIR, "token_ids_public.pt"))

        if args.phase == "all":
            # Train decoder on public activations
            print("\n" + "=" * 60)
            print("PHASE 2: Training inversion decoder (PUBLIC weights)")
            print("=" * 60)
            public_results, public_decoder = train_decoder(activations, token_ids, vocab_size, epochs=args.epochs)
            results["public_model"] = public_results
            print(f"\nPublic model results: {json.dumps(public_results, indent=2)}")

            # LoRA phase - test multiple ranks
            lora_ranks = [8, 16, 32]
            for rank in lora_ranks:
                print(f"\n{'=' * 60}")
                print(f"PHASE 3: LoRA rank={rank} perturbation")
                print("=" * 60)
                lora_activations, lora_token_ids = apply_lora_and_collect(
                    model, tokenizer, TEXT_SAMPLES, DEVICE, lora_rank=rank
                )
                print(f"Collected {lora_activations.shape[0]} LoRA activation-token pairs")

                # NON-ADAPTIVE ATTACKER: Public-trained decoder tested on LoRA activations
                # This is the realistic scenario - attacker has public weights only
                print(f"\n--- Non-adaptive attacker (public decoder → LoRA activations, rank={rank}) ---")
                top1, top5, top10 = evaluate_decoder(public_decoder, lora_activations, lora_token_ids, DEVICE)
                nonadaptive_key = f"lora_r{rank}_nonadaptive"
                results[nonadaptive_key] = {
                    "top1_accuracy": top1,
                    "top5_accuracy": top5,
                    "top10_accuracy": top10,
                    "lora_rank": rank,
                    "attacker": "non-adaptive (public decoder)",
                    "test_samples": lora_activations.shape[0],
                }
                print(f"  Non-adaptive: top-1={top1:.2f}%, top-5={top5:.2f}%, top-10={top10:.2f}%")

                # ADAPTIVE ATTACKER: Fresh decoder trained on LoRA activations
                # Worst case - attacker can collect training data from deployed system
                print(f"\n--- Adaptive attacker (fresh decoder trained on LoRA data, rank={rank}) ---")
                adaptive_results, _ = train_decoder(lora_activations, lora_token_ids, vocab_size, epochs=args.epochs)
                adaptive_key = f"lora_r{rank}_adaptive"
                results[adaptive_key] = adaptive_results
                results[adaptive_key]["lora_rank"] = rank
                results[adaptive_key]["attacker"] = "adaptive (retrained decoder)"
                print(f"  Adaptive: top-1={adaptive_results['top1_accuracy']:.2f}%, "
                      f"top-5={adaptive_results['top5_accuracy']:.2f}%")

            # PHASE 4: Split depth experiment - test different split points
            split_layers = [1, 3, 5, 7]
            for layer in split_layers:
                print(f"\n{'=' * 60}")
                print(f"PHASE 4: Split depth = layer {layer} (activations after layer {layer})")
                print("=" * 60)
                depth_act, depth_tok = collect_activations(
                    model, tokenizer, TEXT_SAMPLES, DEVICE, split_layer=layer
                )
                print(f"Collected {depth_act.shape[0]} pairs from layer {layer}")
                depth_results, _ = train_decoder(depth_act, depth_tok, vocab_size, epochs=args.epochs)
                depth_key = f"depth_layer{layer}"
                results[depth_key] = depth_results
                results[depth_key]["split_after_layer"] = layer
                print(f"  Layer {layer}: top-1={depth_results['top1_accuracy']:.2f}%, "
                      f"top-5={depth_results['top5_accuracy']:.2f}%")

    elif args.phase == "train":
        print("Loading saved activations...")
        activations = torch.load(os.path.join(DATA_DIR, "activations_public.pt"))
        token_ids = torch.load(os.path.join(DATA_DIR, "token_ids_public.pt"))
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        vocab_size = 32000  # Mistral 7B
        public_results, _ = train_decoder(activations, token_ids, vocab_size, epochs=args.epochs)
        results["public_model"] = public_results

    elif args.phase == "lora":
        print("Loading model for LoRA experiment...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map=DEVICE
        )
        model.eval()
        vocab_size = model.config.vocab_size
        lora_activations, lora_token_ids = apply_lora_and_collect(
            model, tokenizer, TEXT_SAMPLES, DEVICE, lora_rank=args.lora_rank
        )
        lora_results, _ = train_decoder(lora_activations, lora_token_ids, vocab_size, epochs=args.epochs)
        results["lora_model"] = lora_results

    # Print final summary
    if results:
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        for config, res in results.items():
            print(f"\n{config}:")
            print(f"  Top-1 accuracy: {res['top1_accuracy']}%")
            print(f"  Top-5 accuracy: {res['top5_accuracy']}%")
            print(f"  Top-10 accuracy: {res.get('top10_accuracy', 'N/A')}%")
            if 'random_baseline' in res:
                print(f"  Random baseline: {res['random_baseline']}%")
            if 'train_samples' in res:
                print(f"  Samples: {res['train_samples']} train, {res['test_samples']} test")
            if 'attacker' in res:
                print(f"  Attacker type: {res['attacker']}")
            if 'split_after_layer' in res:
                print(f"  Split after layer: {res['split_after_layer']}")

        # Save results
        results_path = os.path.join(DATA_DIR, "inversion_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
