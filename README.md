# Split Inference: Privacy-Aware LLM Inference over WANs

Split a transformer between a trusted local GPU and an untrusted cloud GPU. Only intermediate activations cross the network—raw tokens never leave your device.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        LOCAL GPU (RTX 3090)                      │
│  ┌────────────┐   ┌────────────┐   ┌──────────────────────────┐ │
│  │ Tokenizer  │──▶│ Layers 0-1 │──▶│ Layers 30-31 + LM Head   │ │
│  │ (text→ids) │   │ (embed)    │   │ (unembed→text)           │ │
│  └────────────┘   └─────┬──────┘   └───────────▲──────────────┘ │
│        ▲                │                       │                │
│     [TEXT]              │ activations           │ activations    │
│                         ▼                       │                │
└─────────────────────────│───────────────────────│────────────────┘
                          │                       │
                  ═══════ WebSocket (binary) ═══════
                          │                       │
┌─────────────────────────▼───────────────────────│────────────────┐
│                      CLOUD GPU (A100 / 4090)                     │
│                  ┌──────────────────────┐                        │
│                  │  Layers 2-29 (7B)    │                        │
│                  │  Layers 2-37 (12B)   │                        │
│                  └──────────────────────┘                        │
│                                                                  │
│   Cloud sees: [1, seq_len, hidden_dim] float16 tensor            │
│   Cloud does NOT see: tokens, text, or vocabulary mapping        │
└──────────────────────────────────────────────────────────────────┘
```

## Privacy Model

This system provides **architectural privacy**—a structural guarantee that raw tokens never cross the network boundary. The embedding matrix (token→vector) and LM head (vector→token) remain local, so the cloud only receives high-dimensional activation vectors with no direct mapping to the vocabulary.

**Important caveats:**
- This is *not* cryptographic privacy. Intermediate activations are, in principle, invertible given the local layer weights ([Nikolaou et al., 2025](https://arxiv.org/abs/2510.15511)).
- Our inversion experiments show a simple MLP attacker recovers ~59% of tokens from layer-2 activations, dropping to ~35% at layer-8 depth.
- For public models used as-is, an attacker with matching weights can attempt inversion. Local layers should be fine-tuned or adapted (e.g., LoRA) to differentiate from published weights.
- Increasing local layers is the most effective defense: each additional layer adds ~3ms latency (negligible vs. ~80ms network RTT) while substantially reducing attack accuracy.

For organizations where the current alternative is sending plaintext to cloud APIs, this represents a meaningful improvement even without formal guarantees. See the [paper](paper.md) for full analysis.

## Performance

Evaluated on Mistral 7B (32 layers) and Mistral NeMo 12B (40 layers) over ~80ms WAN:

| Configuration | 7B | 12B |
|---|---|---|
| Sequential (A100, ~80ms RTT) | 8.3 tok/s | 8.0 tok/s |
| Lookahead n=3 (A100, ~80ms RTT) | 9.3 tok/s | 8.7 tok/s |
| Projected at 20ms RTT (lookahead) | 19.8 tok/s | 19.3 tok/s |
| Local-only (3090, all layers) | ~39 tok/s | N/A (won't fit) |
| Local VRAM usage (split mode) | 2.0 GB | 4.9 GB |

**Latency breakdown (sequential, 7B):** 64% network RTT, 22% local GPU, 13% cloud GPU, 1% serialization.

**Lookahead acceptance rates** (tokens committed per decoding step):
| Content Type | 7B | 12B |
|---|---|---|
| Code | 1.43 | 1.57 |
| Structured | 1.20 | 1.23 |
| Conversational | 1.17 | 1.12 |
| Creative | 1.12 | 1.06 |

All measurements use greedy (argmax) decoding. Lookahead produces token-identical output to sequential decoding.

## Transport Protocol

WebSocket binary protocol on port 5001 with persistent connections:

```
┌──────────────┬──────────────┬─────────────────────────────────┐
│ 4B header_len│ JSON header  │         Tensor data             │
│  (big-endian)│              │  hidden + cos + sin [+ mask]    │
└──────────────┴──────────────┴─────────────────────────────────┘
```

All tensors are raw float16 bytes—no base64, no serialization frameworks. Flask HTTP on port 5000 for health checks only.

## Decoding Modes

The local GPU server (`jacobi_server.py`) supports three modes via `--mode`:

### Sequential (`--mode sequential`)
One token per network round trip. Simplest and most reliable.

### Jacobi (`--mode jacobi`)
Block-parallel fixed-point iteration. Effective locally (~39 tok/s) but convergence issues make it slower than sequential over WAN (~1.5 tok/s).

### Lookahead (`--mode lookahead`)
N-gram speculation from Jacobi trajectories. Accepts 1–5 tokens per round trip. Best mode for split inference.

**How it works:**
1. Build n-gram pool from prompt tokens
2. Each step: get verified next token, look up candidate continuations in pool
3. Verify all candidates in a single forward pass (one network round trip)
4. Accept longest consecutive match; if none, degrade to sequential (no worse)
5. Grow pool from generated output

## Setup

### Requirements

**Local GPU (RTX 3090 or similar):**
- Python 3.10+, PyTorch with CUDA, transformers 5.x
- 24GB VRAM (only ~2–5 GB used in split mode)

**Cloud GPU:**
- NVIDIA GPU with 16GB+ VRAM (A100, 4090, etc.)
- Same Python/PyTorch/transformers stack

### Installation

```bash
# Both local and cloud
python -m venv venv && source venv/bin/activate
pip install torch transformers requests flask websockets

# Download model (both machines need it)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
    --local-dir models/mistral-7b-instruct
```

### Start Cloud Server

```bash
# On cloud GPU
python cloud_server_kv.py --model /path/to/models/mistral-7b-instruct
```

For NeMo 12B:
```bash
python cloud_server_kv.py --model /path/to/models/mistral-nemo-12b-instruct \
    --cloud-start 2 --cloud-end 37
```

### Start Local Server

```bash
# SSH tunnel to cloud (adjust ports/IPs for your setup)
ssh -fN -L 5000:localhost:5000 -L 5001:localhost:5001 user@CLOUD_IP

# Lookahead mode (recommended)
python jacobi_server.py \
    --model /path/to/models/mistral-7b-instruct \
    --mode lookahead \
    --cloud http://localhost:5000

# For NeMo 12B
python jacobi_server.py \
    --model /path/to/models/mistral-nemo-12b-instruct \
    --mode lookahead \
    --cloud http://localhost:5000 \
    --split-after 1 --resume-at 38
```

### Mac Client (Optional)

Lightweight text display client—no GPU needed:
```bash
python mac_client.py --server http://YOUR_LOCAL_GPU_IP:5001
```

## Files

| File | Description |
|------|-------------|
| `jacobi_server.py` | Local GPU server—sequential, Jacobi, or lookahead decoding |
| `cloud_server_kv.py` | Cloud GPU server—WebSocket + HTTP, KV-cache management |
| `mac_client.py` | Lightweight client for text display (no GPU) |
| `interactive.py` | Interactive chat (direct to cloud, no relay) |
| `inversion_experiment.py` | Privacy evaluation—MLP inversion attack at various split depths |
| `lookahead_ablation.py` | N-gram size sweep (n=3..7) across prompt categories |
| `rtt_analysis.py` | RTT decomposition and throughput projection |
| `perplexity_comparison.py` | Output identity verification (sequential vs. lookahead) |
| `experiment_data/` | Raw JSON results from all experiments |
| `paper.md` | Research paper |

## Key Implementation Notes

### KV-Cache
Cloud server maintains per-session KV-caches with 5-minute timeout. Supports cache cropping (rollback on rejected speculation), relocation (moving accepted entries), and custom attention masks for parallel verification.

### Attention Masks
PyTorch's `is_causal=True` creates relative masks that break with existing KV-cache. We build explicit absolute masks transmitted alongside hidden states.

### transformers 5.x Compatibility
- `model.model.rotary_emb()` returns `[batch, seq_len, head_dim]`—do NOT add `unsqueeze(1)`
- Decoder layers can return 2D tensors—guard with `if hidden.dim() == 2: hidden = hidden.unsqueeze(0)`
- DynamicCache uses `cache.layers[idx].keys`/`.values` (not `cache.key_cache[idx]`)

## Cloud Provider Notes

Provider networking architecture matters more than GPU location:

| Provider | SSH Access | Typical RTT | Suitability |
|----------|-----------|-------------|-------------|
| RunPod | Direct | 80–100ms | Good |
| HyperStack | Direct | 78–85ms | Good |
| VAST.ai | Proxied | 200–400ms | Unusable for split inference |

VAST.ai routes all SSH through proxy servers regardless of GPU location, adding 200–400ms RTT.

## Projected Throughput at Lower RTTs

Using our validated RTT decomposition model (<6.2% cross-validation error):

| RTT | 7B Sequential | 7B Lookahead | 12B Lookahead |
|-----|--------------|-------------|--------------|
| 80ms | 8.1 tok/s | 9.2 tok/s | 10.0 tok/s |
| 40ms | 12.1 tok/s | 14.5 tok/s | 14.7 tok/s |
| 20ms | 15.9 tok/s | 19.8 tok/s | 19.3 tok/s |

## License

MIT
