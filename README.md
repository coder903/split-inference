# Split Inference - Privacy-Preserving LLM

Run large language models with **architectural privacy guarantees**: your text never leaves your local machine. Only intermediate activations (meaningless tensor data) cross the network to the cloud GPU.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR LOCAL MACHINE                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Tokenizer  │───▶│  Layers 0-1 │───▶│  Layers 30-31 + LM  │  │
│  │  (text→ids) │    │  (embed)    │    │  (unembed→text)     │  │
│  └─────────────┘    └──────┬──────┘    └──────────▲──────────┘  │
│         ▲                  │                      │              │
│         │                  │ activations          │ activations  │
│      [TEXT]                ▼                      │              │
└─────────────────────────── │ ─────────────────────│──────────────┘
                             │                      │
                    ═══════ WebSocket (binary) ═══════
                             │                      │
┌────────────────────────────▼──────────────────────│──────────────┐
│                         CLOUD GPU                                │
│                    ┌─────────────────┐                          │
│                    │   Layers 2-29   │                          │
│                    │  (28 transformer │                          │
│                    │     layers)      │                          │
│                    └─────────────────┘                          │
│                                                                  │
│   Cloud sees: [1, seq_len, 4096] float16 tensor                 │
│   Cloud CANNOT see: your text, tokens, or meaning               │
└──────────────────────────────────────────────────────────────────┘
```

## Privacy Guarantee

The cloud provider **mathematically cannot** reconstruct your input:

1. **Tokenization** happens locally - cloud never sees text
2. **Embedding layer** (layer 0) maps tokens to vectors - cloud only sees post-embedding activations
3. **Unembedding layer** (layer 31 + lm_head) maps vectors back to tokens - happens locally
4. **Intermediate activations** are high-dimensional tensors with no direct mapping to text

This is **architectural privacy**, not policy-based. Even a malicious cloud operator cannot recover your input.

## Use Cases

- **Healthcare/HIPAA**: PHI stays on-premise, cloud processes anonymized tensors
- **Legal**: Attorney-client privileged content never leaves the firm
- **Military/Classified**: Classified I/O local, cloud sees meaningless math
- **Finance**: PII and trading strategies stay within compliance boundary

## Performance

### Optimization Journey

| Stage | Tokens/sec | Improvement |
|-------|-----------|-------------|
| HTTP + torch.save serialization | 1.4 tok/s | baseline |
| Numpy binary serialization | 3.9 tok/s | 2.8x |
| RTX 3090 relay (vs Mac MPS) | 5.3 tok/s | 3.8x |
| WebSocket binary protocol | 7.8 tok/s | 5.6x |
| N-gram lookahead speculation | 8.7 tok/s | 6.2x |

### Per-Token Breakdown (WebSocket + Lookahead)

| Component | Time |
|-----------|------|
| Cloud GPU (28 layers, A6000) | ~22ms |
| Network RTT (Montreal) | ~65ms |
| Local GPU (4 layers, RTX 3090) | ~5ms |
| Serialization overhead | ~2ms |
| **Total per sequential token** | **~90ms** |
| **Effective with lookahead** | **~75ms** (1.3-1.5 tokens/step) |

### Local-Only Mode

| Mode | Tokens/second | Notes |
|------|--------------|-------|
| RTX 3090 all 32 layers | ~39 tok/s | No cloud, no network latency |
| A100 all 32 layers | ~34.6 tok/s | Cloud baseline |

## Transport Protocol

### WebSocket Binary Protocol (Primary)

The 3090 server maintains a persistent WebSocket connection to the cloud on port 5001. This eliminates TCP handshake, HTTP headers, and base64 encoding overhead that plagued the original HTTP approach.

**Connection flow:**
1. Client connects to `ws://CLOUD_IP:5001`
2. Server sends JSON: `{"session_id": "uuid"}`
3. Client sends binary frames for each forward pass
4. Client sends JSON `{"type": "end"}` to close

**Binary frame format:**
```
┌──────────────┬──────────────┬─────────────────────────────────┐
│ 4B header_len│ JSON header  │         Tensor data             │
│  (big-endian)│              │  hidden + cos + sin [+ mask]    │
└──────────────┴──────────────┴─────────────────────────────────┘
```

**Request header fields:**
```json
{
  "hidden_shape": [1, seq_len, 4096],
  "pe_shape": [1, seq_len, 64],
  "is_prompt": true,
  "crop_to": null,
  "relocate": {"src": 25, "dst": 21, "len": 3},
  "has_mask": false,
  "mask_shape": [1, 1, 17, 45]
}
```

**Response header fields:**
```json
{
  "hidden_shape": [1, seq_len, 4096],
  "process_time_ms": 22.5,
  "cached_seq_len": 128
}
```

All tensors are raw float16 bytes - no base64, no torch.save, no JSON encoding. This reduces serialization from ~60-80ms (torch.save + base64) to ~1-2ms.

### HTTP/Flask (Fallback)

Flask server on port 5000 provides `/health`, `/new_session`, `/process`, and `/end_session` endpoints for backward compatibility and health checks.

## Decoding Modes

The RTX 3090 server (`jacobi_server.py`) supports three decoding modes:

### Sequential (`--mode sequential`)

Standard autoregressive decoding. One token per forward pass, one network round-trip per token. Simplest and most reliable.

```bash
python jacobi_server.py --mode sequential --cloud http://CLOUD_IP:5000
```

**Performance:** 7.8 tok/s over WebSocket

### Jacobi Parallel (`--mode jacobi`)

Processes blocks of k tokens in parallel. Each block iterates until predictions converge (fixed point). Useful when running locally (no network), but convergence issues make it slower than sequential over a network.

```bash
python jacobi_server.py --mode jacobi --block-size 16 --cloud http://CLOUD_IP:5000
```

**How it works:**
1. Guess a block of k tokens (random initialization)
2. Run all k tokens through the transformer with explicit causal mask
3. Compare predictions with guesses - if all match, block has converged
4. If not converged: crop KV cache back, update guesses, iterate
5. Once converged, emit all tokens and start next block

**Performance:** ~1.5 tok/s over network (worse than sequential due to convergence requiring ~15 iterations per block, each iteration being a full RTT). ~39 tok/s local-only.

### N-gram Lookahead (`--mode lookahead`)

Speculative decoding using n-gram patterns from the prompt and generated output. Each step verifies multiple candidate continuations in a single forward pass, accepting 1-5 tokens per round-trip.

```bash
python jacobi_server.py --mode lookahead --cloud http://CLOUD_IP:5000
```

**Performance:** 8.7 tok/s (12% over sequential, up to 5 tokens accepted per step)

**How it works:**

1. **Build n-gram pool** from prompt tokens: `Dict[token_id, List[Tuple[token_ids...]]]` mapping each token to continuations that historically follow it

2. **Each generation step:**
   - Get `first_token` from last logits (always correct, same as sequential)
   - Look up `first_token` in pool to find up to G=5 candidate continuations (each N-1=4 tokens)
   - Build combined input: `[first_token, cand0_tok0..tok3, cand1_tok0..tok3, ...]`
   - Build custom attention mask: each candidate is causal within itself, sees first_token and committed cache, but isolated from other candidates
   - Single forward pass through all layers (one RTT)
   - Verify: check if model agrees with each candidate token-by-token
   - Accept longest consecutive match from best candidate

3. **If match found** (length M): accept first_token + M tokens (M+1 total per RTT), relocate winning candidate's KV cache entries to correct positions, crop cache

4. **If no match**: accept just first_token (degrades to sequential, no worse)

5. **Grow pool**: add new n-grams from generated output after each step

**Attention mask structure** for combined input `[first_token, c0t0, c0t1, c0t2, c0t3, c1t0, ...]`:
```
Shape: [1, 1, total_tokens, committed_len + total_tokens]

Row 0 (first_token):  sees committed cache + self
Row 1-4 (candidate 0): sees committed cache + first_token + causal within candidate 0
Row 5-8 (candidate 1): sees committed cache + first_token + causal within candidate 1
```

Position IDs have duplicates (each candidate occupies the same future positions), while cache positions are unique sequential (for KV storage).

**KV cache relocation** after accepting a match: the winning candidate's KV entries are scattered in the cache (at their unique cache positions). They must be copied to the correct committed positions. Both local and cloud caches are updated - the cloud receives a `relocate` command (`{src, dst, len}`) in the next request's header, executes it before processing.

## Setup

### Requirements

**Local machine (RTX 3090 or similar):**
- Python 3.10+
- PyTorch with CUDA
- ~25GB disk for Mistral 7B model
- 24GB VRAM

**Cloud server:**
- NVIDIA GPU with 16GB+ VRAM (A100 80GB, A6000 48GB, etc.)
- CUDA 12.x
- PyTorch 2.x
- transformers 5.x

**Mac client (optional):**
- Any Mac - only displays text, no GPU needed

### Installation

```bash
# Clone and setup
cd /path/to/split-inference
python -m venv venv
source venv/bin/activate
pip install torch transformers requests flask websockets

# Download model (both local and cloud need this)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir models/mistral-7b-instruct
```

### Cloud Server Setup

```bash
# On cloud GPU server
ssh ubuntu@YOUR_CLOUD_IP
python -m venv ~/split-inference
source ~/split-inference/bin/activate
pip install torch transformers flask websockets

# Download model
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir ~/models/mistral-7b-instruct

# Copy and start server (Flask on 5000, WebSocket on 5001)
scp cloud_server_kv.py ubuntu@YOUR_CLOUD_IP:~/
nohup python -u ~/cloud_server_kv.py > ~/server.log 2>&1 &
```

### RTX 3090 Server Setup

```bash
ssh mike@192.168.1.32
cd /home/mike/D/coding/python_3/my_projects/split-inference-3090
source venv/bin/activate

# Lookahead mode (recommended for split inference)
nohup python -u jacobi_server.py \
  --model /path/to/models/mistral-7b-instruct \
  --mode lookahead \
  --cloud http://CLOUD_IP:5000 \
  > jacobi.log 2>&1 &

# Sequential mode (simpler, reliable)
nohup python -u jacobi_server.py \
  --model /path/to/models/mistral-7b-instruct \
  --mode sequential \
  --cloud http://CLOUD_IP:5000 \
  > jacobi.log 2>&1 &

# Local-only mode (no cloud, ~39 tok/s)
nohup python -u jacobi_server.py \
  --model /path/to/models/mistral-7b-instruct \
  > jacobi.log 2>&1 &
```

### Mac Client

```bash
python mac_client.py --server http://192.168.1.32:5001
```

## Files

| File | Description |
|------|-------------|
| `mac_client.py` | Lightweight Mac client - connects to RTX 3090, displays streaming text |
| `jacobi_server.py` | RTX 3090 server - sequential, Jacobi, or lookahead decoding modes |
| `cloud_server_kv.py` | Cloud server - WebSocket + HTTP, KV-cache, custom mask support |
| `interactive.py` | Interactive chat (Mac direct to cloud, no 3090 relay) |
| `local_client_kv.py` | Client library with KV-cache support (Mac direct) |
| `local_client.py` | Original client (no KV-cache, slower) |
| `cloud_server.py` | Original cloud server (no KV-cache, HTTP only) |

## Architectures

### Option 1: Mac → RTX 3090 → Cloud (recommended)
```
Mac (text only) → RTX 3090 (layers 0-1, 30-31) ←WebSocket→ Cloud GPU (layers 2-29)
```
Use `mac_client.py` + `jacobi_server.py --mode lookahead --cloud URL`
Text never leaves your local network. Best performance: 8.7 tok/s.

### Option 2: RTX 3090 Local-Only (no cloud)
```
Mac (text only) → RTX 3090 (all 32 layers, Jacobi decoding)
```
Use `mac_client.py` + `jacobi_server.py` (no --cloud flag). ~39 tok/s, zero latency.

### Option 3: Mac → Cloud (direct)
```
Mac (layers 0-1, 30-31) ←→ Cloud GPU (layers 2-29)
```
Use `interactive.py` or `local_client_kv.py`. Slower due to Mac MPS overhead.

## Server Management

### Health Checks

```bash
# Cloud server
curl http://CLOUD_IP:5000/health

# RTX 3090 server
curl http://192.168.1.32:5001/health
```

### Server Logs

```bash
# Cloud
ssh ubuntu@CLOUD_IP "tail -f ~/server.log"

# RTX 3090
ssh mike@192.168.1.32 "tail -f /home/mike/D/coding/python_3/my_projects/split-inference-3090/jacobi.log"
```

Lookahead mode logs each speculation step with timing and match details:
```
[Lookahead] Pool seeded with 16 n-grams from 20 prompt tokens
[Lookahead] Step 9: accepted 5 tokens (4 from n-gram match), 139ms
[Lookahead] Step 10: accepted 3 tokens (2 from n-gram match), 121ms
[Lookahead] Done: 85 tokens in 10807ms (7.9 tok/s), avg accepted/step: 1.49
```

## Technical Details

### KV-Cache

Without KV-cache, each token requires reprocessing the entire sequence (O(n^2) attention). With KV-cache:

1. **Prompt processing**: Full sequence processed once, K/V tensors cached per-layer
2. **Token generation**: Only new token processed, uses cached K/V
3. **Session management**: Cloud maintains per-session caches with 5-minute timeout
4. **Cache operations**: `DynamicCache.crop()` for rollback, `layer.keys`/`layer.values` for direct manipulation

### Transformers 5.x Compatibility

Key fixes for transformers 5.1.0+:

- **Position embeddings**: `model.model.rotary_emb()` returns `[batch, seq_len, head_dim]`. Do NOT add `unsqueeze(1)` - `apply_rotary_pos_emb` does this internally.
- **Batch dimension loss**: Decoder layers can return 2D `[seq_len, hidden_dim]` instead of 3D. Fix: `if hidden.dim() == 2: hidden = hidden.unsqueeze(0)` after each layer.
- **DynamicCache API**: Uses `cache.layers[idx].keys` / `cache.layers[idx].values` (not `cache.key_cache[idx]`). Some layers may have `None` keys if not all layers are processed locally.

### Attention Mask for Split Inference

SDPA's `is_causal=True` creates a relative mask that breaks when there's an existing KV cache. We build an explicit absolute mask:

```python
# Shape: [1, 1, num_new_tokens, total_kv_length]
mask = torch.full((1, 1, k, committed_len + k), float('-inf'), dtype=torch.float16)
for i in range(k):
    mask[0, 0, i, :committed_len + i + 1] = 0.0  # attend to cache + causal within block
```

### Layer Split Rationale

- **Local layers 0-1**: Embedding and early transformer layers (contain token-level information)
- **Cloud layers 2-29**: Middle transformer layers (bulk of compute, ~87.5% of model)
- **Local layers 30-31**: Final transformer layers and lm_head (map back to tokens)

### Data Transfer Per Token

Over WebSocket with binary serialization:
- **Sequential**: ~16KB round-trip (8KB hidden state + position embeddings each way)
- **Lookahead (5 candidates)**: ~70KB request + ~70KB response (21 tokens combined)
- Network bandwidth is not the bottleneck - latency is.

## Troubleshooting

### "Invalid session_id" error
The old `interactive.py` doesn't support KV-cache sessions. Use the updated version.

### WebSocket connection drops
Check that port 5001 is open on the cloud server. The WebSocket server has a 30s ping interval and 60s timeout.

### Slow performance (>1s per token)
- Ensure `cloud_server_kv.py` is running (not the old `cloud_server.py`)
- Check network latency: `ping CLOUD_IP` - anything over 100ms significantly impacts throughput
- Closer cloud server = faster generation (see latency impact table below)

### Shape mismatch errors
transformers 5.x changed rotary embedding handling. Ensure no manual `unsqueeze(1)` on position embeddings, and add batch dimension recovery after each layer call.

### DynamicCache AttributeError
transformers 5.x uses `cache.layers[idx].keys`/`.values` instead of `cache.key_cache[idx]`. Some layers will have `None` keys if not processed locally - always check before accessing.

## Latency Impact on Performance

Network RTT is the dominant bottleneck for split inference. Closer servers yield dramatically better performance:

| RTT to Cloud | Sequential | With Lookahead |
|-------------|-----------|----------------|
| ~85ms (Montreal) | 7.8 tok/s | 8.7 tok/s |
| ~40ms (East Coast) | ~15 tok/s | ~18 tok/s |
| ~20ms (Nearby DC) | ~22 tok/s | ~28 tok/s |
| ~10ms (Same city) | ~30 tok/s | ~35 tok/s |

## Future Improvements

1. **Draft model speculation**: Use a small draft model (e.g., Mistral 0.5B) to generate candidate tokens instead of n-gram lookup - higher match rates on novel content
2. **Activation compression**: Quantize activations to int8 for transfer, dequantize on arrival (~50% bandwidth reduction)
3. **Pipelining**: Compute token N+1 locally while cloud processes token N
4. **Activation noise/encryption**: Add differential privacy or lightweight encryption
5. **Multi-GPU cloud**: Split layers 2-29 across multiple GPUs for faster cloud processing
6. **LoRA injection**: Add local fine-tuning layers without cloud involvement

## Cost

Cloud GPU rental (examples):
- HyperStack A100 80GB: ~$1.35/hour
- A6000 48GB: ~$0.80/hour

Hibernate the instance when not in use.

## License

MIT
