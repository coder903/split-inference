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
                    ═════════╪══════════════════════╪═════════════
                             │      NETWORK         │
                    ═════════╪══════════════════════╪═════════════
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

### Sequential Decoding (KV-cache)

| Metric | Value |
|--------|-------|
| Tokens/second | 5.3 tok/s |
| Per-token latency | ~190ms |
| Cloud processing | ~54ms (28 layers on A100) |
| Local processing | ~80ms (4 layers on RTX 3090) |
| Network overhead | ~56ms |

### Jacobi Parallel Decoding

| Mode | Tokens/second | Notes |
|------|--------------|-------|
| RTX 3090 local-only | ~39 tok/s | All 32 layers on 3090, no network |
| Split + cloud (projected) | ~18 tok/s | 3-4x speedup over sequential |

Baseline comparison:
- A100 alone (full model): 34.6 tok/s
- RTX 3090 alone (sequential): ~100 tok/s
- Mac M4 Max alone: ~5-10 tok/s

## Setup

### Requirements

**Local machine:**
- Python 3.10+
- PyTorch with MPS (Mac) or CUDA (Linux/Windows)
- ~25GB disk for Mistral 7B model
- 16GB+ RAM (48GB recommended)

**Cloud server:**
- NVIDIA GPU with 16GB+ VRAM (A100 80GB recommended)
- CUDA 12.x
- PyTorch 2.x
- transformers 5.x

### Installation

```bash
# Clone and setup local
cd /path/to/split-inference
python -m venv venv
source venv/bin/activate
pip install torch transformers requests flask

# Download model (both local and cloud need this)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir models/mistral-7b-instruct
```

### Cloud Server Setup

```bash
# On cloud GPU server
ssh ubuntu@YOUR_CLOUD_IP
python -m venv ~/split-inference
source ~/split-inference/bin/activate
pip install torch transformers flask

# Download model
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir ~/models/mistral-7b-instruct

# Copy server script
scp cloud_server_kv.py ubuntu@YOUR_CLOUD_IP:~/

# Start server
nohup python ~/cloud_server_kv.py > ~/server.log 2>&1 &
```

### Configuration

Edit the following in `interactive.py` or `local_client_kv.py`:

```python
CLOUD_URL = "http://YOUR_CLOUD_IP:5000"
MODEL_PATH = "/path/to/models/mistral-7b-instruct"
```

## Usage

### Interactive Mode

```bash
source venv/bin/activate
python interactive.py
```

### Programmatic Usage

```python
from local_client_kv import SplitInferenceModelKV

model = SplitInferenceModelKV("/path/to/model")
response = model.generate_split("Your prompt here", max_new_tokens=100)
print(response)
```

## Files

| File | Description |
|------|-------------|
| `mac_client.py` | **Recommended** - Lightweight Mac client that connects to RTX 3090 |
| `jacobi_server.py` | **RTX 3090 Jacobi server** - parallel decoding for 3-5x speedup |
| `interactive.py` | Interactive chat (Mac → Cloud direct) |
| `local_client_kv.py` | Client library with KV-cache support |
| `cloud_server_kv.py` | Cloud server with session-based KV-cache |
| `local_client.py` | Original client (no KV-cache, slower) |
| `cloud_server.py` | Original server (no KV-cache, slower) |

### RTX 3090 Server (separate machine)
Located at `/home/mike/D/coding/python_3/my_projects/split-inference-3090/`:
| File | Description |
|------|-------------|
| `jacobi_server.py` | **Recommended** - Jacobi parallel decoding server |
| `local_server.py` | Original sequential server (relays to cloud) |

## Architectures

### Option 1: Mac → Cloud (direct)
```
Mac (layers 0-1, 30-31) ←→ Cloud A100 (layers 2-29)
```
Use `interactive.py` or `local_client_kv.py`

### Option 2: Mac → RTX 3090 → Cloud (recommended)
```
Mac (text only) → RTX 3090 (layers 0-1, 30-31) ←→ Cloud A100 (layers 2-29)
```
Use `mac_client.py` + `jacobi_server.py` - text never leaves your local network!

### Option 3: RTX 3090 Local-Only (no cloud)
```
Mac (text only) → RTX 3090 (all 32 layers, Jacobi decoding)
```
Use `mac_client.py` + `jacobi_server.py` with `CLOUD_URL=None` - ~39 tok/s, no cloud needed

## Server Management

### Cloud A100 Server
```bash
ssh ubuntu@38.128.232.211
source ~/split-inference/bin/activate
nohup python ~/cloud_server_kv.py > ~/server.log 2>&1 &
```

Check health: `curl http://38.128.232.211:5000/health`

### RTX 3090 Jacobi Server (192.168.1.32)
```bash
ssh mike@192.168.1.32
cd /home/mike/D/coding/python_3/my_projects/split-inference-3090
source venv/bin/activate

# Local-only mode (all 32 layers on 3090, no cloud needed)
nohup python jacobi_server.py > jacobi.log 2>&1 &

# Split mode (layers 0-1 + 30-31 on 3090, layers 2-29 on cloud)
nohup python jacobi_server.py --cloud http://CLOUD_IP:5000 > jacobi.log 2>&1 &
```

Check health: `curl http://192.168.1.32:5001/health`

## Technical Details

### KV-Cache Implementation

Without KV-cache, each token requires reprocessing the entire sequence (O(n²) attention). With KV-cache:

1. **Prompt processing**: Full sequence processed once, K/V tensors cached
2. **Token generation**: Only new token processed, uses cached K/V
3. **Session management**: Cloud maintains per-session caches with 5-minute timeout

### Jacobi Parallel Decoding

Standard autoregressive decoding generates 1 token per forward pass. With split inference, each token requires a network round-trip to the cloud (~190ms), making generation slow. Jacobi decoding solves this by processing blocks of tokens in parallel.

**How it works:**

1. **Prefill**: Process the full prompt through all layers, building the KV cache
2. **Block initialization**: Guess a block of k tokens (default k=16) - initially all copies of the last token
3. **Parallel forward pass**: Run all k tokens through the transformer simultaneously with an explicit causal mask
4. **Convergence check**: Compare predictions with the current guesses
   - If all predictions match (fixed point reached): the block has converged, commit to KV cache
   - If not: crop the KV cache back, update guesses, iterate
5. **Emit**: Once converged (or max iterations hit), emit all k tokens at once and move to the next block

**Why it's faster for split inference:**

With sequential decoding, 100 tokens = 100 network round-trips. With Jacobi (k=16), 100 tokens = ~7 blocks, each taking ~7-10 iterations to converge. But each iteration is a single forward pass through the cloud (1 RTT), and it produces up to 16 tokens. Net result: ~50-70 RTTs instead of 100, with each RTT doing more useful work.

**Key implementation details:**
- Explicit causal attention mask (SDPA's `is_causal=True` breaks with existing KV cache)
- Shifted logit mapping: output at position i predicts token i+1
- KV cache crop/rollback on non-convergence using `DynamicCache.crop()`
- EOS checking only after convergence, not during iterations

### Layer Split

- **Local layers 0-1**: Embedding and first transformer layers
- **Cloud layers 2-29**: Middle transformer layers (bulk of compute)
- **Local layers 30-31**: Final transformer layers and unembedding

### Data Transfer

Per token (after prompt):
- Upload: ~8KB (single token hidden state + position embeddings)
- Download: ~8KB (processed hidden state)

## Troubleshooting

### "Invalid session_id" error
The old `interactive.py` doesn't support KV-cache sessions. Use the updated version.

### Slow performance (>1s per token)
You may be using the non-KV-cache version. Check that `cloud_server_kv.py` is running.

### Shape mismatch errors
transformers 5.x changed rotary embedding handling. Ensure you have the latest code with:
- No manual `unsqueeze(1)` on position embeddings
- `if hidden.dim() == 2: hidden = hidden.unsqueeze(0)` after each layer

## Future Improvements

1. **Jacobi + cloud split**: Enable `_send_to_cloud()` in `jacobi_server.py` for 3-5x speedup over sequential split inference
2. **Activation noise/encryption**: Add differential privacy or homomorphic encryption
3. **Model parallelism**: Split layers across multiple cloud GPUs
4. **LoRA injection**: Add local fine-tuning without cloud involvement
5. **Draft-assisted Jacobi**: Use a smaller draft model to initialize blocks instead of naive repetition, improving convergence speed

## Cost

HyperStack A100 80GB: ~$1.35/hour

Hibernate the instance when not in use.

## License

MIT
