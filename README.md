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

With KV-cache implementation (Feb 2026):

| Metric | Value |
|--------|-------|
| Tokens/second | 3.3 tok/s |
| Per-token latency | ~280ms |
| Cloud processing | ~80ms (28 layers on A100) |
| Local processing | ~100ms (4 layers on M4 Max) |
| Network overhead | ~100ms |

Baseline comparison:
- A100 alone (full model): 34.6 tok/s
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
| `interactive.py` | Interactive chat interface with KV-cache |
| `local_client_kv.py` | Client library with KV-cache support |
| `cloud_server_kv.py` | Cloud server with session-based KV-cache |
| `local_client.py` | Original client (no KV-cache, slower) |
| `cloud_server.py` | Original server (no KV-cache, slower) |

## Server Management

### Start server
```bash
ssh ubuntu@185.216.21.60
source ~/split-inference/bin/activate
nohup python ~/cloud_server_kv.py > ~/server.log 2>&1 &
```

### Check health
```bash
curl http://185.216.21.60:5000/health
```

### View logs
```bash
ssh ubuntu@185.216.21.60 "tail -f ~/server.log"
```

## Technical Details

### KV-Cache Implementation

Without KV-cache, each token requires reprocessing the entire sequence (O(n²) attention). With KV-cache:

1. **Prompt processing**: Full sequence processed once, K/V tensors cached
2. **Token generation**: Only new token processed, uses cached K/V
3. **Session management**: Cloud maintains per-session caches with 5-minute timeout

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

1. **Activation noise/encryption**: Add differential privacy or homomorphic encryption
2. **Speculative decoding**: Predict multiple tokens locally, verify on cloud
3. **Model parallelism**: Split layers across multiple cloud GPUs
4. **LoRA injection**: Add local fine-tuning without cloud involvement

## Cost

HyperStack A100 80GB: ~$1.35/hour

Hibernate the instance when not in use.

## License

MIT
