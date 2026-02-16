#!/usr/bin/env python3
"""
Split Inference - RTX 3090 Jacobi Parallel Decoding Server

Instead of generating 1 token per network round-trip, Jacobi decoding
processes blocks of k tokens in parallel. Each block iterates until
predictions converge (fixed point), then commits to KV cache.

For 100 tokens: ~10 iterations instead of 100 round-trips → 3-5x speedup.

Modes:
  CLOUD_URL = None  → All 32 layers run locally on 3090 (for testing)
  CLOUD_URL = "..." → Layers 0-1 + 30-31 local, 2-29 on cloud A100
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from flask import Flask, request, jsonify, Response
from dataclasses import dataclass, field
import json
import time
import io
import base64
import requests as http_requests
import argparse

# === Configuration ===
MODEL_PATH = "/home/mike/D/models/mistral-7b-instruct"
CLOUD_URL = None  # Set to "http://<cloud-ip>:5000" when A100 is available
DEVICE = "cuda"
BLOCK_SIZE = 16
MAX_JACOBI_ITERS = 32  # Must be >= BLOCK_SIZE for vanilla Jacobi (1 token/iter worst case)
NUM_LAYERS = 32
SPLIT_AFTER = 1   # Local: layers 0-1
RESUME_AT = 30    # Local: layers 30-31


@dataclass
class JacobiMetrics:
    """Tracks performance metrics for a single generation."""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_time_ms: float = 0
    prefill_time_ms: float = 0
    blocks_completed: int = 0
    total_iterations: int = 0
    iterations_per_block: list = field(default_factory=list)
    tokens_per_block: list = field(default_factory=list)
    forward_times_ms: list = field(default_factory=list)

    @property
    def avg_iters_per_block(self):
        return self.total_iterations / max(1, self.blocks_completed)

    @property
    def tokens_per_second(self):
        return self.generated_tokens / max(0.001, self.total_time_ms / 1000)

    @property
    def speedup_vs_sequential(self):
        """generated_tokens / total_iterations = tokens gained per forward pass."""
        return self.generated_tokens / max(1, self.total_iterations)

    def summary(self) -> dict:
        return {
            "tokens_generated": self.generated_tokens,
            "total_time_ms": round(self.total_time_ms, 1),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "prefill_time_ms": round(self.prefill_time_ms, 1),
            "jacobi_stats": {
                "block_size": BLOCK_SIZE,
                "blocks": self.blocks_completed,
                "total_iterations": self.total_iterations,
                "avg_iters_per_block": round(self.avg_iters_per_block, 2),
                "iterations_per_block": self.iterations_per_block,
                "tokens_per_block": self.tokens_per_block,
                "avg_forward_ms": round(
                    sum(self.forward_times_ms) / max(1, len(self.forward_times_ms)), 1
                ),
                "speedup_vs_sequential": round(self.speedup_vs_sequential, 2),
            }
        }


class JacobiDecoder:
    """Jacobi parallel decoding engine for Mistral 7B."""

    def __init__(self, model, tokenizer, device, cloud_url=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cloud_url = cloud_url
        self.cloud_session_id = None

    def _make_causal_mask(self, committed_len, k):
        """Create causal attention mask for a Jacobi block of k tokens.

        SDPA's is_causal=True creates a RELATIVE mask that breaks when
        there's an existing KV cache. We need an ABSOLUTE mask where
        query i (at position committed_len + i) attends to positions
        0..committed_len+i.
        """
        kv_len = committed_len + k
        mask = torch.full(
            (1, 1, k, kv_len), float('-inf'),
            device=self.device, dtype=torch.float16
        )
        for i in range(k):
            mask[0, 0, i, :committed_len + i + 1] = 0.0
        return mask

    def generate_stream(self, prompt: str, max_new_tokens: int = 500,
                        block_size: int = BLOCK_SIZE):
        """Generator that yields token dicts as blocks converge.

        Key insight: In a transformer, output at position i predicts token i+1.
        So for a Jacobi block at positions [n, n+1, ..., n+k-1]:
          - Token at position n comes from the PREVIOUS step's logits (already known)
          - Token at position n+j (j>0) comes from logits at position n+j-1
          - Logits at position n+k-1 predict the NEXT block's first token

        Yields:
            {"token": "text"}          for each generated token
            {"done": True, ...stats}   when generation is complete
        """
        metrics = JacobiMetrics()
        gen_start = time.time()

        # Tokenize
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=False
        ).to(self.device)
        metrics.prompt_tokens = input_ids.shape[1]

        # Prefill
        cache = DynamicCache()
        prefill_start = time.time()
        prefill_logits = self._prefill(input_ids, cache)
        metrics.prefill_time_ms = (time.time() - prefill_start) * 1000
        committed_len = input_ids.shape[1]

        # last_logits: logits from the last committed position, used to
        # determine the first token of the next block
        last_logits = prefill_logits[:, -1, :]  # [1, vocab]
        tokens_remaining = max_new_tokens

        # For correct decoding with SentencePiece, accumulate all token IDs
        # and decode incrementally (single-token decode strips spaces)
        all_generated_ids = []
        prev_text = ""

        # Jacobi block loop
        while tokens_remaining > 0:
            k = min(block_size, tokens_remaining)

            # Position 0 of block is determined by last_logits (already known)
            first_token_id = torch.argmax(last_logits, dim=-1)  # [1]

            # Check EOS at position 0
            if first_token_id.item() == self.tokenizer.eos_token_id:
                tokens_remaining = 0
                break

            # Initialize block: position 0 is known, rest are guesses
            block_ids = first_token_id.repeat(k)  # [k] - all same as first token

            block_converged = False
            iters_used = 0

            for iteration in range(MAX_JACOBI_ITERS):
                iters_used += 1
                fwd_start = time.time()

                # Forward pass for the block
                block_logits = self._forward_block_logits(
                    block_ids, committed_len, cache
                )
                metrics.forward_times_ms.append((time.time() - fwd_start) * 1000)

                # Extract predictions using the SHIFTED mapping:
                #   new[0] = argmax(last_logits)           (from previous step)
                #   new[j] = argmax(block_logits[j-1])     (output at j-1 predicts j)
                new_block_ids = torch.empty_like(block_ids)
                new_block_ids[0] = first_token_id  # Always the same
                if k > 1:
                    new_block_ids[1:] = torch.argmax(block_logits[0, :-1, :], dim=-1)

                # Save logits from last position (predicts next block's first token)
                next_last_logits = block_logits[:, -1, :]

                # Check convergence (fixed point)
                if torch.equal(block_ids, new_block_ids):
                    # Converged! KV cache entries from this forward pass are correct.
                    block_converged = True
                    break

                # Not converged: crop cache back and try again
                cache.crop(committed_len)
                block_ids = new_block_ids

            # --- Block finished (converged or max iterations) ---

            # If max iters hit without convergence, commit with a final forward pass
            if not block_converged:
                cache.crop(committed_len)
                final_logits = self._forward_block_logits(
                    block_ids, committed_len, cache
                )
                next_last_logits = final_logits[:, -1, :]

            # Check for EOS in the final block (only after convergence/commit)
            eos_mask = (block_ids == self.tokenizer.eos_token_id) if not block_converged else \
                       (new_block_ids == self.tokenizer.eos_token_id)
            final_block = new_block_ids if block_converged else block_ids

            if eos_mask.any():
                eos_pos = eos_mask.nonzero(as_tuple=True)[0][0].item()
                # Truncate to EOS (exclusive - don't emit EOS itself)
                k = eos_pos
                if k > 0:
                    # Crop cache to only committed + truncated block
                    cache.crop(committed_len + k)
                else:
                    cache.crop(committed_len)
                committed_len += k
                tokens_remaining = 0
            else:
                committed_len += k
                last_logits = next_last_logits
                tokens_remaining -= k

            # Emit tokens (decode incrementally for correct spacing)
            for j in range(k):
                tid = final_block[j].item()
                all_generated_ids.append(tid)
                full_text = self.tokenizer.decode(all_generated_ids)
                new_text = full_text[len(prev_text):]
                prev_text = full_text
                yield {"token": new_text, "token_id": tid}
                metrics.generated_tokens += 1

            metrics.blocks_completed += 1
            metrics.total_iterations += iters_used
            metrics.iterations_per_block.append(iters_used)
            metrics.tokens_per_block.append(k)

            print(f"  [Jacobi] Block {metrics.blocks_completed}: "
                  f"{'converged' if block_converged else 'max iters'} in {iters_used} iters "
                  f"({k} tokens, "
                  f"{metrics.forward_times_ms[-1]:.0f}ms/fwd)")

        metrics.total_time_ms = (time.time() - gen_start) * 1000
        print(f"  [Jacobi] Done: {metrics.generated_tokens} tokens in "
              f"{metrics.total_time_ms:.0f}ms ({metrics.tokens_per_second:.1f} tok/s), "
              f"speedup vs sequential: {metrics.speedup_vs_sequential:.2f}x")

        yield {"done": True, **metrics.summary()}

    def _prefill(self, input_ids, cache):
        """Process full prompt through all layers, building KV cache.
        Returns logits for the entire prompt."""
        seq_len = input_ids.shape[1]
        cache_position = torch.arange(seq_len, device=self.device)
        position_ids = cache_position.unsqueeze(0)

        with torch.no_grad():
            hidden = self.model.model.embed_tokens(input_ids)
            cos, sin = self.model.model.rotary_emb(hidden, position_ids)
            position_embeddings = (cos, sin)

            hidden = self._forward_layers(
                hidden, position_embeddings, cache, cache_position, position_ids
            )

            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden)

        return logits

    def _forward_block_logits(self, block_ids, committed_len, cache):
        """Forward pass for a Jacobi block of k tokens.
        Appends k KV entries to cache. Caller must crop if not converged.

        Returns: logits [1, k, vocab_size]
        """
        k = block_ids.shape[0]
        cache_position = torch.arange(
            committed_len, committed_len + k, device=self.device
        )
        position_ids = cache_position.unsqueeze(0)
        attn_mask = self._make_causal_mask(committed_len, k)

        with torch.no_grad():
            hidden = self.model.model.embed_tokens(block_ids.unsqueeze(0))
            cos, sin = self.model.model.rotary_emb(hidden, position_ids)
            position_embeddings = (cos, sin)

            hidden = self._forward_layers(
                hidden, position_embeddings, cache, cache_position,
                position_ids, attn_mask
            )

            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden)

        return logits

    def _commit_block(self, block_ids, committed_len, cache):
        """Forward a block through all layers with use_cache=True
        to populate the KV cache with correct entries."""
        k = block_ids.shape[0]
        cache_position = torch.arange(
            committed_len, committed_len + k, device=self.device
        )
        position_ids = cache_position.unsqueeze(0)
        attn_mask = self._make_causal_mask(committed_len, k)

        with torch.no_grad():
            hidden = self.model.model.embed_tokens(block_ids.unsqueeze(0))
            cos, sin = self.model.model.rotary_emb(hidden, position_ids)
            position_embeddings = (cos, sin)

            self._forward_layers(
                hidden, position_embeddings, cache, cache_position,
                position_ids, attn_mask
            )

    def _forward_layers(self, hidden, position_embeddings, cache,
                        cache_position, position_ids, attn_mask=None):
        """Route through layers: all local or split with cloud."""
        if self.cloud_url is None:
            return self._forward_all_layers(
                hidden, position_embeddings, cache, cache_position,
                position_ids, attn_mask
            )
        else:
            return self._forward_split_layers(
                hidden, position_embeddings, cache, cache_position,
                position_ids, attn_mask
            )

    def _forward_all_layers(self, hidden, position_embeddings, cache,
                            cache_position, position_ids, attn_mask=None):
        """All 32 layers on local GPU (local-only mode)."""
        for idx in range(NUM_LAYERS):
            out = self.model.model.layers[idx](
                hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = out[0]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
        return hidden

    def _forward_split_layers(self, hidden, position_embeddings, cache,
                              cache_position, position_ids, attn_mask=None):
        """Split mode: layers 0-1 local, 2-29 cloud, 30-31 local."""
        # Early layers (0-1)
        for idx in range(SPLIT_AFTER + 1):
            out = self.model.model.layers[idx](
                hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = out[0]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

        # Cloud layers (2-29)
        hidden = self._send_to_cloud(hidden, position_embeddings, cache_position)

        # Late layers (30-31)
        for idx in range(RESUME_AT, NUM_LAYERS):
            out = self.model.model.layers[idx](
                hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = out[0]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

        return hidden

    def _send_to_cloud(self, hidden, position_embeddings, cache_position):
        """Send intermediate activations to cloud for layers 2-29.
        TODO: Implement when cloud A100 is available.
        """
        raise NotImplementedError(
            "Cloud relay not yet implemented. Run with CLOUD_URL=None for local-only mode."
        )


# === Flask App ===

app = Flask(__name__)
decoder = None


def load_model(model_path, device):
    global decoder
    print(f"Loading Mistral 7B from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Loaded on {torch.cuda.get_device_name(0)}, "
          f"VRAM: {mem_gb:.1f}GB, "
          f"mode: {'split (cloud)' if CLOUD_URL else 'local-only (all 32 layers)'}")

    decoder = JacobiDecoder(model, tokenizer, device, cloud_url=CLOUD_URL)


@app.route('/health', methods=['GET'])
def health():
    mode = "local-only (all 32 layers)" if CLOUD_URL is None else f"split (cloud: {CLOUD_URL})"
    return jsonify({
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0),
        "memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "local_layers": "0-31" if CLOUD_URL is None else f"0-{SPLIT_AFTER}, {RESUME_AT}-31",
        "cloud_url": CLOUD_URL or "disabled",
        "streaming": True,
        "jacobi": True,
        "block_size": BLOCK_SIZE,
        "max_jacobi_iters": MAX_JACOBI_ITERS,
        "mode": mode,
    })


@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 500)

    print(f"\n[Request] prompt={prompt[:80]}... max_tokens={max_new_tokens}")

    def event_stream():
        for event in decoder.generate_stream(prompt, max_new_tokens):
            yield f"data: {json.dumps(event)}\n\n"

    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 500)

    print(f"\n[Request] prompt={prompt[:80]}... max_tokens={max_new_tokens}")

    token_ids = []
    stats = None
    for event in decoder.generate_stream(prompt, max_new_tokens):
        if 'token_id' in event:
            token_ids.append(event['token_id'])
        elif 'done' in event:
            stats = event

    response_text = decoder.tokenizer.decode(token_ids, skip_special_tokens=True)
    return jsonify({
        "response": response_text,
        **stats,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jacobi Parallel Decoding Server')
    parser.add_argument('--model', default=MODEL_PATH, help='Path to Mistral 7B model')
    parser.add_argument('--port', type=int, default=5001, help='Server port')
    parser.add_argument('--cloud', default=None, help='Cloud server URL for split mode')
    parser.add_argument('--block-size', type=int, default=BLOCK_SIZE, help='Jacobi block size k')
    parser.add_argument('--max-iters', type=int, default=MAX_JACOBI_ITERS, help='Max Jacobi iterations per block')
    args = parser.parse_args()

    MODEL_PATH = args.model
    CLOUD_URL = args.cloud
    BLOCK_SIZE = args.block_size
    MAX_JACOBI_ITERS = args.max_iters

    load_model(MODEL_PATH, DEVICE)
    print(f"\nStarting Jacobi server on port {args.port}...")
    print(f"  Block size: {BLOCK_SIZE}, Max iterations: {MAX_JACOBI_ITERS}")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
