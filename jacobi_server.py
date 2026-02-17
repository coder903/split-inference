#!/usr/bin/env python3
"""
Split Inference - Local Decoding Server

Supports sequential, Jacobi parallel, and lookahead (n-gram speculation)
decoding modes. Model-agnostic: works with any HuggingFace causal LM
(tested with Mistral 7B and LLaMA 2 13B).

Modes:
  --cloud None  → All layers run locally (for testing/baseline)
  --cloud URL   → Early/late layers local, middle layers on cloud GPU
                  Layer split configured via --split-after / --resume-at
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from flask import Flask, request, jsonify, Response
from dataclasses import dataclass, field
import json
import struct
import time
import base64
import requests as http_requests
from websockets.sync.client import connect as ws_connect
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
    block_size: int = 16
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
                "block_size": self.block_size,
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
    """Jacobi parallel decoding engine for split inference."""

    def __init__(self, model, tokenizer, device, cloud_url=None,
                 block_size=16, max_iters=32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cloud_url = cloud_url
        self.block_size = block_size
        self.max_iters = max_iters
        self.cloud_session_id = None
        self._ws = None  # WebSocket connection
        # State passed to _send_to_cloud during forward passes
        self._cloud_is_prompt = False
        self._cloud_crop_to = None
        self._cloud_attn_mask = None   # Custom mask tensor (for speculation)
        self._cloud_relocate = None    # {"src": int, "dst": int, "len": int}
        self._step_timings = []        # Per-cloud-call timing decomposition

    def _relocate_local_cache(self, cache, src, dst, length):
        """Copy KV cache entries from src to dst positions, then crop."""
        for layer in cache.layers:
            if layer.keys is None:
                continue
            layer.keys[:, :, dst:dst+length, :] = \
                layer.keys[:, :, src:src+length, :].clone()
            layer.values[:, :, dst:dst+length, :] = \
                layer.values[:, :, src:src+length, :].clone()

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
                        block_size: int = None, return_logits: bool = False):
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
        if block_size is None:
            block_size = self.block_size
        metrics = JacobiMetrics(block_size=block_size)
        gen_start = time.time()

        # Tokenize
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=False
        ).to(self.device)
        metrics.prompt_tokens = input_ids.shape[1]

        # Create cloud session if in split mode
        if self.cloud_url:
            self._cloud_new_session()

        # Prefill
        cache = DynamicCache()
        prefill_start = time.time()
        self._cloud_is_prompt = True
        self._cloud_crop_to = None
        prefill_logits = self._prefill(input_ids, cache)
        self._cloud_is_prompt = False
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

            # Initialize block: position 0 is known, rest are random guesses
            # Random init converges faster than repeat-last (2-4 tok/iter vs 1)
            block_ids = torch.randint(
                0, self.model.config.vocab_size, (k,), device=self.device
            )
            block_ids[0] = first_token_id  # Position 0 is always known

            block_converged = False
            iters_used = 0

            # In split mode, tell cloud to crop to committed_len before each
            # forward pass. First iteration: no-op. Subsequent: rolls back.
            self._cloud_crop_to = committed_len

            for iteration in range(self.max_iters):
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

            # Save logits for return_logits mode (before last_logits gets updated)
            if return_logits:
                _emit_first_logits = last_logits[0].cpu()
                _emit_source = block_logits if block_converged else final_logits

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
                event = {"token": new_text, "token_id": tid}
                if return_logits:
                    if j == 0:
                        event["logits"] = _emit_first_logits
                    else:
                        event["logits"] = _emit_source[0, j - 1].cpu()
                yield event
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

        # Clean up cloud session
        if self.cloud_url:
            self._cloud_end_session()

        yield {"done": True, **metrics.summary()}

    def generate_stream_lookahead(self, prompt: str, max_new_tokens: int = 500,
                                   ngram_size: int = 5, max_candidates: int = 5,
                                   return_logits: bool = False):
        """Generate with n-gram speculation for multi-token acceptance.

        Maintains an n-gram pool (seeded from prompt, grown from output).
        Each step: look up candidate continuations, verify in a single
        forward pass with isolated attention masks, accept longest match.

        Falls back to sequential (1 token/step) when no candidates match.
        """
        from collections import defaultdict

        N = ngram_size       # n-gram size (key + N-1 continuation tokens)
        G = max_candidates   # max candidates to verify per step
        match_len = N - 1    # length of each candidate continuation

        metrics = JacobiMetrics(block_size=1)
        gen_start = time.time()

        # Tokenize
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=False
        ).to(self.device)
        prompt_ids = input_ids[0].tolist()
        metrics.prompt_tokens = len(prompt_ids)

        # Build n-gram pool from prompt
        pool = defaultdict(list)
        for i in range(len(prompt_ids) - N + 1):
            key = prompt_ids[i]
            ngram = tuple(prompt_ids[i + 1:i + N])
            if ngram not in pool[key]:
                pool[key].append(ngram)
        print(f"  [Lookahead] Pool seeded with {sum(len(v) for v in pool.values())} "
              f"n-grams from {len(prompt_ids)} prompt tokens")

        # Create cloud session if in split mode
        if self.cloud_url:
            self._cloud_new_session()

        # Prefill
        cache = DynamicCache()
        prefill_start = time.time()
        self._cloud_is_prompt = True
        self._cloud_crop_to = None
        self._cloud_attn_mask = None
        self._cloud_relocate = None
        prefill_logits = self._prefill(input_ids, cache)
        self._cloud_is_prompt = False
        metrics.prefill_time_ms = (time.time() - prefill_start) * 1000
        committed_len = input_ids.shape[1]

        last_logits = prefill_logits[:, -1, :]
        tokens_remaining = max_new_tokens

        all_generated_ids = []
        prev_text = ""
        total_accepted = 0
        total_steps = 0

        while tokens_remaining > 0:
            first_token_id = torch.argmax(last_logits, dim=-1).item()

            if first_token_id == self.tokenizer.eos_token_id:
                break

            # Look up candidates in n-gram pool
            candidates = pool.get(first_token_id, [])[:G]

            fwd_start = time.time()
            total_steps += 1

            if candidates and tokens_remaining > 1:
                # === SPECULATIVE VERIFICATION ===
                # Build combined input: [first_token, cand0_tokens..., cand1_tokens...]
                combined_ids = [first_token_id]
                for cand in candidates:
                    combined_ids.extend(cand)

                total_tokens = len(combined_ids)

                # Position IDs: first_token at committed_len,
                # each candidate at committed_len+1..+match_len
                pos_ids = [committed_len]
                for cand in candidates:
                    pos_ids.extend(range(committed_len + 1, committed_len + 1 + len(cand)))

                # Cache positions: unique sequential
                cache_pos = list(range(committed_len, committed_len + total_tokens))

                # Build attention mask
                kv_total = committed_len + total_tokens
                mask = torch.full(
                    (1, 1, total_tokens, kv_total), float('-inf'),
                    device=self.device, dtype=torch.float16
                )
                # All tokens attend to committed cache
                mask[:, :, :, :committed_len] = 0.0
                # first_token attends to self
                mask[:, :, 0, committed_len] = 0.0
                # Each candidate: causal within itself + see first_token
                offset = 1
                for cand in candidates:
                    cand_len = len(cand)
                    for j in range(cand_len):
                        row = offset + j
                        mask[:, :, row, committed_len] = 0.0  # see first_token
                        for prev_j in range(j + 1):
                            mask[:, :, row, committed_len + offset + prev_j] = 0.0
                    offset += cand_len

                # Forward pass
                combined_tensor = torch.tensor(
                    combined_ids, device=self.device
                ).unsqueeze(0)
                cache_position = torch.tensor(cache_pos, device=self.device)
                position_ids = torch.tensor(pos_ids, device=self.device).unsqueeze(0)

                self._cloud_crop_to = committed_len
                self._cloud_attn_mask = mask

                with torch.no_grad():
                    hidden = self.model.model.embed_tokens(combined_tensor)
                    cos, sin = self.model.model.rotary_emb(hidden, position_ids)
                    position_embeddings = (cos, sin)

                    hidden = self._forward_layers(
                        hidden, position_embeddings, cache, cache_position,
                        position_ids, mask
                    )

                    hidden = self.model.model.norm(hidden)
                    logits = self.model.lm_head(hidden)

                metrics.forward_times_ms.append((time.time() - fwd_start) * 1000)

                # Verify candidates
                first_guess = torch.argmax(logits[0, 0]).item()

                best_hit = 0
                best_cand_idx = -1
                best_last_logits_row = 0  # row in logits for last accepted token

                offset = 1
                for ci, cand in enumerate(candidates):
                    cand_len = len(cand)
                    hit = 0

                    if cand[0] == first_guess:
                        hit = 1
                        for j in range(1, cand_len):
                            pred = torch.argmax(logits[0, offset + j - 1]).item()
                            if cand[j] == pred:
                                hit = j + 1
                            else:
                                break

                    if hit > best_hit:
                        best_hit = hit
                        best_cand_idx = ci
                        best_last_logits_row = offset + hit - 1

                    offset += cand_len

                # Handle results
                if best_hit > 0:
                    accepted_count = best_hit + 1  # first_token + matched tokens

                    # Relocate winning candidate's KV entries
                    src_cache = committed_len + 1
                    for ci in range(best_cand_idx):
                        src_cache += len(candidates[ci])
                    dst_cache = committed_len + 1

                    if src_cache != dst_cache:
                        self._relocate_local_cache(
                            cache, src_cache, dst_cache, best_hit
                        )
                        # Tell cloud to do the same on next request
                        self._cloud_relocate = {
                            "src": src_cache, "dst": dst_cache, "len": best_hit
                        }

                    cache.crop(committed_len + accepted_count)
                    last_logits = logits[:, best_last_logits_row:best_last_logits_row + 1, :]

                    # Build per-token logits for return_logits mode
                    if return_logits:
                        best_offset = 1 + sum(len(candidates[ci]) for ci in range(best_cand_idx))
                        _emit_logits = [last_logits.squeeze().cpu()]  # first_token_id [vocab]
                        _emit_logits.append(logits[0, 0].cpu())  # cand[0]
                        for ej in range(1, best_hit):
                            _emit_logits.append(logits[0, best_offset + ej - 1].cpu())

                    # Emit tokens
                    emit_ids = [first_token_id] + list(candidates[best_cand_idx][:best_hit])
                    for ei, tid in enumerate(emit_ids):
                        if tid == self.tokenizer.eos_token_id:
                            tokens_remaining = 0
                            break
                        all_generated_ids.append(tid)
                        full_text = self.tokenizer.decode(all_generated_ids)
                        new_text = full_text[len(prev_text):]
                        prev_text = full_text
                        event = {"token": new_text, "token_id": tid}
                        if return_logits:
                            event["logits"] = _emit_logits[ei]
                        yield event
                        metrics.generated_tokens += 1

                    committed_len += accepted_count
                    tokens_remaining -= accepted_count
                    total_accepted += accepted_count

                    metrics.blocks_completed += 1
                    metrics.total_iterations += 1
                    metrics.iterations_per_block.append(1)
                    metrics.tokens_per_block.append(accepted_count)

                    print(f"  [Lookahead] Step {total_steps}: "
                          f"accepted {accepted_count} tokens "
                          f"({best_hit} from n-gram match), "
                          f"{metrics.forward_times_ms[-1]:.0f}ms")
                else:
                    # No match, accept just first_token
                    cache.crop(committed_len + 1)
                    _no_match_logits = last_logits.squeeze().cpu() if return_logits else None
                    last_logits = logits[:, 0:1, :]

                    all_generated_ids.append(first_token_id)
                    full_text = self.tokenizer.decode(all_generated_ids)
                    new_text = full_text[len(prev_text):]
                    prev_text = full_text
                    event = {"token": new_text, "token_id": first_token_id}
                    if return_logits:
                        event["logits"] = _no_match_logits
                    yield event
                    metrics.generated_tokens += 1

                    committed_len += 1
                    tokens_remaining -= 1
                    total_accepted += 1

                    metrics.blocks_completed += 1
                    metrics.total_iterations += 1
                    metrics.iterations_per_block.append(1)
                    metrics.tokens_per_block.append(1)

                    print(f"  [Lookahead] Step {total_steps}: "
                          f"accepted 1 token (no match from {len(candidates)} candidates), "
                          f"{metrics.forward_times_ms[-1]:.0f}ms")
            else:
                # No candidates available, standard sequential forward
                self._cloud_crop_to = committed_len
                self._cloud_attn_mask = None

                block_id = torch.tensor([first_token_id], device=self.device)
                cache_position = torch.tensor([committed_len], device=self.device)
                position_ids = cache_position.unsqueeze(0)

                with torch.no_grad():
                    hidden = self.model.model.embed_tokens(block_id.unsqueeze(0))
                    cos, sin = self.model.model.rotary_emb(hidden, position_ids)
                    position_embeddings = (cos, sin)

                    hidden = self._forward_layers(
                        hidden, position_embeddings, cache, cache_position,
                        position_ids, None
                    )

                    hidden = self.model.model.norm(hidden)
                    logits = self.model.lm_head(hidden)

                metrics.forward_times_ms.append((time.time() - fwd_start) * 1000)
                _no_cand_logits = last_logits.squeeze().cpu() if return_logits else None
                last_logits = logits[:, -1:, :]

                all_generated_ids.append(first_token_id)
                full_text = self.tokenizer.decode(all_generated_ids)
                new_text = full_text[len(prev_text):]
                prev_text = full_text
                event = {"token": new_text, "token_id": first_token_id}
                if return_logits:
                    event["logits"] = _no_cand_logits
                yield event
                metrics.generated_tokens += 1

                committed_len += 1
                tokens_remaining -= 1
                total_accepted += 1

                metrics.blocks_completed += 1
                metrics.total_iterations += 1
                metrics.iterations_per_block.append(1)
                metrics.tokens_per_block.append(1)

            # Add new n-grams from generated output to pool
            all_ids = prompt_ids + all_generated_ids
            if len(all_ids) >= N:
                key = all_ids[-N]
                ngram = tuple(all_ids[-N + 1:])
                if ngram not in pool[key]:
                    pool[key].append(ngram)

        metrics.total_time_ms = (time.time() - gen_start) * 1000
        avg_accepted = total_accepted / max(1, total_steps)
        print(f"  [Lookahead] Done: {metrics.generated_tokens} tokens in "
              f"{metrics.total_time_ms:.0f}ms ({metrics.tokens_per_second:.1f} tok/s), "
              f"avg accepted/step: {avg_accepted:.2f}, "
              f"pool size: {sum(len(v) for v in pool.values())}")

        if self.cloud_url:
            self._cloud_end_session()

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
        """All layers on local GPU (local-only mode)."""
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
        """Split mode: early layers local, middle cloud, late layers local."""
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

        # Cloud layers (2-29) - pass custom mask if set
        if self._cloud_attn_mask is None and attn_mask is not None:
            self._cloud_attn_mask = attn_mask
        hidden = self._send_to_cloud(
            hidden, position_embeddings, cache_position,
            is_prompt=self._cloud_is_prompt,
            crop_to=self._cloud_crop_to
        )

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

    def _cloud_new_session(self):
        """Open WebSocket connection to cloud server."""
        # Derive WebSocket URL from HTTP URL (port 5001)
        ws_url = self.cloud_url.replace('http://', 'ws://').replace(':5000', ':5001')
        self._ws = ws_connect(ws_url, max_size=20 * 1024 * 1024)
        # Server sends session ID on connect
        resp = json.loads(self._ws.recv())
        self.cloud_session_id = resp['session_id']
        print(f"    [cloud-ws] Connected, session={self.cloud_session_id[:8]}")

    def _cloud_end_session(self):
        """Close WebSocket connection."""
        if self._ws:
            try:
                self._ws.send(json.dumps({"type": "end"}))
                self._ws.close()
            except:
                pass
            self._ws = None
            self.cloud_session_id = None

    def _send_to_cloud(self, hidden, position_embeddings, cache_position,
                       is_prompt=False, crop_to=None):
        """Send intermediate activations to cloud via WebSocket.

        Binary protocol: [4B header_len][JSON header][tensor bytes]
        Tensor bytes: hidden + cos + sin + [optional mask] (raw float16, no base64)
        """
        t0 = time.time()

        # Serialize to raw bytes (no base64!)
        hs_np = hidden.cpu().numpy()
        cos_np = position_embeddings[0].cpu().numpy()
        sin_np = position_embeddings[1].cpu().numpy()

        t1 = time.time()

        # Build header with optional mask and relocate
        header_dict = {
            "hidden_shape": list(hs_np.shape),
            "pe_shape": list(cos_np.shape),
            "is_prompt": is_prompt,
            "crop_to": crop_to,
        }

        # Optional relocate command (for n-gram speculation KV management)
        if self._cloud_relocate is not None:
            header_dict["relocate"] = self._cloud_relocate
            self._cloud_relocate = None  # One-shot

        # Optional custom attention mask
        mask_bytes = b''
        if self._cloud_attn_mask is not None:
            mask_np = self._cloud_attn_mask.cpu().numpy()
            mask_bytes = mask_np.tobytes()
            header_dict["has_mask"] = True
            header_dict["mask_shape"] = list(mask_np.shape)
            self._cloud_attn_mask = None  # One-shot

        header = json.dumps(header_dict).encode()

        msg = (struct.pack('>I', len(header))
               + header
               + hs_np.tobytes()
               + cos_np.tobytes()
               + sin_np.tobytes()
               + mask_bytes)

        self._ws.send(msg)

        t2 = time.time()

        # Receive binary response
        resp = self._ws.recv()

        t3 = time.time()

        # Unpack: [4B header_len][JSON header][tensor bytes]
        resp_header_len = struct.unpack('>I', resp[:4])[0]
        result = json.loads(resp[4:4 + resp_header_len])
        tensor_data = resp[4 + resp_header_len:]

        out = torch.frombuffer(
            bytearray(tensor_data), dtype=torch.float16
        ).reshape(result['hidden_shape']).clone().to(self.device)

        if out.dim() == 2:
            out = out.unsqueeze(0)

        t4 = time.time()
        cloud_ms = result.get('process_time_ms', 0)
        print(f"    [cloud-ws] serialize={1000*(t1-t0):.0f}ms "
              f"send={1000*(t2-t1):.0f}ms "
              f"recv={1000*(t3-t2):.0f}ms "
              f"(cloud_gpu={cloud_ms:.0f}ms) "
              f"deserialize={1000*(t4-t3):.0f}ms "
              f"total={1000*(t4-t0):.0f}ms "
              f"seq_len={hidden.shape[1]}")

        self._step_timings.append({
            "serialize_ms": round((t1 - t0) * 1000, 2),
            "send_ms": round((t2 - t1) * 1000, 2),
            "recv_ms": round((t3 - t2) * 1000, 2),
            "deserialize_ms": round((t4 - t3) * 1000, 2),
            "total_ms": round((t4 - t0) * 1000, 2),
            "cloud_gpu_ms": cloud_ms,
            "network_rtt_ms": round((t3 - t1) * 1000 - cloud_ms, 2),
            "seq_len": hidden.shape[1],
        })

        return out


# === Flask App ===

app = Flask(__name__)
decoder = None
DECODE_MODE = "sequential"  # "sequential", "jacobi", or "lookahead"


def load_model(model_path, device, split_after=None, resume_at=None):
    global decoder, NUM_LAYERS, SPLIT_AFTER, RESUME_AT
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if CLOUD_URL:
        # Split mode: load to CPU first, then move only local layers to GPU
        # This allows models larger than GPU VRAM (e.g., 13B on 24GB 3090)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, device_map="cpu",
        )
        model.eval()

        NUM_LAYERS = len(model.model.layers)
        SPLIT_AFTER = split_after if split_after is not None else 1
        RESUME_AT = resume_at if resume_at is not None else NUM_LAYERS - 2

        # Move only local layers + embeddings + head to GPU
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
        for i in range(SPLIT_AFTER + 1):
            model.model.layers[i] = model.model.layers[i].to(device)
        for i in range(RESUME_AT, NUM_LAYERS):
            model.model.layers[i] = model.model.layers[i].to(device)
        model.model.norm = model.model.norm.to(device)
        model.lm_head = model.lm_head.to(device)

        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Split mode: {NUM_LAYERS} layers total, "
              f"local 0-{SPLIT_AFTER} + {RESUME_AT}-{NUM_LAYERS-1} on {torch.cuda.get_device_name(0)}, "
              f"cloud {SPLIT_AFTER+1}-{RESUME_AT-1}, "
              f"VRAM: {mem_gb:.1f}GB")
    else:
        # Local-only mode: load everything to GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, device_map=device,
        )
        model.eval()

        NUM_LAYERS = len(model.model.layers)
        SPLIT_AFTER = split_after if split_after is not None else 1
        RESUME_AT = resume_at if resume_at is not None else NUM_LAYERS - 2

        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Local-only: {NUM_LAYERS} layers on {torch.cuda.get_device_name(0)}, "
              f"VRAM: {mem_gb:.1f}GB")

    decoder = JacobiDecoder(model, tokenizer, device, cloud_url=CLOUD_URL,
                            block_size=BLOCK_SIZE, max_iters=MAX_JACOBI_ITERS)


@app.route('/health', methods=['GET'])
def health():
    mode = f"local-only (all {NUM_LAYERS} layers)" if CLOUD_URL is None else f"split (cloud: {CLOUD_URL})"
    return jsonify({
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0),
        "memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "num_layers": NUM_LAYERS,
        "local_layers": f"0-{NUM_LAYERS-1}" if CLOUD_URL is None else f"0-{SPLIT_AFTER}, {RESUME_AT}-{NUM_LAYERS-1}",
        "cloud_url": CLOUD_URL or "disabled",
        "streaming": True,
        "jacobi": True,
        "block_size": decoder.block_size,
        "max_jacobi_iters": decoder.max_iters,
        "mode": mode,
    })


def _get_generator(prompt, max_new_tokens):
    """Route to the appropriate generation method based on DECODE_MODE."""
    if DECODE_MODE == "lookahead":
        return decoder.generate_stream_lookahead(prompt, max_new_tokens)
    else:
        return decoder.generate_stream(prompt, max_new_tokens)


@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 500)

    print(f"\n[Request] mode={DECODE_MODE} prompt={prompt[:80]}... max_tokens={max_new_tokens}")

    def event_stream():
        for event in _get_generator(prompt, max_new_tokens):
            yield f"data: {json.dumps(event)}\n\n"

    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 500)

    print(f"\n[Request] mode={DECODE_MODE} prompt={prompt[:80]}... max_tokens={max_new_tokens}")

    token_ids = []
    stats = None
    for event in _get_generator(prompt, max_new_tokens):
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
    parser.add_argument('--model', default=MODEL_PATH, help='Path to model (e.g., Mistral 7B, LLaMA 2 13B)')
    parser.add_argument('--port', type=int, default=5001, help='Server port')
    parser.add_argument('--cloud', default=None, help='Cloud server URL for split mode')
    parser.add_argument('--block-size', type=int, default=BLOCK_SIZE, help='Jacobi block size k')
    parser.add_argument('--max-iters', type=int, default=MAX_JACOBI_ITERS, help='Max Jacobi iterations per block')
    parser.add_argument('--mode', default='sequential', choices=['sequential', 'jacobi', 'lookahead'],
                        help='Decoding mode: sequential (block_size=1), jacobi, or lookahead (n-gram speculation)')
    parser.add_argument('--split-after', type=int, default=None,
                        help='Last local layer before cloud (default: 1)')
    parser.add_argument('--resume-at', type=int, default=None,
                        help='First local layer after cloud (default: num_layers - 2)')
    args = parser.parse_args()

    MODEL_PATH = args.model
    CLOUD_URL = args.cloud
    BLOCK_SIZE = args.block_size if args.mode == 'jacobi' else 1
    MAX_JACOBI_ITERS = args.max_iters
    DECODE_MODE = args.mode

    load_model(MODEL_PATH, DEVICE, split_after=args.split_after, resume_at=args.resume_at)
    print(f"\nStarting server on port {args.port}...")
    print(f"  Mode: {DECODE_MODE}, Block size: {BLOCK_SIZE}")
    app.run(host='0.0.0.0', port=args.port, threaded=True)
