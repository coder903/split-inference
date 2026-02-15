#!/usr/bin/env python3
"""
Split Inference - Interactive Mode with KV Cache
Privacy-preserving LLM: your text stays local, only activations go to cloud
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import requests
import io
import base64
import time

CLOUD_URL = "http://38.128.232.211:5000"
MODEL_PATH = "/Users/mike/coding/python_3/my_projects/split-inference/models/mistral-7b-instruct"
SPLIT_AFTER = 1   # Local: layers 0-1
RESUME_AT = 30    # Local: layers 30-31


def check_cloud():
    try:
        r = requests.get(f"{CLOUD_URL}/health", timeout=5)
        if r.status_code == 200:
            info = r.json()
            kv = "with KV cache" if info.get('kv_cache') else "no KV cache"
            print(f"Cloud connected: {info['gpu']}, layers {info['layers']} ({kv})")
            return True
    except:
        pass
    print("Cloud not reachable!")
    return False


def load_model():
    print("\nLoading local model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="mps"
    )
    model.eval()
    print(f"Ready: local layers 0-{SPLIT_AFTER} + {RESUME_AT}-31, cloud layers {SPLIT_AFTER+1}-{RESUME_AT-1}\n")
    return model, tokenizer


def new_session():
    """Create a new cloud session."""
    response = requests.post(f"{CLOUD_URL}/new_session", timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to create session: {response.text}")
    return response.json()['session_id']


def end_session(session_id):
    """End a cloud session."""
    try:
        requests.post(f"{CLOUD_URL}/end_session", json={"session_id": session_id}, timeout=5)
    except:
        pass


def send_to_cloud(session_id, hidden_states, position_embeddings, is_prompt):
    """Send to cloud for middle layer processing."""
    hs_buffer = io.BytesIO()
    torch.save(hidden_states.cpu(), hs_buffer)
    pe_buffer = io.BytesIO()
    torch.save((position_embeddings[0].cpu(), position_embeddings[1].cpu()), pe_buffer)

    response = requests.post(
        f"{CLOUD_URL}/process",
        json={
            "session_id": session_id,
            "hidden_states": base64.b64encode(hs_buffer.getvalue()).decode(),
            "position_embeddings": base64.b64encode(pe_buffer.getvalue()).decode(),
            "is_prompt": is_prompt
        },
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(f"Cloud error: {response.text}")

    result = response.json()
    hidden = torch.load(io.BytesIO(base64.b64decode(result['hidden_states'])), weights_only=True)
    return hidden, result['process_time_ms']


def generate(model, tokenizer, prompt, max_new_tokens=100):
    """Generate response using split inference with KV cache."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=False).to("mps")
    generated_ids = input_ids.clone()

    # Create session and local caches
    session_id = new_session()
    early_cache = DynamicCache()
    late_cache = DynamicCache()
    current_seq_len = 0

    print("Generating", end="", flush=True)
    start_total = time.time()

    try:
        for i in range(max_new_tokens):
            is_prompt = (i == 0)

            if is_prompt:
                current_ids = generated_ids
                seq_len = current_ids.shape[1]
                cache_position = torch.arange(seq_len, device="mps")
            else:
                current_ids = generated_ids[:, -1:]
                seq_len = 1
                cache_position = torch.tensor([current_seq_len], device="mps")

            position_ids = cache_position.unsqueeze(0)

            with torch.no_grad():
                # Embedding
                hidden = model.model.embed_tokens(current_ids)
                cos, sin = model.model.rotary_emb(hidden, position_ids)
                position_embeddings = (cos, sin)

                # Early layers (0-1) with cache
                for idx in range(SPLIT_AFTER + 1):
                    out = model.model.layers[idx](
                        hidden,
                        position_ids=position_ids,
                        past_key_values=early_cache,
                        use_cache=True,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                    hidden = out[0]
                    if hidden.dim() == 2:
                        hidden = hidden.unsqueeze(0)

                # Cloud for middle layers (2-29)
                hidden, cloud_ms = send_to_cloud(session_id, hidden, position_embeddings, is_prompt)
                hidden = hidden.to("mps").half()
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)

                # Late layers (30-31) with cache
                for idx in range(RESUME_AT, 32):
                    out = model.model.layers[idx](
                        hidden,
                        position_ids=position_ids,
                        past_key_values=late_cache,
                        use_cache=True,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                    hidden = out[0]
                    if hidden.dim() == 2:
                        hidden = hidden.unsqueeze(0)

                # Final norm + lm_head
                hidden = model.model.norm(hidden)
                logits = model.lm_head(hidden)

            # Update sequence length
            if is_prompt:
                current_seq_len = generated_ids.shape[1]
            else:
                current_seq_len += 1

            # Next token
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            print(".", end="", flush=True)

    finally:
        end_session(session_id)

    total_time = time.time() - start_total
    tokens_generated = generated_ids.shape[1] - input_ids.shape[1]
    print(f" done ({total_time:.1f}s, {tokens_generated} tokens, {tokens_generated/total_time:.1f} tok/s)")

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def main():
    print("=" * 60)
    print("Split Inference - Privacy-Preserving LLM")
    print("Your text stays local. Only activations go to cloud.")
    print("=" * 60)

    if not check_cloud():
        print("\nCloud server must be running on 185.216.21.60:5000")
        return

    model, tokenizer = load_model()

    print("Type your prompts below. 'quit' to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            response = generate(model, tokenizer, prompt)
            # Extract just the assistant's response
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
