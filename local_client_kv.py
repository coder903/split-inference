"""
Split Inference - Local Client with KV Cache
Uses session-based KV cache for fast token generation
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


class SplitInferenceModelKV:
    def __init__(self, model_path: str):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="mps"
        )
        self.model.eval()
        print(f"Ready: local 0-{SPLIT_AFTER} + {RESUME_AT}-31, cloud {SPLIT_AFTER+1}-{RESUME_AT-1}")

        # Local KV caches for early and late layers
        self.early_cache = None  # Layers 0-1
        self.late_cache = None   # Layers 30-31
        self.session_id = None
        self.current_seq_len = 0

    def _new_session(self):
        """Create new session on cloud and reset local caches."""
        response = requests.post(f"{CLOUD_URL}/new_session", timeout=10)
        if response.status_code != 200:
            raise Exception(f"Failed to create session: {response.text}")
        self.session_id = response.json()['session_id']
        self.early_cache = DynamicCache()
        self.late_cache = DynamicCache()
        self.current_seq_len = 0
        return self.session_id

    def _end_session(self):
        """Clean up session."""
        if self.session_id:
            try:
                requests.post(f"{CLOUD_URL}/end_session",
                             json={"session_id": self.session_id}, timeout=5)
            except:
                pass
            self.session_id = None
            self.early_cache = None
            self.late_cache = None

    def generate_split(self, prompt: str, max_new_tokens: int = 100) -> str:
        print(f"\nGenerating: {prompt[:50]}...")

        # Create new session
        self._new_session()

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=False
        ).to("mps")

        prompt_len = input_ids.shape[1]
        generated_ids = input_ids.clone()

        try:
            for i in range(max_new_tokens):
                start_time = time.time()
                is_prompt = (i == 0)

                if is_prompt:
                    # First iteration: process full prompt
                    current_ids = generated_ids
                    seq_len = current_ids.shape[1]
                    cache_position = torch.arange(seq_len, device="mps")
                else:
                    # Subsequent: process only new token
                    current_ids = generated_ids[:, -1:]
                    seq_len = 1
                    cache_position = torch.tensor([self.current_seq_len], device="mps")

                position_ids = cache_position.unsqueeze(0)

                with torch.no_grad():
                    # Embedding
                    hidden = self.model.model.embed_tokens(current_ids)

                    # Get position embeddings for current positions
                    cos, sin = self.model.model.rotary_emb(hidden, position_ids)
                    position_embeddings = (cos, sin)

                    # Early layers (0-1) with local cache
                    for idx in range(SPLIT_AFTER + 1):
                        layer = self.model.model.layers[idx]
                        out = layer(
                            hidden,
                            position_ids=position_ids,
                            past_key_values=self.early_cache,
                            use_cache=True,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )
                        hidden = out[0]
                        if hidden.dim() == 2:
                            hidden = hidden.unsqueeze(0)

                    # Send to cloud for middle layers
                    cloud_result, cloud_ms = self._send_to_cloud(
                        hidden, position_embeddings, is_prompt
                    )
                    hidden = cloud_result.to("mps").half()
                    if hidden.dim() == 2:
                        hidden = hidden.unsqueeze(0)

                    # Late layers (30-31) with local cache
                    for idx in range(RESUME_AT, 32):
                        layer = self.model.model.layers[idx]
                        out = layer(
                            hidden,
                            position_ids=position_ids,
                            past_key_values=self.late_cache,
                            use_cache=True,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )
                        hidden = out[0]
                        if hidden.dim() == 2:
                            hidden = hidden.unsqueeze(0)

                    # Final norm + lm_head
                    hidden = self.model.model.norm(hidden)
                    logits = self.model.lm_head(hidden)

                # Update sequence length
                if is_prompt:
                    self.current_seq_len = generated_ids.shape[1]
                else:
                    self.current_seq_len += 1

                # Next token
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                token_time = time.time() - start_time
                print(f"  Token {i+1}: '{self.tokenizer.decode(next_token[0])}' ({token_time*1000:.0f}ms, cloud:{cloud_ms:.0f}ms)")

        finally:
            self._end_session()

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def _send_to_cloud(self, hidden_states: torch.Tensor, position_embeddings, is_prompt: bool):
        """Send to cloud for middle layer processing."""
        # Serialize hidden states
        hs_buffer = io.BytesIO()
        torch.save(hidden_states.cpu(), hs_buffer)

        # Serialize position embeddings
        pe_buffer = io.BytesIO()
        torch.save((position_embeddings[0].cpu(), position_embeddings[1].cpu()), pe_buffer)

        response = requests.post(
            f"{CLOUD_URL}/process",
            json={
                "session_id": self.session_id,
                "hidden_states": base64.b64encode(hs_buffer.getvalue()).decode(),
                "position_embeddings": base64.b64encode(pe_buffer.getvalue()).decode(),
                "is_prompt": is_prompt
            },
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"Cloud error: {response.text}")

        result = response.json()
        return torch.load(io.BytesIO(base64.b64decode(result['hidden_states'])), weights_only=True), result['process_time_ms']


def check_cloud():
    try:
        r = requests.get(f"{CLOUD_URL}/health", timeout=5)
        if r.status_code == 200:
            info = r.json()
            kv = "with KV cache" if info.get('kv_cache') else "no KV cache"
            print(f"Cloud: {info['gpu']}, layers {info['layers']} ({kv})")
            return True
    except:
        pass
    print("Cloud not reachable")
    return False


if __name__ == "__main__":
    if not check_cloud():
        exit(1)

    model = SplitInferenceModelKV(MODEL_PATH)
    response = model.generate_split("Describe quantum entanglement to me.", max_new_tokens=100)
    print(f"\n{'='*50}\nResponse: {response}")
