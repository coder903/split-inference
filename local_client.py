"""
Split Inference - Local Client
Fixed position embeddings shape for transformers 5.1.0
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import io
import base64
import time

CLOUD_URL = "http://185.216.21.60:5000"

SPLIT_AFTER = 1   # Local: layers 0-1
RESUME_AT = 30    # Local: layers 30-31


class SplitInferenceModel:
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

    def _get_position_embeddings(self, hidden_states, position_ids):
        """Get position embeddings with correct shape for layers."""
        cos, sin = self.model.model.rotary_emb(hidden_states, position_ids)
        # apply_rotary_pos_emb does unsqueeze internally, so return as-is
        return (cos, sin)

    def generate_split(self, prompt: str, max_new_tokens: int = 50) -> str:
        print(f"\nGenerating: {prompt[:50]}...")

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=False
        ).to("mps")

        generated_ids = input_ids.clone()

        for i in range(max_new_tokens):
            start_time = time.time()
            seq_len = generated_ids.shape[1]

            with torch.no_grad():
                # Embedding
                hidden = self.model.model.embed_tokens(generated_ids)
                position_ids = torch.arange(seq_len, device="mps").unsqueeze(0)
                position_embeddings = self._get_position_embeddings(hidden, position_ids)

                # Early layers (0-1)
                for idx in range(SPLIT_AFTER + 1):
                    layer = self.model.model.layers[idx]
                    out = layer(hidden, position_ids=position_ids, position_embeddings=position_embeddings)
                    hidden = out[0]
                    # Ensure batch dimension is preserved
                    if hidden.dim() == 2:
                        hidden = hidden.unsqueeze(0)

                # Send to cloud for middle layers
                cloud_result = self._send_to_cloud(hidden, position_embeddings, seq_len)
                hidden = cloud_result.to("mps").half()

                # Late layers (30-31)
                for idx in range(RESUME_AT, 32):
                    layer = self.model.model.layers[idx]
                    out = layer(hidden, position_ids=position_ids, position_embeddings=position_embeddings)
                    hidden = out[0]
                    # Ensure batch dimension is preserved
                    if hidden.dim() == 2:
                        hidden = hidden.unsqueeze(0)

                # Final norm + lm_head
                hidden = self.model.model.norm(hidden)
                logits = self.model.lm_head(hidden)

            # Next token
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            token_time = time.time() - start_time
            print(f"  Token {i+1}: '{self.tokenizer.decode(next_token[0])}' ({token_time*1000:.1f}ms)")

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def _send_to_cloud(self, hidden_states: torch.Tensor, position_embeddings, seq_len: int) -> torch.Tensor:
        """Send to cloud for middle layer processing."""
        # Serialize hidden states
        hs_buffer = io.BytesIO()
        torch.save(hidden_states.cpu(), hs_buffer)

        # Serialize position embeddings (cos, sin with unsqueezed num_heads dim)
        pe_buffer = io.BytesIO()
        torch.save((position_embeddings[0].cpu(), position_embeddings[1].cpu()), pe_buffer)

        response = requests.post(
            f"{CLOUD_URL}/process",
            json={
                "hidden_states": base64.b64encode(hs_buffer.getvalue()).decode(),
                "position_embeddings": base64.b64encode(pe_buffer.getvalue()).decode(),
                "seq_len": seq_len
            },
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"Cloud error: {response.text}")

        result = response.json()
        print(f"  Cloud: {result['process_time_ms']:.1f}ms")

        return torch.load(io.BytesIO(base64.b64decode(result['hidden_states'])), weights_only=True)


def check_cloud():
    try:
        r = requests.get(f"{CLOUD_URL}/health", timeout=5)
        if r.status_code == 200:
            info = r.json()
            print(f"Cloud: {info['gpu']}, layers {info['layers']}")
            return True
    except:
        pass
    print("Cloud not reachable")
    return False


if __name__ == "__main__":
    if not check_cloud():
        exit(1)

    MODEL_PATH = "/Users/mike/coding/python_3/my_projects/split-inference/models/mistral-7b-instruct"
    model = SplitInferenceModel(MODEL_PATH)
    response = model.generate_split("What is 2 + 2?", max_new_tokens=20)
    print(f"\n{'='*50}\nResponse: {response}")
