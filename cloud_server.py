"""
Split Inference - Cloud Server
Fixed position embeddings shape for transformers 5.1.0
"""

import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM
import io
import base64
import time

app = Flask(__name__)
model = None

CLOUD_START = 2
CLOUD_END = 29


def load_model():
    global model
    print("Loading Mistral 7B...")
    model = AutoModelForCausalLM.from_pretrained(
        "/home/ubuntu/models/mistral-7b-instruct",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    print(f"Loaded. Layers {CLOUD_START}-{CLOUD_END}, GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0),
        "memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "layers": f"{CLOUD_START}-{CLOUD_END}"
    })


@app.route('/process', methods=['POST'])
def process():
    try:
        start = time.time()
        data = request.json

        # Decode hidden states
        hidden = torch.load(
            io.BytesIO(base64.b64decode(data['hidden_states'])),
            weights_only=True
        ).cuda().half()

        # Decode position embeddings
        cos, sin = torch.load(
            io.BytesIO(base64.b64decode(data['position_embeddings'])),
            weights_only=True
        )
        position_embeddings = (cos.cuda().half(), sin.cuda().half())

        # Ensure hidden states have batch dimension
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        seq_len = data['seq_len']
        position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0)

        # Process middle layers
        with torch.no_grad():
            for idx in range(CLOUD_START, CLOUD_END + 1):
                layer = model.model.layers[idx]
                out = layer(
                    hidden,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                hidden = out[0]
                # Ensure batch dimension is preserved
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)

        # Encode output
        buffer = io.BytesIO()
        torch.save(hidden.cpu(), buffer)

        return jsonify({
            "hidden_states": base64.b64encode(buffer.getvalue()).decode(),
            "shape": list(hidden.shape),
            "process_time_ms": round((time.time() - start) * 1000, 2)
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == '__main__':
    load_model()
    print("Starting server on port 5000...")
    app.run(host='0.0.0.0', port=5000, threaded=False)
