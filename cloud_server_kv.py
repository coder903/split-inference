"""
Split Inference - Cloud Server with KV Cache
Maintains session-based KV cache for fast token generation
"""

import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import io
import base64
import time
import uuid

app = Flask(__name__)
model = None

CLOUD_START = 2
CLOUD_END = 29

# Session-based KV caches: {session_id: {'cache': DynamicCache, 'last_access': time}}
sessions = {}
SESSION_TIMEOUT = 300  # 5 minutes


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


def cleanup_old_sessions():
    """Remove sessions older than timeout."""
    now = time.time()
    expired = [sid for sid, data in sessions.items() if now - data['last_access'] > SESSION_TIMEOUT]
    for sid in expired:
        del sessions[sid]
    if expired:
        print(f"Cleaned up {len(expired)} expired sessions")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0),
        "memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "layers": f"{CLOUD_START}-{CLOUD_END}",
        "active_sessions": len(sessions),
        "kv_cache": True
    })


@app.route('/new_session', methods=['POST'])
def new_session():
    """Create a new session with fresh KV cache."""
    cleanup_old_sessions()
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'cache': None,  # Will be created on first process
        'last_access': time.time(),
        'seq_len': 0
    }
    return jsonify({"session_id": session_id})


@app.route('/process', methods=['POST'])
def process():
    try:
        start = time.time()
        data = request.json

        session_id = data.get('session_id')
        is_prompt = data.get('is_prompt', True)

        # Validate session
        if session_id not in sessions:
            return jsonify({"error": "Invalid session_id. Call /new_session first."}), 400

        session = sessions[session_id]
        session['last_access'] = time.time()

        # Decode hidden states
        hidden = torch.load(
            io.BytesIO(base64.b64decode(data['hidden_states'])),
            weights_only=True
        ).cuda().half()

        # Ensure batch dimension
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        # Decode position embeddings
        cos, sin = torch.load(
            io.BytesIO(base64.b64decode(data['position_embeddings'])),
            weights_only=True
        )
        position_embeddings = (cos.cuda().half(), sin.cuda().half())

        seq_len = hidden.shape[1]

        if is_prompt:
            # First request: process full prompt, create cache
            session['cache'] = DynamicCache()
            cache_position = torch.arange(seq_len, device='cuda')
            session['seq_len'] = seq_len
        else:
            # Subsequent: process only new token
            cache_position = torch.tensor([session['seq_len']], device='cuda')
            session['seq_len'] += 1

        position_ids = cache_position.unsqueeze(0)

        # Process middle layers with KV cache
        with torch.no_grad():
            for idx in range(CLOUD_START, CLOUD_END + 1):
                layer = model.model.layers[idx]

                # Get layer-specific cache
                layer_idx = idx - CLOUD_START  # Relative index for our subset

                out = layer(
                    hidden,
                    position_ids=position_ids,
                    past_key_values=session['cache'],
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden = out[0]

                # Ensure batch dimension preserved
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)

        # Encode output (only last position for non-prompt)
        buffer = io.BytesIO()
        if is_prompt:
            torch.save(hidden.cpu(), buffer)
        else:
            # Only return the new token's hidden state
            torch.save(hidden[:, -1:, :].cpu(), buffer)

        return jsonify({
            "hidden_states": base64.b64encode(buffer.getvalue()).decode(),
            "shape": list(hidden.shape),
            "process_time_ms": round((time.time() - start) * 1000, 2),
            "cached_seq_len": session['seq_len']
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/end_session', methods=['POST'])
def end_session():
    """Clean up a session."""
    data = request.json
    session_id = data.get('session_id')
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({"status": "session ended"})
    return jsonify({"status": "session not found"})


if __name__ == '__main__':
    load_model()
    print("Starting KV-cache server on port 5000...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
