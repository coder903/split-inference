"""
Split Inference - Cloud Server with KV Cache
Maintains session-based KV cache for fast token generation.
Model-agnostic: works with any HuggingFace causal LM.

Transport:
  - WebSocket (primary, low-latency binary protocol)
  - HTTP/Flask (health checks, backward compat)

Usage:
  python cloud_server_kv.py --model ~/models/mistral-7b-instruct
  python cloud_server_kv.py --model ~/models/llama-2-13b-chat --cloud-start 2 --cloud-end 37
"""

import torch
import numpy as np
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import asyncio
import websockets
import struct
import json
import base64
import time
import uuid
import threading
import traceback

app = Flask(__name__)
model = None

CLOUD_START = 2
CLOUD_END = 29

# Session-based KV caches: {session_id: {'cache': DynamicCache, 'last_access': time}}
sessions = {}
SESSION_TIMEOUT = 300  # 5 minutes


def load_model(model_path, cloud_start=None, cloud_end=None):
    global model, CLOUD_START, CLOUD_END
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    num_layers = len(model.model.layers)
    CLOUD_START = cloud_start if cloud_start is not None else 2
    CLOUD_END = cloud_end if cloud_end is not None else num_layers - 3
    print(f"Loaded. {num_layers} layers total, serving {CLOUD_START}-{CLOUD_END}, "
          f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")


def cleanup_old_sessions():
    """Remove sessions older than timeout."""
    now = time.time()
    expired = [sid for sid, data in sessions.items() if now - data['last_access'] > SESSION_TIMEOUT]
    for sid in expired:
        del sessions[sid]
    if expired:
        print(f"Cleaned up {len(expired)} expired sessions")


def process_tensors(session, hidden, position_embeddings, seq_len, is_prompt,
                    crop_to, relocate=None, custom_mask=None):
    """Core processing logic shared by HTTP and WebSocket handlers."""
    # Handle cache management
    if is_prompt:
        session['cache'] = DynamicCache()
        cache_position = torch.arange(seq_len, device='cuda')
        session['seq_len'] = seq_len
    else:
        # Relocate KV entries before cropping (for n-gram speculation)
        if relocate is not None and session['cache'] is not None:
            src, dst, length = relocate['src'], relocate['dst'], relocate['len']
            for layer in session['cache'].layers:
                if layer.keys is None:
                    continue
                layer.keys[:, :, dst:dst+length, :] = \
                    layer.keys[:, :, src:src+length, :].clone()
                layer.values[:, :, dst:dst+length, :] = \
                    layer.values[:, :, src:src+length, :].clone()

        if crop_to is not None and session['cache'] is not None:
            session['cache'].crop(crop_to)
            session['seq_len'] = crop_to
        cache_position = torch.arange(
            session['seq_len'], session['seq_len'] + seq_len, device='cuda'
        )
        session['seq_len'] += seq_len

    position_ids = cache_position.unsqueeze(0)

    # Use custom mask if provided, otherwise build default causal mask
    if custom_mask is not None:
        attn_mask = custom_mask
    else:
        attn_mask = None
        if not is_prompt and seq_len > 1 and session['seq_len'] - seq_len > 0:
            committed_len = session['seq_len'] - seq_len
            kv_len = session['seq_len']
            attn_mask = torch.full(
                (1, 1, seq_len, kv_len), float('-inf'),
                device='cuda', dtype=torch.float16
            )
            for i in range(seq_len):
                attn_mask[0, 0, i, :committed_len + i + 1] = 0.0

    # Process middle layers with KV cache
    with torch.no_grad():
        for idx in range(CLOUD_START, CLOUD_END + 1):
            layer = model.model.layers[idx]
            out = layer(
                hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=session['cache'],
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = out[0]
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

    return hidden


# === WebSocket Handler (primary, low-latency) ===

async def ws_handler(websocket):
    """Handle a WebSocket connection for one generation session.

    Protocol:
      1. Server sends text JSON: {"session_id": "..."}
      2. Client sends binary frames: [4B header_len][JSON header][tensor bytes]
         Header: {"hidden_shape", "pe_shape", "is_prompt", "crop_to"}
         Tensor bytes: hidden_bytes + cos_bytes + sin_bytes
      3. Server responds binary: [4B header_len][JSON header][tensor bytes]
         Header: {"hidden_shape", "process_time_ms", "cached_seq_len"}
      4. Client sends text JSON: {"type": "end"} to close
    """
    cleanup_old_sessions()
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'cache': None,
        'last_access': time.time(),
        'seq_len': 0
    }

    # Send session ID
    await websocket.send(json.dumps({"session_id": session_id}))
    print(f"[WS] New session {session_id[:8]}...")

    try:
        async for message in websocket:
            # Text message = control command
            if isinstance(message, str):
                data = json.loads(message)
                if data.get('type') == 'end':
                    break
                continue

            # Binary message = process request
            start = time.time()

            session = sessions[session_id]
            session['last_access'] = time.time()

            # Unpack: [4B header_len][JSON header][tensor data]
            header_len = struct.unpack('>I', message[:4])[0]
            header = json.loads(message[4:4 + header_len])
            tensor_data = message[4 + header_len:]

            hs_shape = header['hidden_shape']
            pe_shape = header['pe_shape']
            is_prompt = header.get('is_prompt', True)
            crop_to = header.get('crop_to')
            relocate = header.get('relocate')
            has_mask = header.get('has_mask', False)
            mask_shape = header.get('mask_shape')

            # Reconstruct hidden states
            hs_size = 1
            for d in hs_shape:
                hs_size *= d
            hs_bytes = hs_size * 2  # float16

            hidden = torch.frombuffer(
                bytearray(tensor_data[:hs_bytes]), dtype=torch.float16
            ).reshape(hs_shape).clone().cuda()
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

            # Reconstruct position embeddings
            pe_size = 1
            for d in pe_shape:
                pe_size *= d
            pe_byte_count = pe_size * 2

            pe_end = hs_bytes + pe_byte_count * 2  # cos + sin
            cos = torch.frombuffer(
                bytearray(tensor_data[hs_bytes:hs_bytes + pe_byte_count]),
                dtype=torch.float16
            ).reshape(pe_shape).clone().cuda()
            sin = torch.frombuffer(
                bytearray(tensor_data[hs_bytes + pe_byte_count:hs_bytes + pe_byte_count * 2]),
                dtype=torch.float16
            ).reshape(pe_shape).clone().cuda()
            position_embeddings = (cos, sin)

            # Reconstruct custom attention mask if provided
            custom_mask = None
            if has_mask and mask_shape:
                mask_data = tensor_data[pe_end:]
                custom_mask = torch.frombuffer(
                    bytearray(mask_data), dtype=torch.float16
                ).reshape(mask_shape).clone().cuda()

            seq_len = hidden.shape[1]

            # Process through layers
            hidden = process_tensors(
                session, hidden, position_embeddings, seq_len, is_prompt,
                crop_to, relocate=relocate, custom_mask=custom_mask
            )

            # Pack response: [4B header_len][JSON header][tensor bytes]
            out_np = hidden.cpu().numpy()
            out_bytes = out_np.tobytes()

            resp_header = json.dumps({
                "hidden_shape": list(out_np.shape),
                "process_time_ms": round((time.time() - start) * 1000, 2),
                "cached_seq_len": session['seq_len']
            }).encode()

            resp = struct.pack('>I', len(resp_header)) + resp_header + out_bytes
            await websocket.send(resp)

    except websockets.exceptions.ConnectionClosed:
        print(f"[WS] Connection closed for session {session_id[:8]}")
    except Exception as e:
        print(f"[WS] Error in session {session_id[:8]}: {e}")
        traceback.print_exc()
    finally:
        if session_id in sessions:
            del sessions[session_id]
        print(f"[WS] Session {session_id[:8]} ended")


async def start_ws_server(port=5001):
    async with websockets.serve(
        ws_handler, "0.0.0.0", port,
        max_size=20 * 1024 * 1024,  # 20MB max message
        ping_interval=30,
        ping_timeout=60,
    ):
        print(f"WebSocket server running on port {port}")
        await asyncio.Future()  # run forever


# === HTTP/Flask Handlers (health checks, backward compat) ===

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0),
        "memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "layers": f"{CLOUD_START}-{CLOUD_END}",
        "active_sessions": len(sessions),
        "kv_cache": True,
        "serialization": "websocket+binary",
        "jacobi_support": True
    })


@app.route('/new_session', methods=['POST'])
def new_session():
    """Create a new session with fresh KV cache (HTTP fallback)."""
    cleanup_old_sessions()
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'cache': None,
        'last_access': time.time(),
        'seq_len': 0
    }
    return jsonify({"session_id": session_id})


@app.route('/process', methods=['POST'])
def process():
    """Process hidden states through cloud layers (HTTP fallback)."""
    try:
        start = time.time()
        data = request.json

        session_id = data.get('session_id')
        is_prompt = data.get('is_prompt', True)
        crop_to = data.get('crop_to')

        if session_id not in sessions:
            return jsonify({"error": "Invalid session_id. Call /new_session first."}), 400

        session = sessions[session_id]
        session['last_access'] = time.time()

        # Decode hidden states
        hs_shape = data['hidden_shape']
        hidden = torch.frombuffer(
            base64.b64decode(data['hidden_states']),
            dtype=torch.float16
        ).reshape(hs_shape).clone().cuda()
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        # Decode position embeddings
        pe_shape = data['pe_shape']
        pe_raw = base64.b64decode(data['position_embeddings'])
        pe_size = 1
        for d in pe_shape:
            pe_size *= d
        pe_bytes = pe_size * 2
        cos = torch.frombuffer(
            bytearray(pe_raw[:pe_bytes]), dtype=torch.float16
        ).reshape(pe_shape).clone().cuda()
        sin = torch.frombuffer(
            bytearray(pe_raw[pe_bytes:]), dtype=torch.float16
        ).reshape(pe_shape).clone().cuda()
        position_embeddings = (cos, sin)

        seq_len = hidden.shape[1]

        hidden = process_tensors(
            session, hidden, position_embeddings, seq_len, is_prompt, crop_to
        )

        out_tensor = hidden.cpu()
        out_bytes = base64.b64encode(out_tensor.numpy().tobytes()).decode()

        return jsonify({
            "hidden_states": out_bytes,
            "hidden_shape": list(out_tensor.shape),
            "process_time_ms": round((time.time() - start) * 1000, 2),
            "cached_seq_len": session['seq_len']
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/end_session', methods=['POST'])
def end_session():
    """Clean up a session (HTTP fallback)."""
    data = request.json
    session_id = data.get('session_id')
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({"status": "session ended"})
    return jsonify({"status": "session not found"})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Split Inference Cloud Server')
    parser.add_argument('--model', default="/home/ubuntu/models/mistral-7b-instruct",
                        help='Path to model')
    parser.add_argument('--cloud-start', type=int, default=None,
                        help='First cloud layer (default: 2)')
    parser.add_argument('--cloud-end', type=int, default=None,
                        help='Last cloud layer (default: num_layers - 3)')
    parser.add_argument('--http-port', type=int, default=5000,
                        help='HTTP health check port')
    parser.add_argument('--ws-port', type=int, default=5001,
                        help='WebSocket port')
    args = parser.parse_args()

    load_model(args.model, cloud_start=args.cloud_start, cloud_end=args.cloud_end)

    # Start WebSocket server in background thread
    def run_ws():
        asyncio.run(start_ws_server(port=args.ws_port))

    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()

    # Start Flask HTTP server (main thread)
    print(f"Starting HTTP health server on port {args.http_port}...")
    app.run(host='0.0.0.0', port=args.http_port, threaded=True)
