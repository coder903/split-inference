#!/usr/bin/env python3
"""
Split Inference - Mac Client (Streaming)
Lightweight client that connects to RTX 3090 for privacy-preserving LLM inference.
All processing happens on 3090 + cloud - Mac just sends/receives text.
Tokens stream back in real-time as they're generated.
"""

import requests
import json
import sys

# RTX 3090 server on local network
SERVER_URL = "http://YOUR_LOCAL_GPU_IP:5001"


def check_server():
    """Check if 3090 server is running."""
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        if r.status_code == 200:
            info = r.json()
            print(f"Connected to: {info['gpu']}")
            print(f"Local layers: {info['local_layers']}")
            print(f"Cloud relay: {info['cloud_url']}")
            streaming = info.get('streaming', False)
            print(f"Streaming: {'enabled' if streaming else 'disabled'}")
            return True, streaming
    except Exception as e:
        print(f"Cannot reach 3090 server: {e}")
    return False, False


def generate_stream(prompt: str, max_new_tokens: int = 500):
    """Stream tokens from 3090 as they're generated."""
    response = requests.post(
        f"{SERVER_URL}/generate_stream",
        json={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens
        },
        stream=True,
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"Server error: {response.text}")

    stats = None
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if 'token' in data:
                    print(data['token'], end='', flush=True)
                elif 'done' in data:
                    stats = data
                elif 'error' in data:
                    raise Exception(data['error'])

    return stats


def generate(prompt: str, max_new_tokens: int = 500) -> dict:
    """Send prompt to 3090 and get response (non-streaming fallback)."""
    response = requests.post(
        f"{SERVER_URL}/generate",
        json={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens
        },
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"Server error: {response.text}")

    return response.json()


def main():
    print("=" * 60)
    print("Split Inference - Mac Client (Streaming)")
    print("Connected to RTX 3090 -> Cloud A100 -> RTX 3090")
    print("Your text never leaves the local network.")
    print("=" * 60)

    ok, streaming = check_server()
    if not ok:
        print("\nMake sure the local GPU server is running:")
        print("  python jacobi_server.py --model /path/to/model --cloud ws://CLOUD_IP:5001 --mode lookahead")
        return

    print("\nType your prompts below. 'quit' to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            print("\nAssistant: ", end="", flush=True)

            if streaming:
                stats = generate_stream(prompt)
                if stats:
                    print(f"\n\n[{stats['tokens_generated']} tokens, {stats['total_time_ms']:.0f}ms, {stats['tokens_per_second']:.1f} tok/s]")
                    if 'timing' in stats:
                        t = stats['timing']
                        print(f"  embed: {t['embed_avg_ms']:.1f}ms | early: {t['early_layers_avg_ms']:.1f}ms | cloud: {t['cloud_avg_ms']:.1f}ms | late: {t['late_layers_avg_ms']:.1f}ms | lm_head: {t['lm_head_avg_ms']:.1f}ms")
                    print()
            else:
                print("(waiting...)", end="", flush=True)
                result = generate(prompt)
                print(f"\r{result['response']}")
                print(f"\n[{result['tokens_generated']} tokens, {result['total_time_ms']:.0f}ms, {result['tokens_per_second']:.1f} tok/s]\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
