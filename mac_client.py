#!/usr/bin/env python3
"""
Split Inference - Mac Client
Lightweight client that connects to RTX 3090 for privacy-preserving LLM inference.
All processing happens on 3090 + cloud - Mac just sends/receives text.
"""

import requests
import sys

# RTX 3090 server on local network
SERVER_URL = "http://192.168.1.32:5001"


def check_server():
    """Check if 3090 server is running."""
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        if r.status_code == 200:
            info = r.json()
            print(f"Connected to: {info['gpu']}")
            print(f"Local layers: {info['local_layers']}")
            print(f"Cloud relay: {info['cloud_url']}")
            return True
    except Exception as e:
        print(f"Cannot reach 3090 server: {e}")
    return False


def generate(prompt: str, max_new_tokens: int = 100) -> dict:
    """Send prompt to 3090 and get response."""
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
    print("Split Inference - Mac Client")
    print("Connected to RTX 3090 -> Cloud A100 -> RTX 3090")
    print("Your text never leaves the local network.")
    print("=" * 60)

    if not check_server():
        print("\nMake sure the 3090 server is running:")
        print("  ssh mike@192.168.1.32")
        print("  cd /home/mike/D/coding/python_3/my_projects/split-inference-3090")
        print("  source venv/bin/activate")
        print("  python local_server.py")
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

            print("Generating...", end="", flush=True)
            result = generate(prompt)
            print(f" done ({result['total_time_ms']:.0f}ms, {result['tokens_generated']} tokens, {result['tokens_per_second']:.1f} tok/s)")
            print(f"\nAssistant: {result['response']}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
