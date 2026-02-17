#!/bin/bash
# Setup script for HyperStack A100 cloud server
# Run this after SSHing into the new server
#
# Usage: bash setup_cloud_server.sh

set -e

echo "=== Split Inference Cloud Server Setup ==="
echo "Setting up HyperStack A100 for LLaMA 2 13B..."

# System packages
echo ""
echo "--- Installing system packages ---"
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Create project directory and venv
echo ""
echo "--- Creating project environment ---"
mkdir -p ~/split-inference
cd ~/split-inference
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo ""
echo "--- Installing Python packages ---"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate websockets flask huggingface_hub

# Download LLaMA 2 13B Chat
echo ""
echo "--- Downloading LLaMA 2 13B Chat ---"
echo "Note: You may need to run 'huggingface-cli login' first if gated"
mkdir -p ~/models
huggingface-cli download meta-llama/Llama-2-13b-chat-hf \
    --local-dir ~/models/llama-2-13b-chat \
    --local-dir-use-symlinks False

# Also download Mistral 7B for comparison runs
echo ""
echo "--- Downloading Mistral 7B Instruct ---"
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
    --local-dir ~/models/mistral-7b-instruct \
    --local-dir-use-symlinks False

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start the cloud server for LLaMA 2 13B:"
echo "  cd ~/split-inference && source venv/bin/activate"
echo "  python cloud_server_kv.py --model ~/models/llama-2-13b-chat --cloud-start 2 --cloud-end 37"
echo ""
echo "To start for Mistral 7B (comparison):"
echo "  python cloud_server_kv.py --model ~/models/mistral-7b-instruct"
echo ""
echo "SSH tunnel from 3090 machine:"
echo "  ssh -L 5001:localhost:5001 -L 5000:localhost:5000 <user>@<this-server-ip>"
