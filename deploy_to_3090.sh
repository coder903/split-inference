#!/bin/bash
# Deploy updated split-inference code to the 3090 machine
# Run from the Mac (this repo directory)
#
# Usage: bash deploy_to_3090.sh

set -e

REMOTE="mike@192.168.1.32"
REMOTE_DIR="/home/mike/D/coding/python_3/my_projects/split-inference-3090"
CLOUD_IP="69.19.137.3"

echo "=== Deploying to 3090 ==="

# Copy updated scripts
echo "--- Copying scripts ---"
scp jacobi_server.py ${REMOTE}:${REMOTE_DIR}/
scp lookahead_ablation.py ${REMOTE}:${REMOTE_DIR}/
scp perplexity_comparison.py ${REMOTE}:${REMOTE_DIR}/
scp rtt_analysis.py ${REMOTE}:${REMOTE_DIR}/
echo "Scripts copied."

# Check if NeMo model exists on 3090
echo ""
echo "--- Checking models on 3090 ---"
ssh ${REMOTE} "ls /home/mike/D/models/mistral-nemo-12b-instruct/config.json 2>/dev/null && echo 'NeMo 12B: EXISTS' || echo 'NeMo 12B: NEEDS DOWNLOAD'"
ssh ${REMOTE} "ls /home/mike/D/models/mistral-7b-instruct/config.json 2>/dev/null && echo 'Mistral 7B: EXISTS' || echo 'Mistral 7B: NEEDS DOWNLOAD'"

# Set up SSH tunnel to HyperStack A100
echo ""
echo "--- Setting up SSH tunnel ---"
ssh ${REMOTE} "pkill -f 'ssh.*5001.*${CLOUD_IP}' 2>/dev/null; sleep 1; echo 'old tunnels cleaned'"
ssh ${REMOTE} "ssh -fN -L 5000:localhost:5000 -L 5001:localhost:5001 -o StrictHostKeyChecking=accept-new ubuntu@${CLOUD_IP} && echo 'Tunnel established'"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "To run experiments on 3090, SSH in and:"
echo ""
echo "  # Mistral 7B (validation):"
echo "  cd ${REMOTE_DIR}"
echo "  python jacobi_server.py --model /home/mike/D/models/mistral-7b-instruct --cloud ws://localhost:5001 --mode sequential"
echo ""
echo "  # NeMo 12B:"
echo "  python jacobi_server.py --model /home/mike/D/models/mistral-nemo-12b-instruct --cloud ws://localhost:5001 --mode sequential --split-after 1 --resume-at 38"
echo ""
echo "  # RTT Analysis:"
echo "  python rtt_analysis.py --model /home/mike/D/models/mistral-7b-instruct --cloud http://localhost:5000"
echo ""
echo "  # Perplexity Comparison:"
echo "  python perplexity_comparison.py --model /home/mike/D/models/mistral-7b-instruct --cloud http://localhost:5000"
echo ""
echo "  # Ablation (fixed):"
echo "  python lookahead_ablation.py --model /home/mike/D/models/mistral-7b-instruct --cloud http://localhost:5000"
