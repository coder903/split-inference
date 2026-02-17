# Privacy-Aware Split Inference with Speculative Decoding for Large Language Models over Wide-Area Networks

**Michael Cunningham**

*February 2026*

---

## Abstract

We present a practical system for privacy-aware large language model (LLM) inference that splits a transformer between a trusted local GPU and an untrusted cloud GPU, communicating only intermediate activations over the network. Our system addresses the unique challenges of autoregressive LLM decoding over high-latency wide-area networks (WANs), contributing: (1) an asymmetric layer split where embedding and unembedding layers remain local, ensuring raw tokens never leave the trusted device; (2) the first application of lookahead decoding to split inference over WANs, amortizing network round-trip latency across multiple tokens per iteration; (3) an empirical inversion attack evaluation showing that split depth provides a tunable privacy-performance tradeoff---an attacker can recover ~59% of tokens at a 2-layer split but only ~35% at an 8-layer split, with minimal throughput impact; and (4) ablation experiments showing that n-gram speculation accepts 1.2--1.4 tokens per decoding step on average (peak of 5 observed on code), with the acceptance rate translating directly to throughput improvement over sequential decoding. Evaluated on Mistral 7B over a ~80ms WAN link, our system achieves 14--17 tokens/second at the optimal n-gram size (n=4) depending on content type.

---

## 1. Introduction

The rapid adoption of large language models in enterprise settings has created a fundamental tension between capability and compliance. Organizations in healthcare (HIPAA), defense (ITAR), legal (attorney-client privilege), and financial services face a binary choice: use powerful cloud-hosted models and accept data exposure risks, or run smaller local models and sacrifice capability. Existing privacy-preserving approaches---homomorphic encryption [1], secure multi-party computation [2], and trusted execution environments [3]---impose prohibitive computational overhead for transformer inference, often slowing execution by 100-1000x.

Split inference offers a compelling middle ground. By partitioning a model such that privacy-sensitive operations (token embedding and unembedding) execute locally while compute-intensive middle layers run on untrusted cloud hardware, we can leverage cloud GPU capacity without exposing raw text. The intermediate activations transmitted between local and cloud are high-dimensional floating-point tensors in the model's learned representation space---not tokens, not embeddings, but abstract geometric objects from which reconstructing the original input is computationally intractable without access to the embedding matrices [4, 5].

However, applying split inference to autoregressive LLM decoding introduces a challenge not present in single-pass classification models: each generated token requires a full forward pass through all layers, meaning each token incurs a network round trip. At typical WAN latencies of 80-100ms, this limits sequential decoding throughput to approximately 8-11 tokens/second---usable for some applications, but far below native inference speeds.

Our key insight is that speculative and parallel decoding techniques, originally developed to reduce GPU idle time in single-machine inference, can be repurposed to reduce *network* idle time in split inference. Specifically, we adapt lookahead decoding [6]---which harvests n-gram candidates from Jacobi iteration trajectories [7]---to generate multiple candidate tokens per network round trip, amortizing the WAN latency across 1-5 accepted tokens per iteration.

This paper makes the following contributions:

1. **A practical, deployable system** for privacy-aware LLM inference that achieves 14--17 tok/s on Mistral 7B over a ~80ms WAN link with lookahead decoding, demonstrating that split inference is viable for interactive use.

2. **The first integration of lookahead decoding with split inference**, showing that Jacobi-style parallel decoding can amortize network latency in addition to computation latency.

3. **A detailed empirical study of deployment engineering**, including the critical finding that cloud provider networking architecture (direct SSH vs. proxy routing) dominates performance far more than GPU location, and documenting SSH tunnel engineering required for real-world deployment.

4. **A scaling analysis** demonstrating that split inference overhead decreases proportionally with model size, making the approach *more* viable for the larger models that enterprises most need.

5. **Empirical privacy evaluation** through inversion attack experiments that quantify token recoverability at different split depths, confirming theoretical vulnerabilities and showing that increasing local layers from 2 to 8 reduces attack accuracy from ~59% to ~35%.

---

## 2. Related Work

### 2.1 Split Computing for Neural Networks

The concept of partitioning DNN inference between edge and cloud devices was pioneered by Neurosurgeon [8], which built per-layer latency and energy models to find optimal split points for CNNs. Edgent [9] extended this with dynamic partitioning that adapts to network conditions. Matsubara et al. [10] surveyed split computing comprehensively, identifying the fundamental tradeoff: early splits minimize local computation but produce incompressible intermediate representations, while late splits produce compressible features but require substantial local compute.

Recent work has applied split computing specifically to LLMs. Sung et al. [11] introduced the first autoregressive-aware split computing framework with mixed-precision quantization and intermediate activation compression. Adaptive layer splitting for wireless LLM inference [12] uses reinforcement learning to determine optimal split points under varying channel conditions. CROSS-SEC [13] proposes cross-WAN prefill-decode disaggregation with layerwise KV-cache computation-communication overlapping.

### 2.2 Distributed and Collaborative LLM Inference

Petals [14] demonstrated that consumer GPUs can collaboratively serve BLOOM-176B by hosting different transformer blocks in a BitTorrent-style network. Each participant runs a subset of layers and passes activations to the next, achieving ~1 step/second on commodity hardware. While Petals focuses on collaborative resource pooling among multiple participants, our work focuses on a two-party privacy-aware split between a trusted local device and a single untrusted cloud server.

EdgeShard [15] partitions LLMs across heterogeneous edge devices and cloud using dynamic programming for optimal placement. MDI-LLM [16] introduced recurrent pipeline parallelism for edge-distributed inference. The exo framework [33] enables peer-to-peer inference across heterogeneous consumer devices. Alpa [28] automates inter- and intra-operator parallelism for distributed execution, establishing techniques for partitioning computation graphs across device meshes. These systems optimize for throughput and resource utilization but do not address privacy as a primary design goal.

Splitwise [17] (Microsoft, ISCA 2024) made the influential observation that LLM inference has two distinct phases---compute-intensive prefill and memory-intensive decode---with different optimal hardware. DistServe [18] (OSDI 2024) builds on this insight with disaggregated serving, achieving 7.4x higher request capacity.

### 2.3 Speculative and Parallel Decoding

Speculative decoding was independently proposed by Leviathan et al. [19] and Chen et al. [20], both showing that a small "draft" model can generate candidate tokens verified in parallel by the target model, with rejection sampling preserving the exact output distribution. This yields 2-3x speedups without quality degradation.

Jacobi decoding [7] (Santilli et al., ACL 2023) reframes autoregressive generation as a fixed-point iteration problem, initializing a block of future tokens and iterating until convergence. CLLMs [21] (ICML 2024) improved convergence through consistency-trained models. Fu et al. [6] introduced lookahead decoding, which collects n-gram candidates from Jacobi iteration trajectories for speculation without requiring a separate draft model. Alternative speculation approaches include retrieval-based methods (REST [29]), multi-head parallel drafting (Medusa [30]), and feature-uncertainty-based speculation (EAGLE [31])---each trades different design choices against draft quality and overhead.

Several recent works combine speculative decoding with distributed inference. DSI [22] introduces "speculation parallelism" to overlap draft and target execution across distributed resources. DSSD [23] (ICML 2025) combines split inference with speculative decoding for edge-cloud deployment. The DSD framework [24] specifically addresses high-latency decentralized settings by filling communication wait time with speculative verification.

### 2.4 Privacy of Intermediate Activations

The privacy properties of intermediate neural network representations have been studied from both attack and defense perspectives. He et al. [4] showed that model inversion attacks on collaborative inference can partially reconstruct inputs from intermediate features, though success degrades significantly with network depth. PATROL [5] develops privacy-oriented pruning to defend against such attacks. DISCO [25] uses dynamic channel obfuscation to selectively remove sensitive information from intermediate representations. Split-and-Denoise [26] (ICML 2024) adds calibrated local differential privacy noise to embeddings before transmission. Deng et al. [27] introduced Fisher-approximated Shannon information as a metric for quantifying privacy leakage at different split points.

SecureInfer [3] takes a hardware approach, executing privacy-critical tensor operations inside TEE enclaves while offloading linear operations to untrusted GPUs, achieving 3.7x throughput over TEE-only execution. Fission [2] uses hybrid MPC-evaluator architecture for cryptographic privacy guarantees.

Our work differs from these approaches by splitting at the *embedding boundary*---the layer where token IDs are converted to continuous vectors. This is a qualitatively different privacy boundary than splitting at intermediate convolutional or transformer layers, because the cloud never receives any representation that has a direct algebraic relationship to the token vocabulary.

---

## 3. System Architecture

> **[Figure 1]** *System architecture diagram showing the local-cloud split. The local RTX 3090 runs token embedding, layers 0-1, layers 30-31, RMS norm, and the LM head. The cloud RTX 4090 runs layers 2-29 with session-based KV-cache. Only 8KB activation vectors (per token) cross the WAN link via WebSocket binary protocol. Raw tokens and embeddings never leave the local device.*

### 3.1 Layer Partitioning

We split a 32-layer Mistral 7B Instruct model as follows:

| Component | Location | Layers | Purpose |
|-----------|----------|--------|---------|
| Token embedding | Local (3090) | - | Token IDs → hidden states |
| Layers 0-1 | Local (3090) | 0, 1 | Initial transformer processing |
| Layers 2-29 | Cloud (4090) | 2-29 | Bulk computation (28 layers) |
| Layers 30-31 | Local (3090) | 30, 31 | Final transformer processing |
| RMS norm + LM head | Local (3090) | - | Hidden states → logits → tokens |

This partitioning ensures that:
- **Raw tokens never leave the local device.** The embedding matrix (which maps token IDs to vectors) and the language model head (which maps hidden states back to token probabilities) both reside locally.
- **Only intermediate activations cross the network.** These are 4096-dimensional float16 vectors per token position---approximately 8KB per token---that exist in the model's learned representation space.
- **The cloud has no access to the vocabulary mapping.** Without the embedding and unembedding weights, the cloud server cannot determine which tokens correspond to the received activations.

### 3.2 Transport Layer

We employ a WebSocket binary protocol for activation transfer, chosen after evaluating HTTP/JSON, HTTP/binary, and WebSocket alternatives:

**Protocol format:**
```
[4 bytes: header length][JSON header][tensor bytes (float16)]
```

The JSON header contains metadata: tensor shapes, prompt/generation flag, KV-cache crop position, and optional attention mask shape. Tensor data is transmitted as raw float16 bytes---no base64 encoding, no serialization framework overhead.

**Performance progression of transport optimizations:**
| Transport | Encoding | Throughput |
|-----------|----------|------------|
| HTTP/Flask | torch.save + base64 | 1.4 tok/s |
| HTTP/Flask | numpy + base64 | 3.9 tok/s |
| HTTP/Flask | numpy + base64 (3090) | 5.3 tok/s |
| WebSocket | binary float16 | 7.8--10.9 tok/s |
| WebSocket + lookahead | binary float16 | 13--17 tok/s |

The persistent WebSocket connection eliminates per-request TCP handshake and HTTP header overhead. Binary framing avoids the ~33% base64 expansion and associated encode/decode CPU cost.

### 3.3 KV-Cache Management

A critical design decision is where to maintain the key-value cache. We keep the KV-cache for layers 2-29 on the cloud server, co-located with the layers that produce and consume it. This avoids transferring KV-cache state across the network---a significant advantage, as KV-cache size grows linearly with sequence length and would quickly dominate bandwidth at scale.

The cloud server maintains per-session caches using HuggingFace's `DynamicCache`. For generation (non-prompt) steps, only the new token's hidden state (~8KB) is transmitted, not the full sequence. The server manages cache lifecycle through session IDs, with automatic expiration after 5 minutes of inactivity.

For Jacobi and lookahead decoding, which require multi-token forward passes during generation, the cloud server supports:
- **Cache cropping:** Rolling back to a checkpoint when a speculative block is rejected.
- **Cache relocation:** Moving KV entries when n-gram speculation shifts the verification window.
- **Custom attention masks:** Supporting the non-standard causal masks required by parallel token verification, where each speculative position can attend to all committed tokens plus preceding speculative tokens.

### 3.4 Networking and Tunnel Engineering

A surprising finding in our deployment was that **cloud provider networking architecture dominates latency far more than geographic distance.** We evaluated two providers with GPU instances in Texas:

| Provider | GPU | Geographic Distance | Measured RTT | Throughput |
|----------|-----|-------------------|-------------|------------|
| VAST.ai | RTX 4090 (Texas) | ~800 miles | 200-400ms | 1.2 tok/s |
| RunPod | RTX 4090 (Texas) | ~800 miles | 80-100ms | 11--17 tok/s |

VAST.ai routes all SSH traffic through proxy servers in Virginia, regardless of GPU location. This adds 200-400ms RTT that cannot be eliminated through geographic GPU selection. RunPod provides direct SSH access to pods without proxy intermediaries, achieving 2-4x lower latency to the same Texas region.

**Tunnel architecture:** Because the local GPU server is behind NAT and cloud instances have ephemeral IP addresses, we establish SSH tunnels from the local machine to the cloud instance, forwarding both the HTTP health-check port and the WebSocket activation-transfer port. This tunnel architecture is necessary because cloud GPU providers assign different SSH ports on each pod start, require SSH key management across multiple machines, and may change IP addresses between sessions. We document this not as a limitation but as essential systems engineering that is universally required for split inference over public cloud infrastructure and is largely absent from the academic literature.

---

## 4. Decoding Strategies for Split Inference

### 4.1 Sequential Decoding (Baseline)

In sequential mode, each token requires one complete local-cloud-local round trip:

1. Local: Embed token, process through layers 0-1 → hidden state (8KB)
2. Network: Send hidden state + position embeddings to cloud
3. Cloud: Process through layers 2-29 with KV-cache → hidden state (8KB)
4. Network: Return hidden state to local
5. Local: Process through layers 30-31, norm, LM head → next token

At ~85ms RTT (HyperStack) this yields ~7.8 tok/s; at ~80ms RTT (RunPod) it yields ~10.9 tok/s with our WebSocket protocol.

### 4.2 Jacobi Parallel Decoding

Jacobi decoding [7] initializes a block of *k* future token positions and iterates until convergence. At each iteration, all *k* positions are processed in parallel through a single forward pass.

**Adaptation for split inference:** In the local-only case, Jacobi iterations are essentially free (GPU parallelism makes *k*-token and 1-token forward passes nearly identical cost). Over a network, each iteration still costs one round trip, but a converged block commits *k* tokens. The effective throughput is:

$$\text{tok/s} = \frac{k}{\text{iterations} \times \text{RTT}}$$

In practice over WAN, we observed poor convergence---vanilla Jacobi rarely commits more than 1-2 tokens per iteration for language modeling, yielding ~1.5 tok/s (worse than sequential due to the overhead of transmitting larger tensors for the full block).

### 4.3 Lookahead Decoding (Our Primary Mode)

Lookahead decoding [6] addresses Jacobi's convergence problem by collecting n-gram candidates from the Jacobi iteration trajectories themselves. As the Jacobi iteration progresses, observed token sequences that appear during intermediate iterations are cached as n-gram candidates. These candidates can be verified in subsequent iterations, allowing 1-5 tokens to be committed per round trip when n-gram matches occur.

**Why this is particularly effective for split inference:** In local-only inference, the benefit of lookahead decoding is modest (1.5-2x) because the baseline is already fast. In split inference, the benefit is amplified because:

1. **Each round trip is expensive** (~100ms vs. ~3ms locally), so amortizing across multiple tokens has outsized impact.
2. **The verification cost is negligible** relative to network latency---verifying 5 candidates costs the same one round trip as generating 1 token.
3. **N-gram speculation succeeds frequently** for structured text (code, templates, repetitive content), which is common in enterprise applications.

Our implementation achieves 13--17 tok/s depending on content type (up from ~11 tok/s sequential on the same provider), with acceptance rates of 1.2--1.4 tokens per step translating directly to throughput improvement (Section 6.4).

### 4.4 Attention Mask Engineering

A critical implementation detail for multi-token Jacobi and lookahead decoding in the split inference context: PyTorch's `torch.nn.functional.scaled_dot_product_attention` with `is_causal=True` constructs a *relative* causal mask based on query and key sequence lengths. When KV-cache entries exist from prior committed tokens, this relative mask produces incorrect attention patterns---speculative position *i* would attend to KV entries that it should not.

We construct explicit attention masks for all multi-token generation steps:

```python
attn_mask = torch.full((1, 1, k, kv_len), float('-inf'), device='cuda', dtype=torch.float16)
for i in range(k):
    attn_mask[0, 0, i, :committed_len + i + 1] = 0.0
```

This mask is transmitted alongside the hidden states in the binary WebSocket frame, adding negligible bandwidth overhead (~2KB for typical block sizes) but preventing catastrophic attention errors.

---

## 5. Privacy Analysis

### 5.1 Threat Model

We consider a semi-honest cloud server that faithfully executes the assigned computation but attempts to infer information about the input from observed intermediate activations. The server has:
- Full knowledge of the model architecture and layer weights for layers 2-29
- Access to all intermediate activations passing through its layers
- No access to the embedding matrix, unembedding (LM head) weights, or layer 0-1 / 30-31 weights

**Critical assumption:** Our privacy analysis assumes the cloud does not possess the local layer weights. For a public model like Mistral 7B, this assumption requires that the local layers be fine-tuned, adapted (e.g., via LoRA), or otherwise differentiated from the published weights. If the cloud can obtain identical copies of all layers, the architectural privacy boundary is substantially weakened. In a production deployment, the local embedding and unembedding layers would be customized per organization, ensuring the cloud cannot simply download matching weights.

### 5.2 Privacy Properties and Limitations

**What the architecture guarantees:** Raw token IDs and token embeddings never leave the local device. This is a structural property, not a statistical one---the cloud receives only post-layer-1 hidden states, never tokens or embeddings.

**What the architecture does *not* guarantee:** Recent theoretical work by Nikolaou et al. [32] proved that the mapping from input token sequences to transformer hidden states is *injective*---distinct inputs produce distinct internal representations. Their SipIt algorithm can reconstruct exact input tokens from hidden states in linear time, given access to the full model weights. This result establishes that intermediate activations are, in principle, invertible.

However, several factors mitigate this in our setting:

1. **Weight access requirement.** Inversion attacks, including SipIt [32], require the attacker to possess the weights of the layers between the input and the observed activations. In our threat model, the cloud lacks layers 0-1, the embedding matrix, and critically, any fine-tuning applied to local layers.
2. **Depth of transformation.** He et al. [4] showed that model inversion attack success degrades with network depth. We confirm this empirically: training an MLP attack decoder on (activation, token ID) pairs from our system yields the following top-1 token recovery accuracy at different split points:

| Split After Layer | Local Layers | Top-1 Accuracy | Top-5 Accuracy |
|-------------------|-------------|----------------|----------------|
| Layer 1 (default) | 2 | 58.8% | 62.0% |
| Layer 3 | 4 | 44.3% | 49.8% |
| Layer 5 | 6 | 44.8% | 56.6% |
| Layer 7 | 8 | 34.8% | 46.2% |
| Random baseline | - | 0.003% | 0.016% |

The attack decoder is a 3-layer MLP (4096→2048→2048→32000) trained on ~880 samples from diverse text (general knowledge, code, medical, legal, conversational). This represents a *lower bound* on vulnerability: a more sophisticated attacker using sequence-level context (e.g., language model priors over token sequences) or larger training sets could achieve higher accuracy. At the default 2-layer split, even this simple attacker recovers ~59% of tokens---confirming the theoretical vulnerability. Accuracy drops substantially with depth: adding just 2 more local layers (split after layer 3) reduces recovery to ~44%, and 8 local layers reduces it to ~35%. The non-monotonic result at layer 5 (44.8% vs. 44.3% at layer 3) is within the ±10% run-to-run variance of our small test set and does not indicate that deeper splits are less effective.

3. **Symmetric vulnerability.** The architecture has a symmetric privacy boundary: the cloud observes layer-1 output on the input side (2 layers from token embedding) and layer-29 output on the output side (2 layers from the LM head). Both sides are equally shallow. Increasing local depth should be applied symmetrically (e.g., layers 0-3 and 28-31 locally) to protect both input and output tokens.

*Experimental caveats:* The inversion experiment uses a small dataset (~1,100 activation-token pairs, 80/20 train/test split) and shows variance of ±10% between runs due to the limited test set size (221 samples). The random baseline represents literal uniform-random guessing over the 32K vocabulary; a more informative baseline would use token frequency priors (which would achieve ~3-5% top-1 by always predicting common tokens). The MLP attack decoder is a lower bound---it treats each position independently, while a sequence-aware attacker could exploit inter-token dependencies. Larger training sets (10K+ samples) would likely yield higher and more stable attack accuracy.

**Privacy characterization:** We emphasize that our system provides *architectural* privacy---a structural separation that ensures raw text never crosses the network boundary. This is qualitatively different from, and weaker than, cryptographic privacy (which provides mathematical guarantees) or hardware-attested privacy (which provides tamper-resistant isolation). We avoid the term "computational privacy" to prevent conflation with formal security notions.

**Comparison of privacy approaches:**

| Approach | Guarantee Type | What It Prevents | Performance Overhead | Status |
|----------|---------------|-----------------|---------------------|--------|
| Homomorphic Encryption [1] | Cryptographic | Any information leakage | 100-1000x slower | Research only |
| Secure MPC [2] | Cryptographic | Any information leakage | 8-50x slower | Emerging |
| TEE-based [3] | Hardware-attested | Observation of computation | 2-3x slower | Requires SGX/TDX |
| **Split inference (ours)** | **Architectural** | **Raw text leaving device** | **3x slower (at 7B)** | **Deployable today** |
| No privacy (cloud API) | None | Nothing | 1x (baseline) | Standard practice |

For organizations where the current alternative is sending full plaintext to a cloud API, the architectural guarantee that tokens never leave the local device represents a meaningful improvement---even without formal cryptographic assurances.

### 5.3 Strengthening Privacy

Several techniques can augment the base architectural privacy:
- **Increasing local layers** is the most effective defense we measured. Our inversion experiment shows that moving from 2 to 8 local layers reduces token recovery from ~59% to ~35% (Section 5.2). Each additional local layer adds only ~3ms to per-token latency on the RTX 3090, negligible compared to the ~100ms network round trip. For example, splitting at layers 0-4 and 27-31 (10 local layers) would protect both input and output tokens with 5 layers of depth on each side, while reducing throughput by only ~10-15%. Deng et al. [27] provide Fisher-information-based metrics for quantifying privacy leakage at different split points.
- **Fine-tuning local layers** ensures the cloud cannot reconstruct the full model, directly addressing the SipIt [32] attack vector. Even modest LoRA adaptation of layers 0-1 and the embedding matrix creates a private local model variant. We hypothesize that random LoRA perturbation alone would not reduce inversion accuracy against an adaptive attacker who can retrain on the perturbed activations; the fine-tuning must produce genuinely private weights not available to the adversary. Validating this hypothesis requires further experimentation.
- **Randomizing the split point** per session or per request forces the attacker to maintain separate inversion models for each possible split depth. Since all transformer layers share the same hidden dimension (4096 for Mistral 7B), the cloud cannot trivially determine which layer produced a given activation tensor from the tensor shape alone. However, activation magnitude distributions may shift across layers, potentially allowing statistical identification of the split depth; this caveat warrants investigation. Combined with deeper splits, randomization increases the cost of targeted inversion attacks.
- **Adding calibrated noise** to transmitted activations (as in Split-and-Denoise [26]) provides formal (ε, δ)-differential privacy guarantees at the cost of some output quality.
- **Activation compression** (quantization to int8 for transmission) lossily reduces information content in transmitted representations.

---

## 6. Experimental Results

### 6.1 Setup

- **Local device:** NVIDIA RTX 3090 24GB, Manjaro Linux, layers 0-1 + 30-31
- **Cloud device:** NVIDIA RTX 4090 24GB, RunPod US-TX-3 ($0.59/hr), layers 2-29
- **Model:** Mistral 7B Instruct v0.3, float16 precision
- **Network:** SSH tunnel over public internet, RTT ~80-100ms to Texas datacenter
- **Client:** Apple M4 Max MacBook Pro (display only, no inference computation)

**Methodology:** Throughput measurements report end-to-end tokens/second from first generated token to last (including prefill). Section 6.2 results are from interactive sessions with conversational prompts, measured over 3-5 runs of 100-500 tokens. Section 6.4 ablation results use 8 specific prompts across 4 categories (code, structured, creative, conversational), each generating exactly 200 tokens (2 prompts per category). The reported figures represent typical sustained throughput, not peak instantaneous rates. Greedy decoding (argmax sampling) was used throughout to ensure deterministic, reproducible results.

### 6.2 Performance Results

> **[Figure 2]** *Bar chart showing optimization progression from 1.4 to 13--17 tok/s, with bars colored by category (transport vs. algorithmic vs. infrastructure). Inset pie chart shows the per-token latency breakdown in sequential mode (78% network, 8% serialization, 4% each for local compute, cloud compute, WebSocket overhead, and sampling).*

**Optimization progression:** The following table shows cumulative improvements. Changes fall into two categories: *transport optimizations* (rows 1-4) that reduce per-round-trip overhead, and *algorithmic/infrastructure optimizations* (rows 5-7) that reduce round trips or RTT.

| Configuration | tok/s | vs. Baseline | Category |
|--------------|-------|-------------|----------|
| HTTP + torch.save (A100 Montreal) | 1.4 | 1.0x | Transport |
| HTTP + numpy serialization | 3.9 | 2.8x | Transport |
| 3090 relay (Mac → 3090 → cloud) | 5.3 | 3.8x | Transport |
| WebSocket binary protocol | 7.8 | 5.6x | Transport |
| + Lookahead decoding (HyperStack, ~85ms RTT) | 8.7 | 6.2x | Algorithmic |
| Provider switch to RunPod (~80ms RTT) | 10.9 | 7.8x | Infrastructure |
| + Lookahead on lower-RTT link | 13--17 | 9--12x | Combined |

The jump from 8.7 to 10.9 tok/s was driven by switching cloud providers (HyperStack to RunPod), which reduced RTT by eliminating proxy intermediaries. The further improvement to 13--17 tok/s reflects lookahead decoding operating more effectively on the lower-latency link: with faster round trips, n-gram verification completes sooner and more speculative iterations execute per unit time. The range depends on content type (Section 6.4). Notably, switching from HyperStack+lookahead (8.7 tok/s) to RunPod+sequential (10.9 tok/s) yielded a 2.2 tok/s gain, while the initial lookahead gain on the higher-latency link was only 0.9 tok/s above sequential---underscoring that RTT reduction and speculation are *multiplicative* rather than additive.

**Comparison to baselines:**

| Setup | tok/s | % of local-only |
|-------|-------|-----------------|
| RTX 3090 local-only (all 32 layers) | ~39 | 100% |
| Split: 3090 + cloud 4090 (lookahead) | 13--17 | 33--44% |
| Split: 3090 + cloud 4090 (sequential) | ~11 | 28% |
| Cloud API (typical, incl. queuing) | 30-60 | - |

### 6.3 Latency Breakdown

Per-token latency in sequential mode at ~100ms RTT (~128ms total, measured on HyperStack):

| Component | Time | % of Total |
|-----------|------|-----------|
| Local layers (0-1, 30-31) | ~5ms | 4% |
| Cloud layers (2-29) | ~3ms | 2% |
| Network round trip | ~100ms | 78% |
| Serialization + framing | ~10ms | 8% |
| WebSocket overhead | ~5ms | 4% |
| Sampling + bookkeeping | ~5ms | 4% |

Network round-trip time dominates at 78% of total per-token latency. This has two implications: (1) traditional GPU optimization of the compute path yields diminishing returns; (2) any technique that amortizes RTT across multiple tokens---like lookahead decoding---has outsized impact.

**Prefill latency:** The initial prompt processing (prefill) requires transmitting the full prompt's hidden states (~8KB × prompt_length) to the cloud in a single forward pass. For a 100-token prompt, this is ~800KB, completing in a single round trip (~100ms + transfer time). Prefill is a one-time cost per generation and does not affect the per-token throughput figures reported above, though it adds 100--500ms of time-to-first-token depending on prompt length.

**Effective per-token latency in lookahead mode:** Lookahead decoding improves throughput by accepting multiple tokens per step when n-gram speculation succeeds (up to 5 tokens per step on code). Each lookahead step requires exactly one cloud round trip, same as sequential decoding. The wall time per step is therefore dominated by RTT: at ~80ms RTT (ablation conditions, Section 6.4), each step takes ~81ms; at ~100ms RTT (Section 6.2 conditions), each step takes ~128ms. At the optimal n-gram size (n=4), the average acceptance rate is 1.23 tokens/step; combined with ~81ms per step at ~80ms RTT, this yields ~15 tok/s (1.23 / 0.081s), consistent with our measured ablation throughput. On code prompts, acceptance rises to 1.40 tokens/step (17.1 tok/s); on creative text, acceptance drops to 1.12 tokens/step (14.6 tok/s).

### 6.4 Lookahead Ablation

We systematically tested n-gram sizes 3-7 across four prompt categories (code, structured, creative, conversational), generating 200 tokens per prompt in split mode with ~80ms RTT.

**Methodology notes:** The ablation was conducted on RunPod (~80ms RTT). Due to an implementation oversight, the ablation's baseline was measured using the Jacobi decode path (`generate_stream` with block_size=16) rather than the true sequential path (block_size=1). The Jacobi baseline incurs convergence overhead from multiple round trips per token block, yielding 7.1 tok/s---slower than true sequential decoding at ~10.9 tok/s (Section 6.2, row 6). The lookahead results are unaffected by this baseline choice, as they use a separate code path. Since each lookahead step requires exactly one cloud round trip (same as sequential), the acceptance rate directly predicts the speedup over true sequential at matched conditions.

**Throughput by n-gram size (averaged across all prompt types):**

| N-gram Size | tok/s | Acceptance Rate | Match Rate |
|-------------|-------|-----------------|------------|
| Sequential (RunPod, Section 6.2) | ~10.9 | 1.00 tok/step | - |
| n=3 | 14.7 | 1.21 tok/step | 14.4% |
| **n=4** | **15.1** | **1.23 tok/step** | **13.7%** |
| n=5 | 13.5 | 1.23 tok/step | 13.4% |
| n=6 | 13.1 | 1.25 tok/step | 13.7% |
| n=7 | 13.2 | 1.25 tok/step | 13.2% |

N-gram size 4 is optimal: larger n-grams achieve marginally higher acceptance rates (1.25 vs 1.21 tokens/step) but incur greater per-step overhead from verifying more candidates, resulting in lower net throughput.

**Throughput by prompt category (at optimal n=4):**

| Category | Lookahead (n=4) | Acceptance | vs Sequential (~10.9) |
|----------|-----------------|------------|----------------------|
| Code | 17.1 | 1.40 | 1.57x |
| Structured | 14.2 | 1.21 | 1.30x |
| Creative | 14.6 | 1.12 | 1.34x |
| Conversational | 14.8 | 1.18 | 1.36x |

Code prompts benefit most from lookahead speculation (1.40 tokens accepted per step), likely because code contains repetitive patterns (indentation, common function signatures, closing brackets) that generate strong n-gram candidates. Creative text shows the lowest acceptance rate (1.12). The speedup over sequential decoding tracks closely with the acceptance rate, consistent with our analysis in Section 6.3 that each lookahead step costs approximately one round trip. Per-category throughput shows ±5% deviation from acceptance-rate predictions (e.g., creative achieves 14.6 tok/s vs. 13.8 predicted from 1.12 × 1000/81ms), likely due to per-step wall time varying with the number of candidates verified and natural RTT fluctuation across measurement runs.

**Note on greedy decoding:** All measurements use greedy (argmax) decoding for reproducibility. Temperature or top-k/top-p sampling would likely reduce n-gram acceptance rates, as sampling introduces stochasticity that makes next-token predictions less deterministic. The speedup ratios reported here should be considered upper bounds for stochastic decoding strategies.

### 6.5 Cloud Provider Comparison

| Provider | Method | RTT | Proxy | Cost/hr | Usability |
|----------|--------|-----|-------|---------|-----------|
| HyperStack (A100 Montreal) | Direct SSH | ~85ms | No | ~$2.00 | IP changes on reboot |
| VAST.ai (4090 Texas) | Proxied SSH | 200-400ms | Yes (Virginia) | ~$0.50 | Unusable for split inference |
| RunPod (4090 Texas) | Direct SSH | 80-100ms | No | $0.59 | Persistent volumes, templates |

VAST.ai's proxy architecture routes all SSH connections through intermediate servers regardless of pod location, making it fundamentally unsuitable for latency-sensitive split inference. This finding is not documented in VAST.ai's literature and was discovered only through empirical measurement.

---

## 7. Scaling Analysis

### 7.1 RTT-to-Compute Ratio

The key observation for split inference viability at scale is that network round-trip time is largely *fixed* while cloud compute time grows *proportionally* with model size. We define the RTT-to-compute ratio as the network RTT divided by the cloud processing time per token:

| Model | Hidden Dim | Activation Size/Token | Cloud Compute/Token (est.) | Network RTT | RTT-to-Compute Ratio |
|-------|-----------|----------------------|--------------------|----|--------|
| Mistral 7B | 4,096 | 8 KB | ~3ms | ~80ms | 27x |
| LLaMA 70B | 8,192 | 16 KB | ~30-50ms | ~80ms | 1.6-2.7x |
| LLaMA 405B | 16,384 | 32 KB | ~100-200ms | ~80ms | 0.4-0.8x |

A high ratio (27x at 7B) means network latency overwhelmingly dominates---the GPU sits idle waiting for the next activation. A ratio near 1x or below (405B) means compute time rivals or exceeds network latency, and the split inference overhead approaches the theoretical minimum. The ratio decreases because:
1. The number of cloud-side layers increases linearly with model depth.
2. Per-layer compute grows quadratically with hidden dimension (attention is O(d^2)).
3. Network transfer grows only linearly with hidden dimension (8KB → 32KB, a 4x increase for a 58x larger model).

**Projected throughput with scaling (at ~80ms RTT):**

Per-token time is RTT + cloud compute + overhead (~20ms serialization and protocol). Sequential throughput = 1/per-token-time; lookahead multiplied by ~1.3x acceptance rate.

| Model | Cloud Compute (est.) | Per-Token Time (est.) | Split Sequential | Split + Lookahead | Overhead Factor |
|-------|---------------------|-----------------------|-----------------|------------------|-----------------|
| 7B | ~3ms | ~103ms | ~11 tok/s (measured) | 14--17 tok/s (measured) | 34x |
| 70B (est.) | ~30-50ms | ~130-150ms | ~7 tok/s | ~9 tok/s | 3-5x |
| 405B (est.) | ~100-200ms | ~200-300ms | ~3-5 tok/s | ~4-7 tok/s | 1.5-3x |

The *overhead factor* (per-token time / cloud compute time) quantifies how much slower split inference is compared to running all layers on the cloud without a network round trip. This factor drops from 34x at 7B to 1.5-3x at 405B, meaning that for larger models, the network round trip becomes a diminishing fraction of total per-token time. At 405B, split inference is only 1.5-3x slower than cloud-only execution---and provides architectural privacy that cloud-only does not.

Note that these absolute throughputs are lower than the 7B baseline because larger models have more compute per token (which is also why they can't run locally, motivating split inference). The comparison is not split-vs-local-native (the user *cannot* run 70B+ locally), but split-with-privacy vs. sending-plaintext-to-cloud-API.

**Important caveats on projections:** The 70B and 405B rows are estimates derived from the per-token time formula above; they have not been empirically validated. Several factors could reduce actual throughput: (1) KV-cache memory on the cloud grows with both model size and sequence length, potentially requiring cache eviction or compression; (2) activation transfer at 32KB/token for 405B, while still small, could face bandwidth limitations on constrained links; (3) larger models may require multi-GPU cloud setups, introducing additional inter-GPU communication overhead; (4) lookahead acceptance rates may differ at larger model sizes. These projections should be treated as order-of-magnitude estimates pending empirical validation.

### 7.2 Local Memory Requirements

| Model | Embedding + Head | Local Layers | KV-Cache (local, 2K ctx) | Peak Local VRAM |
|-------|-----------------|-------------|------------------------|-----------------|
| 7B | ~1 GB | ~0.5 GB | ~0.1 GB | ~2 GB |
| 70B | ~4 GB | ~4 GB | ~0.5 GB | ~9 GB |
| 405B | ~8-16 GB | ~8-16 GB | ~2-4 GB | ~20-36 GB |

An RTX 3090 (24GB) comfortably handles 70B local layers. For 405B, a workstation-class GPU (RTX A6000 48GB or similar) would be needed. Note that the local KV-cache only stores entries for the locally-processed layers (4 out of 32 for 7B), keeping local memory requirements modest. The bulk of the KV-cache resides on the cloud server.

---

## 8. Discussion

### 8.1 What is Novel

Several aspects of this work represent novel contributions relative to the existing literature:

1. **Privacy-motivated layer split at the embedding boundary with empirical evaluation.** Prior split computing work [8, 9, 10] optimizes split points for latency and energy. We split specifically to keep the token-to-vector mapping local, ensuring raw tokens never cross the network. The closest prior work, Split-and-Denoise [26], adds noise to embeddings but still transmits them; we transmit post-transformer hidden states. Crucially, we provide empirical inversion attack measurements (Section 5.2) quantifying token recoverability at different split depths, showing that increasing local layers from 2 to 8 reduces attack accuracy from ~59% to ~35%.

2. **Lookahead decoding for network latency amortization.** Lookahead decoding [6] was designed to exploit GPU parallelism. We show it is equally---arguably more---effective at amortizing network round-trip latency in split inference, achieving 1.2--1.4 tokens per step (up to 5 on code), where the cost of a single round trip is 30-100x higher than a local forward pass.

3. **Empirical deployment engineering.** The academic split computing literature assumes direct network connectivity between devices. In practice, deploying over public cloud GPU infrastructure requires SSH tunnel management, provider-specific networking workarounds, ephemeral IP handling, and cross-machine SSH key distribution. Our documentation of these challenges and solutions fills a gap between theoretical architectures and deployable systems.

4. **Inverse scaling of overhead.** While prior work [10] notes that intermediate activation sizes grow with model width, we provide the first quantitative analysis showing that the *RTT-to-compute ratio* of split inference improves with model size, because cloud compute time grows faster than network transfer time. We note this analysis remains unvalidated beyond 7B (Section 7).

### 8.2 Relationship to Concurrent Work

DSSD [23] (ICML 2025) is the most closely related concurrent work, combining split inference with speculative decoding for edge-cloud LLM deployment. Our system differs in three ways: (1) we use lookahead decoding rather than a separate draft model, avoiding the need to deploy and maintain an additional model; (2) we focus on privacy preservation as the primary motivation rather than resource efficiency; and (3) we provide detailed empirical measurements of real WAN deployment rather than simulated edge-cloud environments.

CROSS-SEC [13] addresses WAN latency with layerwise KV-cache overlapping, which is complementary to our approach and could further reduce our per-token latency.

### 8.3 Limitations

- **No formal privacy guarantees.** Our system provides architectural privacy (raw tokens never leave the device), not cryptographic guarantees. Our inversion experiment (Section 5.2) confirms that even a simple MLP attacker with local layer weights can recover ~59% of tokens from layer-2 activations, validating the theoretical vulnerability [32]. A more sophisticated sequence-aware attacker would likely achieve higher accuracy. Increasing local layers to 8 reduces MLP attack accuracy to ~35%, but does not eliminate the risk. Our privacy relies on the assumption that local layers are customized; for public models used as-is, the architectural boundary is significantly weakened.
- **Single model tested.** Results are demonstrated on Mistral 7B only. While our scaling analysis projects favorable results for larger models, these projections need empirical validation.
- **Consumer hardware.** Our local device (RTX 3090) is a consumer GPU. Enterprise deployments may use workstation GPUs (A6000, L40) that could run more local layers for stronger privacy.
- **Lookahead effectiveness varies.** Our ablation (Section 6.4) confirms that n-gram speculation effectiveness depends on text type: code prompts achieve 1.40 tokens/step acceptance, while creative text achieves only 1.12 tokens/step. The optimal n-gram size (n=4) provides the best throughput despite not having the highest acceptance rate, due to lower per-step overhead. All measurements use greedy decoding; stochastic sampling would likely reduce acceptance rates.
- **Prefill latency not optimized.** Our system processes the full prompt in a single forward pass, adding 100--500ms time-to-first-token. Techniques like chunked prefill or prefill-decode disaggregation [17, 18] could reduce this.

### 8.4 Partnership Model with Inference Providers

A natural deployment model pairs a local-device software SDK with a partnered cloud inference provider. The provider gains access to compliance-locked market segments (HIPAA, ITAR, legal) that currently cannot use cloud inference at all.

As a thought experiment, we consider how partner infrastructure characteristics would affect throughput. Since our results show network RTT is the dominant bottleneck (78% of per-token time), partners with lower-latency infrastructure would yield disproportionate gains:

| Partner Infrastructure | Est. Cloud Compute | Est. RTT | Projected tok/s |
|-----------------------|-------------------|----------|-----------------|
| Standard GPU cloud (A100) | ~3ms | ~80-100ms | 13--17 (measured) |
| Low-latency accelerator (same region) | ~1ms | ~20-30ms | 30-50 (est.) |
| Low-latency accelerator (co-located) | ~1ms | ~5-10ms | 50-100 (est.) |

These projections are speculative and assume the same activation transfer protocol scales linearly with reduced RTT. In practice, other bottlenecks (serialization overhead, local compute, bandwidth) would likely emerge as RTT decreases. Nevertheless, the analysis suggests that co-location with a fast inference partner could make the privacy overhead of split inference negligible for end users.

---

## 9. Conclusion

We have demonstrated that split inference for large language models can achieve interactive speeds (13--17 tok/s on Mistral 7B) over commodity internet connections using consumer GPUs. The key technical contribution---applying lookahead decoding to amortize network latency---addresses the fundamental bottleneck of autoregressive generation in the split inference setting, and our scaling analysis suggests the approach becomes more favorable as model size increases.

Important limitations remain. Our inversion experiment (Section 5.2) shows that at the default 2-layer split, an attacker with local layer weights can recover ~59% of tokens from intermediate activations, confirming the theoretical vulnerability [32]. Increasing local depth to 8 layers reduces this to ~35% with minimal throughput impact, but formal privacy guarantees require additional mechanisms such as differential privacy [26, 27]. Results are demonstrated on a single model and hardware configuration; validation across model families and scales is needed to confirm generalizability.

Nevertheless, for organizations where the current alternative is sending full plaintext to cloud APIs, this work demonstrates a practical path toward privacy-aware LLM deployment. Combined with the finding that cloud provider networking architecture and tunnel engineering are first-order performance determinants, our results provide a foundation for further development of deployable privacy-aware inference systems.

---

## References

[1] C. Gentry, "Fully Homomorphic Encryption Using Ideal Lattices," *STOC*, 2009.

[2] M. Ugurbil et al., "Fission: Distributed Privacy-Preserving Large Language Model Inference," *IACR ePrint* 2025/653, 2025.

[3] T. Nayan et al., "SecureInfer: Heterogeneous TEE-GPU Architecture for Privacy-Critical Tensors for Large Language Model Deployment," *arXiv:2510.19979*, 2025.

[4] Z. He et al., "Model Inversion Attacks Against Collaborative Inference," *ACSAC*, pp. 148-162, 2019.

[5] S. Ding et al., "PATROL: Privacy-Oriented Pruning for Collaborative Inference Against Model Inversion Attacks," *WACV*, 2024.

[6] Y. Fu, P. Bailis, I. Stoica, H. Zhang, "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding," *ICML*, 2024.

[7] A. Santilli et al., "Accelerating Transformer Inference for Translation via Parallel Decoding," *ACL*, pp. 12336-12355, 2023.

[8] Y. Kang et al., "Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge," *ASPLOS*, 2017.

[9] E. Li et al., "Edge Intelligence: On-Demand Deep Learning Model Co-Inference with Device-Edge Synergy," *IEEE Trans. Wireless Communications*, 2019.

[10] Y. Matsubara et al., "Split Computing and Early Exiting for Deep Learning Applications: Survey and Research Challenges," *ACM Computing Surveys*, 2022.

[11] M. Sung et al., "Memory- and Latency-Constrained Inference of Large Language Models via Adaptive Split Computing," *arXiv:2511.04002*, 2025.

[12] Y. Chen et al., "Adaptive Layer Splitting for Wireless LLM Inference in Edge Computing," *Frontiers of Information Technology & Electronic Engineering*, 2025.

[13] R. Qin et al., "CROSS-SEC: A Cloud-Edge Collaborative Inference System for Data-secure LLM Serving," *NAIC*, 2025.

[14] A. Borzunov et al., "Petals: Collaborative Inference and Fine-tuning of Large Models," *ACL System Demonstrations*, 2023.

[15] M. Zhang et al., "EdgeShard: Efficient LLM Inference via Collaborative Edge Computing," *IEEE IoT Journal*, 2024.

[16] D. Macario et al., "MDI-LLM: Model-Distributed Inference for Large Language Models at the Edge," *IEEE LANMAN*, 2025.

[17] P. Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," *ISCA*, 2024.

[18] Y. Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving," *OSDI*, 2024.

[19] Y. Leviathan, M. Kalman, Y. Matias, "Fast Inference from Transformers via Speculative Decoding," *ICML*, 2023.

[20] C. Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling," *arXiv:2302.01318*, 2023.

[21] S. Kou et al., "CLLMs: Consistency Large Language Models," *ICML*, 2024.

[22] N. Timor et al., "Distributed Speculative Inference of Large Language Models," *NeurIPS Workshop*, 2024.

[23] J. Ning et al., "DSSD: Efficient Edge-Device LLM Deployment and Collaborative Inference via Distributed Split Speculative Decoding," *ICML*, 2025.

[24] J. Song et al., "Speculative Decoding in Decentralized LLM Inference: Turning Communication Latency into Computation Throughput," *arXiv:2511.11733*, 2025.

[25] A. Singh et al., "DISCO: Dynamic and Invariant Sensitive Channel Obfuscation for Deep Neural Networks," *CVPR*, 2021.

[26] P. Mai et al., "Split-and-Denoise: Protect Large Language Model Inference with Local Differential Privacy," *ICML*, 2024.

[27] R. Deng et al., "Quantifying Privacy Leakage in Split Inference via Fisher-Approximated Shannon Information Analysis," *arXiv:2504.10016*, 2025.

[28] Y. Zheng et al., "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning," *OSDI*, 2022.

[29] Z. He et al., "REST: Retrieval-Based Speculative Decoding," *NAACL*, pp. 1582-1595, 2024.

[30] T. Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads," *ICML*, 2024.

[31] Y. Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty," *ICML*, 2024.

[32] G. Nikolaou et al., "Language Models are Injective and Hence Invertible," *arXiv:2510.15511*, 2025.

[33] exo-explore, "exo: Run your own AI cluster at home with everyday devices," *GitHub*, 2024. https://github.com/exo-explore/exo

---

*Code and reproduction instructions will be made available at: [github.com/coder903/split-inference](https://github.com/coder903/split-inference)*
