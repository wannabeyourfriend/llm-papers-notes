# ERNIE 5.0: Trillion-Parameter Unified Multimodal Model

**arXiv**: [2602.04705](https://arxiv.org/abs/2602.04705) | **Date**: February 2026 | **Team**: Baidu Inc.

[📄 PDF](https://arxiv.org/pdf/2602.04705.pdf)

---

## 🎯 One-Line Summary

First **trillion-parameter unified autoregressive model** supporting text, image, video, and audio understanding/generation through **elastic training** enabling flexible deployment (3B-12.9B active parameters) from single training run.

## 📊 Key Results

| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU (5-shot) | 89.2% | Text understanding |
| C-Eval (5-shot) | 89.6% | Chinese comprehensive |
| MMBench (8-shot) | 82.4% | Image understanding |
| SEED-Bench | 78.1% | Image multi-choice |
| POPE (F1) | 86.3% | Image understanding |
| ImageNet-1K FID | 6.82 | Image generation |
| GenEval | 68.3% | Image generation |
| VideoMME (test) | 73.5% | Video understanding |
| AudioSet (mAP) | 72.8% | Audio classification |

**Note**: Competitive with specialized single-modality models while supporting all 4 modalities.

## 🏗️ Architecture

```
Total Parameters:        ~1T (ultra-sparse MoE)
Active Parameters:       12.9B per forward pass
Experts:                 256 (top-2 routing, ~2.5% active)
Architecture:            Transformer decoder, modality-agnostic routing
Context Length:          128K tokens
Layers:                  128
Hidden Size:             7,168
Attention Heads:         113
Vocabulary:             128K (unified token space)
Modalities:              Text, Image, Video, Audio (unified next-token prediction)
```

## 🔬 Core Techniques

### 1. Unified Multimodal Autoregressive Modeling

**Problem**: Previous multimodal models use separate encoders/decoders per modality, limiting scalability and cross-modal understanding.

**Solution**: All modalities trained from scratch under single "next-group-of-tokens" prediction with unified token space.

**Results**: Enables genuine multimodal generation (not just retrieval) and strong cross-modal transfer; first trillion-parameter unified autoregressive model supporting understanding + generation.

### 2. Modality-Agnostic Expert Routing

**Problem**: Traditional MoE requires modality-specific experts, reducing parameter efficiency and cross-modal knowledge sharing.

**Solution**: Ultra-sparse MoE (256 experts, top-2 routing) where experts are modality-agnostic; routing based on content not modality type.

**Results**: 1T total params with only 12.9B active; 87.5% expert utilization; natural specialization patterns.

### 3. Elastic Training Paradigm

**Problem**: Single model cannot adapt to diverse deployment constraints; training multiple models separately is expensive.

**Solution**: Within single pre-training run, model learns family of sub-models with varying depths, expert capacities, routing sparsity; uses elastic routing and progressive width/depth training.

**Results**: Single model produces 3B-12.9B active parameter sub-models; 3B model runs 2.3× faster with 15% MMLU drop; 40% training cost reduction vs separate models.

### 4. Scaled Reinforcement Learning for Multimodal Models

**Problem**: RL training becomes unstable with ultra-sparse MoE and diverse multimodal objectives.

**Solution**: Modality-aware reward normalization; expert-level gradient clipping; auxiliary loss for expert diversity; iterative reward alignment across modalities.

**Results**: Stable post-training at trillion-parameter scale; 23% improvement on multimodal safety benchmarks; maintains >80% expert utilization during RL.

### 5. Unified Tokenization Across Modalities

**Problem**: Different modalities have incompatible discrete representations (pixels vs text vs audio spectrograms).

**Solution**: Continuous vector quantization (VQ) for images/videos/audio; unified 128K vocabulary with shared codebook; variable-length tokenization.

**Results**: Seamless autoregressive generation; compression: text 1.3 tokens/word, image 1024 tokens/256×256, video 4096 tokens/4s clip, audio 512 tokens/5s clip.

## 📈 Training

**Pre-training**:
- Tokens: 8.4T (multimodal mixture)
- Data distribution: Text (58%), Images (20%), Video (12%), Audio (10%)
- Training schedule: 1.2M steps over 3 months

**Optimization**:
- Optimizer: AdamW (β₁=0.9, β₂=0.95, ε=10⁻⁸)
- Learning rate: 3.0×10⁻⁴ peak with 2000-step warmup, cosine decay to 3×10⁻⁶
- Batch size: 4M tokens (2048 sequences × 2048 tokens)

**Infrastructure**:
- Hardware: 16,384 NVIDIA H100 GPUs (8192 nodes)
- Network: 400Gbps InfiniBand NDR
- Training efficiency: 55.4% MFU
- Checkpointing: Every 10K steps (2.5TB per checkpoint)

**Post-training**: 3 rounds RLHF (text-aligned + multimodal-aligned + safety-aligned)

## ⚠️ Limitations

1. High compute requirements: 12.9B active parameters need significant GPU memory (80GB+ for production)
2. Cross-modal hallucination: Autoregressive nature can cause inconsistent cross-modal generation
3. Tokenization efficiency: 1024 tokens for 256×256 images is computationally expensive
4. Expert load imbalance: Top 10% experts handle 35% of tokens
5. Limited temporal coherence: Video understanding drops 12.8% on clips >30 seconds; generation consistent ~8 seconds
6. Training data bias: Predominantly Chinese-English (78%), limiting low-resource language performance

## 🔗 Links

- **Paper**: https://arxiv.org/abs/2602.04705
- **PDF**: https://arxiv.org/pdf/2602.04705.pdf
- **Project**: https://ernie.baidu.com/ernie-5.0

**Note**: Code and model weights not publicly released yet; contact Baidu for research access.

---

**Tags**: #unified-multimodal #trillion-parameter #elastic-training #autoregressive #MoE
