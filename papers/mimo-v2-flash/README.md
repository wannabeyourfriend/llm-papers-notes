# MIMO-V2-FLASH: Extreme Context Length Training

**arXiv**: [2601.02780](https://arxiv.org/abs/2601.02780) | **Date**: January 2026 | **Team**: Project Mc2, McQUIC

[📄 PDF](https://arxiv.org/pdf/2601.02780) | [💻 Project](https://mc2-projects.github.io/mimo-v2/)

---

## 🎯 One-Line Summary

Enables training with **extreme context lengths** (up to 128M tokens) through novel MoE approach with FlashAttention-3 and 3D parallelism, achieving SOTA long-context performance while maintaining training efficiency.

## 📊 Key Results

| Benchmark | Context | Score | Rank |
|-----------|---------|-------|------|
| RULER | 128K | 81.4% | SOTA (+6.2% over GPT-4o) |
| InfiniteBench | 1M | 55.9% | SOTA |
| PG19 (perplexity) | 1M | 1.85 | Competitive |
| Needle-in-Haystack | 128M | 100% | Perfect retrieval |
| Line-by-Line Retrieval | 1M | 99.9% | Near-perfect |
| HuggingFace (128K) | 128K | 60.8% | SOTA |
| Training Efficiency | - | 2.3× | Throughput improvement |

## 🏗️ Architecture

```
Parameters:              7B (base model)
Experts:                 64 per MoE layer
Active Experts/Token:    8 (out of 64)
Layers:                  32
Hidden Size:             4,096
Attention Heads:         32
FFN Dimension:           10,240 per expert
Context Length:          Up to 128M tokens (training)
Sequence Length:         32K (pre-training) → 128M (long-context)
```

## 🔬 Core Techniques

### 1. MIMO-V2 Architecture (Multi-Input Multi-Output)

**Problem**: Standard Transformers suffer from quadratic complexity and memory limitations with long contexts, making pre-training beyond 32K tokens infeasible.

**Solution**: Sparse MoE routing (8/64 experts per token) with optimized attention patterns and 3D parallelism (data, tensor, pipeline).

**Results**: Stable training at 128M token context with 2.3× throughput improvement and linear compute scaling.

### 2. FlashAttention-3 Integration

**Problem**: Standard attention mechanisms are memory-inefficient and slow for extreme context lengths.

**Solution**: FlashAttention-3 with sequence-level parallelism and specialized Hopper GPU kernels (O(N) memory).

**Results**: 3.2× speedup on attention; 128M token sequences on 512 H100 GPUs.

### 3. 3D Hybrid Parallelism

**Problem**: Training with extreme contexts requires distributing sequences across multiple GPUs while maintaining efficiency.

**Solution**: 8-way data + 8-way tensor + 8-way pipeline parallelism with optimized NCCL communication patterns.

**Results**: 48.5% MFU on 512 H100 GPUs with 128K context; 90%+ parallel efficiency.

### 4. Long-Context Curriculum Learning

**Problem**: Direct training on 128M token contexts is unstable and computationally wasteful.

**Solution**: Three-stage curriculum: (1) Pre-train 32K, (2) Gradually extend to 1M via interpolation, (3) Fine-tune on 128M with RoPE scaling and ALiBi bias.

**Results**: Stable convergence without catastrophic forgetting; maintains short-context performance.

### 5. Sparsity-Aware Optimizer

**Problem**: MoE models have imbalanced gradient updates due to expert routing, causing training instability.

**Solution**: ZO loss-aware expert selection, load-balancing regularization, per-expert gradient clipping, adaptive learning rates.

**Results**: Load imbalance reduced from 3.2× to 1.08×; 40% faster convergence.

## 📈 Training

**Pre-training**: 3.2T tokens (32K context length)

**Long-Context Adaptation**: 100B tokens (gradually extended to 128M)

**Optimizer**: AdamW with per-expert LR adaptation

**Schedule**: Cosine decay with warmup (3,000 steps), peak LR 3e-4

**Batch Size**: 4M tokens (512 H100 GPUs, 8-way gradient accumulation)

**Infrastructure**: 512 NVIDIA H100 80GB GPUs, NCCL backend, custom PyTorch with Megatron-LM core

**Training Time**: 92 days (3.2T tokens at 32K context)

## ⚠️ Limitations

1. Inference latency: 128M context requires 80GB GPU memory, >5 seconds/token
2. RoPE extrapolation: Performance degrades beyond 128M tokens due to saturation
3. Expert imbalance: At >64M contexts, load imbalance reaches 1.4×
4. Transfer gap: Long-context specialization leads to 2-3% drop on <4K benchmarks
5. Hardware requirements: 512+ H100 GPUs required for training

## 🔗 Links

- **Paper**: https://arxiv.org/abs/2601.02780
- **PDF**: https://arxiv.org/pdf/2601.02780
- **Project**: https://mc2-projects.github.io/mimo-v2/
- **Code**: (Not yet released, likely at https://github.com/mc2-projects/mimo)

---

**Tags**: #long-context #MoE #FlashAttention #3D-parallelism #extreme-context
