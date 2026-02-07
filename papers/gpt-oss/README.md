# GPT-OSS: Fully Open-Source 120B & 20B Models

**arXiv**: [2508.10925](https://arxiv.org/abs/2508.10925) | **Date**: August 2025 | **Team**: Multi-institutional (Berkeley, UW, Stanford, CMU, Princeton, etc.)

[📄 PDF](https://arxiv.org/pdf/2508.10925) | [🤗 Models](https://huggingface.co/EleutherAI/gpt-oss-120b)

---

## 🎯 One-Line Summary

First **fully transparent open-source LLMs** (120B & 20B MoE) releasing complete training stack—2.6T tokens of data, all code, infrastructure, and 26 intermediate checkpoints—enabling full reproducibility and scientific study.

## 📊 Key Results

**GPT-OSS-120B:**
| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU | 80.6 | Competitive with GPT-4 |
| MMLU-Pro | 54.8 | More robust variant |
| GPQA (4-shot) | 43.5 | Graduate-level QA |
| Math Vista | 40.8 | Visual math reasoning |
| BBH (3-shot) | 75.9 | Big-Bench Hard |
| HumanEval | 68.3 | Zero-shot coding |
| MATH (4-shot) | 50.5 | Problem solving |

**GPT-OSS-20B:**
| Benchmark | Score |
|-----------|-------|
| MMLU | 66.8 |
| MMLU-Pro | 39.9 |
| GPQA (4-shot) | 27.0 |
| Math Vista | 26.2 |
| BBH (3-shot) | 61.4 |
| HumanEval | 35.4 |
| MATH (4-shot) | 22.5 |

## 🏗️ Architecture

**GPT-OSS-120B:**
```
Total Parameters:        120B (16 experts)
Active Parameters:       ~8.4B per token (~7%)
Layers:                  80
Hidden Size:             6,144
Context Length:          8,192 tokens
Attention Heads:         48
Intermediate Size:       16,896
Routing:                 Top-2 router with load balancing
```

**GPT-OSS-20B:**
```
Total Parameters:        20B (16 experts)
Active Parameters:       ~1.4B per token (~7%)
Layers:                  48
Hidden Size:             4,096
Context Length:          8,192 tokens
Attention Heads:         32
Intermediate Size:       11,008
Routing:                 Top-2 router with load balancing
```

## 🔬 Core Techniques

### 1. Mixture-of-Experts Architecture

**Problem**: Training and serving large dense models is computationally prohibitive.

**Solution**: Sparse MoE with 16 experts and top-2 routing, activating only ~7% of parameters per token.

**Results**: ~4.5× reduced training costs; 120B model competitive with closed models using only ~8.4B active parameters.

### 2. Fully Open Training Pipeline

**Problem**: Lack of transparency in LLM training prevents reproducibility and scientific progress.

**Solution**: Released complete training stack: 2.6T tokens data, training code, infrastructure configs, all 26 intermediate checkpoints (every 100B tokens), detailed logs.

**Results**: Enables full reproducibility; researchers can study training dynamics and intervene at any point.

### 3. High-Quality Open Data Pipeline

**Problem**: Publicly available data sources often insufficient for high-performance models.

**Solution**: Carefully curated 2.6T tokens from diverse web sources, books, code, academic papers with aggressive deduplication and quality filtering.

**Results**: Competitive with closed-source models using only open data sources.

### 4. Expert Load Balancing

**Problem**: MoE models can suffer from expert collapse where few experts are utilized frequently.

**Solution**: Load balancing loss during training for uniform expert utilization and routing stability.

**Results**: Stable training with balanced expert utilization; maintained quality with computational efficiency.

### 5. Training Stability Techniques

**Problem**: Training large models at scale often encounters optimization instabilities.

**Solution**: Careful LR scheduling, gradient clipping, warmup strategies; close monitoring of training metrics.

**Results**: Stable training across 2.6T tokens without catastrophic loss spikes; consistent improvement across all checkpoints.

## 📈 Training

**Data**: 2.6 trillion tokens from web text (CommonCrawl, RefinedWeb), books, code, academic papers

**Infrastructure**:
- Hardware: 512 H100 80GB GPUs (for 120B)
- Framework: JAX + Orax
- Parallelization: 3D (data, tensor, pipeline)
- Training time: ~3 months for 120B

**Optimization**:
- Optimizer: AdamW
- Learning rate: Peak 2e-4 with cosine decay
- Batch size: 4M tokens
- Sequence length: 8,192 tokens
- Precision: bfloat16
- Checkpoints: Every 100B tokens (26 total, all released)

## ⚠️ Limitations

1. Context window: 8,192 tokens shorter than recent models (128K+)
2. No alignment: Base pre-trained models without RLHF or instruction tuning
3. Performance gap: Still lags best closed-source models on some reasoning tasks
4. Data contamination: Potential test set contamination in training data
5. English-centric: Training predominantly English, limiting multilingual performance

## 🔗 Links

- **Paper**: https://arxiv.org/abs/2508.10925
- **PDF**: https://arxiv.org/pdf/2508.10925
- **120B Model**: https://huggingface.co/EleutherAI/gpt-oss-120b
- **20B Model**: https://huggingface.co/EleutherAI/gpt-oss-20b
- **Training Data**: Available through open source release
- **Checkpoints**: All 26 intermediate checkpoints released

---

**Tags**: #open-source #MoE #reproducibility #transparency #training-data #JAX
