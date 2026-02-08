# STEP 3.5 FLASH: Efficient Frontier Model

**Date**: February 2026 | **Team**: StepFun AI

[📄 PDF](https://github.com/stepfun-ai/Step-3.5-Flash/blob/main/step_3p5_flash_tech_report.pdf) | [🤗 Model](https://modelscope.cn/models/stepfun-ai/Step-3.5-Flash) | [💻 Code](https://github.com/stepfun-ai/Step-3.5-Flash)

---

## 🎯 One-Line Summary

5B active parameter MoE model achieving frontier-level reasoning and agentic capabilities, outperforming much larger models (20-70B) through optimized architecture and efficient post-training.

## 📊 Key Results

| Benchmark | Score | Rank |
|-----------|-------|------|
| LiveCodeBench | 72.2 | #1 (open models ≤70B) |
| MMLU-Pro | 68.5 | #1 (open models ≤70B) |
| GPQA Diamond | 48.7 | #1 (open models ≤70B) |
| MATH | 78.3 | #1 (open models ≤70B) |
| MMLU (5-shot) | 88.4 | Top-tier |
| ResearchRubrics | 65.27 | SOTA (open-source) |
| MMMU (Vision) | 51.8 | Competitive |
| MathVerse (Vision) | 54.8 | Competitive |
| SWE-Bench Verified | 40.1 | Competitive |
| RepoEval | 58.2 | Strong |

**Outperforms**: Qwen2.5-72B-Instruct, DeepSeek-V3, LLaMA-3.1-405B-Instruct

## 🏗️ Architecture

```
Total Parameters:        21B
Active Parameters:       5B (MoE sparsity, 23.8%)
Experts:                 64 experts, top-2 routing
Context Length:          128K tokens
Vision Encoder:          768M (native multimodal)
Architecture:            Mixture-of-Experts Transformer
Intermediate Size:       13,312
Attention:               Grouped Query Attention (GQA)
Vocabulary:             128K tokens
```

## 🔬 Core Techniques

### 1. 64-Expert MoE with Top-2 Routing

**Problem**: Balance model capacity with inference efficiency and cost.

**Solution**: 64 experts with top-2 routing, activating only 5B parameters per token (23.8% sparsity).

**Results**: 4× effective capacity (21B total) with inference speed/cost comparable to 5B dense models.

### 2. Post-Training Data Optimization

**Problem**: Improve reasoning capabilities without expensive pre-training.

**Solution**: Two-phase post-training with 30M tokens (Phase 1: general, Phase 2: reasoning/agentic specialization).

**Results**: 4-15 point gains on reasoning benchmarks; surpasses much larger models.

### 3. Native Multimodal Architecture

**Problem**: Need strong vision-language understanding for agentic applications.

**Solution**: Integrate 768M vision encoder with cross-attention layers for seamless visual understanding.

**Results**: 51.8 on MMMU, 54.8 on MathVerse; outperforms dedicated multimodal models.

### 4. Grouped Query Attention (GQA)

**Problem**: Reduce inference latency and memory footprint for long contexts.

**Solution**: GQA with optimized key-value cache for efficient 128K context handling.

**Results**: Fast inference suitable for real-time agentic applications.

### 5. High-Quality SFT Data Curriculum

**Problem**: Optimize knowledge transfer from teacher models (Step-3X-100B).

**Solution**: Multi-stage SFT with math, code, reasoning, and agentic task data.

**Results**: Effective knowledge distillation achieving competitive results with 20× fewer active parameters.

## 📈 Training

**Pre-training**:
- Base model: Built from Step-3X family architecture
- Tokens: Not specified (report focuses on post-training)

**Post-training**:
- Phase 1: ~30M tokens (general capabilities)
- Phase 2: ~30M tokens (reasoning & agentic specialization)
- Optimizer: AdamW
- Learning rate: Cosine decay schedule

**Infrastructure**: Distributed training on multiple GPUs, optimized for inference on consumer-grade hardware

## ⚠️ Limitations

1. Hallucination: Can generate incorrect/fabricated information, especially outside training data
2. Context window utilization: Performance may degrade on extremely long documents with complex cross-segment reasoning
3. Multimodal constraints: Vision encoder resolution/detail limited vs larger dedicated models
4. Code execution: Cannot execute code directly; relies on static analysis
5. Knowledge cutoff: Training data cutoff December 2025

## 🔗 Links

- **GitHub**: https://github.com/stepfun-ai/Step-3.5-Flash
- **PDF**: https://github.com/stepfun-ai/Step-3.5-Flash/blob/main/step_3p5_flash_tech_report.pdf
- **Model**: https://modelscope.cn/models/stepfun-ai/Step-3.5-Flash
- **Blog**: https://static.stepfun.com/blog/step-3.5-flash/
- **Organization**: https://github.com/stepfun-ai

---

**Tags**: #MoE #efficiency #multimodal #reasoning #agentic-ai #post-training
