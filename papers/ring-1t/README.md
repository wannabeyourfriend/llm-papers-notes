# RING-1T: Scaling RL for Trillion-Scale Thinking Model

**arXiv**: [2510.18855](https://arxiv.org/abs/2510.18855) | **Date**: October 2025 | **Team**: L Team (inclusionAI)

[📄 PDF](https://arxiv.org/pdf/2510.18855.pdf) | [🤗 Model](https://huggingface.co/inclusionAI/Ring-1T) | [💻 Code](https://github.com/inclusionAI/asystem-amem)

---

## 🎯 One-Line Summary

First open-source **trillion-parameter reasoning model** achieving SOTA math performance through multi-stage RL (SFT → reasoning RL → general RL) on mixed-quality reasoning data.

## 📊 Key Results

| Benchmark | Score | Rank |
|-----------|-------|------|
| AIME 2024 | 63.3% (23.3/30) | SOTA |
| AIME 2025 | 80.0% (24.0/30) | SOTA |
| HMMT 2025 | 86.72% | SOTA |
| IMO 2025 | 53.33% (8/15) | SOTA |
| GPQA Diamond | 76.1% | SOTA |
| MATH | 95.3% | SOTA |
| MMLU-Pro | 86.9% | SOTA |
| Codeforces Rating | 2088 | Expert level |
| ARC-AGI v1 | 55.94% | Competitive with o1-preview |

## 🏗️ Architecture

```
Total Parameters:        1T (1 trillion)
Active Parameters:       ~10B per forward pass
Architecture:            Mixture-of-Experts (MoE)
Experts:                 256 total
Active Experts/Token:    8 (out of 256)
Layers:                  128
Hidden Size:             6,144
Context Length:          128K tokens
```

## 🔬 Core Techniques

### 1. Long-CoT Supervised Fine-Tuning

**Problem**: Need high-quality reasoning demonstrations as foundation for RL.

**Solution**: Curated 1.2M long chain-of-thought samples with intermediate verification from math, code, and general reasoning sources.

**Results**: Established strong baseline for subsequent RL with diverse reasoning patterns.

### 2. Two-Stage Reinforcement Learning

**Problem**: How to improve reasoning through RL without environment rewards.

**Solution**: (a) Reasoning RL: Outcome verification reward model using verifier/ground-truth for math/code; (b) General RL: GRPO on diverse mixed-quality data (3M+ reasoning traces from various models).

**Results**: Continuous improvement across multiple RL iterations with expanding capabilities to general tasks.

### 3. Iterative Self-Evolution

**Problem**: Need sustainable improvement without expensive human annotation.

**Solution**: Each RL iteration uses best model to generate new training data (Ring-0 → Ring-1T-preview → Ring-1T).

**Results**: Performance gains across iterations with efficient data curation and automated reward construction.

### 4. Expert-Specific Router Optimization

**Problem**: Standard MoE routing doesn't differentiate between reasoning and non-reasoning tasks.

**Solution**: Train separate router configurations for different task types during SFT and RL phases.

**Results**: Better expert utilization for reasoning-intensive tasks with 10B active parameters.

## 📈 Training

**Base Model**: DeepSeek-V3 (1T MoE)

**Training Stages**:
- Ring-0 (initial) → Ring-1T-preview → Ring-1T (final)
- Multi-month iterative training with continuous RL cycles

**Data**:
- SFT: 1.2M long-CoT samples with intermediate reasoning
- RL: 3M+ mixed-quality reasoning traces from diverse sources

**Infrastructure**: 2,000+ NVIDIA H100 GPUs

**Optimization**: AdamW with weight decay, custom LR schedule with warmup/decay

**Reward Signals**: Verifier-based (math/code), automated scoring (general tasks)

## ⚠️ Limitations

1. Incomplete reasoning chains: Model may not fully explore all intermediate steps (average 11K reasoning tokens)
2. General task trade-offs: Heavy reasoning optimization may degrade pure language task performance
3. Continuous training required: Model under active training; preview version may have instabilities
4. Compute intensity: Requires significant infrastructure (2000+ H100 GPUs) and substantial inference compute
5. Reward signal quality: Performance limited by accuracy of automatic reward signals

## 🔗 Links

- **Paper**: https://arxiv.org/abs/2510.18855
- **PDF**: https://arxiv.org/pdf/2510.18855.pdf
- **Model (HF)**: https://huggingface.co/inclusionAI/Ring-1T
- **Model (ModelScope)**: https://modelscope.cn/models/inclusionAI/Ring-1T-preview
- **Code**: https://github.com/inclusionAI/asystem-amem

---

**Tags**: #reasoning #RL #MoE #math #open-source #thinking-model
