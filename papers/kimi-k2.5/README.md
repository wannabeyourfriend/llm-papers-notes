# Kimi K2.5: Visual Agentic Intelligence

**arXiv**: [2602.02276](https://arxiv.org/abs/2602.02276) | **Date**: February 2026 | **Team**: Moonshot AI

[📄 PDF](https://arxiv.org/pdf/2602.02276) | [🤗 Model](https://huggingface.co/moonshotai/Kimi-K2.5)

---

## 🎯 One-Line Summary

Introduces **joint text-vision optimization** and **Agent Swarm** (parallel multi-agent orchestration) reducing inference latency by 4.5× while achieving SOTA across coding, vision, reasoning, and agentic tasks.

## 📊 Key Results

| Benchmark | Score | Rank |
|-----------|-------|------|
| AIME 2025 | 96.1% | SOTA (approaches GPT-5.2) |
| LiveCodeBench v6 | 85.0% | Outperforms Claude Opus 4.5 (82.2%) |
| BrowseComp (Agent Swarm) | 78.4% | SOTA (surpasses GPT-5.2 Pro 77.9%) |
| WideSearch (Agent Swarm) | 79.0% | SOTA (+6.3% vs single-agent) |
| MMMU-Pro | 78.5% | Competitive with GPT-5.2 (79.5%) |
| VideoMMUU | 86.6% | SOTA video understanding |
| LVBench | 75.9% | SOTA long-video comprehension |
| OCRBench | 92.3% | SOTA OCR |
| HLE-Full w/ tools | 50.2% | Outperforms Gemini 3 Pro (45.8%) |
| SWE-Bench Verified | 76.8% | Competitive with frontier models |

## 🏗️ Architecture

```
Total Parameters:        1.04T (Mixture-of-Experts)
Activated Parameters:     32B per token
Experts:                  384 total, 8 activated (48:1 sparsity)
Vision Encoder:           MoonViT-3D (NaViT-based, native-resolution)
Context Length:           Up to 262k tokens
Sequence Length:          4096 → 32768 → 262144
Video Compression:        4× temporal (patch-level averaging)
Training Tokens:          ~15T vision-text tokens (joint)
```

## 🔬 Core Techniques

### 1. Native Multimodal Pre-training with Early Fusion

**Problem**: Late-stage vision token addition (50%+ ratio) causes "dip-and-recover" pattern where text performance degrades.

**Solution**: Early fusion with lower vision ratios (10%:90% vision:text) throughout entire training.

**Results**: 25.8 vision knowledge vs 24.2 for late fusion with 50% vision ratio; no representation collapse.

### 2. Zero-Vision SFT

**Problem**: VLMs don't naturally perform vision-based tool-calling; manual visual trajectories limit generalization.

**Solution**: Text-only SFT activates visual capabilities through IPython programmatic operations (pixel tasks, object localization, OCR).

**Results**: Enables diverse visual reasoning without vision data; paired with vision RL achieves robust capabilities.

### 3. Joint Multimodal Reinforcement Learning

**Problem**: Balancing text and vision performance during RL; risk of visual RL degrading text capabilities.

**Solution**: Joint RL across modalities organized by abilities (knowledge, reasoning, coding, agentic) not input type; GRM optimizes across heterogeneous traces.

**Results**: Vision RL improves text—MMLU-Pro 84.7%→86.4%, GPQA-Diamond 84.3%→86.4%.

### 4. Agent Swarm with PARL

**Problem**: Sequential agent execution leads to linear inference time scaling; complex tasks face unacceptable latency.

**Solution**: Trainable orchestrator dynamically creates frozen subagents, decomposes tasks into parallelizable subproblems; reward function includes parallel instantiation and finish rate.

**Results**: 3-4.5× latency reduction on WideSearch; BrowseComp 60.6%→78.4%, WideSearch 72.7%→79.0%.

### 5. Toggle: Token-Efficient RL

**Problem**: Length-overfitting—models trained under rigid budget constraints fail to generalize to higher compute scales.

**Solution**: Alternating optimization between budget-limited phase (conditional enforcement when accuracy > λ) and standard scaling phase.

**Results**: 25-30% reduction in output tokens with negligible performance impact; strong domain generalization.

## 📈 Training

**Pre-training**:
- Total tokens: ~15T vision-text (joint)
- Vision ratio: 10%:90% (early fusion, constant)
- Stages: ViT Training (1T) → Joint Pre-training (15T) → Mid-training (500B→200B)

**Post-training**:
- SFT: Large-scale instruction-tuning
- RL: Joint multimodal RL + PARL for Agent Swarm
- Toggle algorithm for token efficiency

**Infrastructure**:
- Hardware: NVIDIA H800 clusters
- Parallelism: 16-way PP + 16-way EP + ZeRO-1
- DEP: 90% efficiency vs text-only

## ⚠️ Limitations

1. Computer-Use: OSWorld-Verified (63.3%) lags Claude Opus 4.5 (66.3%)
2. SimpleQA Verified: 36.9% significantly lower than GPT-5.2 (72.1%)
3. Token efficiency: Toggle requires careful tuning of λ and budget estimation
4. Parallel agent complexity: Learned policies for dynamic subagent creation; credit assignment challenges
5. Vision RL data quality: Lack of high-quality vision data affects text-vision SFT

## 🔗 Links

- **Paper**: https://arxiv.org/abs/2602.02276
- **PDF**: https://arxiv.org/pdf/2602.02276
- **Model**: https://huggingface.co/moonshotai/Kimi-K2.5
- **Team**: Moonshot AI (https://www.moonshot.cn)

---

**Tags**: #multimodal #agentic-ai #agent-swarm #RL #joint-optimization #vision
