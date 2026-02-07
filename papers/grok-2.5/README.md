# GROK-2: Open-Weight 270B Mixture-of-Experts Model

**Date**: August 2024 (release), August 2025 (open-source) | **Team**: xAI

[🤗 Model](https://huggingface.co/xai-org/grok-2) | [📝 Announcement](https://x.ai/news/grok-2)

---

## 🎯 One-Line Summary

**270B-parameter MoE language model** with sparse activation (115B active parameters) achieving competitive reasoning performance, bridging gap between proprietary and open-source AI.

## 📊 Key Results

| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU | 87.5% | Competitive with GPT-4 class |
| MMLU-Pro | 75.5% | More robust variant |
| GPQA | 56.0% | Graduate-level QA |
| GSM8K | ~92% | Strong math (vs 62.9% Grok-1) |
| HumanEval | High | Strong coding capabilities |

## 🏗️ Architecture

```
Total Parameters:        269,515,497,472 (~270B)
Activated Parameters:    ~115B per forward pass
Architecture:            Mixture-of-Experts (MoE)
Number of Experts:       8
Active Experts:          2 per forward pass
Activation Rate:         42.6% (115B/270B)
Model Weight Size:       ~500GB
Precision:               FP16 optimized
```

## 🔬 Core Techniques

### 1. Mixture-of-Experts Architecture

**Problem**: Dense models with 270B parameters computationally prohibitive for inference.

**Solution**: Sparse MoE with 8 experts, activating only 2 experts per forward pass.

**Results**: Large model capacity (270B total) with efficient inference (115B active); ~57% computational cost reduction.

### 2. Expert Routing and Selection

**Problem**: Need efficient selection of which experts should process each token.

**Solution**: Learned routing mechanism activating most relevant 2 out of 8 experts per token.

**Results**: Optimized parameter utilization; faster inference vs dense equivalents.

### 3. Multi-Benchmark Training Optimization

**Problem**: Achieving strong performance across diverse reasoning tasks (math, coding, knowledge).

**Solution**: Training methodology focused on reasoning-intensive datasets and academic benchmarks.

**Results**: Competitive on MMLU (87.5%), MMLU-Pro (75.5%), GPQA (56%); GSM8K improved 62.9% → ~92%.

### 4. Open-Weight Release

**Problem**: Limited access to state-of-the-art model weights for research.

**Solution**: Released full model weights (~500GB) under community license on HuggingFace.

**Results**: Enables research community to study, fine-tune, and deploy capable open-weight model.

## 📈 Training

**Training Period**: 2024

**Focus**: Text-based tasks with emphasis on reasoning capabilities

**Infrastructure**: Optimized for NVIDIA H100/H200 GPUs (1979 TFLOPS FP16)

**Note**: Specific training methodology (tokens, schedule, data) not publicly disclosed; xAI has not released comprehensive technical documentation.

## ⚠️ Limitations

1. Limited documentation: No comprehensive technical report or training details published
2. License restrictions: Grok 2 Community License more restrictive than fully open-source (Apache/MIT)
3. Infrastructure requirements: ~500GB model size requires significant storage and compute
4. Training transparency: No published paper on training methodology, data composition, or hyperparameters
5. Closed development: Model developed behind closed doors without peer review

## 🔗 Links

- **HuggingFace**: https://huggingface.co/xai-org/grok-2
- **Discussions**: https://huggingface.co/xai-org/grok-2/discussions
- **Announcement**: https://x.ai/news/grok-2
- **License**: https://huggingface.co/xai-org/grok-2/blob/main/LICENSE

---

**Tags**: #MoE #open-weights #reasoning #xAI #large-model
