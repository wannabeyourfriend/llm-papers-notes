# Mistral Large 3: 675B Open-Weight Frontier Model

**Date**: December 2025 | **Team**: Mistral AI (Paris, France)

[🤗 Model](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) | [📝 Announcement](https://mistral.ai/news/mistral-3)

---

## 🎯 One-Line Summary

**Frontier-grade open-weight MoE model** (675B total, 41B active) achieving competitive performance with proprietary models while using Apache 2.0 license—Europe's most capable open-source LLM.

## 📊 Key Results

| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU (Multilingual) | Top-tier | Strong open model performance |
| SimpleQA | ~23.8% | Independent testing |
| SWE-Bench | Competitive | Solves software engineering tasks |
| Artificial Analysis Index | 23 | Above average |
| vs GPT-5.2 | ~92% | Claimed performance |

## 🏗️ Architecture

```
Total Parameters:        675 billion
Active Parameters:       41 billion (per token)
Architecture:            Sparse Mixture-of-Experts (MoE)
Context Window:          256k tokens
Multimodal:              Yes (vision + text)
Model Formats:           FP8, NVFP4, Eagle, GGUF
License:                 Apache 2.0 (fully open-source)
```

## 🔬 Core Techniques

### 1. Granular Sparse Mixture-of-Experts

**Problem**: How to achieve frontier performance while maintaining inference efficiency.

**Solution**: Sparse MoE with 675B total parameters but only 41B active per token, allowing specialization for different tasks.

**Results**: Frontier-grade performance with significantly lower active parameters; efficient deployment.

### 2. Open-Weight Apache 2.0 Release

**Problem**: Limited accessibility to frontier-grade models for research and commercial use.

**Solution**: Full model weights under permissive Apache 2.0 license in multiple formats.

**Results**: Widespread adoption, modification, and deployment without licensing restrictions.

### 3. Multimodal Architecture

**Problem**: Need unified model handling text and visual inputs.

**Solution**: Integrated vision capabilities within MoE framework.

**Results**: Multimodal understanding without separate vision models.

### 4. Large Context Window (256k)

**Problem**: Limited context capacity in previous models.

**Solution**: Extended context window to 256k tokens.

**Results**: Long-document analysis, extended conversations, RAG applications.

### 5. NVIDIA Hardware Optimization

**Problem**: Efficient deployment on modern GPU infrastructure.

**Solution**: Comprehensive stack optimizations for NVIDIA H200 and GB200 NVL72 platforms.

**Results**: Best-in-class performance on GB200 NVL72 with multiple quantization formats.

## 📈 Training

**Infrastructure**: ~3,000 NVIDIA H200 GPUs

**Hardware Partners**: NVIDIA (optimized for H200, GB200 NVL72)

**Training Duration**: Frontier-class training run (specifics not disclosed)

**Release**: Open weights (Apache 2.0)

**Note**: Detailed training methodology, tokens, data composition not publicly disclosed.

## ⚠️ Limitations

1. Factuality: SimpleQA ~23.8% suggests limitations in factual accuracy
2. Arena Hard: Ranked #16 (outside top 10) on instruction-following benchmark
3. Resource requirements: 256k context and 675B total parameters require substantial compute
4. Training transparency: Limited public information about training data composition
5. Benchmark variance: Performance varies significantly across different benchmark types

## 🔗 Links

- **Announcement**: https://mistral.ai/news/mistral-3
- **Model**: https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512
- **NVFP4**: https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4
- **Eagle**: https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle
- **Documentation**: https://docs.mistral.ai/models/mistral-large-3-25-12
- **NVIDIA Blog**: https://developer.nvidia.com/blog/nvidia-accelerated-mistral-3-open-models/
- **vLLM Guide**: https://docs.vllm.ai/projects/recipes/en/latest/Mistral/Mistral-Large-3.html
- **GGUF**: https://huggingface.co/unsloth/Mistral-Large-3-675B-Instruct-2512-GGUF

---

**Tags**: #MoE #open-weights #Apache-2.0 #multimodal #frontier-model #Mistral
