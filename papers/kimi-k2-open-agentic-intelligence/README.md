# Kimi K2: Open Agentic Intelligence

**arXiv**: [2507.20534](https://arxiv.org/abs/2507.20534) | **Date**: July 2025 (v2: Feb 2026) | **Team**: Moonshot AI

[📄 PDF](https://arxiv.org/pdf/2507.20534) | [🤗 Model](https://huggingface.co/moonshotai/Kimi-K2-Instruct) | [💻 Checkpoint Engine](https://github.com/MoonshotAI/checkpoint-engine)

---

## 🎯 One-Line Summary

A 1.04T-parameter MoE model achieving SOTA agentic capabilities through **MuonClip optimizer** (stable training), **large-scale tool-use synthesis**, and **general RL** (verifiable + self-critique rewards).

## 📊 Key Results (Non-Thinking)

| Benchmark | Score | Rank |
|-----------|-------|------|
| SWE-Bench Verified | 65.8% (71.6% multi) | #1 Open-source |
| SWE-Bench Multilingual | 47.3% | #1 Open-source |
| τ²-Bench | 66.1 | #1 Open-source |
| ACEBench (En) | 76.5 | #1 Open-source |
| LiveCodeBench v6 | 53.7% | #1 All models |
| OJBench | 27.1% | #1 All models |
| AIME 2025 | 49.5% | Top tier |
| GPQA-Diamond | 75.1% | Top tier |
| LMSYS Arena | #5 overall | #1 Open-source |

## 🏗️ Architecture

```
Total Parameters:        1.04T (↑54% vs DeepSeek-V3)
Activated Parameters:     32.6B (↓13% vs DeepSeek-V3)
Total Experts:              384 (↑50% vs DeepSeek-V3)
Active Experts/Token:         8
Attention Heads:             64 (↓50% vs DeepSeek-V3)
Sparsity:                   48 (ratio: 384/8)
Layers:                     61
Context Length:         128k (extended via YaRN)
```

**Key Design**: Higher sparsity (48) → 1.69× FLOP reduction vs sparsity 8 at same loss

## 🔬 Core Techniques

### 1. MuonClip: Stable Training at Scale

**Problem**: Muon optimizer causes exploding attention logits (>1000) at 1T+ scale

**Solution**: QK-Clip mechanism
- Monitor max attention logit per head: `S_max^h = max(Q_i · K_j) / √d`
- When `S_max^h > τ` (τ=100), rescale weights:
  ```
  γ = τ / S_max^h
  q^C ← q^C · √γ
  k^C ← k^C · √γ
  q^R ← q^R · γ
  ```
- **Per-head clipping**: Only affects problematic heads (minimal intervention)
- **Self-deactivating**: Stops triggering after ~70k steps

**Results**:
- ✅ **Zero loss spikes** across 15.5T tokens
- ✅ No performance degradation

### 2. Token Efficiency via Synthetic Data

**Knowledge Rephrasing**:
- Style- and perspective-diverse LLM prompting
- Chunk-wise autoregressive generation
- Fidelity verification

**Results** (SimpleQA):
- 10 rephrasings × 1 epoch: **28.94%**
- Raw data × 10 epochs: **23.76%**

### 3. Agentic Data Synthesis Pipeline

**Three-Stage Process**:
1. **Tool Spec Generation** (23,000+ tools):
   - 3,000+ real MCP tools from GitHub
   - 20,000+ synthetic tools (hierarchical domain evolution)

2. **Agent & Task Generation**:
   - Thousands of diverse agents (varied system prompts)
   - Rubric-based tasks (explicit success criteria)

3. **Trajectory Generation**:
   - User simulation (LLM-generated personas)
   - Tool simulator (stateful, stochastic)
   - Quality filter (LLM judge vs rubrics)
   - **Hybrid**: Real execution sandboxes for coding

### 4. General RL Framework

**A. Verifiable Rewards (RLVR)**:
- Math/STEM: Moderate difficulty, diverse coverage
- Instruction Following: Code interpreter + LLM judge + hack-check
- Faithfulness: Sentence-level judge
- Coding: Competition problems + GitHub PRs
- Safety: Adversarial prompt evolution

**B. Self-Critique Rubric Reward**:
- Actor generates responses, Critic ranks via pairwise evaluation
- Rubrics: Core (clarity, fluency) + Prescriptive (no reward hacking) + Task-specific
- **Closed-loop**: Critic updated with verifiable signals (grounds subjective in objective)

**Algorithm Enhancements**:
- Budget control (token limits)
- PTX loss (prevent forgetting)
- Temperature decay (exploration → exploitation)

### 5. RL Infrastructure

**Checkpoint Engine**:
- **30 seconds** for full 1T parameter update
- Open-sourced: github.com/MoonshotAI/checkpoint-engine

**Colocated Architecture**:
- Training + inference engines on same workers
- GPU resource sharing (offload when idle)

## 📈 Training

**Pre-training**:
- Tokens: 15.5T
- Optimizer: MuonClip (τ=100, weight decay=0.1)
- Schedule: WSD (2e-4 constant → cosine decay → 7e-6 annealing)
- Context: 4k → 128k (YaRN extension)
- Batch: 67M tokens (global)

**Infrastructure**:
- Parallelism: 16-way PP + 16-way EP + ZeRO-1 DP
- Memory: Selective recomputation, FP8 storage, CPU offload
- Hardware: NVIDIA H800 cluster

## ⚠️ Limitations

1. Verbose outputs on hard reasoning or unclear tool definitions
2. Performance decline when tool use unnecessarily enabled
3. One-shot prompting less effective than agentic frameworks for software projects
4. Potential overconfidence from rubric constraints

## 🚀 Impact

- ✅ First stable 1T+ parameter training (zero loss spikes)
- ✅ State-of-the-art agentic capabilities (tool use, coding)
- ✅ Token-efficient optimization (Muon + QK-Clip)
- ✅ Open-source checkpoints + infrastructure

## 🔗 Links

- **Paper**: https://arxiv.org/abs/2507.20534
- **PDF**: https://arxiv.org/pdf/2507.20534
- **Hugging Face**: https://huggingface.co/moonshotai/Kimi-K2-Instruct
- **Checkpoint Engine**: https://github.com/MoonshotAI/checkpoint-engine
- **Moonshot AI**: https://www.moonshot.cn

---

**Tags**: #moE #optimizer #agentic-ai #tool-use #RL #synthetic-data #open-source
