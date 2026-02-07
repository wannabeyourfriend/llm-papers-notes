# Kimi K2: Open Agentic Intelligence - Technical Summary

## 📄 Paper Metadata

- **Title**: Kimi K2: Open Agentic Intelligence
- **arXiv**: [2507.20534](https://arxiv.org/abs/2507.20534)
- **PDF**: [arxiv.org/pdf/2507.20534](https://arxiv.org/pdf/2507.20534)
- **Authors**: Kimi Team (Moonshot AI) - Yulun Du et al.
- **Version**: v2 (February 3, 2026)
- **Institution**: Moonshot AI
- **Release**: [Hugging Face](https://huggingface.co/moonshotai/Kimi-K2-Instruct)
- **GitHub**: [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)

## 🎯 One-Line Summary

A 1.04 trillion-parameter Mixture-of-Experts model achieving state-of-the-art agentic capabilities through stable training (MuonClip optimizer), large-scale synthetic tool-use data, and a general RL framework combining verifiable rewards with self-critique.

## 🏆 Key Results (Non-Thinking)

| Benchmark | Score | Rank |
|-----------|-------|------|
| **SWE-Bench Verified** | 65.8% (71.6% multi-attempt) | #1 Open-source |
| **SWE-Bench Multilingual** | 47.3% | #1 Open-source |
| **τ²-Bench** | 66.1 | #1 Open-source |
| **ACEBench (En)** | 76.5 | #1 Open-source |
| **LiveCodeBench v6** | 53.7% | #1 All models |
| **OJBench** | 27.1% | #1 All models |
| **AIME 2025** | 49.5% | Top tier |
| **GPQA-Diamond** | 75.1% | Top tier |
| **LMSYS Arena** | #5 overall | #1 Open-source |

## 📊 Model Architecture

### Specifications

```
Total Parameters:        1.04T (↑54% vs DeepSeek-V3)
Activated Parameters:     32.6B (↓13% vs DeepSeek-V3)
Total Experts:              384 (↑50% vs DeepSeek-V3)
Active Experts/Token:         8
Attention Heads:             64 (↓50% vs DeepSeek-V3)
Sparsity:                   48 (ratio: 384/8)
Hidden Dimension:         7168
MoE Hidden Dimension:     2048
Layers:                     61
Context Length:        128k (4k pre-training, extended via YaRN)
```

### Key Design Decisions

1. **Increased Sparsity (48)**: Scaling law shows better performance under fixed FLOPs
   - Sparsity 48 achieves 1.69×, 1.39×, 1.15× FLOP reduction vs sparsity 8, 16, 32
   - Validated loss: 1.5 at same compute

2. **Fewer Attention Heads (64)**:
   - 83% inference FLOP reduction at 128k context vs 128 heads
   - Only 0.5-1.2% validation loss improvement from doubling heads (not worth cost)

3. **Multi-Head Latent Attention (MLA)**: From DeepSeek-V3 design

## 🔬 Core Technical Contributions

### 1. MuonClip: Stable Training at Scale

**Problem**: Muon optimizer causes exploding attention logits (>1000) when scaling to 1T+ parameters

**Solution**: QK-Clip mechanism
- Monitor max attention logit per head: `S_max^h = max(Q_i · K_j) / √d`
- When `S_max^h > τ` (τ=100):
  ```
  γ = τ / S_max^h
  q^C ← q^C · √γ
  k^C ← k^C · √γ
  q^R ← q^R · γ
  k^R ← k^R (untouched - shared)
  ```
- **Per-head clipping**: Only affects problematic heads (minimal intervention)
- **Self-deactivating**: Stops triggering after ~70k steps

**Results**:
- ✅ **Zero loss spikes** across 15.5T tokens
- ✅ No performance degradation (ablation studies confirm)
- ✅ Maximum logits capped at 100, then naturally decay

**Why Muon Explodes More Than Adam**:
- Muon updates have **higher effective rank** (all singular values equal)
- Adam updates are **low-rank** (few large singular values dominate)
- Higher rank → higher probability of singular vector alignment → additive singular value growth
- Attention compounds this: `W_q · W_k^T` squares spectral norm

### 2. Token Efficiency via Synthetic Data

**Knowledge Rephrasing**:
- Style- and perspective-diverse LLM prompting
- Chunk-wise autoregressive generation (preserves long-document coherence)
- Fidelity verification checks semantic alignment
- **Results**:
  - 10 rephrasings × 1 epoch: **28.94%** SimpleQA accuracy
  - 1 rephrasing × 10 epochs: **27.39%**
  - Raw data × 10 epochs: **23.76%**

**Mathematics Rephrasing**:
- Convert to "learning-note" style (SwallowMath methodology)
- Translate high-quality non-English materials to English

**Impact**: Enables 15.5T token training with limited high-quality data

### 3. Agentic Data Synthesis Pipeline

**Three-Stage Process**:

1. **Tool Spec Generation** (23,000+ tools):
   - 3,000+ real MCP tools from GitHub
   - 20,000+ synthetic tools via hierarchical domain evolution
   - Categories: financial trading, software, robot control, etc.

2. **Agent & Task Generation**:
   - Thousands of diverse system prompts (personas, expertise, behaviors)
   - Rubric-based tasks with explicit success criteria
   - Range from simple to complex

3. **Trajectory Generation**:
   - **User Simulation**: LLM-generated personas with distinct communication styles
   - **Tool Simulator**: Maintains state, introduces controlled stochasticity
   - **Quality Filter**: LLM judge evaluates against rubrics
   - **Real Execution**: Hybrid with coding sandboxes for authenticity

**Scale**: Tens of thousands of diverse, high-quality demonstrations

### 4. General RL Framework

**A. Verifiable Rewards (RLVR)**:
- Math/STEM: Moderate difficulty, diverse coverage
- Instruction Following: Code interpreter + LLM-as-judge + hack-check
- Faithfulness: Sentence-level judge for unsupported claims
- Coding: Competition problems + GitHub PRs/issues
- Safety: Adversarial prompt evolution (attack → target → judge)

**B. Self-Critique Rubric Reward**:

*Actor-Critic Loop*:
1. Actor generates responses for general prompts
2. Critic ranks via pairwise evaluation against rubrics:
   - **Core**: Clarity, relevance, fluency, objectivity
   - **Prescriptive**: No initial praise, no explicit justification
   - **Human**: Task-specific criteria
3. Critic continuously updated with verifiable signals
4. Grounds subjective judgments in objective reality

**Algorithm Enhancements**:
- **Budget Control**: Per-sample token limits
- **PTX Loss**: High-quality data mixture (prevents forgetting)
- **Temperature Decay**: High exploration → low exploitation

### 5. RL Infrastructure Innovations

**Checkpoint Engine**:
- Manages parameter state across cluster
- **30 seconds** for full 1T parameter update
- Pipelined: H2D → Broadcast → Reload
- Open-sourced: github.com/MoonshotAI/checkpoint-engine

**Colocated Architecture**:
- Training + inference engines on same workers
- GPU resources released/offloaded when engine idle
- Centralized controller orchestrates iterations

**Agentic Optimizations**:
- Heavy environments as dedicated services
- Large concurrent rollouts (amortize latency)
- Partial rollout (pause/resume long-horizon tasks)
- OpenAI Gym-like unified interface

## 📈 Training Details

### Pre-training

**Optimizer**: MuonClip (τ=100, weight decay=0.1)
**Schedule**: WSD (warmup → constant → cosine decay)
```
First 10T tokens:   2e-4 (after 500-step warmup)
Next 5.5T tokens:   cosine decay 2e-4 → 2e-5
Annealing:          2e-5 → 7e-6
                    (400B tokens @ 4k + 60B @ 32k)
```
**Context**: 4,096 → 128,000 (via YaRN)
**Batch Size**: 67M tokens (global)
**Total Tokens**: 15.5T

### Infrastructure

**Parallelism** (any multiple of 32 nodes):
- 16-way Pipeline Parallelism (virtual stages)
- 16-way Expert Parallelism
- ZeRO-1 Data Parallelism

**Memory Optimization**:
- Selective recomputation (LayerNorm, SwiGLU, MLA up, MoE down)
- FP8 storage for MoE up-projections and SwiGLU inputs
- Activation CPU offload with streaming overlap

**Communication**:
- Interleaved 1F1B with EP all-to-all overlap
- Decoupled weight-gradient ∥ PP communication
- Smallest EP size (16) for full overlap

## 🔍 Evaluation Details

### Benchmarks

**Coding**:
- LiveCodeBench v6 (Aug 2024 - May 2025)
- OJBench (competition-level)
- SWE-Bench Verified (agentless + agentic)
- SWE-Bench Multilingual
- SWE-Lancer
- MultiPL-E, TerminalBench, PaperBench, Aider-Polyglot

**Tool Use**:
- τ²-Bench (multi-turn tool calling)
- ACEBench (3,000+ tools)

**Reasoning**:
- AIME 2024/2025, MATH-500, HMMT 2025
- CNMO 2024, PolyMath-en
- ZebraLogic, AutoLogi
- GPQA-Diamond, SuperGPQA
- Humanity's Last Exam

**Long Context**:
- MRCR (retrieval)
- DROP, FRAMES, LongBench v2 (reasoning)

**Factuality**:
- FACTS Grounding
- Vectara Hallucination Leaderboard
- FaithJudge

**General**:
- MMLU, MMLU-Redux, MMLU-Pro
- IFEval, Multi-Challenge
- SimpleQA, LiveBench

### Settings

- All models in **non-thinking mode**
- Output capped at 8,192 tokens (16,384 for SWE-Bench Verified)
- Context window: 128k
- Repeated sampling (Avg@k) for high-variance benchmarks
- SWE-Bench: Agentless (single patch) + Agentic (bash/editor tools)

## ⚠️ Limitations

1. **Verbose outputs** on hard reasoning or unclear tool definitions
2. **Performance decline** when tool use unnecessarily enabled
3. **One-shot prompting** less effective than agentic coding frameworks
4. **Potential overconfidence** from rubric constraints (may disfavor epistemic humility)

## 🚀 Impact & Significance

**Technical Achievements**:
- ✅ First stable 1T+ parameter training (zero loss spikes)
- ✅ Token-efficient optimization (Muon + QK-Clip)
- ✅ State-of-the-art agentic capabilities (tool use, coding)
- ✅ Scalable synthetic data pipeline
- ✅ General RL framework (verifiable + subjective)

**Open Source Contributions**:
- Base + post-trained checkpoints released
- Checkpoint engine open-sourced
- Enables agentic intelligence research at scale

**Paradigm Shift**:
- Demonstrates **agentic intelligence** as the next frontier
- Shows token efficiency > scale (limited high-quality data era)
- Proves synthetic + real data hybrid is optimal
- Establishes self-critique as viable for general RL

## 📚 Key References

**Optimizer Background**:
- [Muon](https://arxiv.org/abs/2408.04455): Token-efficient optimizer
- [Moonlight](https://arxiv.org/abs/2501.xxx): Muon scaling experiments

**Agentic Data**:
- [ACEBench](https://arxiv.org/abs/2501.xxx): Comprehensive tool-use benchmark
- [ToolLLM](https://arxiv.org/abs/2307.xxx): Teaching thousands of tools
- [AgentInstruct](https://arxiv.org/abs/2406.xxx): Synthetic agent data

**RL Methods**:
- [Kimi K1.5](https://arxiv.org/abs/2501.xxx): Verifiable rewards foundation
- [FACTS Grounding](https://arxiv.org/abs/2406.xxx): Faithfulness evaluation

## 📝 Notes Structure

- `notes.md` - Detailed section-by-section notes
- `implementation-notes.md` - Practical implementation insights
- `architecture/` - Diagrams and visualizations
- `references.md` - Related papers and future work

---

**Paper Status**: ✅ Complete
**Notes Status**: ✅ Comprehensive
**Last Updated**: February 2026
