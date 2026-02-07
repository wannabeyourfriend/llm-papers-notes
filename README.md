# LLM Papers Technical Notes

Concise technical summaries of cutting-edge Large Language Model research papers, focusing on core techniques, performance, and official resources.

## 📚 Purpose

Quick reference for impactful LLM papers with:
- **Core Techniques** - Key innovations at a glance
- **Performance** - Benchmark results and comparisons
- **Links** - Official codebases and resources

No fluff, just essentials.

## 📖 Papers

### [1. Kimi K2: Open Agentic Intelligence](./papers/kimi-k2-open-agentic-intelligence/)

**arXiv**: 2507.20534 | **Date**: July 2025 | **Team**: Moonshot AI

**Core Techniques**:
- **MuonClip**: Stable 1T+ parameter training (zero loss spikes)
- **Agentic Data Synthesis**: 23K+ tools, multi-stage pipeline
- **General RL**: Verifiable rewards + self-critique

**Key Results**:
- SWE-Bench Verified: **65.8%** (#1 open-source)
- τ²-Bench: **66.1** (#1 open-source)
- LiveCodeBench v6: **53.7%** (#1 all models)

**Links**:
- [Paper](https://arxiv.org/abs/2507.20534) | [PDF](https://arxiv.org/pdf/2507.20534)
- [🤗 Model](https://huggingface.co/moonshotai/Kimi-K2-Instruct)
- [💻 Checkpoint Engine](https://github.com/MoonshotAI/checkpoint-engine)

[→ Read Summary](./papers/kimi-k2-open-agentic-intelligence/)

---

### [2. Kimi K2.5: Visual Agentic Intelligence](./papers/kimi-k2.5/)

**arXiv**: 2602.02276 | **Date**: February 2026 | **Team**: Moonshot AI

**Core Techniques**:
- **Joint Text-Vision Optimization**: Early fusion (10%:90% ratio)
- **Agent Swarm**: Parallel multi-agent orchestration (PARL)
- **Toggle RL**: Token-efficient training

**Key Results**:
- AIME 2025: **96.1%** (SOTA)
- BrowseComp: **78.4%** (#1, surpasses GPT-5.2 Pro)
- LiveCodeBench v6: **85.0%** (outperforms Claude Opus 4.5)

**Links**:
- [Paper](https://arxiv.org/abs/2602.02276) | [PDF](https://arxiv.org/pdf/2602.02276)
- [🤗 Model](https://huggingface.co/moonshotai/Kimi-K2.5)

[→ Read Summary](./papers/kimi-k2.5/)

---

### [3. MIROTHINKER V1.0: Interactive Scaling](./papers/mirothinker-v1.0/)

**arXiv**: 2511.11793 | **Date**: November 2025 | **Team**: MiroMind AI

**Core Techniques**:
- **Interactive Scaling**: Deeper agent-environment interactions (600 tool calls)
- **GRPO RL**: Environment feedback for error correction
- **Recency-Based Context**: K=5 retention for long trajectories

**Key Results**:
- GAIA-Text-Only: **81.9%** (#1 open-source)
- BrowseComp: **47.1%** (#1 open-source)
- SEAL-0: **51.0%** (comparable to GPT-5-high)

**Links**:
- [Paper](https://arxiv.org/abs/2511.11793) | [PDF](https://arxiv.org/pdf/2511.11793.pdf)
- [🤗 Model](https://huggingface.co/miromind-ai/MiroThinker-v1.0-72B)
- [💻 Code](https://github.com/MiroMindAI/MiroThinker)

[→ Read Summary](./papers/mirothinker-v1.0/)

---

### [4. RING-1T: Trillion-Scale Thinking Model](./papers/ring-1t/)

**arXiv**: 2510.18855 | **Date**: October 2025 | **Team**: L Team (inclusionAI)

**Core Techniques**:
- **Multi-Stage RL**: SFT → reasoning RL → general RL
- **Iterative Self-Evolution**: Automated data curation
- **Long-CoT Training**: 1.2M reasoning samples

**Key Results**:
- AIME 2025: **80.0%** (SOTA)
- MATH: **95.3%** (SOTA)
- GPQA Diamond: **76.1%** (SOTA)

**Links**:
- [Paper](https://arxiv.org/abs/2510.18855) | [PDF](https://arxiv.org/pdf/2510.18855.pdf)
- [🤗 Model](https://huggingface.co/inclusionAI/Ring-1T)
- [💻 Code](https://github.com/inclusionAI/asystem-amem)

[→ Read Summary](./papers/ring-1t/)

---

### [5. MIMO-V2-FLASH: Extreme Context Training](./papers/mimo-v2-flash/)

**arXiv**: 2601.02780 | **Date**: January 2026 | **Team**: Project Mc2, McQUIC

**Core Techniques**:
- **128M Token Context**: Extreme length training
- **FlashAttention-3**: O(N) memory with sequence parallelism
- **3D Hybrid Parallelism**: 8-way data + tensor + pipeline

**Key Results**:
- RULER (128K): **81.4%** (SOTA, +6.2% over GPT-4o)
- Needle-in-Haystack (128M): **100%** retrieval
- Training efficiency: **2.3×** throughput improvement

**Links**:
- [Paper](https://arxiv.org/abs/2601.02780) | [PDF](https://arxiv.org/pdf/2601.02780)
- [📐 Project](https://mc2-projects.github.io/mimo-v2/)

[→ Read Summary](./papers/mimo-v2-flash/)

---

### [6. GPT-OSS: Fully Open-Source Models](./papers/gpt-oss/)

**arXiv**: 2508.10925 | **Date**: August 2025 | **Team**: Multi-institutional

**Core Techniques**:
- **Full Transparency**: 2.6T tokens data + all code + 26 checkpoints
- **MoE Architecture**: 16 experts, top-2 routing
- **Open Training Pipeline**: Complete reproducibility

**Key Results**:
- GPT-OSS-120B: MMLU **80.6%** (competitive with GPT-4)
- GPT-OSS-20B: MMLU **66.8%**
- **~4.5×** reduced training costs via MoE

**Links**:
- [Paper](https://arxiv.org/abs/2508.10925) | [PDF](https://arxiv.org/pdf/2508.10925)
- [🤗 120B Model](https://huggingface.co/EleutherAI/gpt-oss-120b)
- [🤗 20B Model](https://huggingface.co/EleutherAI/gpt-oss-20b)

[→ Read Summary](./papers/gpt-oss/)

---

### [7. GROK-2: Open-Weight 270B MoE](./papers/grok-2.5/)

**Date**: August 2025 | **Team**: xAI

**Core Techniques**:
- **Mixture-of-Experts**: 8 experts, 2 active (42.6% activation)
- **Sparse Architecture**: 270B total, 115B active parameters
- **Multi-Benchmark Optimization**: Reasoning-focused training

**Key Results**:
- MMLU: **87.5%** (GPT-4 class)
- MMLU-Pro: **75.5%**
- GSM8K: **~92%** (vs 62.9% Grok-1)

**Links**:
- [🤗 Model](https://huggingface.co/xai-org/grok-2)
- [📝 Announcement](https://x.ai/news/grok-2)

[→ Read Summary](./papers/grok-2.5/)

---

### [8. Mistral Large 3: 675B Frontier Model](./papers/mistral-large-3/)

**Date**: December 2025 | **Team**: Mistral AI

**Core Techniques**:
- **Granular MoE**: 675B total, 41B active (6% activation)
- **Apache 2.0 License**: Fully open-source
- **Multimodal**: Vision + text capabilities

**Key Results**:
- MMLU: **Top-tier** (competitive with frontier models)
- ~**92%** of GPT-5.2 performance (claimed)
- **256k** context window

**Links**:
- [🤗 Model](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512)
- [📝 Announcement](https://mistral.ai/news/mistral-3)

[→ Read Summary](./papers/mistral-large-3/)

---

## 🔍 How to Use

Each paper summary includes:
1. **One-Line Summary** - What and why it matters
2. **Key Results** - Benchmark performance table
3. **Architecture** - Model specs and design decisions
4. **Core Techniques** - Main technical contributions (numbered list)
5. **Training Details** - Essential hyperparameters and infrastructure
6. **Limitations** - Known weaknesses
7. **Links** - Paper, code, model weights

## 🎯 Paper Selection Criteria

- **Novel techniques** (not just benchmarks)
- **Reproducible** (implementation details provided)
- **Paradigm-shifting** (fundamental advances)
- **Open resources** (code/models available)

## 🛠️ Adding New Papers

```bash
cd ~/llm-papers-notes/papers
mkdir new-paper-arxivid
# Create README.md following the template
git add .
git commit -m "Add [paper title]"
git push
```

### Template

```markdown
# Paper Title

**arXiv**: XXXX.XXXXX | **Date**: Month Year | **Team**: Institution

[📄 PDF](link) | [🤗 Model](link) | [💻 Code](link)

---

## 🎯 One-Line Summary

One sentence capturing the main contribution.

## 📊 Key Results

| Benchmark | Score | Rank |
|-----------|-------|------|
| Benchmark | Score | #X |

## 🏗️ Architecture

```
Key specs
```

## 🔬 Core Techniques

### 1. Technique Name

**Problem**: What issue it solves

**Solution**: How it works

**Results**: Key outcomes

## 📈 Training

**Pre-training**:
- Key hyperparameters

## 🔗 Links

- **Paper**: link
- **Code**: link

---
```

## 📊 Repository Stats

- **Papers Covered**: 8
- **Last Updated**: February 2026
- **Focus Areas**: Agentic AI, Training Techniques, Architecture, Reasoning, Long Context, Open Models

## 🔗 Related Resources

- [Papers with Code](https://paperswithcode.com/)
- [arXiv](https://arxiv.org/list/cs.AI/recent)
- [Hugging Face Papers](https://huggingface.co/papers)

---

**Maintained by**: [Zixuan Wang](https://github.com/wannabeyourfriend)
**License**: MIT
**Last Updated**: February 2026
