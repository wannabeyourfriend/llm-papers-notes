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

- **Papers Covered**: 1
- **Last Updated**: February 2026
- **Focus Areas**: Agentic AI, Training Techniques, Architecture

## 🔗 Related Resources

- [Papers with Code](https://paperswithcode.com/)
- [arXiv](https://arxiv.org/list/cs.AI/recent)
- [Hugging Face Papers](https://huggingface.co/papers)

---

**Maintained by**: [Zixuan Wang](https://github.com/wannabeyourfriend)
**License**: MIT
**Last Updated**: February 2026
