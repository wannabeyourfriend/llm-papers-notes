# MIROTHINKER V1.0: Interactive Scaling for Research Agents

**arXiv**: [2511.11793](https://arxiv.org/abs/2511.11793) | **Date**: November 2025 | **Team**: MiroMind AI

[📄 PDF](https://arxiv.org/pdf/2511.11793.pdf) | [🤗 Model](https://huggingface.co/miromind-ai/MiroThinker-v1.0-72B) | [💻 Code](https://github.com/MiroMindAI/MiroThinker)

---

## 🎯 One-Line Summary

Open-source research agent achieving SOTA through **interactive scaling**—training models to handle deeper agent-environment interactions (up to 600 tool calls) as third performance dimension beyond model size and context length.

## 📊 Key Results (72B Model)

| Benchmark | Score | Rank |
|-----------|-------|------|
| GAIA-Text-Only | 81.9% | #1 Open-source (+6.2 pts) |
| BrowseComp-ZH | 55.6% | #1 Open-source (+6.1 pts) |
| xbench-DeepSearch | 77.8% | #1 Open-source |
| HLE | 37.7% | #1 Open-source (+4.8 pts) |
| BrowseComp | 47.1% | #1 Open-source (+2.0 pts) |
| SEAL-0 | 51.0% | Comparable to GPT-5-high (51.4%) |
| FRAMES | 87.1% | #1 Open-source |
| WebWalkerQA | 62.1% | #1 Open-source |

## 🏗️ Architecture

```
Model Variants:        8B, 30B, 72B (Qwen2.5/Qwen3 base)
Context Window:        Up to 256K tokens
Max Tool Calls:        600 per task
Agent Paradigm:        ReAct (Reasoning + Acting)
Tools:                 Python, Linux sandbox, web search/scrape, file management
Context Management:    Recency-based retention (K=5), result truncation
```

## 🔬 Core Techniques

### 1. Interactive Scaling via GRPO RL

**Problem**: Previous agents only scaled model size or context length; test-time scaling operates in isolation with degradation risks.

**Solution**: Train via GRPO reinforcement learning to handle deeper/frequent agent-environment interactions, leveraging environment feedback for error correction.

**Results**: 8-10 point accuracy gains; RL model 41.2% vs SFT 32.2% on BrowseComp.

### 2. Recency-Based Context Management

**Problem**: Standard ReAct retains all tool outputs, wasting context on old observations.

**Solution**: Keep only recent K=5 tool responses while preserving full thought/action sequences; truncate long outputs.

**Results**: Enables 600 tool calls within 256K context without performance degradation.

### 3. MiroVerse v1.0 Data Construction

**Problem**: Need high-quality multi-hop reasoning and agentic trajectory data.

**Solution**: Two-stage synthesis—(1) MultiDocQA via knowledge graph construction from interlinked web docs with constraint obfuscation; (2) Agentic trajectories via ReAct single-agent + MiroFlow multi-agent paradigms using diverse LLMs.

**Results**: Large-scale synthetic dataset enabling SOTA across 8 benchmarks.

### 4. Three-Stage Training Pipeline

**Problem**: Need to establish agentic behaviors, refine decisions, and enable creative exploration.

**Solution**: (1) Agentic SFT on expert trajectories with data repair/filtering; (2) DPO preference optimization based on answer correctness; (3) GRPO RL with trajectory curation.

**Results**: SFT→DPO→RL gains of 15+ points on some benchmarks.

## 📈 Training

**Pre-training Base**: Qwen2.5 and Qwen3 models

**Post-training**:
- Agentic SFT on expert trajectories
- DPO preference optimization
- GRPO RL with streaming rollout acceleration

**Infrastructure**: Scalable environments supporting thousands of concurrent rollouts (web, Python, Linux VM)

**Inference**: temperature=1.0, top-p=0.95, max turns=600, retention=5

## ⚠️ Limitations

1. Tool-use efficiency: RL-tuned model makes more frequent but sometimes redundant tool invocations
2. Overlong CoT: RL encourages excessively long, repetitive reasoning chains
3. Language mixing: Non-English inputs may trigger multilingual mixing
4. Limited sandbox proficiency: Occasional timeouts, misuse of code execution for web scraping

## 🔗 Links

- **Paper**: https://arxiv.org/abs/2511.11793
- **PDF**: https://arxiv.org/pdf/2511.11793.pdf
- **Code**: https://github.com/MiroMindAI/MiroThinker
- **Model**: https://huggingface.co/miromind-ai/MiroThinker-v1.0-72B
- **Demo**: https://dr.miromind.ai

---

**Tags**: #agentic-ai #reinforcement-learning #tool-use #interactive-scaling #research-agent
