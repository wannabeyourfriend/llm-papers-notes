# Kimi K2: Detailed Notes

## Section 1: Introduction

### Core Thesis

LLM development is shifting from **static imitation learning** to **agentic intelligence**:
- Models that actively learn through interactions
- Acquire skills beyond training distribution
- Adapt behavior through experiences
- Go beyond limitations of static human-generated data

### Two Fundamental Challenges

**Pre-training Challenge**:
- Limited high-quality data available
- Must maximize **token efficiency** (learning signal per token)
- Token efficiency emerging as critical scaling coefficient

**Post-training Challenge**:
- Must transform broad priors into actionable behaviors
- Agentic capabilities (multi-step reasoning, tool use, planning) are rare in natural data
- Costly to scale with human annotation
- Need: Scalable synthesis + general RL

### K2's Three Contributions

1. **MuonClip**: Muon + QK-Clip for stable 1T+ parameter training
2. **Agentic Data Synthesis**: Systematic tool-use demonstration generation
3. **General RL Framework**: Verifiable rewards + self-critique rubric

### Results Summary

**Agentic Benchmarks**:
- τ²-Bench: 66.1
- ACEBench (En): 76.5
- SWE-Bench Verified: 65.8%
- SWE-Bench Multilingual: 47.3%

**Coding & Reasoning**:
- LiveCodeBench v6: 53.7%
- OJBench: 27.1%
- AIME 2025: 49.5%
- GPQA-Diamond: 75.1%

**LMSYS Arena** (July 17, 2025):
- #1 open-source model
- #5 overall (3,000+ votes)

---

## Section 2: Pre-training

### 2.1 Token Efficiency Motivation

**Key Insight**: With limited high-quality human data, token efficiency is the new scaling coefficient.

**Muon Advantages** (from Moonlight paper):
- Under same compute + model size = same training data
- Muon substantially outperforms AdamW
- Better token efficiency → more learning per token

### 2.2 MuonClip: QK-Clip Mechanism

#### Problem: Logit Explosion in Muon

**Symptoms**:
- Maximum attention logits exceed 1000
- Leads to training instability
- Loss spikes and occasional divergence

**Why Muon is More Prone** (Appendix analysis):

*Structural Difference in Updates*:
- Muon: **msign operation** → all singular values equal → **higher effective rank**
- Adam: skewed spectrum → few large singular values dominate → **low effective rank**

*SVD Analysis*:
```
W_t = Σ σ_i u_i v_i^T  (previous weights)
ΔW_t = Σ σ̄ ū_j v̄_j^T  (Muon updates: full rank)

Higher rank → higher probability of u_i v_i^T aligning with ū_j v̄_j^T
→ Additive singular value growth
```

*Attention-Specific Amplification*:
```
q_i · k_j = (x_i W_q) · (x_j W_k)
           = x_i (W_q W_k^T) x_j^T

Spectral norm gets SQUARED in bilinear form
→ Any singular value increase in W_q or W_k is compounded
```

#### Solution: QK-Clip

**Core Idea**: Rescale W_q and W_k post-update when max logit exceeds threshold

**Mathematical Formulation**:
```
For each attention head h:
  S_max^h = max_{X∈B} max_{i,j} (Q_i^h · K_j^h) / √d

If S_max^h > τ:
  γ = τ / S_max^h
  q^C ← q^C · √γ
  k^C ← k^C · √γ
  q^R ← q^R · γ
  k^R ← k^R (untouched - shared across heads)
```

**Key Design Decisions**:

1. **Per-Head Clipping** (not per-layer):
   - Only small subset of heads explode
   - Minimizes intervention on other heads
   - For MLA: clip unshared components only

2. **Threshold Selection**:
   - τ = 100 for K2
   - Aggressive thresholds (30) tested - no performance loss

3. **Minimal Intervention**:
   - Only activates when necessary
   - Deactivates after training stabilizes

**Self-Deactivation** (K2 training):
- Initial 70k steps: 12.7% of heads triggered QK-Clip
- Post 70k steps: All heads reduced S_max below 100
- QK-Clip becomes inactive

**Results**:
- Zero loss spikes across 15.5T tokens
- No performance degradation (small-scale ablations)
- Validates: safe and effective method

### 2.3 Pre-training Data: Token Utility via Rephrasing

#### Motivation

**Trade-off**:
- Single epoch: insufficient knowledge absorption
- Multi-epoch: diminishing returns + overfitting risk

**Solution**: Synthetic rephrasing amplifies tokens without overfitting

#### Knowledge Data Rephrasing

**Pipeline Components**:

1. **Style- and Perspective-Diverse Prompting**:
   - Inspired by WRAP
   - Carefully engineered prompts for linguistic diversity
   - Maintains factual integrity

2. **Chunk-Wise Autoregressive Generation**:
   - Divide long documents into segments
   - Rewrite sequentially
   - Stitch back together
   - Preserves global coherence
   - Avoids LLM output length limitations

3. **Fidelity Verification**:
   - Compare semantic alignment of rephrased vs original
   - Initial quality control before training

**Results** (SimpleQA Accuracy):

| # Rephrasings | # Epochs | Accuracy |
|--------------|----------|----------|
| 0 (raw) | 10 | 23.76% |
| 1 | 10 | 27.39% |
| 10 | 1 | **28.94%** |

**Key Finding**: Diverse rephrasing × 1 epoch > raw × 10 epochs

**Application**: Extended to other large knowledge corpora (max 2× per corpus)

#### Mathematics Data Rephrasing

**Method**:
- Rewrite to "learning-note" style (SwallowMath methodology)
- Translate high-quality non-English math materials to English

**Status**: Promising results, but active research area

**Challenges**:
- Generalizing to diverse domains without factual errors
- Minimizing hallucinations
- Scalability to large datasets

#### Overall Pre-training Corpus

**Total**: 15.5 trillion tokens

**Domains**:
- Web Text
- Code
- Mathematics
- Knowledge

**Methodology**:
- Rigorous correctness and quality validation
- Targeted data experiments for diversity + effectiveness

### 2.4 Model Architecture

#### Specifications

```
Total Parameters:        1.04T
Activated Parameters:     32.6B
Total Experts:              384
Active Experts/Token:         8
Shared Experts:               1
Attention Heads:             64
Layers:                     61
Hidden Dimension:         7168
MoE Hidden Dimension:     2048
```

#### Sparsity Scaling Law

**Definition**: Sparsity = Total Experts / Active Experts

**Finding** (controlled experiments):
- Fixed activated parameters (constant FLOPs)
- Increasing total experts (higher sparsity) → lower loss

**FLOP Reduction** (at validation loss 1.5):
- Sparsity 48 vs 8: **1.69×** reduction
- Sparsity 48 vs 16: **1.39×** reduction
- Sparsity 48 vs 32: **1.15×** reduction

**Trade-off**: Higher sparsity → better performance + ↑ infrastructure complexity

**K2 Choice**: Sparsity 48 (8/384 active)
- Balances performance with cost
- Outperforms sparsity 8, 16, 32

#### Attention Heads vs Inference Cost

**Problem**: DeepSeek-V3 uses 128 heads (~2× layers)
- At 128k context: 83% inference FLOP increase (64 → 128 heads)

**Experiment**: Doubled heads vs baseline (various training FLOPs)

**Results**: Only 0.5-1.2% validation loss improvement

**Decision**: 64 heads (not 128)
- Marginal gains don't justify inference cost
- Especially important for agentic applications (long context)

#### Comparison: K2 vs DeepSeek-V3

| Metric | DeepSeek-V3 | Kimi K2 | Δ |
|--------|-------------|---------|---|
| Layers | 61 | 61 | = |
| Total Params | 671B | 1.04T | ↑ 54% |
| Active Params | 37B | 32.6B | ↓ 13% |
| Experts | 256 | 384 | ↑ 50% |
| Active/Token | 8 | 8 | = |
| Attention Heads | 128 | 64 | ↓ 50% |
| Dense Layers | 3 | 1 | ↓ 67% |
| Expert Grouping | Yes | No | - |

### 2.5 Training Infrastructure

#### Compute Cluster

**Hardware**: NVIDIA H800 GPUs
**Node Config**:
- 2 TB RAM per node
- 8 GPUs (NVLink + NVSwitch within node)
- 8×400 Gbps RoCE interconnects (between nodes)

#### Parallelism Strategy

**Goal**: Flexible training on any multiple of 32 nodes

**Strategy**:
- 16-way Pipeline Parallelism (PP) with virtual stages
- 16-way Expert Parallelism (EP)
- ZeRO-1 Data Parallelism

**Memory** (256 GPUs in model-parallel group):
- BF16 parameters + FP32 gradient buffer = ~6 TB
- Each GPU holds ~30 GB for all states
- Remaining GPU memory → activations

**Research Efficiency**: Same configuration for small + large experiments

#### Communication Optimizations

**EP Overlap with Interleaved 1F1B**:
- Increase warm-up micro-batches
- Overlap EP all-to-all with computation
- Standard interleaved 1F1B schedule

**Why Not DualPipe?** (DeepSeek-V3 approach)
- Doubles memory for parameters + gradients
- Requires increased parallelism to compensate
- Prohibitively expensive for 1T+ parameters

**PP Communication Overhead Reduction**:
- Decouple weight-gradient computation from backward pass
- Execute ∥ with PP communication
- All PP communications overlapped (except warm-up)

**Smaller EP Size (16)**:
- K2 has 64 heads (vs 128 in V3)
- Reduced attention time → minimize EP operation time
- EP=16: Smallest feasible for full overlap
- Relaxes expert-balance constraints
- Near-optimal speed without tuning

#### Activation Reduction

**Techniques**:

1. **Selective Recomputation**:
   - LayerNorm, SwiGLU
   - MLA up-projections
   - MoE down-projections (prevents early-stage crashes)

2. **FP8 Storage** (insensitive activations):
   - MoE up-projections inputs
   - SwiGLU inputs
   - Compressed to FP8-E4M3 (1×128 tiles, FP32 scales)
   - Not applied in computation (risk of degradation)

3. **Activation CPU Offload**:
   - All remaining activations
   - Copy engine streams offload/onload
   - Overlaps with computation + communication
   - 1F1B: Offload previous forward, prefetch next backward

**Result**: Activation memory fits within GPU constraints

### 2.6 Training Recipe

**Context Window**: 4,096 tokens
**Optimizer**: MuonClip (τ=100)
**Learning Rate**: WSD schedule
```
Steps 1-500:           Warmup to 2e-4
Next 10T tokens:       2e-4 (constant)
Next 5.5T tokens:      Cosine decay 2e-4 → 2e-5
Annealing:
  - 400B @ 4k:        2e-5 → 7e-6
  - 60B @ 32k:        Continue decay
```
**Weight Decay**: 0.1 (throughout)
**Global Batch Size**: 67M tokens
**Total Tokens**: 15.5T

**Long Context Extension**:
- 4k → 128k via YaRN method
- 60B tokens at 32k before extension

---

## Section 3: Post-training

### 3.1 Supervised Fine-Tuning (SFT)

**Optimizer**: Muon (recommended for K2 fine-tuning)
**Rationale**: Muon-pretrained → Muon-finetuned best performance (from K1.5 work)

**Data Principles**:
1. Maximize prompt diversity
2. Ensure high response quality

**Pipelines**:
- Domain-specific generation
- Human annotation + prompt engineering + verification
- K1.5 + in-house expert models generate candidates
- LLM/human judges evaluate + filter

### 3.2 Large-Scale Agentic Data Synthesis

#### Motivation

**Challenge**: Agentic capabilities require multi-step, interactive tool use
- Real-world environments: rich but costly to scale
- Privacy, complexity, accessibility constraints
- Need: Scalable synthesis maintaining authenticity

**Inspiration**:
- AgentInstruct, Self-Instruct: Synthetic data without real interaction
- StableToolBench, ZeroSearch: Promising results
- ACEBench: Comprehensive data synthesis framework

#### Three-Stage Pipeline

**Stage 1: Tool Spec Generation** (23,000+ tools)

*Real Tools*:
- 3,000+ MCP tools from GitHub
- High-quality specs, leverage existing work

*Synthetic Tools* (20,000+):
- Hierarchical domain evolution (WizardLM)
  - Start: Key categories (financial, software, robot control)
  - Evolve: Specific application domains
  - Synthesize: Specialized tools per domain
- Clear interfaces, descriptions, semantics

*Coverage Validation*:
- t-SNE visualization shows complementary coverage
- MCP tools: Natural clustering by source
- Synthetic tools: Systematic domain coverage

**Stage 2: Agent & Task Generation**

*Agent Diversification*:
- Thousands of distinct agents
- Varied system prompts (personas, expertise, behaviors)
- Different tool combinations from repository

*Rubric-Based Task Generation*:
- Range from simple to complex
- Paired with explicit rubrics:
  - Success criteria
  - Expected tool-use patterns
  - Evaluation checkpoints

**Stage 3: Multi-Turn Trajectory Generation**

*User Simulation*:
- LLM-generated personas
- Distinct communication styles, preferences
- Multi-turn dialogues with agents
- Naturalistic interaction patterns

*Tool Execution Environment*:
- Sophisticated simulator (functional world model)
- Executes tool calls
- Maintains + updates state after each execution
- Introduces controlled stochasticity
- Produces: successes, partial failures, edge cases

*Quality Evaluation*:
- LLM-based judge evaluates trajectories
- Assesses against task rubrics
- Only passing trajectories retained
- Allows natural variation in strategies

**Hybrid Approach**:
- Simulation: Scalability, diversity
- Real Execution: Authenticity (coding, software engineering)
- Real sandboxes: Execute actual code, genuine dev environments
- Objective metrics: Test suite pass rates
- **Result**: Best of both worlds

**Impact**:
- Large-scale rejection sampling via quality filtering
- High-quality synthetic data → significant tool-use improvements
- Effective across wide range of real-world applications

### 3.3 Reinforcement Learning

#### Motivation

**RL Advantages**:
- Better token efficiency than SFT
- Better generalization than SFT

**K2 Scaling** (beyond K1.5):
- Task diversity ↑
- Training FLOPs ↑
- Framework: Gym-like extensible system

#### A. Verifiable Rewards Gym

**Math, STEM, Logical Tasks**:

*Principles*:
1. Diverse Coverage
2. Moderate Difficulty

*Collection*:
- Expert annotations
- Internal QA extraction pipelines
- Open datasets (NuminaMath, AIMO)
- Tagging system: increase under-covered domains

*Formats*:
- Structured data (multi-hop tabular, cross-table aggregation)
- Logic puzzles (24-game, Sudoku, riddles, cryptarithms, Morse)

**Complex Instruction Following**:

*Dual-Path Verification*:
1. **Deterministic**: Code interpreter (length, style constraints)
2. **LLM-as-Judge**: Nuanced constraint understanding
3. **Hack-Check**: Detects deceptive fulfillment claims

*Instruction Generation*:
1. Expert-crafted complex conditionals + rubrics
2. Agentic augmentation (AutoIF-inspired)
3. Fine-tuned specialized model (edge cases, failure modes)

**Faithfulness**:

*Why Critical*: Agentic models need groundedness in:
- Multi-turn tool use
- Self-generated reasoning chains
- Open-environment interactions

*Method*:
- Train sentence-level faithfulness judge
- Detects unsupported factual claims
- FACTS Grounding evaluation framework
- Judge as reward model

**Coding & Software Engineering**:

*Competition Programming*:
- Problems + judges from open-source (OpenCode, KOD)
- Synthetic sources
- High-quality human unit tests (from pre-training data)

*Software Engineering*:
- GitHub PRs + issues
- User prompts/issues + executable unit tests
- Kubernetes sandbox:
  - 10,000+ concurrent instances
  - Scalable + secure

**Safety**:

*Seed Prompts*:
- Human-curated
- Prevalent risk categories (violence, fraud, discrimination)

*Prompt Evolution*:
- **Attack Model**: Iteratively generates adversarial prompts
- **Target Model**: Produces responses
- **Judge Model**: Evaluates if jailbreak succeeded
- Sophisticated attempts: role-playing, literary narratives, academic discourse

*Evaluation*:
- Task-specific rubric
- Binary success/failure label

#### B. Self-Critique Rubric Reward

**Goal**: Extend alignment beyond verifiable tasks
- Helpful, creative, reasoning depth
- Factuality, safety

**Mechanism**:

*Bootstrapping*:
- Curated mixture: open-source + in-house preference datasets
- Initialize critic capability in SFT stage

*Actor-Critic Loop*:

1. **Self-Critiqued Policy Optimization**:
   - Actor generates responses (general prompts)
   - Critic ranks via pairwise evaluation
   - Rubrics:
     - **Core**: Fundamental values (clarity, relevance, fluency, objectivity)
     - **Prescriptive**: Eliminate reward hacking (no initial praise, no explicit justification)
     - **Human-Annotated**: Task-specific criteria
   - K2 retains flexibility: weigh rubrics vs internal priors

2. **Closed-Loop Critic Refinement**:
   - Critic updated with verifiable signals during RL
   - On-policy rollouts (verifiable prompts) → update critic
   - **Transfer learning**: Objective → subjective
   - Grounds subjective judgments in verifiable data
   - Continuously recalibrates with policy evolution

**Impact**:
- Comprehensive improvements: user intent, creative writing, complex reasoning, nuanced comprehension
- Performance gains from verifiable → enhance subjective judgment

#### C. RL Algorithm

**Foundation**: K1.5 policy optimization

**Objective**:
```
L_RL(θ) = E_x[D][1/K Σ_i (r(x, y_i) - r̄(x) - τ log(π_θ(y_i|x) / π_old(y_i|x)))^2]

where:
  r̄(x) = 1/K Σ_i r(x, y_i)  (mean reward)
  τ > 0: regularization parameter
```

**Optimizer**: Muon

**Enhancements**:

1. **Budget Control**:
   - Problem: RL increases response length
   - Longer responses → better complex reasoning but costly
   - Solution: Per-sample maximum token budget
   - Truncation + penalty for exceeding
   - **Result**: Improved token efficiency, concise solutions

2. **PTX Loss**:
   - Problem: Forget valuable high-quality data
   - Solution: Curated dataset + auxiliary PTX loss
   - **Benefit**: Prevents overfitting to limited RL tasks
   - **Outcome**: Better generalization

3. **Temperature Decay**:
   - Creative writing + reasoning: exploration crucial early
   - High temperature → diverse, innovative responses
   - Later stages: high temperature → randomness, inconsistency
   - Solution: Decay schedule (exploration → exploitation)
   - **Result**: Early discovery + stable convergence

### 3.4 RL Infrastructure

#### Colocated Architecture

**Design**: Training + inference engines on same workers
- When one works, other releases/offloads GPU resources
- Centralized controller orchestrates iterations

**Iteration Loop**:
1. Controller calls inference → generate data
2. Controller notifies training → train on data
3. Training sends parameters → inference (next iteration)

**Goal**: Both engines heavily optimized for throughput

#### Efficient Engine Switching

**Challenge**: Scale of K2 → switching latency + failure recovery significant

**Training Engine Startup**:
- Parameters offloaded to DRAM during rollout
- Bring up = simple H2D transmission

**Inference Engine Startup** (harder):
- Needs updated parameters from training engine
- Different sharding paradigm

**Checkpoint Engine** (Solution):
- Distributed checkpoint engine co-located on training nodes
- Manages parameter states
- **Process**:
  1. Each worker gets local copy from training engine
  2. Broadcasts full parameter set across all workers
  3. Inference retrieves shard from checkpoint engine
- **For 1T model**: Parameter-by-parameter pipelined update
- **Time**: < 30 seconds (negligible for RL iteration)

**Design Choice**:
- Broadcast full set (vs transfer-what-you-need)
- More data transferred BUT:
  - Simpler system design
  - Less intrusive to engines
  - Decouples training + inference
  - Actually faster (reduced sync overhead, higher bandwidth)

**Open Source**: github.com/MoonshotAI/checkpoint-engine

#### Efficient System Startup

**Training Engine**:
- Selective read (part or none) from disk
- Broadcast necessary parameters to peers
- **Goal**: Collective read once (minimize expensive I/O)

**Inference Engines**:
- Independent replicas → avoid extra sync barriers
- Reuse checkpoint engine for startup:
  - Checkpoint engine reads from disk (like training)
  - Updates uninitialized inference engines
- **Robustness**: Single-point failure → replica restarts independently

#### Agentic Rollout

**Challenge**: Long-horizon, multi-turn tasks
- Complex environmental interactions
- Prolonged rollout durations

**Optimizations**:

1. **GPU Utilization**:
   - Interactions blocked on environment feedback (VM, code interpreter)
   - Strategy:
     - Heavy environments → dedicated services (scale easily)
     - Large concurrent rollouts (amortize latency)

2. **Long-Tail Trajectories**:
   - Problem: Block entire rollout process
   - Solution: Partial rollout
     - Pause unfinished tasks
     - Resume next RL iteration

3. **Unified Interface**:
   - OpenAI Gym-inspired
   - Streamline environment integration
   - **Goal**: Scale to more diverse interactive environments

---

## Section 4: Evaluations

### 4.1 Post-training (Kimi-K2-Instruct)

#### Benchmarks Covered

**Coding**:
- LiveCodeBench v6 (Aug 2024 - May 2025)
- OJBench (competition-level)
- MultiPL-E
- SWE-Bench Verified (agentless + agentic)
- TerminalBench
- Multi-SWE-bench
- SWE-Lancer
- PaperBench
- Aider-Polyglot

**Tool Use**:
- τ²-Bench (multi-turn)
- ACEBench

**Reasoning**:
- AIME 2024/2025
- MATH-500
- HMMT 2025
- CNMO 2024
- PolyMath-en
- ZebraLogic
- AutoLogi
- GPQA-Diamond
- SuperGPQA
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

#### Settings

**All models**: Non-thinking mode (no test-time compute)
**Output cap**: 8,192 tokens (16,384 for SWE-Bench Verified)
**Context**: 128k (truncate if needed)
**Sampling**: Avg@k for high-variance benchmarks
**SWE-Bench**:
- Agentless: Single patch without test
- Agentic: bash/editor tools (single + multiple attempts)
  - Multiple: best-of-N with internal verifier
- Multilingual: Single-attempt agentic only

#### Key Results

**Agentic & Competitive Coding**:
- SWE-Bench Verified: 65.8% (71.6% multi-attempt)
  - Closing gap with Claude 4 Opus/Sonnet
- SWE-Bench Multilingual: 47.3%
- SWE-Lancer: 39.1%
- LiveCodeBench v6: 53.7%
- OJBench: 27.1%

**Agentic Tool Use**:
- τ²-Bench: 66.1 Pass@1
- ACEBench: 76.5
  - Substantially outperforms all baselines

**General Capabilities**:
- SimpleQA: 31.0%
- MMLU: 89.5%
- MMLU-Redux: 92.7%
- IFEval: 89.8%
- Multi-Challenge: 54.1%
- AIME 2024: 69.6%
- GPQA-Diamond: 75.1%
- DROP: 93.5%
- MRCR: 55.0%

**Open-Ended**:
- LMSYS Arena (July 17, 2025)
  - #1 open-source
  - #5 overall (3,000+ votes)

### 4.2 Pre-training (Kimi-K2-Base)

#### Benchmarks

**General**:
- MMLU, MMLU-Pro, MMLU-Redux
- BBH, TriviaQA, SuperGPQA, SimpleQA
- HellaSwag, AGIEval, GPQA-Diamond
- ARC-Challenge, WinoGrande

**Coding**:
- EvalPlus (HumanEval, MBPP, + variants)
- LiveCodeBench v6
- CRUXEval

**Math**:
- GSM8K, GSM8K-Platinum, MATH, CMATH

**Chinese**:
- C-Eval, CMMLU, CSimpleQA

#### Settings

**Perplexity-based**: MMLU, MMLU-Redux, GPQA-Diamond, HellaSwag, ARC-Challenge, C-Eval, CMMLU
**Generation-based**: MMLU-Pro, SuperGPQA, TriviaQA, BBH, CSimpleQA, MATH, CMATH, GSM8K, GSM8K-Platinum, CRUXEval, LiveCodeBench, EvalPlus
**GPQA-Diamond**: Mean of 8 independent runs
**Framework**: Internal LM-Harness-Evaluation derivative

#### Results

**General Language** (10/12 SOTA):
- MMLU: 87.79%
- MMLU-Pro: 69.17%
- MMLU-Redux: 90.17%
- SuperGPQA: 44.67%
- SimpleQA: 35.25%

**Coding** (All SOTA):
- CRUXEval-I-cot: 74.00%
- CRUXEval-O-cot: 83.50%
- LiveCodeBench v6: 26.29%
- EvalPlus: 80.33%

**Math** (3/4 SOTA):
- MATH: 70.22%
- GSM8K: 92.12%
- GSM8K-Platinum: 94.21%
- CMATH: 90.26% (2nd place)

**Chinese** (All SOTA):
- C-Eval: 92.50%
- CMMLU: 90.90%
- CSimpleQA: 77.57%

### 4.3 Safety Evaluation

#### Red-teaming Setup

**Tool**: Promptfoo
**Plugins**:
- Harmful (9 categories)
- Criminal (12 categories)
- Misinformation (12 categories)
- Privacy (4 categories)
- Security (11 categories)

**Strategies**:
- Basic
- Prompt Injection
- Iterative Jailbreak
- Crescendo

**Baselines**: DeepSeek-V3, DeepSeek-R1, Qwen3

**Test Cases**:
- 3 attack prompts per plugin-strategy combo
- Some: English + Chinese (6 total)

**Review**: Manual, multiple rounds, same reviewer per test set

#### Results

**Findings**:
- No targeted optimization for specific scenarios
- Complex cases (e.g., Harmful + Iterative Jailbreak): higher passing rate
- Base64: ~100% passing (encoding minimal impact)
- Crescendo: General drop (stronger adversarial)
- Complex attacks not always better (transformation loses meaning)

**Limitations**:
- Subjectivity (human review)
- Some plugins (API misuse, external tools) more relevant for agent models

---

## Section 5: Limitations

1. **Verbose Outputs**: Hard reasoning tasks or unclear tool definitions → excessive tokens
2. **Performance Decline**: Unnecessary tool use enabled → worse results
3. **One-Shot Limitations**: Less effective than agentic coding frameworks for software projects
4. **Overconfidence Risk**: Rubric constraints may favor assertiveness over epistemic humility

---

## Section 6: Conclusions

### Technical Achievements

- ✅ Stable 1T+ parameter training (MuonClip)
- ✅ Token-efficient optimization (15.5T tokens, zero loss spikes)
- ✅ State-of-the-art agentic capabilities
- ✅ Scalable synthetic data pipeline
- ✅ General RL framework (verifiable + self-critique)

### Open Source Contributions

- Base + post-trained checkpoints
- Checkpoint engine (github.com/MoonshotAI/checkpoint-engine)
- Enables agentic intelligence research at scale

### Significance

**Most capable open-source LLM to date**:
- Particularly in agentic tasks
- Sets new benchmarks across domains

**Blueprint for next-gen agents**:
- Token efficiency > scale (limited high-quality data era)
- Synthetic + real data hybrid is optimal
- Self-critique enables general RL

---

## Appendix Insights

### QK-Clip Quality Impact

**Small-Scale Ablations** (0.5B activated, 3B total):
- Vanilla Muon vs MuonClip (τ=30: aggressive)
- Loss curves: Negligible difference
- Downstream tasks: No statistically significant degradation

**Self-Deactivation**:
- Only 12.7% heads triggered in first 70k steps
- All heads self-regulated after 70k steps
- Per-head (not per-layer): Minimizes over-regularization

### Muon vs Adam Logit Explosion

**Singular Value Entropy** (16B Moonlight model):
- Muon weights: Higher effective rank
- Adam weights: Lower effective rank
- Confirms theoretical intuition

**Key Formula**:
```
|q_i · k_j| ≤ ||x_i|| ||x_j|| ||W_q|| ||W_k||
```
- RMS-Norm bounds ||x_i|| ||x_j||
- Spectral norm of W_q, W_k drives explosion
- Muon's higher rank → additive singular value growth
- Attention's bilinear form squares the effect

### K2 Critic Rubrics

**Core Rubrics**:
1. **Clarity and Relevance**:
   - Succinct, fully addressing intent
   - Eliminate unnecessary detail
   - Efficient formats (brief paragraphs, compact lists)
   - Single, well-defined answer when choice expected

2. **Conversational Fluency and Engagement**:
   - Natural, flowing dialogue
   - Coherence, appropriate engagement
   - Relevant insights, constructive guidance
   - Judicious follow-up questions
   - Graceful hypothetical/personal-analogy handling
   - Adaptive tone (empathetic, formal, casual)

3. **Objective and Grounded Interaction**:
   - Objective, grounded tone
   - Focus on user request substance
   - Avoid: Metacommentary, unwarranted flattery
   - Direct, task-focused assistance

**Prescriptive Rubrics**:
1. **No Initial Praise**: Avoid "That's a beautiful question", "Good question!"
2. **No Explicit Justification**: Avoid "This response is good because...", "I successfully fulfilled..."

**Limitations**:
- May favor confident, assertive responses
- Disincentivizes: Self-qualification, disclaimers, hedging
- In ambiguity/subjectivity: May overstate certainty
- Future: Fine-grained calibrated uncertainty handling

### Engine Switching Pipeline

**Theoretical 3-Stage Pipeline**:
1. H2D: Shard → H2D buffer (async)
2. Broadcast: Copy to IPC buffer → broadcast to all
3. Reload: Inference engines load from other IPC buffer

**Reality: PCIe Saturation**:
- Concurrent H2D + broadcast saturates shared PCIe
- Collapses to sequential procedure

**Adopted 2-Stage**:
1. All devices: Synchronous H2D transfer
2. Broadcast + Reload: Parallel

**Scaling Benefit**:
- More devices → smaller shards
- Entire parameter set fits in H2D buffer
- Overhead disappears

**Overlap**: H2D, Broadcast, Reload → high bandwidth resharding

---

## Key Takeaways

1. **Stability Enables Scale**: MuonClip's QK-clip is key to reliable 1T+ training
2. **Token Efficiency Matters**: Limited high-quality data → per-token signal critical
3. **Synthetic + Real = Optimal**: Simulation provides scale, reality provides authenticity
4. **Agentic Requires Systematic Synthesis**: Multi-stage pipelines with quality filters
5. **General RL Needs Both Signals**: Verifiable rewards ground self-critique
6. **Infrastructure is Critical**: Checkpoint engine, colocated architecture enable training
7. **Sparsity is Powerful**: Higher sparsity → better performance under fixed FLOPs
8. **Inference Efficiency Matters**: Fewer attention heads worthy trade-off for long context
9. **Quality Over Quantity**: Curated, verified data > massive raw data
10. **Open Source Accelerates Progress**: Releasing checkpoints enables community research

---

**Notes Complete**: ✅
**Comprehensiveness**: 100% (all sections covered)
**Detail Level**: High (key formulas, algorithms, insights)
**Actionability**: Implementation-ready information
