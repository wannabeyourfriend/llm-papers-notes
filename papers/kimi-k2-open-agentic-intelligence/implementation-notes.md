# Kimi K2: Implementation Notes

## Practical Takeaways for LLM Development

### Optimizer Choice

**When to Use Muon/MuonClip**:
- ✅ Large-scale training (10B+ parameters)
- ✅ Limited high-quality data (need token efficiency)
- ✅ Want better performance than AdamW at same compute

**When to Use AdamW**:
- ✅ Small-scale models (<10B)
- ✅ Sufficient training data
- ✅ Want stable, well-understood optimizer

**MuonClip Implementation Checklist**:
```python
# Pseudo-code for QK-Clip
def muonclip_step(weights, grads, tau=100):
    # 1. Standard Muon update
    updated_weights = muon_update(weights, grads)

    # 2. Compute max logits for each head (from forward pass)
    for head in attention_heads:
        S_max = head.max_logit

        # 3. Apply QK-Clip if needed
        if S_max > tau:
            gamma = tau / S_max
            updated_weights.q[head] *= sqrt(gamma)
            updated_weights.k[head] *= sqrt(gamma)
            updated_weights.q_rotary[head] *= gamma
            # Don't touch k_rotary (shared)

    return updated_weights
```

**Key Hyperparameters**:
- τ (threshold): 100 (tested down to 30 with no degradation)
- Weight decay: 0.1
- Learning rate: 2e-4 (constant) → 2e-5 (cosine decay) → 7e-6 (annealing)
- Warmup: 500 steps

### Synthetic Data Generation

**Knowledge Rephrasing Pipeline**:

```python
# Chunk-wise autoregressive rephrasing
def rephrase_long_document(document, chunk_size=2000, overlap=200):
    chunks = split_with_overlap(document, chunk_size, overlap)
    rephrased = []

    for i, chunk in enumerate(chunks):
        # Include previous context for coherence
        context = chunks[i-1] if i > 0 else ""
        prompt = f"Previous: {context}\n\nCurrent: {chunk}"

        # Style-diverse prompting
        style = random.choice(["academic", "conversational", "technical"])
        rephrased_chunk = llm_generate(
            prompt,
            style=style,
            maintain_facts=True
        )
        rephrased.append(rephrased_chunk)

    # Stitch back together
    return merge_chunks(rephrased, overlap)

# Fidelity verification
def verify_fidelity(original, rephrased):
    score = semantic_similarity(original, rephrased)
    return score > 0.85  # Threshold
```

**Math Rewriting**:
- Convert to "learning-note" style
- Add step-by-step explanations
- Worked examples
- Conceptual intuition before formalism

**Agentic Tool Use Synthesis**:

```python
# Three-stage pipeline
def generate_agentic_training_data():
    # Stage 1: Tool repository
    tools = load_real_mcp_tools() + generate_synthetic_tools(
        domains=["financial", "software", "robotics"],
        num_synthetic=20000
    )

    # Stage 2: Agent + task generation
    agents = create_diverse_agents(num_agents=1000)
    tasks = generate_rubric_tasks(
        agents=agents,
        tools=tools,
        difficulty="simple_to_complex"
    )

    # Stage 3: Trajectory generation
    trajectories = []
    for agent, task in zip(agents, tasks):
        user_sim = simulate_user(agent.persona)
        tool_env = ToolSimulator(stateful=True, stochastic=True)

        traj = generate_multi_turn_trajectory(
            agent=agent,
            user=user_sim,
            env=tool_env,
            task=task,
            max_turns=20
        )

        # Quality filter
        if evaluate_against_rubric(traj, task.rubric):
            trajectories.append(traj)

    return trajectories
```

### Agentic Data Synthesis Best Practices

**Tool Spec Generation**:
- Start with real tools (MCP, GitHub) → build authenticity
- Synthetic tools: Hierarchical evolution
  - Category → Domain → Specific tools
- Validate coverage with embeddings (t-SNE clustering)

**Agent Diversification**:
- Vary system prompts (personas, expertise, tone)
- Different tool combinations
- Behavioral patterns (cautious vs aggressive)

**Task Quality**:
- Explicit rubrics: Success criteria, checkpoints
- Range from simple API calls to multi-step workflows
- Include edge cases, error handling

**Trajectory Generation**:
- **User simulator**: LLM with distinct persona
- **Tool environment**: Stateful, stochastic feedback
  - Success, partial failure, timeout, errors
- **Quality gate**: LLM judge vs rubric
- **Real execution**: Hybrid for critical domains (coding)

### Reinforcement Learning Setup

**Verifiable Rewards Gym**:

```python
class VerifiableGym:
    def __init__(self):
        self.tasks = {
            "math": MathTaskSet(moderate_difficulty=True),
            "instruction_following": InstructionTasks(
                verification="hybrid"  # code + LLM judge
            ),
            "coding": CodingTasks(
                unit_tests=True,
                execution=True
            ),
            "safety": SafetyTasks(
                adversarial_evolution=True
            )
        }

    def compute_reward(self, task, response):
        if task.verifiable:
            return task.execute_tests(response)
        else:
            # Use self-critique
            return self.critic.evaluate(response, task.rubric)
```

**Self-Critique Implementation**:

```python
class SelfCritiqueTrainer:
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic

    def train_step(self, prompts):
        # 1. Generate responses
        responses = [self.actor.generate(p) for p in prompts]

        # 2. Critic ranks (pairwise)
        rankings = self.critic.rank_pairs(
            responses,
            rubrics=self.get_rubrics(prompts)
        )

        # 3. Update actor based on preferences
        actor_loss = self.compute_dpo_loss(
            responses,
            rankings
        )

        # 4. Update critic with verifiable signals
        verifiable = self.filter_verifiable(prompts)
        critic_loss = self.train_critic_on_verifiable(
            verifiable
        )

        return actor_loss, critic_loss

    def get_rubrics(self, prompts):
        return {
            "core": CORE_RUBRICS,
            "prescriptive": PRESCRIPTIVE_RUBRICS,
            "task_specific": self.load_task_rubrics(prompts)
        }
```

**RL Algorithm Enhancements**:

```python
# Budget control
def compute_budgeted_loss(response, reward, max_tokens, budget_penalty=0.5):
    actual_tokens = len(response)
    if actual_tokens > max_tokens:
        truncated_reward = reward * (max_tokens / actual_tokens)
        return truncated_reward - budget_penalty
    return reward

# PTX loss
def compute_rl_loss_with_ptx(rl_loss, ptx_data, alpha=0.1):
    ptx_loss = compute_nll_loss(model, ptx_data)
    return rl_loss + alpha * ptx_loss

# Temperature decay
def get_temperature(step, total_steps, init_temp=1.0, final_temp=0.7):
    progress = step / total_steps
    return init_temp - (init_temp - final_temp) * progress
```

### Infrastructure Design

**Checkpoint Engine**:

```python
class DistributedCheckpointEngine:
    def __init__(self, training_workers, inference_workers):
        self.workers = training_workers + inference_workers
        self.buffers = {
            "h2d": allocate_gpu_buffer(PARAM_SIZE),
            "ipc1": allocate_ipc_buffer(PARAM_SIZE),
            "ipc2": allocate_ipc_buffer(PARAM_SIZE)
        }

    def update_inference_engines(self):
        # Stage 1: H2D copy (async)
        for worker in self.workers:
            copy_from_training_to_buffer(worker, self.buffers["h2d"])

        # Stage 2: Broadcast
        broadcast_to_all(self.buffers["h2d"], self.buffers["ipc1"])

        # Stage 3: Reload (inference engines read from ipc2)
        while not all_inference_updated():
            continue

        return  # ~30 seconds for 1T model
```

**Colocated Training**:

```python
class ColocatedRLTrainer:
    def __init__(self):
        self.training_engine = TrainingEngine()
        self.inference_engine = InferenceEngine()
        self.checkpoint_engine = CheckpointEngine()
        self.mode = None  # "training" or "inference"

    def iteration(self):
        # Rollout phase
        self.switch_to_inference()
        data = self.inference_engine.generate_rollouts()

        # Training phase
        self.switch_to_training()
        metrics = self.training_engine.train_on(data)

        # Update inference with new weights
        self.checkpoint_engine.update_inference(
            self.training_engine.get_weights()
        )

        return metrics

    def switch_to_inference(self):
        # Offload training states to CPU
        self.training_engine.offload_states()
        # Load inference weights from checkpoint engine
        self.inference_engine.load_weights(
            self.checkpoint_engine.get_inference_weights()
        )

    def switch_to_training(self):
        # Offload inference (already checkpointed)
        self.inference_engine.offload()
        # Load training states from CPU
        self.training_engine.load_states()
```

**Agentic Rollout Optimization**:

```python
class AgenticRolloutManager:
    def __init__(self, environments):
        self.envs = environments
        self.heavy_envs = [e for e in environments if e.is_heavy()]
        self.light_envs = [e for e in environments if not e.is_heavy()]

    def deploy_heavy_as_services(self):
        # Deploy VMs, code interpreters as scalable services
        for env in self.heavy_envs:
            env.deploy_as_service(scale="auto")

    def run_concurrent_rollouts(self, agents, tasks, num_concurrent=1000):
        # Amortize latency with many concurrent rollouts
        futures = []
        for agent, task in zip(agents, tasks):
            future = self.submit_rollout_async(agent, task)
            futures.append(future)
            if len(futures) >= num_concurrent:
                self.wait_for_some(futures)

        return self.collect_results(futures)

    def partial_rollout(self, long_horizon_tasks):
        # Pause/resume long-tail tasks
        results = {}
        for task in long_horizon_tasks:
            if task.is_taking_too_long():
                task.pause()
                results[task.id] = "resume_next_iteration"
            else:
                results[task.id] = task.complete()

        return results
```

### Model Architecture Decisions

**Sparsity Selection**:

```python
def compute_optimal_sparsity(target_loss, compute_budget):
    # Run scaling law experiments
    for sparsity in [8, 16, 32, 48, 64]:
        model = MoEModel(
            total_params=compute_budget_to_params(compute_budget),
            sparsity=sparsity
        )
        loss = train_and_evaluate(model)

        if loss <= target_loss:
            return sparsity  # Higher sparsity = better
    return max(sparsity_options)  # Return best achievable
```

**Attention Heads vs Context Length**:

```python
def analyze_heads_vs_context(heads, context_length, expert_count):
    # Inference FLOPs
    attention_flops = compute_attention_flops(heads, context_length)
    moe_flops = compute_moe_flops(expert_count)
    total_flops = attention_flops + moe_flops

    # For 128k context, 384 experts:
    # 64 heads:  1.0x baseline
    # 128 heads: 1.83x (83% increase!)

    return total_flops

# Decision: 64 heads (not 128)
# Rationale: 0.5-1.2% validation loss improvement not worth 83% cost
```

### Training Configuration

**Complete Recipe**:

```python
config = {
    # Architecture
    "total_params": 1.04e12,
    "active_params": 32.6e9,
    "experts": 384,
    "active_experts": 8,
    "attention_heads": 64,
    "layers": 61,
    "hidden_dim": 7168,
    "moe_hidden_dim": 2048,

    # Training
    "tokens": 15.5e12,
    "context": 4096,
    "final_context": 128000,
    "batch_size": 67e6,

    # Optimizer
    "optimizer": "MuonClip",
    "tau": 100,
    "weight_decay": 0.1,
    "lr_schedule": "WSD",
    "peak_lr": 2e-4,
    "final_lr": 7e-6,
    "warmup_steps": 500,

    # Schedule
    "constant_tokens": 10e12,  # 2e-4
    "cosine_tokens": 5.5e12,   # 2e-4 -> 2e-5
    "anneal_4k_tokens": 400e9, # 2e-5 -> 7e-6
    "anneal_32k_tokens": 60e9, # 2e-5 -> 7e-6

    # Extension
    "extension_method": "YaRN",
    "target_context": 128000
}
```

### Evaluation Best Practices

**Non-Thinking Mode**:
- All models evaluated without extended thinking
- Output token limits enforced
- No best-of-N unless specified
- Ensures fair comparison

**Benchmark Selection**:
- Coding: LiveCodeBench (temporal split), OJBench (competition)
- Tool Use: τ²-Bench, ACEBench (multi-turn)
- Reasoning: AIME (hard math), GPQA (STEM)
- Agentic: SWE-Bench (real GitHub issues)

**Repeated Sampling**:
- High-variance benchmarks: Avg@k (typically k=5-10)
- Report mean ± std
- Ensures stable scores

### Common Pitfalls

❌ **Don't**: Use AdamW for 1T+ models with limited data
✅ **Do**: Use MuonClip for token efficiency

❌ **Don't**: Clip all heads at once
✅ **Do**: Per-head clipping (minimal intervention)

❌ **Don't**: Use synthetic data exclusively
✅ **Do**: Hybrid synthetic + real for authenticity

❌ **Don't**: Skip verifiable rewards in RL
✅ **Do**: Ground self-critique in objective signals

❌ **Don't**: Double attention heads for long context
✅ **Do**: Fewer heads (64) for inference efficiency

❌ **Don't**: Ignore budget control in RL
✅ **Do**: Enforce token limits to prevent verbosity

❌ **Don't**: Use naive aggregation for RL system startup
✅ **Do**: Checkpoint engine for efficient weight loading

### Reproducibility Checklist

- [ ] MuonClip optimizer implementation (τ=100, per-head)
- [ ] QK-clip self-deactivation monitoring
- [ ] Synthetic data pipeline (chunk-wise rephrasing)
- [ ] Agentic data synthesis (3-stage with quality filter)
- [ ] Verifiable rewards gym (math, coding, instruction following)
- [ ] Self-critique rubrics (core + prescriptive + task-specific)
- [ ] RL algorithm enhancements (budget, PTX, temperature decay)
- [ ] Checkpoint engine infrastructure
- [ ] Colocated training/inference architecture
- [ ] Evaluation in non-thinking mode

---

## Quick Reference

### Key Formulas

**QK-Clip**:
```
S_max^h = max(Q_i · K_j) / √d
If S_max^h > τ:
    γ = τ / S_max^h
    q ← q · √γ
    k ← k · √γ
```

**Sparsity**:
```
Sparsity = Total Experts / Active Experts
FLOP reduction (sparsity 48 vs 8): 1.69×
```

**RL Objective**:
```
L_RL = E[(r - r̄ - τ log(π_θ/π_old))^2]
```

### Critical Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| τ (QK-Clip) | 100 | Caps logits, tested down to 30 |
| Weight decay | 0.1 | Regularization |
| Peak LR | 2e-4 | WSD schedule |
| Final LR | 7e-6 | Annealing |
| Sparsity | 48 | Optimal compute-performance |
| Active experts | 8 | Balance capacity + efficiency |
| Attention heads | 64 | Inference efficiency @ 128k |
| Batch size | 67M tokens | Global effective batch |

### Code References

- **Checkpoint Engine**: https://github.com/MoonshotAI/checkpoint-engine
- **Model Weights**: https://huggingface.co/moonshotai/Kimi-K2-Instruct
- **Paper**: https://arxiv.org/abs/2507.20534

---

**Implementation Status**: Ready to use
**Reproducibility**: High (detailed specifications provided)
**Production Readiness**: Proven at 1T scale
