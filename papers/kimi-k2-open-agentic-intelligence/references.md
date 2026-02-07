# References & Related Work

## Direct References

### Core Papers

1. **Muon Optimizer**
   - Jordan et al. (2024). "Muon: Token-Efficient Optimizer for LLMs"
   - arXiv:2408.04455
   - **Key insight**: msign operation → higher effective rank updates

2. **Kimi K1.5**
   - Team (2025). "Kimi K1.5: Verifiable Rewards for RL"
   - **Foundation**: RLVR methodology, policy optimization algorithm

3. **DeepSeek-V3**
   - DeepSeek AI (2024). "DeepSeek-V3 Technical Report"
   - **Architecture**: MLA, MoE design (256 experts, 128 heads)

### Data Synthesis

4. **WRAP**
   - Maini et al. (2024). "Rephrasing Web Recipe for Compute"
   - **Method**: Style-diverse prompting for rephrasing

5. **SwallowMath**
   - Fujii et al. (2025). "Rewriting Pre-training Data Boosts Math"
   - **Method**: Learning-note style rephrasing

6. **ACEBench**
   - Chen et al. (2025). "ACEBench: Comprehensive Tool-Use Benchmark"
   - **Framework**: Data synthesis pipeline inspiration

7. **ToolLLM**
   - Qin et al. (2023). "ToolLLM: Teaching Thousands of Tools"
   - **Method**: Tool-use at scale

8. **AgentInstruct**
   - Mitra et al. (2024). "AgentInstruct: Synthetic Agent Data"
   - **Method**: Synthetic without real interaction

9. **StableToolBench**
   - Guo et al. (2025). "StableToolBench: Stable Large-Scale Benchmarking"
   - **Method**: Scalable synthetic tool data

10. **ZeroSearch**
    - Sun et al. (2025). "ZeroSearch: Incentivize Search Capability"
    - **Method**: Synthetic data for search

### Reinforcement Learning

11. **FACTS Grounding**
    - Jacovi et al. (2025). "FACTS Grounding: Leaderboard and Benchmarking"
    - **Method**: Faithfulness evaluation

12. **AutoIF**
    - Dong et al. (2024). "SelfPlay and Execution for Instruction Following"
    - **Method**: Agentic instruction augmentation

13. **OpenAI Gym**
    - Brockman et al. (2016). "OpenAI Gym: A API for RL"
    - **Framework**: Unified environment interface

### Architecture

14. **Transformer**
    - Vaswani et al. (2017). "Attention Is All You Need"
    - **Foundation**: Base architecture

15. **Multi-Head Latent Attention (MLA)**
    - Liu et al. (2024). "DeepSeek-V2 and MLA"
    - **Method**: Efficient attention mechanism

### Evaluation Benchmarks

**Coding**:
- LiveCodeBench v6 (Jain et al., 2024)
- OJBench (Wang et al., 2025)
- SWE-Bench (Jimenez et al., 2024; Yang et al., 2025)
- Multi-SWE-Bench (Zan et al., 2025)
- SWE-Lancer (Miserendino et al., 2025)
- PaperBench (Starace et al., 2025)
- Aider-Polyglot

**Tool Use**:
- τ²-Bench (Barresi et al., 2025)
- ACEBench (Chen et al., 2025)

**Reasoning**:
- AIME (American Invitational Mathematics Examination)
- MATH-500
- HMMT (Harvard-MIT Mathematics Tournament)
- CNMO (Chinese National Mathematical Olympiad)
- PolyMath-en
- ZebraLogic (Lin et al., 2025)
- AutoLogi (Zhu et al., 2025)
- GPQA-Diamond (Rein et al., 2024)
- SuperGPQA (Du et al., 2025)
- Humanity's Last Exam (Phan et al., 2025)

**Long Context**:
- MRCR (Massive Reference Corpus Retrieval)
- DROP (Dua et al., 2019)
- FRAMES (Krishna et al., 2025)
- LongBench v2 (Bai et al., 2025)

**Factuality**:
- FACTS Grounding
- Vectara Hallucination Leaderboard
- FaithJudge (Tamber et al., 2025)

**General**:
- MMLU (Hendrycks et al., 2021)
- MMLU-Pro (Wang et al., 2024)
- MMLU-Redux (Gema et al., 2024)
- IFEval (Zhou et al., 2023)
- Multi-Challenge (Sirdeshmukh et al., 2025)
- SimpleQA (Wei et al., 2024)
- LiveBench

**Chinese**:
- C-Eval (Huang et al., 2023)
- CMMLU (Li et al., 2024)
- CSimpleQA

### Optimization

16. **AdamW**
    - Kingma & Ba (2015); Loshchilov & Hutter (2018)
    - **Baseline**: Standard optimizer

17. **WSD Learning Rate Schedule**
    - Hu et al. (2024). "MiniCPM: WSD Schedule"
    - **Method**: Warmup → Stay → Decay

18. **YaRN**
    - Peng et al. (2023). "YaRN: Context Window Extension"
    - **Method**: Extending context to 128k

### Infrastructure

19. **GPipe**
    - Huang et al. (2019)
    - **Method**: Pipeline parallelism

20. **PipeDream**
    - Harlap et al. (2018)
    - **Method**: Interleaved 1F1B schedule

21. **GShard**
    - Lepikhin et al. (2020)
    - **Method**: Expert parallelism

22. **ZeRO**
    - Rajbhandari et al. (2020)
    - **Method**: Data parallelism with optimizer state sharding

## Related Work by Category

### Optimizer Stability

- **Logit Cap**: Team et al. (2024). "Gemma: Logit Soft-Cap"
  - *Limitation*: Clips after computation, Q·K already exploded

- **QK-Norm**: Dehghani et al. (2023); Wortsman et al. (2023)
  - *Limitation*: Not applicable to MLA (keys not materialized)

- **MuonClip**: This paper
  - *Advantage*: Clips weights before computation, works with MLA

### Token Efficiency

- **Chinchilla Scaling Laws**: Hoffmann et al. (2022)
  - *Finding*: Compute-optimal model:token ratio

- **Data quality over quantity**: Brown et al. (2020); Muennighoff et al. (2023)
  - *Finding*: High-quality data > more data

- **Synthetic data**: This paper, WRAP, SwallowMath
  - *Finding*: Rephrasing > repetition for knowledge retention

### Agentic Capabilities

- **ReAct**: Yao et al. (2022)
  - *Method*: Reasoning + Acting paradigm

- **Reflexion**: Shinn et al. (2023)
  - *Method*: Self-reflection for task improvement

- **Voyager**: Wang et al. (2023)
  - *Method**: Lifelong learning agents

- **AutoGPT**: (2023)
  - *Method*: Autonomous task execution

- **K2 Agentic Pipeline**: This paper
  - *Advantage*: Systematic large-scale synthesis + real execution hybrid

### Reinforcement Learning for LLMs

- **RLHF**: Ouyang et al. (2022). "Training LLMs to Follow Instructions"
  - *Method*: Reward model + PPO

- **DPO**: Rafailov et al. (2023). "Direct Preference Optimization"
  - *Method*: Directly optimize preferences

- **RLAIF**: Lee et al. (2023). "Constitutional AI"
  - *Method*: AI feedback (not human)

- **K2 RL Framework**: This paper
  - *Advantage*: Verifiable + self-critique (no separate RM)

### Tool Use at Scale

- **ToolFormer**: Schick et al. (2023)
  - *Method*: Self-supervised tool learning

- **Gorilla**: Fan et al. (2023)
  - *Method*: API calling training

- **ToolAlpaca**: Qin et al. (2023)
  - *Method*: Instruction tuning for tool use

- **K2 Tool Synthesis**: This paper
  - *Advantage*: 23,000+ tools, quality filtering, real+sim hybrid

### Faithfulness & Grounding

- **RAG**: Lewis et al. (2020). "Retrieval-Augmented Generation"
  - *Method*: Ground responses in retrieved context

- **Attribution**: Gao et al. (2023)
  - *Method*: Attribute claims to sources

- **K2 Faithfulness**: This paper
  - *Method*: Sentence-level judge + RL reward

## Comparison to Contemporary Models

### vs DeepSeek-V3

| Aspect | DeepSeek-V3 | Kimi K2 |
|--------|-------------|---------|
| Total Params | 671B | 1.04T (↑54%) |
| Active Params | 37B | 32.6B (↓13%) |
| Experts | 256 | 384 (↑50%) |
| Attention Heads | 128 | 64 (↓50%) |
| Optimizer | Unknown | MuonClip |
| Training Tokens | Unknown | 15.5T |
| Loss Spikes | Unknown | Zero |

**K2 Advantages**:
- Higher sparsity (48 vs 32)
- Fewer attention heads (inference efficiency @ 128k)
- Stable training (MuonClip)
- Token efficiency (15.5T with synthetic data)

### vs Qwen3-235B-A22B

| Aspect | Qwen3 | Kimi K2 |
|--------|-------|---------|
| Architecture | MoE | MoE |
| Total Params | 235B | 1.04T |
| Active Params | 22B | 32.6B |
| Open Sourced | Instruct only | Base + Instruct |
| Optimizer | Unknown | MuonClip |

**K2 Advantages**:
- Larger capacity (1.04T vs 235B)
- Both base and instruct released
- Open optimizer (MuonClip)

### vs Claude 4 (Opus/Sonnet)

| Aspect | Claude 4 | Kimi K2 |
|--------|----------|---------|
| Weights | Closed | Open |
| Access | API only | Self-hostable |
| Evaluation mode | Thinking available | Non-thinking |
| SWE-Bench Verified | ~70% | 65.8% (71.6% multi) |

**K2 Advantages**:
- Open weights (research, customization)
- Competitive performance (non-thinking mode)
- Local deployment possible

### vs Llama 4

| Aspect | Llama 4 | Kimi K2 |
|--------|---------|---------|
| Variants | Maverick, Behemoth | K2 (1 size) |
| Open Sourced | Maverick only | Full model |
| Focus | General | Agentic |

**K2 Advantages**:
- Full model released (not just base)
- Agentic specialization
- Tool-use focus

## Future Work Directions

### Immediate Extensions

1. **QK-Clip Analysis**:
   - Theoretical analysis of per-head vs per-layer clipping
   - Adaptive threshold selection
   - Application to other optimizers (Sophia, Lion)

2. **Synthetic Data Quality**:
   - Hallucination detection in rephrased data
   - Fidelity verification improvements
   - Domain expansion beyond knowledge/math

3. **Agentic Benchmarks**:
   - More diverse tool-use scenarios
   - Longer-horizon tasks (100+ turns)
   - Multi-agent collaboration

4. **Self-Critique**:
   - Calibrated uncertainty handling
   - Adversarial rubric optimization
   - Multi-objective rubric learning

### Long-term Research

1. **Beyond Verifiable Rewards**:
   - Learning from subjective feedback
   - Open-ended goal optimization
   - Intrinsic motivation for exploration

2. **Agentic Generalization**:
   - Zero-shot tool learning
   - Tool composition
   - Meta-tool use (creating new tools)

3. **Safety at Scale**:
   - Adversarial robustness
   - Controllable agent behaviors
   - Alignment with human values

4. **Efficiency**:
   - Further sparsity increases
   - Dynamic expert routing
   - Context compression for agentic tasks

## Open Questions

1. **Muon Limits**: What is the largest model MuonClip can train? (Tested to 1.04T)

2. **Synthetic Data Saturation**: At what point does more rephrasing stop helping?

3. **Self-Critique Scalability**: Can rubric-based RL scale to billions of diverse prompts?

4. **Agentic Generalization**: How to measure true agentic intelligence (vs tool-use memorization)?

5. **Safety-Utility Trade-off**: How to enable powerful agents without catastrophic risks?

## Citation

```bibtex
@article{kimi2024k2,
  title={Kimi K2: Open Agentic Intelligence},
  author={Kimi Team and Bai, Yifan and Bao, Yiping and Charles, Y. and Chen, Cheng and Chen, Guanduo and Chen, Haiting and Chen, Huarong and Chen, Jiahao and Chen, Ningxin and Chen, Ruijue and Chen, Yanru and Chen, Yuankun and Chen, Yutian and Chen, Zhuofu and Cui, Jialei and Ding, Hao and Dong, Mengnan and Du, Angang and Du, Chenzhuang and Du, Dikang and Du, Yulun and Fan, Yu and Feng, Yichen and Fu, Kelin and Gao, Bofei and Gao, Chenxiao and Gao, Hongcheng and Gao, Peizhong and Gao, Tong and Ge, Yuyao and Geng, Shangyi and Gu, Qizheng and Gu, Xinran and Guan, Longyu and Guo, Haiqing and Guo, Jianhang and Hao, Xiaoru and He, Tianhong and He, Weiran and He, Wenyang and He, Yunjia and Hong, Chao and Hu, Hao and Hu, Yangyang and Hu, Zhenxing and Huang, Weixiao and Huang, Zhiqi and Huang, Zihao and Jiang, Tao and Jiang, Zhejun and Jin, Xinyi and Kang, Yongsheng and Lai, Guokun and Li, Cheng and Li, Fang and Li, Haoyang and Li, Ming and Li, Wentao and Li, Yang and Li, Yanhao and Li, Yiwei and Li, Zhaowei and Li, Zheming and Lin, Hongzhan and Lin, Xiaohan and Lin, Zongyu and Liu, Chengyin and Liu, Chenyu and Liu, Hongzhang and Liu, Jingyuan and Liu, Junqi and Liu, Liang and Liu, Shaowei and Liu, Tian-Yang and Liu, Tianwei and Liu, Weizhou and Liu, Yangyang and Liu, Yibo and Liu, Yiping and Liu, Yue and Liu, Zhengying and Lu, Enzhe and Lu, Haoyu and Lu, Lijun and Luo, Yashuo and Ma, Shengling and Ma, Xinyu and Ma, Yingwei and Mao, Shaoguang and Mei, Jie and Men, Xin and Miao, Yibo and Pan, Siyuan and Peng, Yebo and Qin, Ruoyu and Qin, Zeyu and Qu, Bowen and Shang, Zeyu and Shi, Lidong and Shi, Shengyuan and Song, Feifan and Su, Jianlin and Su, Zhengyuan and Sui, Lin and Sun, Xinjie and Sung, Flood and Tai, Yunpeng and Tang, Heyi and Tao, Jiawen and Teng, Qifeng and Tian, Chaoran and Wang, Chensi and Wang, Dinglu and Wang, Feng and Wang, Hailong and Wang, Haiming and Wang, Jianzhou and Wang, Jiaxing and Wang, Jinhong and Wang, Shengjie and Wang, Shuyi and Wang, Si and Wang, Xinyuan and Wang, Yao and Wang, Yejie and Wang, Yiqin and Wang, Yuxin and Wang, Yuzhi and Wang, Zhaoji and Wang, Zhengtao and Wang, Zhexu and Wei, Chu and Wei, Qianqian and Wu, Haoning and Wu, Wenhao and Wu, Xingzhe and Wu, Yuxin and Xiao, Chenjun and Xie, Jin and Xie, Xiaotong and Xiong, Weimin and Xu, Boyu and Xu, Jinjing and Xu, L. H. and Xu, Lin and Xu, Suting and Xu, Weixin and Xu, Xinran and Xu, Yangchuan and Xu, Ziyao and Xu, Jing and Xu, Jing and Yan, Junjie and Yan, Yuzi and Yang, Hao and Yang, Xiaofei and Yang, Yi and Yang, Ying and Yang, Zhen and Yang, Zhilin and Yang, Zonghan and Yao, Haotian and Yao, Xingcheng and Ye, Wenjie and Ye, Zhuorui and Yin, Bohong and Yu, Longhui and Yuan, Enming and Yuan, Hongbang and Yuan, Mengjie and Yuan, Siyu and Zhan, Haobing and Zhang, Dehao and Zhang, Hao and Zhang, Wanlu and Zhang, Xiaobin and Zhang, Yadong and Zhang, Yangkun and Zhang, Yichi and Zhang, Yizhi and Zhang, Yongting and Zhang, Yu and Zhang, Yutao and Zhang, Yutong and Zhang, Zheng and Zhao, Haotian and Zhao, Yikai and Zhao, Zijia and Zheng, Huabin and Zheng, Shaojie and Zhong, Longguang and Zhou, Jianren and Zhou, Xinyu and Zhou, Zaida and Zhu, Jinguo and Zhu, Zhen and Zhuang, Weiyu and Zu, Xinxing},
  journal={arXiv preprint arXiv:2507.20534},
  year={2025},
  note={v2: February 2026}
}
```

---

**Last Updated**: February 2026
**Total References**: 50+
**Coverage**: Comprehensive
