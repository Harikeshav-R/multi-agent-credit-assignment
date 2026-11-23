---
title: "Critical Memo: Multi-Agent Allocation, Skill Credit, and Empirical Governance"
author: "Hari"
papersize: us-letter
fontsize: 11pt
margin:
  top: 0.7in
  bottom: 0.7in
  left: 0.75in
  right: 0.75in
---

## 1. Resource-Aware Allocation in Multi-Agent Systems (BAMAS)

Multi-agent LLM systems frequently deploy frontier models across all operational roles, wasting inference budgets on routine formatting or redundant checks. Heterogeneous provisioning pairs lightweight models with capable frontier models to maximize accuracy under token cost limits. BAMAS [1] formalizes team deployment as a two-stage optimization problem: integer linear programming (ILP) for model instance provisioning, followed by offline reinforcement learning (RL) for workflow topology selection.

The provisioning formulation introduces notable structural compromises. Decisions select integer counts of candidate models under an estimated token budget, constrained to at least two agent instances. Critically, the ILP objective maximizes lexicographic model-tier weights derived from Chatbot Arena Elo rankings rather than measured task accuracy or learned interaction utility. Arena Elo reflects single-turn conversational preference, not collaborative problem-solving. Feasibility constraints rely on pre-execution heuristics using fixed 500-token prompt lengths and maximum observed completion lengths over 50 training instances. These estimates do not guarantee spending caps; Table 3 reports realized budget overruns on 30/1000 MATH tasks at budget 2000, requiring runtime halts once spending limits are breached (Appendix B).

BAMAS decouples provisioning from collaboration topology, offering Linear, Star, Feedback, and Planner-Driven structures. Topology governs information visibility, intermediate verification, and error-recovery rounds. The workflow selector maps task embeddings and budget levels to topology decisions via Q-learning from final task rewards. A simpler baseline omitted from the evaluation is validation-set enumeration: evaluating the four fixed workflows across budget tiers and deploying the empirical frontier. In published benchmarks, BAMAS reports up to 86% cost reduction on MBPP (spending 529.2 tokens at 82.6% accuracy versus MetaGPT's 3735.1 tokens at 82.2% accuracy [1, Tab. 1]). However, the component comparison in Table 4 evaluates Naive-CostAware using identical-model teams in a fixed linear chain against BAMAS's dynamically provisioned and routed teams. Because model heterogeneity and interaction topology vary simultaneously, this comparison fails to isolate the individual causal contribution of tier-based provisioning.

**BAMAS Research Question:** *At a fixed collaboration topology and common spending cap, does model-tier-first allocation outperform task-calibrated model allocation?*

To isolate provisioning from routing, we propose a factorial study on a two-agent generator/checker system. We hold topology, prompts, per-step token caps, and termination rules strictly fixed while evaluating four ordered allocations: cheap/cheap, cheap/strong, strong/cheap, and strong/strong. Using 20 validation and 50 held-out reasoning problems under identical spending caps, we compare BAMAS's tier-first allocation (highest Arena-tier feasible combination) against a task-calibrated baseline that selects the allocation maximizing validation accuracy per dollar. We record realized expenditures, cap violations, and account separately for calibration and execution costs.

```{=typst}
#pagebreak()
```

## 2. Skill Evolution and Credit Assignment (MASkills)

MASkills [2] optimizes multi-agent teams by refining modular, text-based skill libraries rather than training model weights. Each agent maintains procedural instructions, metadata, and optional resources. During execution, logged skill invocations produce trajectories evaluated by an LLM language critic. The critic estimates skill contributions relative to an unassisted baseline, generating textual feedback to guide four operators: refinement, induction, consolidation, and pruning. Candidate edits undergo task-subset validation and are accepted or rolled back.

While MASkills introduces language analogies for gradients, advantages, and momentum, these metaphors are not mathematical derivatives; updating text prompts does not adjust internal model weights. Credit assignment remains challenging because team rewards conflate individual agent contributions, tool executions, and complementary skill synergies. Table 2 provides valuable ablations: on LoCoMo-MH, full MASkills achieves 17.2 versus 14.2 without credit assignment and 6.6 without rollback; on GAIA, it scores 23.3 versus 17.1 without credit and 13.5 without rollback. Furthermore, Figure 3 tests prompt-generated baselines and cross-task transfer. However, these ablations do not prove language critics accurately recover causal contributions. Critic judgments rely on retrospective prompt queries over recorded execution logs rather than true counterfactual branch executions, leaving feedback vulnerable to hallucinated attribution.

**MASkills Research Question:** *How reliably does skill-level credit identify complementary and irrelevant skills when skill usage is correlated?*

To examine this question, our synthetic implementation investigates counterfactual intervention in multi-agent environments. When skill usage is correlated, observational usage-conditioned attribution rewards inert skills that co-occur with high-performing teammate skills, whereas counterfactual ablation isolates true effects. In real LLM deployments, evaluating true counterfactuals requires re-executing non-deterministic trajectories, incurring substantial inference overhead. A rigorous LLM benchmark must compare skill-specific critic feedback against agent-level holistic feedback while strictly matching optimizer model (GPT-5.1), feedback volume, candidate edit counts, and validation token budgets.

Regarding recursive self-improvement (RSI), MASkills implements external artifact adaptation. The meta-prompt optimizer and foundation models remain static; only external library text is edited. Structured storage, explicit invocation, and validation rollback guard against catastrophic degradation, but do not demonstrate open-ended algorithmic self-improvement. The system is conceptually grounded in multi-agent RL credit assignment, notably COMA [3], which computes counterfactual advantages using a centralized critic that marginalizes out an agent's policy while holding teammate actions fixed. MASkills adapts this counterfactual principle to natural language, yet substitutes retrospective prompting for COMA's marginalization over alternative joint actions.

## 3. Research Direction and Synthesis

My research trajectory focuses on resource-aware multi-agent systems, informed by systems orchestration and distributed inference. The technical exercise on causal attribution (Option B) provided essential grounding in evaluation cost and attribution mechanics. It demonstrated that precise causal credit demands substantial extra compute—a trade-off directly aligned with designing cost-governed agent architectures.

## References

1. L. Yang et al., "BAMAS: Structuring Budget-Aware Multi-Agent Systems," arXiv:2511.21572, Nov 2025. [https://arxiv.org/abs/2511.21572](https://arxiv.org/abs/2511.21572)
2. H. Yao et al., "MASkills: Continual Skills Optimization for Multi-Agent LLM Systems," arXiv:2609.02094, Sep 2026. [https://arxiv.org/abs/2609.02094](https://arxiv.org/abs/2609.02094)
3. J. N. Foerster et al., "Counterfactual Multi-Agent Policy Gradients," *AAAI*, 2018 (v3 erratum Dec 2024). [https://arxiv.org/abs/1705.08926](https://arxiv.org/abs/1705.08926)
