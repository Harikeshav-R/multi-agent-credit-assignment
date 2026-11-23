---
title: "Experimental Summary: Multi-Agent Skill Attribution"
author: "Hari"
papersize: us-letter
fontsize: 11pt
margin:
  top: 0.7in
  bottom: 0.7in
  left: 0.75in
  right: 0.75in
---

### Research Setup and Findings

We evaluated when counterfactual ablation justifies its information cost over observational usage attribution in two-agent teams across three environments (additive, synergy, confounded; 900 episodes/seed, 30 seeds). Individual attribution computes mean success conditioned on skill usage; ablation computes the paired difference replacing the active skill with a no-skill baseline while freezing teammate actions and outcome noise $u$.

| Scenario | Skill | Usage mean ± SD | Exact usage | Ablation mean ± SD | Exact delta |
|---|---|---:|---:|---:|---:|
| additive | A1 | 0.391 ± 0.027 | 0.400 | 0.201 ± 0.023 | 0.200 |
| additive | A2 | 0.156 ± 0.020 | 0.150 | -0.049 ± 0.011 | -0.050 |
| additive | B1 | 0.447 ± 0.035 | 0.450 | 0.298 ± 0.027 | 0.300 |
| additive | B2 | 0.147 ± 0.019 | 0.150 | 0.000 ± 0.000 | 0.000 |
| synergy | A1 | 0.288 ± 0.027 | 0.300 | 0.197 ± 0.023 | 0.200 |
| synergy | A2 | 0.053 ± 0.013 | 0.050 | -0.051 ± 0.011 | -0.050 |
| synergy | B1 | 0.279 ± 0.030 | 0.283 | 0.196 ± 0.022 | 0.200 |
| synergy | B2 | 0.083 ± 0.015 | 0.083 | 0.000 ± 0.000 | 0.000 |
| confounded | A1 | 0.474 ± 0.079 | 0.467 | 0.197 ± 0.045 | 0.200 |
| confounded | A2 | 0.576 ± 0.017 | 0.574 | 0.000 ± 0.000 | 0.000 |
| confounded | B1 | 0.607 ± 0.016 | 0.605 | 0.504 ± 0.015 | 0.500 |
| confounded | B2 | 0.176 ± 0.051 | 0.167 | 0.000 ± 0.000 | 0.000 |

In additive environments, usage-conditioned success and ablation delta rank skills consistently. In synergy, A1 and B1 have zero solo gain over baseline, but their joint interaction adds $+0.600$; ablation assigns $+0.200$ conditional credit to each, exposing that leave-one-out differences do not uniquely divide joint gains. In confounded environments, inert skill A2 frequently co-occurs with strong skill B1; observational usage incorrectly ranks A2 over useful skill A1 on 26/30 seeds (mean 0.576 vs 0.474). Counterfactual ablation cleanly isolates A2's true null effect (mean 0.000, exact 0.000), correctly ranking A1 over A2 on 30/30 seeds. Across 81,000 factual episodes, this causal precision required 122,385 additional counterfactual evaluations (+151% computational cost).

### Optimization Risks, Real-World LLMs, and Limitations

In an iterative skill-evolution loop, observational credit risks preserving inert skills (like A2) due to spurious correlation, while solo testing risks pruning complementary skills (like A1/B1) that yield zero value in isolation. In live LLM deployments, causal intervention is far more difficult: hidden chains of thought, non-deterministic outputs, and branching environment state make frozen-noise counterfactuals impossible via standard APIs. Downstream teammates react dynamically to missing skills rather than holding actions static, requiring full trajectory re-executions that multiply inference costs.

**Limitations:** The simulation uses constructed stationary rewards, two single-skill agents, a known no-skill baseline, and coupled noise; it does not measure multi-step tool dynamics, sequential library editing, LLM prompt tokens, or language-critic veracity.
