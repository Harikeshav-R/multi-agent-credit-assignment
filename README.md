# Multi-Agent Attribution Experiment & Exploration Submission

This repository contains the code, evaluation suite, and deliverables for the multi-agent exploration assignment evaluating causal attribution in collaborative agent teams.

## 1. Prerequisites

- **Python 3.11+** (uses standard library only: `argparse`, `dataclasses`, `json`, `math`, `pathlib`, `random`, `statistics`, `sys`, `unittest`).
- **No external Python packages, LLM API keys, model downloads, or GPU training required.**
- *(Optional)* **Pandoc 3.1+** with **Typst 0.11+** if recompiling `memo.pdf` and `experiment-summary.pdf` from source markdown.

## 2. Verification and Execution

### Run Behavioral and Unit Tests
Execute the 13-test behavioral suite covering analytical expectations, joint probability distributions, paired counterfactual coupling, confounding rank reversals, edge cases, and CLI validation:

```bash
python3 -m unittest -v test_experiment
```

### Run Full Experiment
Run the 30-seed simulation across additive, synergy, and confounded scenarios:

```bash
python3 experiment.py --episodes 900 --seeds 30 --out results
```

Default parameters: `--episodes 900`, `--seeds 30`, `--out results`.
Execution generates 81,000 factual episodes and 122,385 counterfactual interventions in ~0.5 seconds.

*(Optional) Recompile PDFs from Markdown:*
```bash
pandoc memo.md --pdf-engine=typst -o memo.pdf
pandoc experiment-summary.md --pdf-engine=typst -o experiment-summary.pdf
```

## 3. Environment & Methodology

The experiment models two collaborating agents (Agent A and Agent B) choosing among three actions: no skill (`0`), skill 1 (`1`), or skill 2 (`2`). There are 9 joint configurations $(a, b) \in \{0, 1, 2\} \times \{0, 1, 2\}$.

Success probability $p(a, b) \in [0, 1]$ follows three scenario definitions:
- **Additive**: $p(a, b) = 0.10 + 0.20[a=1] - 0.05[a=2] + 0.30[b=1]$ (uniform sampling across all 9 pairs, $p=1/9$).
- **Synergy**: $p(a, b) = 0.10 + 0.60[a=1 \land b=1] - 0.05[a=2]$ (uniform sampling, $p=1/9$; solo skills add $0.00$, joint pair adds $+0.60$).
- **Confounded**: $p(a, b) = 0.10 + 0.20[a=1] + 0.50[b=1]$ (correlated sampling: $P(a=2, b=1) = 0.8 + 0.2/9 \approx 0.822$; other pairs $0.2/9 \approx 0.022$).

### Counterfactual Coupling
For each episode, outcome noise $u \sim \text{Uniform}(0, 1)$ determines factual success $\mathbf{1}(u < p(a, b))$. For active skills ($a > 0$ or $b > 0$), counterfactual evaluations replace the active skill with action $0$ holding the teammate action and the exact same noise $u$ fixed: $\mathbf{1}(u < p(0, b))$ and $\mathbf{1}(u < p(a, 0))$.

### Attribution Estimands
For each skill $s$ over active episodes $U_s$:
- **Individual Attribution (observational usage conditioning):**
  $$\text{Individual}(s) = \frac{1}{|U_s|} \sum_{i \in U_s} \text{success}_i$$
- **Ablation Attribution (counterfactual difference):**
  $$\text{Ablation}(s) = \frac{1}{|U_s|} \sum_{i \in U_s} \left(\text{success}_i - \text{success}_i^{(-s)}\right)$$

  where $\text{success}_i^{(-s)}$ is the paired counterfactual outcome with skill $s$ removed (`without_a` or `without_b` in `episodes.csv`).

## 4. Generated Artifacts

Executing `experiment.py` writes four reproducible files to `--out` (`results/`):
- `results/episodes.csv`: 81,000 episode rows containing scenario, seed, episode ID, actions $(a, b)$, noise $u$, factual success, and counterfactual outcomes (`without_a`, `without_b`).
- `results/attribution.csv`: Per-seed attribution records containing sample counts, empirical estimates, and exact oracle expectations for each skill.
- `results/table.md`: 12-row markdown table reporting empirical means $\pm$ sample standard deviations, exact oracle values, interaction diagnostics, rank-reversal frequencies, and evaluation counts.
- `results/metadata.json`: Environment specifications, mathematical formulas, sampling distributions, evaluation counters, and Python environment metadata (timestamps omitted for byte-level reproducibility).

## 5. Repository Structure and Deliverables

- `memo.pdf` / `memo.md`: Two-page critical memo analyzing BAMAS, MASkills, COMA, and proposing factorial validation protocols.
- `experiment.py`: Minimal, self-contained implementation of the two-agent simulation, counterfactual ablation, exact oracle calculations, and CLI.
- `test_experiment.py`: Comprehensive test suite verifying analytical targets, noise coupling, confounding rank reversals, reproducibility, and CLI error handling.
- `results/`: Directory containing all generated experimental outputs and tables.
- `experiment-summary.pdf` / `experiment-summary.md`: One-page summary detailing experimental setup, findings, optimization risks, real-world LLM challenges, and limitations.
- `reflection.md`: Five-paragraph reflection detailing research interests, critical judgments, immediate next steps, self-assessment, and forward-looking research questions.
