#!/usr/bin/env python3
"""Multi-agent attribution experiment.

Compares individual (usage-conditioned success) and ablation (counterfactual delta)
attribution methods in additive, synergy, and confounded two-agent environments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Literal, Sequence

Action = Literal[0, 1, 2]
Scenario = Literal["additive", "synergy", "confounded"]
Skill = Literal["A1", "A2", "B1", "B2"]

SCENARIOS: tuple[Scenario, ...] = ("additive", "synergy", "confounded")
SKILLS: tuple[Skill, ...] = ("A1", "A2", "B1", "B2")
ACTIONS: tuple[Action, ...] = (0, 1, 2)


@dataclass(frozen=True)
class Episode:
    scenario: Scenario
    seed: int
    episode_id: int
    a: Action
    b: Action
    u: float
    success: int
    without_a: int | None
    without_b: int | None


@dataclass(frozen=True)
class Attribution:
    n_used: int
    individual: float | None
    ablation: float | None


def _validate_scenario(scenario: Scenario) -> None:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; must be one of {SCENARIOS}")


def _validate_action(action: Action, name: str) -> None:
    if action not in ACTIONS:
        raise ValueError(f"Invalid action {action!r} for {name}; must be 0, 1, or 2")


def _validate_skill(skill: Skill) -> None:
    if skill not in SKILLS:
        raise ValueError(f"Unknown skill {skill!r}; must be one of {SKILLS}")


def success_probability(scenario: Scenario, a: Action, b: Action) -> float:
    """Compute binary success probability p(a, b) for given actions."""
    _validate_scenario(scenario)
    _validate_action(a, "a")
    _validate_action(b, "b")

    if scenario == "additive":
        return 0.10 + (0.20 if a == 1 else 0.0) - (0.05 if a == 2 else 0.0) + (0.30 if b == 1 else 0.0)
    elif scenario == "synergy":
        return 0.10 + (0.60 if (a == 1 and b == 1) else 0.0) - (0.05 if a == 2 else 0.0)
    elif scenario == "confounded":
        return 0.10 + (0.20 if a == 1 else 0.0) + (0.50 if b == 1 else 0.0)
    else:
        raise ValueError(f"Unhandled scenario: {scenario}")


def joint_probabilities(scenario: Scenario) -> tuple[tuple[Action, Action, float], ...]:
    """Return joint distribution over all nine (a, b) pairs in lexicographical order."""
    _validate_scenario(scenario)
    pairs: list[tuple[Action, Action, float]] = []
    for a in ACTIONS:
        for b in ACTIONS:
            if scenario in ("additive", "synergy"):
                p = 1.0 / 9.0
            elif scenario == "confounded":
                p = 0.8 + 0.2 / 9.0 if (a == 2 and b == 1) else 0.2 / 9.0
            else:
                raise ValueError(f"Unhandled scenario: {scenario}")
            pairs.append((a, b, p))
    return tuple(pairs)


def simulate(scenario: Scenario, seed: int, n_episodes: int) -> list[Episode]:
    """Generate simulated episodes with paired counterfactual interventions."""
    _validate_scenario(scenario)
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")

    rng = random.Random(seed)
    joint = joint_probabilities(scenario)
    episodes: list[Episode] = []

    for ep_id in range(n_episodes):
        v = rng.random()
        cum = 0.0
        chosen_a: Action = 2
        chosen_b: Action = 2
        for a, b, p in joint:
            cum += p
            if v < cum:
                chosen_a = a
                chosen_b = b
                break

        u = rng.random()
        p_fact = success_probability(scenario, chosen_a, chosen_b)
        success = 1 if u < p_fact else 0

        without_a: int | None = None
        without_b: int | None = None

        if chosen_a > 0:
            p_no_a = success_probability(scenario, 0, chosen_b)
            without_a = 1 if u < p_no_a else 0

        if chosen_b > 0:
            p_no_b = success_probability(scenario, chosen_a, 0)
            without_b = 1 if u < p_no_b else 0

        episodes.append(
            Episode(
                scenario=scenario,
                seed=seed,
                episode_id=ep_id,
                a=chosen_a,
                b=chosen_b,
                u=u,
                success=success,
                without_a=without_a,
                without_b=without_b,
            )
        )

    return episodes


def estimate(episodes: Sequence[Episode], skill: Skill) -> Attribution:
    """Compute individual and ablation attribution for a skill over episodes."""
    _validate_skill(skill)
    if not episodes:
        return Attribution(n_used=0, individual=None, ablation=None)

    if skill == "A1":
        used = [ep for ep in episodes if ep.a == 1]
        diff_fn = lambda ep: ep.success - (ep.without_a if ep.without_a is not None else 0)
    elif skill == "A2":
        used = [ep for ep in episodes if ep.a == 2]
        diff_fn = lambda ep: ep.success - (ep.without_a if ep.without_a is not None else 0)
    elif skill == "B1":
        used = [ep for ep in episodes if ep.b == 1]
        diff_fn = lambda ep: ep.success - (ep.without_b if ep.without_b is not None else 0)
    elif skill == "B2":
        used = [ep for ep in episodes if ep.b == 2]
        diff_fn = lambda ep: ep.success - (ep.without_b if ep.without_b is not None else 0)
    else:
        raise ValueError(f"Unhandled skill: {skill}")

    n_used = len(used)
    if n_used == 0:
        return Attribution(n_used=0, individual=None, ablation=None)

    indiv = sum(ep.success for ep in used) / n_used
    abl = sum(diff_fn(ep) for ep in used) / n_used
    return Attribution(n_used=n_used, individual=indiv, ablation=abl)


def exact_scores(scenario: Scenario, skill: Skill) -> tuple[float, float]:
    """Compute exact (individual, ablation) oracle expectations under logging distribution."""
    _validate_scenario(scenario)
    _validate_skill(skill)

    joint = joint_probabilities(scenario)

    if skill == "A1":
        matches = lambda a, b: a == 1
        delta_fn = lambda a, b: success_probability(scenario, 1, b) - success_probability(scenario, 0, b)
    elif skill == "A2":
        matches = lambda a, b: a == 2
        delta_fn = lambda a, b: success_probability(scenario, 2, b) - success_probability(scenario, 0, b)
    elif skill == "B1":
        matches = lambda a, b: b == 1
        delta_fn = lambda a, b: success_probability(scenario, a, 1) - success_probability(scenario, a, 0)
    elif skill == "B2":
        matches = lambda a, b: b == 2
        delta_fn = lambda a, b: success_probability(scenario, a, 2) - success_probability(scenario, a, 0)
    else:
        raise ValueError(f"Unhandled skill: {skill}")

    p_used = sum(p for a, b, p in joint if matches(a, b))
    if p_used <= 0.0:
        return (0.0, 0.0)

    exact_indiv = sum(p * success_probability(scenario, a, b) for a, b, p in joint if matches(a, b)) / p_used
    exact_abl = sum(p * delta_fn(a, b) for a, b, p in joint if matches(a, b)) / p_used

    return (exact_indiv, exact_abl)


def interaction_diagnostic(scenario: Scenario) -> float:
    """Compute exact A1/B1 interaction: p(1,1) - p(1,0) - p(0,1) + p(0,0)."""
    return (
        success_probability(scenario, 1, 1)
        - success_probability(scenario, 1, 0)
        - success_probability(scenario, 0, 1)
        + success_probability(scenario, 0, 0)
    )


def run(output_dir: Path, n_episodes: int, n_seeds: int) -> None:
    """Run simulation across all scenarios and seeds, saving artifacts to output_dir."""
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    if n_seeds < 2:
        raise ValueError(f"n_seeds must be at least 2 for sample standard deviation, got {n_seeds}")

    # Verify output_dir is not an existing non-directory file
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Target path {output_dir} exists and is not a directory")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_episodes: list[Episode] = []
    total_factual_evals = 0
    total_ablation_evals = 0

    # scenario -> skill -> list of (indiv, abl) across seeds
    estimates_by_scenario_skill: dict[Scenario, dict[Skill, list[tuple[float | None, float | None]]]] = {
        sc: {sk: [] for sk in SKILLS} for sc in SCENARIOS
    }
    # scenario -> skill -> list of n_used across seeds
    n_used_by_scenario_skill: dict[Scenario, dict[Skill, list[int]]] = {
        sc: {sk: [] for sk in SKILLS} for sc in SCENARIOS
    }

    # Tracking for rank reversal in confounded scenario across seeds
    # Seed -> (indiv_a1, indiv_a2, abl_a1, abl_a2)
    confounded_seed_attributions: list[dict[Skill, Attribution]] = []

    attribution_rows: list[list[str]] = []

    for sc in SCENARIOS:
        for seed in range(n_seeds):
            eps = simulate(sc, seed, n_episodes)
            all_episodes.extend(eps)
            total_factual_evals += len(eps)
            for ep in eps:
                if ep.a > 0:
                    total_ablation_evals += 1
                if ep.b > 0:
                    total_ablation_evals += 1

            seed_attr: dict[Skill, Attribution] = {}
            for sk in SKILLS:
                attr = estimate(eps, sk)
                seed_attr[sk] = attr
                estimates_by_scenario_skill[sc][sk].append((attr.individual, attr.ablation))
                n_used_by_scenario_skill[sc][sk].append(attr.n_used)

                ex_indiv, ex_abl = exact_scores(sc, sk)
                indiv_str = f"{attr.individual:.17g}" if attr.individual is not None else ""
                abl_str = f"{attr.ablation:.17g}" if attr.ablation is not None else ""
                attribution_rows.append(
                    [
                        sc,
                        str(seed),
                        sk,
                        str(attr.n_used),
                        indiv_str,
                        abl_str,
                        f"{ex_indiv:.17g}",
                        f"{ex_abl:.17g}",
                    ]
                )

            if sc == "confounded":
                confounded_seed_attributions.append(seed_attr)

    # 1. Write episodes.csv
    episodes_file = output_dir / "episodes.csv"
    with episodes_file.open("w", newline="", encoding="utf-8") as f:
        f.write("scenario,seed,episode_id,a,b,u,success,without_a,without_b\n")
        for ep in all_episodes:
            wa = str(ep.without_a) if ep.without_a is not None else ""
            wb = str(ep.without_b) if ep.without_b is not None else ""
            f.write(f"{ep.scenario},{ep.seed},{ep.episode_id},{ep.a},{ep.b},{ep.u:.17g},{ep.success},{wa},{wb}\n")

    # 2. Write attribution.csv
    attribution_file = output_dir / "attribution.csv"
    with attribution_file.open("w", newline="", encoding="utf-8") as f:
        f.write("scenario,seed,skill,n_used,individual,ablation,exact_individual,exact_ablation\n")
        for row in attribution_rows:
            f.write(",".join(row) + "\n")

    # 3. Compute summary statistics for table.md
    missing_estimates_count = 0
    table_lines: list[str] = [
        "| Scenario | Skill | Usage mean ± SD | Exact usage | Ablation mean ± SD | Exact delta |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for sc in SCENARIOS:
        for sk in SKILLS:
            ex_indiv, ex_abl = exact_scores(sc, sk)
            indiv_vals = [i for i, _ in estimates_by_scenario_skill[sc][sk] if i is not None]
            abl_vals = [a for _, a in estimates_by_scenario_skill[sc][sk] if a is not None]

            missing_indiv = len(estimates_by_scenario_skill[sc][sk]) - len(indiv_vals)
            missing_abl = len(estimates_by_scenario_skill[sc][sk]) - len(abl_vals)
            missing_estimates_count += missing_indiv + missing_abl

            if len(indiv_vals) >= 2:
                indiv_mean = statistics.mean(indiv_vals)
                indiv_sd = statistics.stdev(indiv_vals)
                indiv_str = f"{indiv_mean:.3f} ± {indiv_sd:.3f}"
            elif len(indiv_vals) == 1:
                indiv_str = f"{indiv_vals[0]:.3f} ± NA"
            else:
                indiv_str = "NA"

            if len(abl_vals) >= 2:
                abl_mean = statistics.mean(abl_vals)
                abl_sd = statistics.stdev(abl_vals)
                abl_str = f"{abl_mean:.3f} ± {abl_sd:.3f}"
            elif len(abl_vals) == 1:
                abl_str = f"{abl_vals[0]:.3f} ± NA"
            else:
                abl_str = "NA"

            table_lines.append(
                f"| {sc} | {sk} | {indiv_str} | {ex_indiv:.3f} | {abl_str} | {ex_abl:.3f} |"
            )

    # Diagnostics and counts
    inter_additive = interaction_diagnostic("additive")
    inter_synergy = interaction_diagnostic("synergy")
    inter_confounded = interaction_diagnostic("confounded")

    # Rank reversal counts for confounded scenario
    a2_over_a1_indiv = 0
    a1_over_a2_abl = 0
    both_correct = 0

    for attr_map in confounded_seed_attributions:
        a1_attr = attr_map["A1"]
        a2_attr = attr_map["A2"]
        indiv_reversed = (
            a2_attr.individual is not None
            and a1_attr.individual is not None
            and a2_attr.individual > a1_attr.individual
        )
        abl_corrected = (
            a1_attr.ablation is not None
            and a2_attr.ablation is not None
            and a1_attr.ablation > a2_attr.ablation
        )
        if indiv_reversed:
            a2_over_a1_indiv += 1
        if abl_corrected:
            a1_over_a2_abl += 1
        if indiv_reversed and abl_corrected:
            both_correct += 1

    table_md = "\n".join(table_lines) + "\n\n"
    table_md += "### Diagnostic Summary\n\n"
    table_md += f"- **A1/B1 Interaction Diagnostic**: additive = {inter_additive:.3f}, synergy = {inter_synergy:.3f}, confounded = {inter_confounded:.3f}\n"
    table_md += (
        f"- **Confounded Rank Reversal**: A2 outranked A1 in individual attribution on {a2_over_a1_indiv}/{n_seeds} seeds; "
        f"A1 outranked A2 in ablation attribution on {a1_over_a2_abl}/{n_seeds} seeds "
        f"(both holds simultaneously on {both_correct}/{n_seeds} seeds).\n"
    )
    table_md += f"- **Missing Estimates Count**: {missing_estimates_count}\n"
    table_md += f"- **Environment Evaluations**: factual = {total_factual_evals}, ablation = {total_ablation_evals}, total = {total_factual_evals + total_ablation_evals}\n"

    table_file = output_dir / "table.md"
    with table_file.open("w", encoding="utf-8") as f:
        f.write(table_md)

    # 4. Write metadata.json
    metadata = {
        "ablation_evaluations": total_ablation_evals,
        "episode_count_per_scenario_seed": n_episodes,
        "factual_evaluations": total_factual_evals,
        "joint_sampling_probabilities": {
            "additive": "uniform 1/9 across all (a,b) in {0,1,2}x{0,1,2}",
            "confounded": "0.8 * [a=2 and b=1] + 0.2/9 across (a,b) in {0,1,2}x{0,1,2}",
            "synergy": "uniform 1/9 across all (a,b) in {0,1,2}x{0,1,2}",
        },
        "missing_estimates_count": missing_estimates_count,
        "no_skill_intervention": "Action of active agent replaced by 0; teammate action and outcome noise u held fixed.",
        "python_version": sys.version,
        "rank_reversal_counts": {
            "a1_over_a2_ablation": a1_over_a2_abl,
            "a2_over_a1_individual": a2_over_a1_indiv,
            "both_simultaneous": both_correct,
            "total_seeds": n_seeds,
        },
        "reward_formulas": {
            "additive": "p(a,b) = 0.10 + 0.20[a=1] - 0.05[a=2] + 0.30[b=1]",
            "confounded": "p(a,b) = 0.10 + 0.20[a=1] + 0.50[b=1]",
            "synergy": "p(a,b) = 0.10 + 0.60[a=1 and b=1] - 0.05[a=2]",
        },
        "scenarios": list(SCENARIOS),
        "seeds": list(range(n_seeds)),
        "shared_noise_coupling": "Outcome success = int(u < p(a,b)); counterfactual without agent = int(u < p(0,b)) using same uniform u.",
        "skills": list(SKILLS),
        "total_evaluations": total_factual_evals + total_ablation_evals,
    }

    metadata_file = output_dir / "metadata.json"
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent attribution experiment.")
    parser.add_argument("--episodes", type=int, default=900, help="Number of episodes per scenario per seed (default: 900)")
    parser.add_argument("--seeds", type=int, default=30, help="Number of random seeds (default: 30)")
    parser.add_argument("--out", type=Path, default=Path("results"), help="Output directory path (default: results)")

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error(f"--episodes must be positive, got {args.episodes}")
    if args.seeds < 2:
        parser.error(f"--seeds must be at least 2 for sample standard deviation, got {args.seeds}")

    run(args.out, args.episodes, args.seeds)


if __name__ == "__main__":
    main()
