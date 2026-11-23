"""Unit and behavioral tests for multi-agent attribution experiment."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiment import (
    ACTIONS,
    SCENARIOS,
    SKILLS,
    Action,
    Attribution,
    Episode,
    Scenario,
    Skill,
    estimate,
    exact_scores,
    interaction_diagnostic,
    joint_probabilities,
    run,
    simulate,
    success_probability,
)


class TestExactTargetsAndProbabilities(unittest.TestCase):
    def test_probabilities_in_bounds_and_no_clipping(self) -> None:
        """Verify all 9 joint reward probabilities lie in [0, 1] without clipping."""
        for sc in SCENARIOS:
            for a in ACTIONS:
                for b in ACTIONS:
                    p = success_probability(sc, a, b)
                    self.assertGreaterEqual(p, 0.0)
                    self.assertLessEqual(p, 1.0)

    def test_joint_probabilities_sum_to_one(self) -> None:
        """Verify sampling distribution sums to 1.0 across all 9 configurations."""
        for sc in SCENARIOS:
            joint = joint_probabilities(sc)
            self.assertEqual(len(joint), 9)
            total_p = sum(p for _, _, p in joint)
            self.assertAlmostEqual(total_p, 1.0, places=12)

    def test_exact_analytical_targets(self) -> None:
        """Verify exact rational targets for all scenarios and skills."""
        # Additive
        indiv, abl = exact_scores("additive", "A1")
        self.assertAlmostEqual(indiv, 0.40, places=12)
        self.assertAlmostEqual(abl, 0.20, places=12)

        indiv, abl = exact_scores("additive", "A2")
        self.assertAlmostEqual(indiv, 0.15, places=12)
        self.assertAlmostEqual(abl, -0.05, places=12)

        indiv, abl = exact_scores("additive", "B1")
        self.assertAlmostEqual(indiv, 0.45, places=12)
        self.assertAlmostEqual(abl, 0.30, places=12)

        indiv, abl = exact_scores("additive", "B2")
        self.assertAlmostEqual(indiv, 0.15, places=12)
        self.assertAlmostEqual(abl, 0.0, places=12)

        # Synergy
        indiv, abl = exact_scores("synergy", "A1")
        self.assertAlmostEqual(indiv, 0.30, places=12)
        self.assertAlmostEqual(abl, 0.20, places=12)

        indiv, abl = exact_scores("synergy", "A2")
        self.assertAlmostEqual(indiv, 0.05, places=12)
        self.assertAlmostEqual(abl, -0.05, places=12)

        indiv, abl = exact_scores("synergy", "B1")
        self.assertAlmostEqual(indiv, 17.0 / 60.0, places=12)
        self.assertAlmostEqual(abl, 0.20, places=12)

        indiv, abl = exact_scores("synergy", "B2")
        self.assertAlmostEqual(indiv, 1.0 / 12.0, places=12)
        self.assertAlmostEqual(abl, 0.0, places=12)

        # Confounded
        indiv, abl = exact_scores("confounded", "A1")
        self.assertAlmostEqual(indiv, 7.0 / 15.0, places=12)
        self.assertAlmostEqual(abl, 0.20, places=12)

        indiv, abl = exact_scores("confounded", "A2")
        self.assertAlmostEqual(indiv, 112.0 / 195.0, places=12)
        self.assertAlmostEqual(abl, 0.0, places=12)

        indiv, abl = exact_scores("confounded", "B1")
        self.assertAlmostEqual(indiv, 118.0 / 195.0, places=12)
        self.assertAlmostEqual(abl, 0.50, places=12)

        indiv, abl = exact_scores("confounded", "B2")
        self.assertAlmostEqual(indiv, 1.0 / 6.0, places=12)
        self.assertAlmostEqual(abl, 0.0, places=12)


class TestSynergyAndConfoundingMechanisms(unittest.TestCase):
    def test_synergy_intervention_and_interaction(self) -> None:
        """Verify synergy pair success 0.70, singletons 0.10, delta -0.60 on ablation."""
        p_pair = success_probability("synergy", 1, 1)
        p_a1_only = success_probability("synergy", 1, 0)
        p_b1_only = success_probability("synergy", 0, 1)
        p_none = success_probability("synergy", 0, 0)

        self.assertAlmostEqual(p_pair, 0.70, places=12)
        self.assertAlmostEqual(p_a1_only, 0.10, places=12)
        self.assertAlmostEqual(p_b1_only, 0.10, places=12)
        self.assertAlmostEqual(p_none, 0.10, places=12)

        # Solo gains over none
        self.assertAlmostEqual(p_a1_only - p_none, 0.0, places=12)
        self.assertAlmostEqual(p_b1_only - p_none, 0.0, places=12)

        # Ablation of either active skill in the pair loses 0.60
        self.assertAlmostEqual(p_pair - p_b1_only, 0.60, places=12)
        self.assertAlmostEqual(p_pair - p_a1_only, 0.60, places=12)

        # Interaction diagnostic
        inter = interaction_diagnostic("synergy")
        self.assertAlmostEqual(inter, 0.60, places=12)

    def test_confounding_reversal_mechanism(self) -> None:
        """Verify A2 has zero ablation delta for every b, yet oracle usage exceeds A1."""
        for b in ACTIONS:
            delta = success_probability("confounded", 2, b) - success_probability("confounded", 0, b)
            self.assertAlmostEqual(delta, 0.0, places=12)

        a1_usage, a1_abl = exact_scores("confounded", "A1")
        a2_usage, a2_abl = exact_scores("confounded", "A2")

        self.assertGreater(a2_usage, a1_usage)
        self.assertGreater(a1_abl, a2_abl)
        self.assertAlmostEqual(a2_abl, 0.0, places=12)
        self.assertAlmostEqual(a1_abl, 0.20, places=12)


class TestPairedStochasticityAndCoupling(unittest.TestCase):
    def test_paired_stochasticity_synergy_pair(self) -> None:
        """Check explicit u=0.08 and u=0.40 in synergy pair (1,1)."""
        # p(1,1) = 0.70, p(0,1) = 0.10, p(1,0) = 0.10
        # At u = 0.08: factual succeeds (0.08 < 0.70), removal succeeds (0.08 < 0.10) -> delta = 0
        # At u = 0.40: factual succeeds (0.40 < 0.70), removal fails (0.40 >= 0.10) -> delta = 1
        ep_08 = Episode(
            scenario="synergy",
            seed=0,
            episode_id=0,
            a=1,
            b=1,
            u=0.08,
            success=int(0.08 < 0.70),
            without_a=int(0.08 < 0.10),
            without_b=int(0.08 < 0.10),
        )
        self.assertEqual(ep_08.success, 1)
        self.assertEqual(ep_08.without_a, 1)
        self.assertEqual(ep_08.without_b, 1)
        attr_08_a1 = estimate([ep_08], "A1")
        self.assertEqual(attr_08_a1.ablation, 0.0)

        ep_40 = Episode(
            scenario="synergy",
            seed=0,
            episode_id=1,
            a=1,
            b=1,
            u=0.40,
            success=int(0.40 < 0.70),
            without_a=int(0.40 < 0.10),
            without_b=int(0.40 < 0.10),
        )
        self.assertEqual(ep_40.success, 1)
        self.assertEqual(ep_40.without_a, 0)
        self.assertEqual(ep_40.without_b, 0)
        attr_40_a1 = estimate([ep_40], "A1")
        self.assertEqual(attr_40_a1.ablation, 1.0)

    def test_simulated_episodes_noise_coupling(self) -> None:
        """Verify generated episodes strictly match stored u and exact probabilities."""
        episodes = simulate("additive", seed=42, n_episodes=200)
        for ep in episodes:
            p_fact = success_probability("additive", ep.a, ep.b)
            expected_success = 1 if ep.u < p_fact else 0
            self.assertEqual(ep.success, expected_success)

            if ep.a > 0:
                p_no_a = success_probability("additive", 0, ep.b)
                expected_no_a = 1 if ep.u < p_no_a else 0
                self.assertEqual(ep.without_a, expected_no_a)
            else:
                self.assertIsNone(ep.without_a)

            if ep.b > 0:
                p_no_b = success_probability("additive", ep.a, 0)
                expected_no_b = 1 if ep.u < p_no_b else 0
                self.assertEqual(ep.without_b, expected_no_b)
            else:
                self.assertIsNone(ep.without_b)

    def test_irrelevant_skill_ablation_invariance(self) -> None:
        """In confounded scenario, A2 has zero effect: success == without_a for all episodes with a=2."""
        episodes = simulate("confounded", seed=123, n_episodes=500)
        a2_episodes = [ep for ep in episodes if ep.a == 2]
        self.assertGreater(len(a2_episodes), 0)
        for ep in a2_episodes:
            self.assertEqual(ep.success, ep.without_a)


class TestEdgeCasesAndValidation(unittest.TestCase):
    def test_missing_skill_and_empty_input(self) -> None:
        """Empty input or unobserved skill returns n_used=0, None, None (not 0.0)."""
        attr_empty = estimate([], "A1")
        self.assertEqual(attr_empty, Attribution(n_used=0, individual=None, ablation=None))

        # Episodes with only a=0, b=0
        episodes = [
            Episode("additive", 0, 0, 0, 0, 0.5, 0, None, None),
            Episode("additive", 0, 1, 0, 0, 0.05, 1, None, None),
        ]
        attr_unobserved = estimate(episodes, "A1")
        self.assertEqual(attr_unobserved, Attribution(n_used=0, individual=None, ablation=None))
        self.assertIsNone(attr_unobserved.individual)
        self.assertIsNone(attr_unobserved.ablation)

    def test_invalid_parameters_raise_value_error(self) -> None:
        """Invalid scenarios, actions, skills, or episode counts must raise ValueError."""
        with self.assertRaises(ValueError):
            success_probability("invalid_scenario", 0, 0)  # type: ignore

        with self.assertRaises(ValueError):
            success_probability("additive", 3, 0)  # type: ignore

        with self.assertRaises(ValueError):
            success_probability("additive", 0, -1)  # type: ignore

        with self.assertRaises(ValueError):
            joint_probabilities("bogus")  # type: ignore

        with self.assertRaises(ValueError):
            simulate("additive", seed=0, n_episodes=0)

        with self.assertRaises(ValueError):
            simulate("additive", seed=0, n_episodes=-10)

        with self.assertRaises(ValueError):
            estimate([], "InvalidSkill")  # type: ignore

        with self.assertRaises(ValueError):
            exact_scores("synergy", "C1")  # type: ignore


class TestReproducibilityAndCLI(unittest.TestCase):
    def test_reproducibility_same_seed(self) -> None:
        """Same seed must produce identical episodes; different seed must differ."""
        eps1 = simulate("additive", seed=99, n_episodes=50)
        eps2 = simulate("additive", seed=99, n_episodes=50)
        eps3 = simulate("additive", seed=100, n_episodes=50)

        self.assertEqual(eps1, eps2)
        self.assertNotEqual(eps1, eps3)

    def test_deterministic_rerun_byte_equality(self) -> None:
        """Two runs with identical parameters must produce byte-identical files."""
        with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
            path1 = Path(dir1)
            path2 = Path(dir2)

            run(path1, n_episodes=100, n_seeds=3)
            run(path2, n_episodes=100, n_seeds=3)

            for fname in ("episodes.csv", "attribution.csv", "table.md", "metadata.json"):
                bytes1 = (path1 / fname).read_bytes()
                bytes2 = (path2 / fname).read_bytes()
                self.assertEqual(bytes1, bytes2, f"File {fname} differs across identical runs")

    def test_cli_invalid_args_and_file_as_output_dir(self) -> None:
        """CLI must fail with nonzero exit code for invalid episodes, seeds, or file output path."""
        # --episodes <= 0
        res = subprocess.run(
            [sys.executable, "experiment.py", "--episodes", "0", "--seeds", "5"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--episodes must be positive", res.stderr)

        # --seeds < 2
        res = subprocess.run(
            [sys.executable, "experiment.py", "--episodes", "10", "--seeds", "1"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("--seeds must be at least 2", res.stderr)

        # Target output is an existing file
        with tempfile.NamedTemporaryFile() as tmp_file:
            res = subprocess.run(
                [sys.executable, "experiment.py", "--episodes", "10", "--seeds", "2", "--out", tmp_file.name],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(res.returncode, 0)


if __name__ == "__main__":
    unittest.main()
