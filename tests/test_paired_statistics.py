"""Offline paired, multi-seed statistics tests."""

import unittest

from evaluation.paired_statistics import (
    compare_paired_methods,
    paired_bootstrap_ci,
    paired_permutation_test,
)


class PairedStatisticsTests(unittest.TestCase):
    def test_pair_clustered_multi_seed_comparison_and_failure_counts(self):
        records = []
        values = {
            ("p1", 1): (0.2, 0.5),
            ("p1", 2): (0.3, 0.6),
            ("p2", 1): (0.1, 0.5),
            ("p2", 2): (0.2, 0.6),
        }
        for (pair_id, seed), (baseline, candidate) in values.items():
            records.extend(
                [
                    {
                        "pair_id": pair_id, "seed": seed,
                        "method": "standard", "score": baseline,
                    },
                    {
                        "pair_id": pair_id, "seed": seed,
                        "method": "adaptive", "score": candidate,
                    },
                ]
            )
        records.append(
            {
                "pair_id": "p3", "seed": 1, "method": "adaptive",
                "status": "failed",
            }
        )
        result = compare_paired_methods(
            records,
            baseline_method="standard",
            metric="score",
            bootstrap_samples=200,
            seed=9,
        )
        comparison = result["comparisons"]["adaptive"]
        self.assertEqual(comparison["n_pairs"], 2)
        self.assertEqual(comparison["n_observations"], 4)
        self.assertEqual(comparison["seeds_per_pair"], {"p1": 2, "p2": 2})
        self.assertAlmostEqual(comparison["mean_improvement"], 0.35)
        self.assertLessEqual(
            comparison["bootstrap_ci"]["low"],
            comparison["mean_improvement"],
        )
        self.assertGreaterEqual(
            comparison["bootstrap_ci"]["high"],
            comparison["mean_improvement"],
        )
        self.assertAlmostEqual(
            comparison["permutation_test"]["p_value"], 0.5
        )
        self.assertEqual(
            result["failure_counts"]["adaptive"]["failed_records"], 1
        )

    def test_lower_is_better_orients_improvement(self):
        records = [
            {"pair_id": "p1", "method": "base", "seed": 1, "error": 0.5},
            {"pair_id": "p1", "method": "new", "seed": 1, "error": 0.3},
        ]
        result = compare_paired_methods(
            records,
            baseline_method="base",
            metric="error",
            higher_is_better=False,
            bootstrap_samples=20,
        )
        comparison = result["comparisons"]["new"]
        self.assertAlmostEqual(comparison["mean_raw_difference"], -0.2)
        self.assertAlmostEqual(comparison["mean_improvement"], 0.2)

    def test_bootstrap_and_permutation_are_deterministic(self):
        first = paired_bootstrap_ci(
            {"a": [1.0, 2.0], "b": [3.0]}, n_resamples=100, seed=4
        )
        second = paired_bootstrap_ci(
            {"a": [1.0, 2.0], "b": [3.0]}, n_resamples=100, seed=4
        )
        self.assertEqual(first, second)
        permutation = paired_permutation_test([1.0, 1.0])
        self.assertTrue(permutation["exact"])
        self.assertEqual(permutation["p_value"], 0.5)


if __name__ == "__main__":
    unittest.main()
