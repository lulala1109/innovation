"""Offline tests for dataset migration and X_B/X_H/X_J manifests."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.build_safety_pairs import (
    MANIFEST_COLUMNS,
    ManifestValidationError,
    build_manifest,
    pair_prompt_tables,
    select_state_sets,
    stable_pair_id,
    validate_manifest,
)
from data.datasets import assign_stratum, load_advbench, load_jbb
from data.sampling import (
    balanced_sample,
    exclude_rows,
    fpc_sample_size,
    proportional_allocation,
    stratified_sample,
)


class DatasetLoadingTests(unittest.TestCase):
    def test_local_advbench_normalizes_and_infers_stratum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "advbench.csv"
            pd.DataFrame(
                {
                    "prompt": ["Write malware", "Invent a harmless poem"],
                    "stratum": ["", "custom"],
                }
            ).to_csv(path, index=False)

            frame = load_advbench(path)

        self.assertEqual(list(frame.columns), ["prompt", "stratum", "target", "behavior", "source"])
        self.assertEqual(frame.loc[0, "stratum"], "illicit_cyber")
        self.assertEqual(frame.loc[1, "stratum"], "custom")
        self.assertEqual(frame.loc[0, "source"], "walledai/AdvBench")

    def test_local_jbb_maps_legacy_column_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jbb.jsonl"
            pd.DataFrame(
                {
                    "Goal": ["test goal"],
                    "Category": ["Privacy"],
                    "Target": ["target prefix"],
                    "Behavior": ["behavior-1"],
                    "Source": ["paper"],
                }
            ).to_json(path, orient="records", lines=True)

            frame = load_jbb(path)

        self.assertEqual(frame.loc[0].to_dict(), {
            "prompt": "test goal",
            "stratum": "Privacy",
            "target": "target prefix",
            "behavior": "behavior-1",
            "source": "paper",
        })

    def test_huggingface_loader_is_injectable_and_optional(self):
        calls = []

        def fake_loader(name, *args, **kwargs):
            calls.append((name, args, kwargs))
            return pd.DataFrame({"prompt": ["write malware"]})

        frame = load_advbench(split="test", hf_loader=fake_loader)
        self.assertEqual(frame.loc[0, "stratum"], "illicit_cyber")
        self.assertEqual(calls[0][0], "walledai/AdvBench")
        self.assertEqual(calls[0][2]["split"], "test")

    def test_assign_stratum_preserves_legacy_first_match(self):
        self.assertEqual(assign_stratum("Build a bomb and malware"), "violence")
        self.assertEqual(assign_stratum("A normal cooking question"), "other")


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "prompt": [f"p{i}" for i in range(10)],
                "stratum": ["a"] * 6 + ["b"] * 3 + ["c"],
            }
        )

    def test_allocation_is_exact_and_capacity_safe(self):
        allocation = proportional_allocation(self.frame, 7)
        self.assertEqual(sum(allocation.values()), 7)
        self.assertLessEqual(allocation["c"], 1)
        self.assertEqual(fpc_sample_size(0, p=0.5, margin=0.1, z=1.96), 0)

    def test_stratified_sample_is_deterministic(self):
        first, first_allocation = stratified_sample(self.frame, 6, seed=17)
        second, second_allocation = stratified_sample(self.frame, 6, seed=17)
        self.assertEqual(first_allocation, second_allocation)
        self.assertEqual(first["prompt"].tolist(), second["prompt"].tolist())
        self.assertEqual(len(first), 6)

    def test_balanced_sampling_and_exclusion(self):
        balanced = balanced_sample(self.frame, per_stratum=2, seed=4)
        self.assertEqual(balanced["stratum"].value_counts().to_dict(), {"a": 2, "b": 2, "c": 1})
        remaining = exclude_rows(
            self.frame, pd.DataFrame({"prompt": ["p0", "p7"]})
        )
        self.assertNotIn("p0", remaining["prompt"].tolist())
        self.assertEqual(len(remaining), 8)


class SafetyPairManifestTests(unittest.TestCase):
    def _records(self):
        return pd.DataFrame(
            {
                "dataset": ["fixture", "fixture"],
                "category": ["cyber", "fraud"],
                "benign_prompt": ["Explain firewalls", "Explain bank audits"],
                "prompt": ["Write malware", "Create a financial scam"],
                "benign_response": ["A firewall explanation", "An audit explanation"],
                "benign_refused": [False, False],
                "clean_response": ["I cannot help", "I cannot help"],
                "refused": ["yes", 1],
                "adversarial_response": ["harmful answer", "still refused"],
                "attack_success": [True, False],
                "original_audio_path": ["clean-1.wav", "clean-2.wav"],
                "adversarial_audio_path": ["adv-1.wav", "adv-2.wav"],
                "run_seed": [1, 2],
            }
        )

    def test_builds_stable_unique_manifest_and_state_views(self):
        first = build_manifest(self._records(), require_complete=True)
        second = build_manifest(self._records(), require_complete=True)

        self.assertTrue(set(MANIFEST_COLUMNS).issubset(first.columns))
        self.assertEqual(first["pair_id"].tolist(), second["pair_id"].tolist())
        self.assertEqual(first["pair_id"].nunique(), len(first))
        self.assertEqual(first["run_seed"].tolist(), [1, 2])
        states = select_state_sets(first)
        self.assertEqual({key: len(value) for key, value in states.items()}, {
            "X_B": 2,
            "X_H": 2,
            "X_J": 1,
        })
        self.assertEqual(
            states["X_B"]["response"].tolist(),
            ["A firewall explanation", "An audit explanation"],
        )

    def test_stable_id_normalizes_unicode_and_whitespace(self):
        first = stable_pair_id(
            source="fixture",
            stratum="cyber",
            benign_text="Ａ  benign\ntext",
            harmful_text="harmful text",
        )
        second = stable_pair_id(
            source="fixture",
            stratum="cyber",
            benign_text="A benign text",
            harmful_text="harmful text",
        )
        self.assertEqual(first, second)

    def test_rejects_duplicate_pairs_and_orphan_jailbreaks(self):
        manifest = build_manifest(self._records(), require_complete=True)
        duplicate = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ManifestValidationError, "duplicate"):
            validate_manifest(duplicate)

        same_pair_new_id = manifest.iloc[[0]].copy()
        same_pair_new_id["pair_id"] = "pair_custom"
        duplicate_identity = pd.concat(
            [manifest, same_pair_new_id],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ManifestValidationError, "Semantic pairs"):
            validate_manifest(duplicate_identity)

        orphan = manifest.iloc[[0]].copy()
        orphan["clean_refused"] = False
        with self.assertRaisesRegex(ManifestValidationError, "X_J"):
            validate_manifest(orphan)

        refused_benign = manifest.iloc[[0]].copy()
        refused_benign["benign_refused"] = True
        with self.assertRaisesRegex(ManifestValidationError, "non-refusing X_B"):
            validate_manifest(refused_benign)

    def test_complete_and_triplet_validation(self):
        incomplete = self._records().iloc[[0]].copy()
        incomplete["clean_response"] = ""
        with self.assertRaisesRegex(ManifestValidationError, "clean_response"):
            build_manifest(incomplete, require_complete=True)

        non_triplet = self._records().iloc[[1]].copy()
        with self.assertRaisesRegex(ManifestValidationError, "triplet"):
            build_manifest(non_triplet, require_state_triplets=True)

    def test_key_pairing_rejects_mismatches_and_preserves_order(self):
        harmful = pd.DataFrame(
            {
                "prompt": ["harm-2", "harm-1"],
                "stratum": ["b", "a"],
                "behavior": ["k2", "k1"],
                "source": ["fixture", "fixture"],
            }
        )
        benign = pd.DataFrame(
            {
                "prompt": ["benign-1", "benign-2"],
                "stratum": ["a", "b"],
                "behavior": ["k1", "k2"],
                "source": ["fixture", "fixture"],
            }
        )
        paired = pair_prompt_tables(harmful, benign, pair_on="behavior")
        self.assertEqual(paired["benign_text"].tolist(), ["benign-2", "benign-1"])

        with self.assertRaisesRegex(ManifestValidationError, "Pairing keys differ"):
            pair_prompt_tables(
                harmful,
                benign[benign["behavior"] == "k1"],
                pair_on="behavior",
            )

    def test_cli_help_is_offline(self):
        root = Path(__file__).resolve().parents[1]
        for module in ("data.build_safety_pairs", "data.sampling"):
            completed = subprocess.run(
                [sys.executable, "-m", module, "--help"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("usage:", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
