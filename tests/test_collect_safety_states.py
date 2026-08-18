"""Offline tests for paired hidden-state collection and X_H references."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch

from data.build_safety_pairs import (
    ManifestValidationError,
    build_manifest,
    select_state_sets,
)
from experiments.build_pair_references import (
    attach_reference_paths,
    build_pair_references,
    save_per_pair_references,
    validate_reference_checkpoint,
)
from experiments.collect_safety_states import (
    StateCollectionError,
    collect_safety_states,
    validate_collection_payload,
)
from experiments.train_safety_probes import validate_training_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Forward:
    def __init__(self, hidden_states, attention_mask, token_spans):
        self.hidden_states = hidden_states
        self.attention_mask = attention_mask
        self.token_spans = token_spans


class _ToyModel:
    sample_rate = 16_000

    def generate(self, waveform, max_tokens=100, temperature=1.0, do_sample=False):
        return "benign answer"

    def forward_attack(self, waveform, target_text, *, output_hidden_states=False):
        self.assert_hidden_requested = output_hidden_states
        value = waveform.float().mean()
        positions = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
        first = torch.cat(
            [
                value.expand(1, 4, 1) + positions,
                (value * 2).expand(1, 4, 1),
                torch.ones(1, 4, 1),
            ],
            dim=-1,
        )
        second = first + 10.0
        embedding = torch.zeros_like(first)
        return _Forward(
            (embedding, first, second),
            torch.ones(1, 4, dtype=torch.long),
            {"audio": (0, 3), "target": (3, 4)},
        )


class _Judge:
    def evaluate(self, prompt, response):
        return {"refusal": int("refuse" in response.casefold())}


class _RefusalScorer:
    def eval(self):
        return self

    def score_refusal(self, hidden_states):
        return OrderedDict(
            (layer, torch.sigmoid(value.mean(dim=-1)))
            for layer, value in hidden_states.items()
        )


class StateCollectionTests(unittest.TestCase):
    def _fixture(self, root: Path, *, trajectory_owner="pair-a") -> Path:
        audio_names = []
        rows = []
        for index, pair_id in enumerate(("pair-a", "pair-b")):
            names = {
                "benign_audio_path": f"b-{index}.wav",
                "clean_audio_path": f"h-{index}.wav",
                "jailbreak_audio_path": f"j-{index}.wav",
            }
            audio_names.extend(names.values())
            rows.append(
                {
                    "pair_id": pair_id,
                    "source": "fixture",
                    "stratum": "cyber",
                    "benign_text": f"benign {index}",
                    "harmful_text": f"harmful {index}",
                    "clean_response": "I refuse",
                    "clean_refused": True,
                    "jailbreak_response": "harmful answer",
                    "jailbreak_success": True,
                    **names,
                }
            )
        for name in audio_names:
            (root / name).touch()

        trajectory = root / "case-a"
        trajectory.mkdir()
        torch.save(
            {
                "metadata": {
                    "step": 3,
                    "pair_id": trajectory_owner,
                    "response": "harmful answer",
                    "refused": False,
                },
                "tensors": {"adversarial_wav": torch.full((1, 8), 0.75)},
            },
            trajectory / "checkpoint_000003.pt",
        )
        with (trajectory / "trajectory_index.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "metadata": {
                            "step": 3,
                            "pair_id": trajectory_owner,
                        }
                    }
                )
                + "\n"
            )
        rows[0]["trajectory_path"] = "case-a"
        manifest = root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        return manifest

    @staticmethod
    def _audio_loader(path, *, target_sr):
        stem = Path(path).stem
        offset = sum(ord(character) for character in stem) % 7
        return torch.full((1, 8), float(offset) / 10.0)

    def test_manifest_benign_audio_alias_and_state_views(self):
        frame = pd.DataFrame(
            {
                "source": ["fixture"],
                "stratum": ["cyber"],
                "benign_text": ["benign"],
                "harmful_text": ["harmful"],
                "clean_response": ["refuse"],
                "clean_refused": [True],
                "jailbreak_response": ["answer"],
                "jailbreak_success": [True],
                "benign_wav_path": ["b.wav"],
                "harmful_audio_path": ["h.wav"],
                "attack_audio_path": ["j.wav"],
            }
        )
        manifest = build_manifest(frame, require_state_triplets=True)
        self.assertEqual(manifest.loc[0, "benign_audio_path"], "b.wav")
        views = select_state_sets(manifest)
        self.assertEqual(views["X_B"].loc[0, "audio_path"], "b.wav")
        self.assertEqual(views["X_H"].loc[0, "refusal_label"], 1)
        self.assertFalse(views["X_H"].loc[0, "state_jailbreak_success"])
        self.assertEqual(views["X_J"].loc[0, "harmfulness_label"], 1)
        self.assertTrue(views["X_J"].loc[0, "state_jailbreak_success"])

        frame["benign_wav_path"] = ""
        with self.assertRaisesRegex(ManifestValidationError, "benign_audio_path"):
            build_manifest(frame, require_state_triplets=True)

    def test_collects_triplets_and_new_artifact_trajectory_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root)
            output = root / "states.pt"
            payload = collect_safety_states(
                manifest,
                model=_ToyModel(),
                output_path=output,
                audio_loader=self._audio_loader,
                judge=_Judge(),
                shard_size=2,
            )

            self.assertEqual(validate_collection_payload(payload), 7)
            compatible = validate_training_payload(payload)
            self.assertEqual(compatible.num_samples, 7)
            self.assertEqual(list(payload["hidden_states"]), [0, 1])
            self.assertEqual(payload["hidden_states"][0].shape, (7, 3))
            self.assertEqual(payload["states"].count("X_B"), 2)
            self.assertEqual(payload["states"].count("X_H"), 2)
            self.assertEqual(payload["states"].count("X_J"), 2)
            self.assertEqual(payload["states"].count("trajectory"), 1)
            trajectory_index = payload["states"].index("trajectory")
            self.assertEqual(int(payload["steps"][trajectory_index]), 3)
            self.assertEqual(payload["pair_ids"][trajectory_index], "pair-a")
            self.assertEqual(payload["responses"][trajectory_index], "harmful answer")
            self.assertTrue(output.is_file())
            shard_index_path = root / "states.pt.shards.json"
            self.assertTrue(shard_index_path.is_file())
            with shard_index_path.open("r", encoding="utf-8") as handle:
                shard_index = json.load(handle)
            self.assertTrue(shard_index["pair_safe"])
            self.assertEqual(len(shard_index["shards"]), 2)
            shard_pairs = [set(item["pair_ids"]) for item in shard_index["shards"]]
            self.assertEqual(shard_pairs, [{"pair-a"}, {"pair-b"}])
            self.assertEqual(len(list((root / "states.shards").glob("part-*.pt"))), 2)
            self.assertEqual(list(root.rglob("*.tmp")), [])
            loaded = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(loaded["pair_ids"], payload["pair_ids"])

    def test_rejects_cross_pair_trajectory_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root, trajectory_owner="pair-b")
            with self.assertRaisesRegex(StateCollectionError, "does not match"):
                collect_safety_states(
                    manifest,
                    model=_ToyModel(),
                    audio_loader=self._audio_loader,
                    judge=_Judge(),
                )

    def test_collection_schema_rejects_row_pair_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = collect_safety_states(
                self._fixture(root),
                model=_ToyModel(),
                audio_loader=self._audio_loader,
                judge=_Judge(),
                include_trajectories=False,
            )
            payload["row_metadata"][0]["pair_id"] = "different-pair"
            with self.assertRaisesRegex(StateCollectionError, "does not own"):
                validate_collection_payload(payload)

    def test_builds_aggregate_and_per_pair_clean_references(self):
        from experiments.batch_safety_attack import (
            _default_reference_loader,
            _reference_for_pair,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root)
            states = collect_safety_states(
                manifest,
                model=_ToyModel(),
                audio_loader=self._audio_loader,
                include_trajectories=False,
            )
            output = root / "references.pt"
            checkpoint = build_pair_references(
                states,
                scorer=_RefusalScorer(),
                output_path=output,
            )
            self.assertEqual(validate_reference_checkpoint(checkpoint), 2)
            self.assertEqual(checkpoint["pair_ids"], ["pair-a", "pair-b"])
            paths = save_per_pair_references(checkpoint, root / "by-pair")
            loaded = _default_reference_loader(paths["pair-a"])
            selected = _reference_for_pair(
                loaded, "pair-a", require_pair_mapping=True
            )
            self.assertEqual(list(selected), [0, 1])
            self.assertTrue(all(value.ndim == 0 for value in selected.values()))

            attached_path = root / "manifest-with-references.csv"
            attached = attach_reference_paths(manifest, paths, attached_path)
            self.assertTrue(attached_path.is_file())
            self.assertEqual(set(attached["reference_refusal_path"].str.len() > 0), {True})
            self.assertEqual(list((root / "by-pair").glob("*.tmp")), [])

    def test_help_does_not_import_torch(self):
        source = textwrap.dedent(
            """
            import builtins
            import runpy
            import sys

            real_import = builtins.__import__
            def guarded_import(name, *args, **kwargs):
                if name == "torch" or name.startswith("torch."):
                    raise AssertionError("--help imported torch")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = guarded_import
            sys.argv = ["module", "--help"]
            runpy.run_module(sys.argv_module, run_name="__main__")
            """
        )
        for module in (
            "experiments.collect_safety_states",
            "experiments.build_pair_references",
        ):
            script = "import sys\nsys.argv_module = " + repr(module) + "\n" + source
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("usage:", completed.stdout.casefold())


if __name__ == "__main__":
    unittest.main()
