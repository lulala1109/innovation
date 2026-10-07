import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rq2.config import DEV_SCREEN_STAGE_ORDER, STAGE_ORDER, RQ2ConfigError, load_rq2_config
from rq2.artifacts import atomic_json, atomic_jsonl, file_sha256
from rq2.dev_screen import DevScreenError, prepare_dev_screen, summarize_records, summarize_dev_screen


def _records(positive=0):
    ids = [f"dev_{i}" for i in range(20)]
    trials, labels = [], []
    for i, pair in enumerate(ids):
        for key in ["scan:clean", *(f"scan:{s}" for s in range(101))]:
            tid = hashlib.sha256(f"{pair}/{key}".encode()).hexdigest()
            room = key == "scan:10" and i < positive
            trials.append({
                "trial_id": tid, "pair_id": pair, "rq2_role": "rq2_dev",
                "state_key": key, "step": 0 if key == "scan:clean" else int(key[5:]),
                "intervention": "trajectory_baseline", "run_fingerprint": "a" * 64,
                "response_sha256": tid,
            })
            labels.append({
                "trial_id": tid, "response_sha256": tid, "judge_fingerprint": "b" * 64,
                "label_status": "ok", "strongreject_score": 0.75 if room else 0.0,
                "refusal_label": not room, "compliance_label": room,
            })
    return ids, trials, labels


class SummaryTests(unittest.TestCase):
    def test_opportunity_boundary_does_not_claim_causality(self):
        for positive, expected in ((0, "fixed_state_infeasible"), (11, "fixed_state_infeasible"), (12, "fixed_state_feasible_for_oracle_pilot")):
            with self.subTest(positive=positive):
                ids, trials, labels = _records(positive)
                result = summarize_records(trials, labels, pair_ids=ids)
                self.assertEqual(result["decision"], expected)
                self.assertEqual(result["states"][1]["minimum_positive_pairs_on_observed_valid_population"], 12)
                self.assertFalse(result["causal_evidence"])
                self.assertFalse(result["automatically_launches_next_stage"])
                self.assertEqual(len(result["pair_audit"]), 100)

    def test_unknown_does_not_shrink_decision_denominator(self):
        ids, trials, labels = _records(12)
        label = next(l for t, l in zip(trials, labels) if t["pair_id"] == "dev_19" and t["state_key"] == "scan:10")
        label.update(label_status="unknown", strongreject_score=None, refusal_label=None, compliance_label=None)
        result = summarize_records(trials, labels, pair_ids=ids)
        self.assertEqual(result["decision"], "needs_data_repair")
        self.assertEqual(result["states"][1]["valid_pairs"], 19)
        self.assertEqual(result["states"][1]["unknown_pairs"], 1)
        self.assertEqual(result["states"][1]["positive_score_pairs"], 12)

    def test_clean_reference_failure_blocks_advancement(self):
        ids, trials, labels = _records(12)
        labels[0]["refusal_label"] = False
        result = summarize_records(trials, labels, pair_ids=ids)
        self.assertEqual(result["decision"], "needs_data_repair")
        self.assertEqual(result["clean_reference_failures"], ["dev_0"])

    def test_misaligned_or_cross_population_sidecars_fail_closed(self):
        for corruption in ("sha", "duplicate", "test_pair", "intervention"):
            with self.subTest(corruption=corruption):
                ids, trials, labels = _records(12)
                if corruption == "sha":
                    labels[0]["response_sha256"] = "c" * 64
                elif corruption == "duplicate":
                    labels.append(copy.deepcopy(labels[0]))
                elif corruption == "test_pair":
                    trials[0]["rq2_role"] = "rq2_causal_test"
                else:
                    trials[0]["intervention"] = "r_direction"
                with self.assertRaises(DevScreenError):
                    summarize_records(trials, labels, pair_ids=ids)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name) / "project"
        (root / "configs").mkdir(parents=True)
        self.source_path = root / "configs" / "source.json"
        raw = json.loads(Path("configs/stage2_rq2_qwen7b_advbench_sure_here_is_v2_run01.json").read_text())
        for key in ("preregistration", "statistical_preregistration"):
            raw[key]["path"] = str(Path(raw[key]["path"]).resolve())
        source_manifest = root / "source.jsonl"
        rows = [{"pair_id": f"pair_{i}", "rq2_role": "rq2_dev" if i < 20 else "rq2_causal_test", "split_rank": i} for i in range(60)]
        source_manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        raw["manifest"] = str(source_manifest)
        self.source_path.write_text(json.dumps(raw))
        self.protocol = root / "protocol.md"
        self.protocol.write_text("Frozen feasibility rules for this test fixture.\n")
        result = prepare_dev_screen(self.source_path, name="unit_dev_screen", protocol=self.protocol)
        self.config = load_rq2_config(result["config"])

    def test_exact_population_and_idempotent_preparation(self):
        rows = [json.loads(line) for line in self.config.manifest.read_text().splitlines()]
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(r["rq2_role"] == "rq2_dev" for r in rows))
        self.assertEqual([r["pair_id"] for r in rows], self.config.dev_screen["dev_pair_ids"])
        result = prepare_dev_screen(self.source_path, name="unit_dev_screen", protocol=self.protocol)
        self.assertEqual(result["config_fingerprint"], self.config.fingerprint)

    def test_config_rejects_test_pair_in_subset(self):
        with self.config.manifest.open("a") as out:
            out.write(json.dumps({"pair_id": "pair_20", "rq2_role": "rq2_causal_test", "split_rank": 20}) + "\n")
        with self.assertRaisesRegex(RQ2ConfigError, "exactly copy"):
            load_rq2_config(self.config.path)

    def test_protocol_change_is_rejected(self):
        self.protocol.write_text("Changed rules.\n")
        with self.assertRaisesRegex(RQ2ConfigError, "protocol SHA"):
            load_rq2_config(self.config.path)

    def test_summary_requires_current_stage_hashes(self):
        ids, trials, labels = _records(12)
        for trial in trials:
            trial["pair_id"] = trial["pair_id"].replace("dev_", "pair_")
        directory = self.config.output_root / "trajectory_behavior"
        trial_path = atomic_jsonl(directory / "trials.jsonl", trials)
        label_path = atomic_jsonl(directory / "labels.jsonl", labels)
        atomic_json(self.config.output_root / "pipeline_state.json", {
            "config_fingerprint": self.config.fingerprint,
            "stages": {
                "trajectory_behavior_generate": {"artifacts": {str(trial_path): file_sha256(trial_path)}},
                "trajectory_behavior_judge": {"artifacts": {str(label_path): file_sha256(label_path)}},
            },
        })
        self.assertEqual(summarize_dev_screen(self.config.path)["decision"], "fixed_state_feasible_for_oracle_pilot")
        with label_path.open("a") as out:
            out.write("\\n")
        with self.assertRaisesRegex(DevScreenError, "SHA"):
            summarize_dev_screen(self.config.path)

    def test_all_forbidden_stages_fail_before_loading_model(self):
        from rq2.pipeline import RQ2Pipeline, RQ2PipelineError

        def forbidden_model(_):
            self.fail("static guard must never load a model")

        pipeline = RQ2Pipeline(self.config, model_factory=forbidden_model)
        self.assertEqual([r["stage"] for r in pipeline.plan()], list(DEV_SCREEN_STAGE_ORDER))
        self.assertEqual(pipeline.validate()["formal_pair_count"], 0)
        for stage in STAGE_ORDER:
            if stage not in DEV_SCREEN_STAGE_ORDER:
                with self.subTest(stage=stage), self.assertRaisesRegex(RQ2PipelineError, "dev_screen"):
                    pipeline.run([stage])
        with self.assertRaisesRegex(RQ2PipelineError, "dev_screen"):
            pipeline.validate(require_protocol=True)


if __name__ == "__main__":
    unittest.main()
