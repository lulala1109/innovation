"""Synthetic CPU-only fixtures; no model weights or live Judge access."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rq2.artifacts import (
    BehaviorLabel, RQ2ArtifactError, TrialKey, TrialRecord, atomic_json,
    atomic_jsonl, atomic_text, canonical_sha256, file_sha256,
)
from rq2.direction_diagnostics import (
    CENTER, LAYERS, build, csv_text, diagnose, expected_conditions, load_source,
    stats, write_report,
)


def fixture(directory):
    root = directory / "source"
    pairs = [f"p{i:02d}" for i in range(20)]
    contract = {"fixture": "no_api"}
    config = {"format": "rq2-post-oracle-exploration", "version": 1,
              "output_root": str(root), "pair_ids": pairs, "scoring_contract": contract,
              "design": {"candidate_layers": list(LAYERS), "source_pairs": 20,
                         "random_replicates": 3, "token_scope": "audio", "application": "prefill_once",
                         "allow_formal": False, "allow_subspace": False, "protocol_locked": False}}
    config_path = directory / "config.json"
    atomic_json(config_path, config)
    fingerprint = canonical_sha256({"exploration_config": canonical_sha256(config), "phase": "mechanism"})
    response_hash = "a" * 64
    trials, labels, effects = [], [], []
    baseline = {}

    def label(tid, score, refused):
        compliance = not refused and score >= .5
        return BehaviorLabel(tid, "ok", "refusal" if refused else "compliance", refused, compliance,
                             compliance, float(refused), score, 1-score, 1 if refused else 5,
                             1 if refused else 5, response_hash, canonical_sha256(contract)).to_record()

    for pair in pairs:
        for state in (CENTER, "clean"):
            key = TrialKey(fingerprint, pair, state, None, "baseline")
            baseline[pair, state] = key.trial_id
            trials.append(TrialRecord(key, "rq2_dev", 0 if state == "clean" else 10, None,
                                      {}, {}, response_hash).to_record())
            labels.append(label(key.trial_id, 0. if state == "clean" else 1., state == "clean"))
    for pair, layer, state, kind, dose, rep in sorted(expected_conditions(pairs)):
        reverse = state == "clean"
        before = 20. if reverse else 10.
        gap = 10.
        magnitude = (2. if reverse else gap) * dose
        delta = (-magnitude if reverse else magnitude)
        if kind in ("h_direction_control", "reverse_h_control"):
            delta *= .6
        elif kind in ("random_direction_control", "reverse_random_control"):
            delta *= (-.01, 0., .01)[rep]
        elif kind in ("sham", "reverse_sham"):
            delta, magnitude = 0., 0.
        audit = {"kind": kind, "layer": layer, "dose": dose, "apply_count": 1,
                 "token_scope": "audio", "token_count": 4, "shift_l2": magnitude*2,
                 "mean_shift_l2": magnitude, "projection_before": before,
                 "projection_after": before+delta, "projection_delta": delta,
                 "projection_reference": 20.}
        key = TrialKey(fingerprint, pair, state, layer, kind, dose=dose, replicate=rep)
        bid = baseline[pair, state]
        trials.append(TrialRecord(key, "rq2_dev", 0 if reverse else 10, bid, audit, {}, response_hash).to_record())
        score = 0. if reverse else (1. if kind == "sham" else .5)
        labels.append(label(key.trial_id, score, reverse))
        effects.append({"trial_id": key.trial_id, "baseline_trial_id": bid, "pair_id": pair,
                        "layer": layer, "state_key": state, "intervention": kind, "dose": dose,
                        "replicate": rep, "utility_effect": 0. if reverse else 1-score,
                        "refusal_effect": 0, "compliance_effect": 0})
    atomic_json(root / "provenance/population.json", {"expected_dev_pair_ids": pairs, "event_pair_ids": pairs})
    atomic_json(root / "provenance/scoring_contract.json", contract)
    for name in ("scoring_consistency_audit.json", "baseline_replay_audit.json"):
        atomic_json(root / "mechanism" / name, {"passed": True})
    atomic_jsonl(root / "mechanism/trials.jsonl", trials)
    atomic_jsonl(root / "mechanism/labels.jsonl", labels)
    atomic_text(root / "mechanism/pair_effects.csv", csv_text(effects))
    bound = {str(p): file_sha256(p) for p in root.rglob("*") if p.is_file()}
    state = {"config_fingerprint": canonical_sha256(config),
             "stages": {s: {"artifacts": bound if s == "prepare" else {}} for s in
                        ("prepare", "mechanism_generate", "mechanism_judge", "mechanism_analyze", "report")}}
    atomic_json(root / "pipeline_state.json", state)
    return config_path, root


def rebind(root, path):
    state_path = root / "pipeline_state.json"
    state = json.loads(state_path.read_text())
    state["stages"]["prepare"]["artifacts"][str(path)] = file_sha256(path)
    atomic_json(state_path, state)


class DirectionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.config, self.source = fixture(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_grid_and_expected_magnitudes(self):
        summary, files = build(self.config)
        self.assertEqual(summary["trial_rows"], 920)
        self.assertEqual(len(list(csv.DictReader(files["pair_comparisons.csv"].splitlines()))), 60)
        for row in summary["layers"]:
            self.assertEqual(row["independent_pairs"], 20)
            self.assertEqual(row["H_signed_R_gap_fraction"]["median"], .6)
            self.assertEqual(row["reverse_absolute_R_gap_fraction"]["median"], .2)
            self.assertEqual(row["R_target_error_over_gap"]["max"], 0.)
            self.assertEqual(row["control_mean_norm_ratio_to_R_per_trial"]["min"], 1.)
        self.assertFalse(summary["formal_allowed"])

    def test_exact_idempotence_and_no_source_mutation(self):
        summary, files = build(self.config)
        output = self.directory / "diagnosis"
        kwargs = {"source_root": self.source, "inputs": summary["source_files_sha256"]}
        write_report(output, files, **kwargs)
        times = {p: p.stat().st_mtime_ns for p in output.iterdir()}
        write_report(output, files, **kwargs)
        self.assertEqual(times, {p: p.stat().st_mtime_ns for p in output.iterdir()})
        for path, digest in kwargs["inputs"].items():
            self.assertEqual(file_sha256(path), digest)
        with self.assertRaisesRegex(RQ2ArtifactError, "differs"):
            write_report(output, {**files, "report.md": "changed"}, **kwargs)

    def test_rejects_modified_frozen_artifact(self):
        path = self.source / "mechanism/pair_effects.csv"
        atomic_text(path, path.read_text() + "\n")
        with self.assertRaisesRegex(RQ2ArtifactError, "modified/unbound"):
            build(self.config)

    def test_rejects_missing_condition_even_with_updated_hash(self):
        path = self.source / "mechanism/trials.jsonl"
        records = [json.loads(s) for s in path.read_text().splitlines()]
        atomic_jsonl(path, records[:-1])
        rebind(self.source, path)
        with self.assertRaisesRegex(RQ2ArtifactError, "coverage"):
            build(self.config)

    def test_rejects_effect_sign_corruption(self):
        path = self.source / "mechanism/pair_effects.csv"
        rows = list(csv.DictReader(path.read_text().splitlines()))
        rows[0]["utility_effect"] = 99
        atomic_text(path, csv_text(rows))
        rebind(self.source, path)
        with self.assertRaisesRegex(RQ2ArtifactError, "recomputation"):
            build(self.config)

    def test_reverse_effect_is_post_minus_baseline(self):
        labels_path = self.source / "mechanism/labels.jsonl"
        rows = [json.loads(s) for s in labels_path.read_text().splitlines()]
        effects_path = self.source / "mechanism/pair_effects.csv"
        effects = list(csv.DictReader(effects_path.read_text().splitlines()))
        effect = next(e for e in effects if e["intervention"] == "reverse_suppression")
        label = next(l for l in rows if l["trial_id"] == effect["trial_id"])
        label.update(refusal_label=False, compliance_label=True, jailbreak_success=True,
                     refusal_score=0., strongreject_score=1., refusal_orientation=0.)
        effect.update(utility_effect=1., refusal_effect=1, compliance_effect=1)
        atomic_jsonl(labels_path, rows)
        atomic_text(effects_path, csv_text(effects))
        rebind(self.source, labels_path)
        rebind(self.source, effects_path)
        summary, _ = build(self.config)
        self.assertEqual(summary["trial_rows"], 920)

    def test_zero_gap_retained_as_undefined_and_half_dose_target(self):
        _, pairs, trials, effects, _ = load_source(self.config)
        for t in trials:
            if t["pair_id"] == pairs[0] and t["intervention"] == "r_direction" and t["dose"] == 1:
                t["audit"]["projection_reference"] = t["audit"]["projection_before"]
        rows, _, summary = diagnose(trials, effects, pairs)
        self.assertEqual(len(rows), 920)
        self.assertTrue(all(s["R_target_error_over_gap"]["undefined"] == 1 for s in summary))
        half = next(r for r in rows if r["pair_id"] == pairs[1] and r["intervention"] == "r_direction" and r["dose"] == .5)
        self.assertEqual(half["R_target_error_over_gap"], 0.)
        self.assertEqual(stats([None])["n"], 0)

    def test_nonfinite_or_inconsistent_audit_stops(self):
        _, pairs, trials, effects, _ = load_source(self.config)
        for value in (float("nan"), 123.):
            modified = copy.deepcopy(trials)
            modified[0]["audit"]["projection_delta"] = value
            with self.assertRaises(RQ2ArtifactError):
                diagnose(modified, effects, pairs)

    def test_random_behavior_is_averaged_within_pair(self):
        _, pairs, trials, effects, _ = load_source(self.config)
        for t in trials:
            if t["intervention"] == "random_direction_control":
                effects[t["trial_id"]]["utility_effect"] = (0., .5, 1.)[t["replicate"]]
        _, comparisons, summaries = diagnose(trials, effects, pairs)
        self.assertEqual(len(comparisons), 60)
        self.assertTrue(all(p["random_utility_effect"] == .5 for p in comparisons))
        self.assertTrue(all(s["independent_pairs"] == 20 for s in summaries))
        self.assertTrue(all(s["R_minus_random_utility_mean"] == 0 for s in summaries))

    def test_opposing_H_projection_sign_is_preserved(self):
        _, pairs, trials, effects, _ = load_source(self.config)
        t = next(t for t in trials if t["intervention"] == "h_direction_control" and t["layer"] == 26)
        t["audit"]["projection_delta"] = -6.
        t["audit"]["projection_after"] = t["audit"]["projection_before"] - 6.
        _, _, summaries = diagnose(trials, effects, pairs)
        s = next(s for s in summaries if s["layer"] == 26)
        self.assertEqual(s["H_signed_R_gap_fraction"]["min"], -.6)
        self.assertEqual(s["H_opposes_clean_event_gap_pairs"], 1)

    def test_sources_and_changed_inputs_are_never_overwritten(self):
        summary, files = build(self.config)
        for output in (self.source, self.source / "diagnosis", self.source.parent):
            with self.assertRaisesRegex(RQ2ArtifactError, "separate"):
                write_report(output, files, source_root=self.source, inputs=summary["source_files_sha256"])
        atomic_text(self.config, self.config.read_text() + "\n")
        with self.assertRaisesRegex(RQ2ArtifactError, "changed during"):
            write_report(self.directory / "diagnosis", files, source_root=self.source,
                         inputs=summary["source_files_sha256"])

    def test_fresh_process_has_no_torch_or_network_dependency(self):
        program = ("import sys,socket,urllib.request; "
                   "deny=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network forbidden')); "
                   "socket.create_connection=deny; urllib.request.urlopen=deny; "
                   "from rq2.direction_diagnostics import build; "
                   "build(sys.argv[1]); "
                   "assert 'torch' not in sys.modules and 'dotenv' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", program, str(self.config)],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
