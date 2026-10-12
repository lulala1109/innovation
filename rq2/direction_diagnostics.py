"""Offline projection/dose diagnostics for completed exploratory dev runs.

Only the standard library and artifact helpers are imported. No model, Judge,
environment credentials, or experiment pipeline is constructed.
"""
from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

from rq2.artifacts import (
    BehaviorLabel, RQ2ArtifactError, atomic_text, canonical_sha256, file_sha256,
    read_jsonl, records_by_id, validate_trial_record,
)

ROOT = Path(__file__).resolve().parents[1]
CENTER = "event:first_non_refusal_step:+0"
LAYERS = (19, 24, 26)
EPS = 1e-8


def require(condition, message):
    if not condition:
        raise RQ2ArtifactError(message)


def finite(value):
    require(not isinstance(value, bool) and isinstance(value, (float, int))
            and math.isfinite(value), "missing/nonfinite numerical audit")
    return float(value)


def ratio(numerator, denominator):
    return numerator / denominator if abs(denominator) > EPS else None


def stats(values):
    values = list(values)
    valid = [finite(v) for v in values if v is not None]
    return {"n": len(valid), "undefined": len(values) - len(valid),
            "min": min(valid) if valid else None,
            "median": median(valid) if valid else None,
            "max": max(valid) if valid else None}


def expected_conditions(pairs):
    conditions = set()
    for pair in pairs:
        for layer in LAYERS:
            for state, main, h, random, sham in (
                (CENTER, "r_direction", "h_direction_control", "random_direction_control", "sham"),
                ("clean", "reverse_suppression", "reverse_h_control", "reverse_random_control", "reverse_sham"),
            ):
                for kind, dose, rep in ((main, 1., 0), (main, .5, 0), (h, 1., 0), (sham, 1., 0),
                                        *((random, 1., i) for i in range(3))):
                    conditions.add((pair, layer, state, kind, dose, rep))
        for layer in (18, 20, 23, 25):
            conditions.add((pair, layer, CENTER, "r_direction", 1., 0))
    return conditions


def load_source(config_path):
    inputs = {}

    def track(path):
        path = Path(path).resolve()
        inputs[str(path)] = file_sha256(path)
        return path

    config = json.loads(track(config_path).read_text(encoding="utf-8"))
    require(config.get("format") == "rq2-post-oracle-exploration" and config.get("version") == 1,
            "only post-Oracle exploratory v1 is supported")
    design = config["design"]
    require(design["candidate_layers"] == list(LAYERS) and design["source_pairs"] == 20
            and design["random_replicates"] == 3 and design["token_scope"] == "audio"
            and design["application"] == "prefill_once" and design["allow_formal"] is False
            and design["allow_subspace"] is False and design["protocol_locked"] is False,
            "unexpected or non-exploratory design")
    root = (ROOT / config["output_root"]).resolve()
    state = json.loads(track(root / "pipeline_state.json").read_text(encoding="utf-8"))
    require(state["config_fingerprint"] == canonical_sha256(config), "config fingerprint mismatch")
    for stage in ("prepare", "mechanism_generate", "mechanism_judge", "mechanism_analyze", "report"):
        require(stage in state["stages"], f"missing completed stage: {stage}")
    bound = {}
    for stage in state["stages"].values():
        for path, digest in stage["artifacts"].items():
            key = str(Path(path).resolve())
            require(key not in bound or bound[key] == digest, "conflicting artifact hashes")
            bound[key] = digest

    def frozen(name, kind="json"):
        path = track(root / name)
        require(bound.get(str(path)) == inputs[str(path)], f"modified/unbound source: {name}")
        if kind == "jsonl":
            return read_jsonl(path)
        if kind == "csv":
            with path.open(encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        return json.loads(path.read_text(encoding="utf-8"))

    population = frozen("provenance/population.json")
    pairs = config["pair_ids"]
    require(len(pairs) == len(set(pairs)) == 20 and pairs == population["expected_dev_pair_ids"]
            == population["event_pair_ids"], "inexact frozen dev population")
    contract = frozen("provenance/scoring_contract.json")
    require(contract == config["scoring_contract"], "scoring contract differs")
    for name in ("scoring_consistency_audit.json", "baseline_replay_audit.json"):
        require(frozen("mechanism/" + name)["passed"] is True, "saved technical audit failed")
    trials = records_by_id(validate_trial_record(t) for t in frozen("mechanism/trials.jsonl", "jsonl"))
    labels = records_by_id(frozen("mechanism/labels.jsonl", "jsonl"))
    effects = records_by_id(frozen("mechanism/pair_effects.csv", "csv"))
    require(set(trials) == set(labels), "inexact label coverage")
    fingerprint = canonical_sha256({"exploration_config": canonical_sha256(config), "phase": "mechanism"})
    for tid, trial in trials.items():
        label = labels[tid]
        BehaviorLabel(**{k: label.get(k) for k in BehaviorLabel.__dataclass_fields__})
        require(trial["rq2_role"] == "rq2_dev" and trial["pair_id"] in pairs
                and trial["run_fingerprint"] == fingerprint and trial["token_scope"] == "audio"
                and label["label_status"] == "ok"
                and label["judge_fingerprint"] == canonical_sha256(contract)
                and label["response_sha256"] == trial["response_sha256"], "trial/label identity mismatch")
    baselines = [t for t in trials.values() if t["intervention"] == "baseline"]
    require(Counter((t["pair_id"], t["state_key"]) for t in baselines)
            == Counter((p, s) for p in pairs for s in (CENTER, "clean")), "baseline coverage mismatch")
    active = [t for t in trials.values() if t["intervention"] != "baseline"]
    conditions = [(t["pair_id"], t["layer"], t["state_key"], t["intervention"], t["dose"], t["replicate"])
                  for t in active]
    require(len(conditions) == len(set(conditions)) and set(conditions) == expected_conditions(pairs),
            "incomplete/duplicate intervention grid")
    require(set(effects) == {t["trial_id"] for t in active}, "inexact effect coverage")
    for t in active:
        tid, bid = t["trial_id"], t["baseline_trial_id"]
        require(bid in trials and trials[bid]["intervention"] == "baseline"
                and all(trials[bid][k] == t[k] for k in ("pair_id", "state_key", "step")),
                "not a same-pair same-state baseline")
        base, post, e = labels[bid], labels[tid], effects[tid]
        sign = -1 if t["state_key"] == "clean" else 1
        values = {
            "utility_effect": sign * (base["strongreject_score"] - post["strongreject_score"]),
            "refusal_effect": sign * (int(post["refusal_label"]) - int(base["refusal_label"])),
            "compliance_effect": sign * (int(base["compliance_label"]) - int(post["compliance_label"])),
        }
        require(e["baseline_trial_id"] == bid and e["pair_id"] == t["pair_id"]
                and e["intervention"] == t["intervention"] and e["state_key"] == t["state_key"]
                and int(e["layer"]) == t["layer"] and float(e["dose"]) == t["dose"]
                and int(e["replicate"]) == t["replicate"], "effect metadata differs")
        for key, value in values.items():
            require(math.isclose(float(e[key]), value, rel_tol=0, abs_tol=1e-12), "effect recomputation differs")
        a = t["audit"]
        require(a["kind"] == t["intervention"] and a["layer"] == t["layer"]
                and a["dose"] == t["dose"] and a["apply_count"] == 1
                and a["token_scope"] == "audio" and a["token_count"] > 0, "invalid application audit")
        if t["intervention"] in ("sham", "reverse_sham"):
            require(t["response_sha256"] == trials[bid]["response_sha256"]
                    and a["shift_l2"] == 0 and all(v == 0 for v in values.values()), "no-op drift")
    return root, pairs, active, effects, inputs


def diagnose(trials, effects, pairs, layers=LAYERS):
    lookup = {(t["pair_id"], t["layer"], t["intervention"], t["dose"], t["replicate"]): t for t in trials}
    require(len(lookup) == len(trials), "duplicate diagnostic condition")
    rows, comparisons, summaries = [], [], []
    for t in trials:
        a, pair, layer = t["audit"], t["pair_id"], t["layer"]
        r = lookup[pair, layer, "r_direction", 1., 0]
        gap = finite(r["audit"]["projection_reference"]) - finite(r["audit"]["projection_before"])
        before, after = finite(a["projection_before"]), finite(a["projection_after"])
        delta, norm = finite(a["projection_delta"]), finite(a["mean_shift_l2"])
        require(math.isclose(after - before, delta, rel_tol=0, abs_tol=1e-6) and norm >= 0,
                "projection delta mismatch or negative norm")
        reverse = t["state_key"] == "clean"
        matched = lookup[pair, layer, "reverse_suppression" if reverse else "r_direction", 1., 0]
        residual = (abs(after - (before + float(t["dose"]) * gap))
                    if t["intervention"] == "r_direction" else None)
        e = effects[t["trial_id"]]
        rows.append({"pair_id": pair, "trial_id": t["trial_id"], "layer": layer,
                     "state_key": t["state_key"], "intervention": t["intervention"],
                     "dose": t["dose"], "replicate": t["replicate"], "token_count": a["token_count"],
                     "clean_minus_event_R_gap": gap, "projection_before": before,
                     "projection_after": after, "projection_delta": delta,
                     "mean_shift_l2": norm, "full_tensor_shift_l2": finite(a["shift_l2"]),
                     "signed_R_gap_fraction": ratio(delta, gap),
                     "absolute_R_gap_fraction": ratio(abs(delta), abs(gap)),
                     "R_target_error_over_gap": ratio(residual, abs(gap)) if residual is not None else None,
                     "mean_norm_ratio_to_primary_R": ratio(norm, finite(matched["audit"]["mean_shift_l2"])),
                     "R_component_of_mean_shift": ratio(delta, norm),
                     **{k: float(e[k]) for k in ("utility_effect", "refusal_effect", "compliance_effect")}})
    for layer in layers:
        lr = [r for r in rows if r["layer"] == layer]
        for pair in pairs:
            group = [r for r in lr if r["pair_id"] == pair and r["dose"] == 1]
            p = {"pair_id": pair, "layer": layer}
            for prefix, kind in (("R", "r_direction"), ("H", "h_direction_control"),
                                 ("random", "random_direction_control"), ("reverse_R", "reverse_suppression")):
                selected = [r for r in group if r["intervention"] == kind]
                require(len(selected) == (3 if prefix == "random" else 1), "missing matched condition")
                for key in ("utility_effect", "refusal_effect", "compliance_effect"):
                    p[prefix + "_" + key] = mean(r[key] for r in selected)
            p["R_minus_H_utility"] = p["R_utility_effect"] - p["H_utility_effect"]
            p["R_minus_random_utility"] = p["R_utility_effect"] - p["random_utility_effect"]
            comparisons.append(p)
        r_main = [r for r in lr if r["intervention"] == "r_direction" and r["dose"] == 1]
        h_main = [r for r in lr if r["intervention"] == "h_direction_control"]
        random_main = [r for r in lr if r["intervention"] == "random_direction_control"]
        reverse = [r for r in lr if r["intervention"] == "reverse_suppression" and r["dose"] == 1]
        pc = [p for p in comparisons if p["layer"] == layer]
        summaries.append({
            "layer": layer, "independent_pairs": len(pairs),
            "clean_minus_event_R_gap": stats(r["clean_minus_event_R_gap"] for r in r_main),
            "R_target_error_over_gap": stats(r["R_target_error_over_gap"] for r in r_main),
            "H_signed_R_gap_fraction": stats(r["signed_R_gap_fraction"] for r in h_main),
            "H_opposes_clean_event_gap_pairs": sum(r["signed_R_gap_fraction"] is not None
                                                   and r["signed_R_gap_fraction"] < 0 for r in h_main),
            "random_absolute_R_gap_fraction_per_trial": stats(r["absolute_R_gap_fraction"] for r in random_main),
            "control_mean_norm_ratio_to_R_per_trial": stats(r["mean_norm_ratio_to_primary_R"]
                                                             for r in h_main + random_main),
            "reverse_absolute_R_gap_fraction": stats(r["absolute_R_gap_fraction"] for r in reverse),
            "reverse_R_projection_delta": stats(r["projection_delta"] for r in reverse),
            "R_and_H_both_improve_pairs": sum(p["R_utility_effect"] > 0 and p["H_utility_effect"] > 0 for p in pc),
            "R_and_random_mean_both_improve_pairs": sum(p["R_utility_effect"] > 0 and p["random_utility_effect"] > 0 for p in pc),
            "R_minus_H_utility_mean": mean(p["R_minus_H_utility"] for p in pc),
            "R_minus_random_utility_mean": mean(p["R_minus_random_utility"] for p in pc),
        })
    return rows, comparisons, summaries


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build(config_path):
    root, pairs, trials, effects, inputs = load_source(config_path)
    rows, comparisons, summaries = diagnose(trials, effects, pairs)
    summary = {"format": "rq2-direction-diagnostics", "version": 1,
               "source_root": str(root), "scope": "post_hoc_descriptive_dev_only",
               "gpu_used": False, "api_calls": 0, "formal_allowed": False, "protocol_locked": False,
               "original_decisions_changed": False, "trial_rows": len(rows), "pair_ids": pairs,
               "layers": summaries,
               "validation_scope": "Consumed artifact hashes, exact dev grid, label identities, effects and numeric audits; not full upstream provenance revalidation or semantic rejudging.",
               "limitations": ["H overlap is realized local projection, not proof of a semantic mechanism.",
                               "Random repetitions are averaged within pair for behavior contrasts.",
                               "Reverse ratio is relative to each pair's gap, not a proof of insufficient dose.",
                               "No new orthogonal controls, calibration, readout measurement, or manual review was run.",
                               "Zero/near-zero denominators are undefined and retained, not silently dropped."],
               "source_files_sha256": inputs,
               "diagnostic_code_sha256": file_sha256(__file__)}
    report = ["# Direction overlap and dose diagnostics", "",
              "Post-hoc description only. No GPU, API, rejudging, or formal authorization.", "",
              "| Layer | Pairs | Max R target error / gap | Median H R-gap fraction | Median reverse / gap |",
              "|---|---:|---:|---:|---:|"]
    def percent(value):
        return "undefined" if value is None else f"{100 * value:.3f}%"
    for s in summaries:
        report.append(f"| {s['layer']} | {s['independent_pairs']} | "
                      f"{percent(s['R_target_error_over_gap']['max'])} | "
                      f"{percent(s['H_signed_R_gap_fraction']['median'])} | "
                      f"{percent(s['reverse_absolute_R_gap_fraction']['median'])} |")
    report += ["", "H fractions are signed medians; individual signs can differ. "
               "Reverse fractions are pairwise magnitudes, not percentages of refusal removed.", "",
               "## Files", "", "- projection_diagnostics.csv: all 920 mechanism intervention rows.",
               "- pair_comparisons.csv: 60 candidate pair-layer rows; random behavior averaged within pair.",
               "- summary.json: ranges, undefined counts, paired contrasts and input hashes.", "",
               "## Limits", "", *["- " + s for s in summary["limitations"]], "",
               "Human review and any new calibration require separate work. Original Oracle decisions remain unchanged.", ""]
    files = {"summary.json": json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
             "projection_diagnostics.csv": csv_text(rows), "pair_comparisons.csv": csv_text(comparisons),
             "report.md": "\n".join(report)}
    return summary, files


def write_report(output, files, *, source_root, inputs):
    output, source = Path(output).resolve(), Path(source_root).resolve()
    require(output != source and source not in output.parents and output not in source.parents,
            "diagnostic output must be separate from the source run")
    for path, digest in inputs.items():
        require(file_sha256(path) == digest, "source changed during diagnosis")
    for name, content in files.items():
        path = output / name
        require(Path(name).name == name and not path.is_symlink(), "invalid report destination")
        require(not path.exists() or path.read_bytes() == content.encode("utf-8"),
                "existing report differs; choose a new --name")
    for name, content in files.items():
        if not (output / name).exists():
            atomic_text(output / name, content)
