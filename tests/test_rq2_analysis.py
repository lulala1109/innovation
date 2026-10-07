import pytest

from rq2.analysis import (
    bh_fdr,
    causal_prior,
    evaluate_causal_gate,
    fit_layer_state_model,
    join_effect_rows,
    pair_cluster_bootstrap,
    paired_binary_tests,
    paired_specificity_contrasts,
    summarize_effects,
)
from rq2.artifacts import TrialKey, TrialRecord
from rq2.behavior import label_from_judge_result


def test_pair_cluster_bootstrap_clusters_repeated_rows():
    rows = [
        {"pair_id": "a", "effect": 1.0},
        {"pair_id": "a", "effect": 3.0},
        {"pair_id": "b", "effect": 0.0},
    ]
    result = pair_cluster_bootstrap(
        rows, value_field="effect", replicates=100, confidence=0.95, seed=1
    )
    assert result["pair_count"] == 2
    assert result["mean"] == pytest.approx(1.0)


def test_actual_shift_norm_audit_blocks_mismatched_control():
    def contrast(control_norm):
        rows = [
            {"pair_id": "p", "state_key": "fixed:2", "layer": 4,
             "intervention": "r_direction", "dose": 1.0, "token_scope": "audio",
             "causal_effect": 0.4, "generation_shift_l2": 1.0},
            {"pair_id": "p", "state_key": "fixed:2", "layer": 4,
             "intervention": "h_direction_control", "dose": 1.0, "token_scope": "audio",
             "causal_effect": 0.0, "generation_shift_l2": control_norm},
        ]
        result = paired_specificity_contrasts(
            rows, candidate_layers=(4,), distant_control_layers=(),
            replicates=20, confidence=0.95, seed=1,
        )
        return next(item for item in result if item["contrast"] == "r_vs_h")
    assert contrast(1.02)["norm_audit_status"] == "pass"
    assert contrast(1.2)["norm_audit_status"] == "mismatch"
    assert contrast(None)["norm_audit_status"] == "missing"
    for sham_norm, expected in ((0.0, "pass"), (0.01, "mismatch"), (None, "missing")):
        rows = [
            {"pair_id": "p", "state_key": "fixed:2", "layer": 4,
             "intervention": kind, "dose": 1.0, "token_scope": "audio",
             "causal_effect": effect, "generation_shift_l2": norm}
            for kind, effect, norm in (("r_direction", 0.4, 1.0), ("sham", 0.0, sham_norm))
        ]
        result = paired_specificity_contrasts(
            rows, candidate_layers=(4,), distant_control_layers=(),
            replicates=20, confidence=0.95, seed=1,
        )
        assert next(item for item in result if item["contrast"] == "r_vs_sham")["norm_audit_status"] == expected


def test_bh_fdr_is_monotone_in_rank():
    adjusted = bh_fdr([0.01, 0.02, 0.5])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_paired_specificity_contrast_uses_within_pair_difference():
    rows = []
    for pair in ("a", "b", "c"):
        rows.extend([
            {"pair_id": pair, "state_key": "fixed:2", "layer": 4,
             "intervention": "r_direction", "dose": 1.0, "token_scope": "audio", "causal_effect": 0.4},
            {"pair_id": pair, "state_key": "fixed:2", "layer": 4,
             "intervention": "sham", "dose": 1.0, "token_scope": "audio", "causal_effect": 0.0},
        ])
    contrasts = paired_specificity_contrasts(
        rows, candidate_layers=(4,), distant_control_layers=(),
        replicates=100, confidence=0.95, seed=3,
    )
    sham = next(row for row in contrasts if row["contrast"] == "r_vs_sham")
    assert sham["contrast_effect_mean"] == pytest.approx(0.4)


def _gate_inputs():
    summaries = [
        {
            "layer": 4, "state_key": "fixed:2", "intervention": "r_direction",
            "token_scope": "audio", "causal_effect_mean": 0.4,
            "fdr_q_value": 0.01, "sign_consistency": 0.9,
        },
        {
            "layer": 4, "state_key": "clean", "intervention": "reverse_suppression",
            "token_scope": "audio", "causal_effect_mean": 0.3,
            "fdr_q_value": 0.01, "sign_consistency": 0.9,
        },
    ]
    for row in summaries:
        row.update({
            "dose": 1.0, "pair_count": 40, "utility_pair_count": 40,
            "utility_effect_mean": row["causal_effect_mean"],
            "utility_ci_low": 0.1, "utility_fdr_q_value": 0.01,
            "utility_inference_status": "ok",
            "utility_fdr_family": (
                "F4_reverse" if row["intervention"] == "reverse_suppression"
                else "F1_fixed_utility"
            ),
            "refusal_effect_mean": 0.3, "refusal_ci_low": 0.1,
            "refusal_pair_count": 40, "refusal_inference_status": "ok",
            "refusal_fdr_q_value": 0.01,
            "refusal_fdr_family": (
                "F4_reverse" if row["intervention"] == "reverse_suppression"
                else "F2_candidate_refusal"
            ),
        })
    names = (
        "r_vs_h", "r_vs_random", "r_vs_sham", "audio_vs_position",
    )
    contrasts = [
        {
            "layer": 4, "state_key": "fixed:2", "contrast": name,
            "control_layer": 4, "primary_intervention": "r_direction",
            "dose": 1.0, "token_scope": "audio", "contrast_scope": "same_layer_specificity",
            "contrast_effect_mean": 0.2, "fdr_q_value": 0.01,
            "pair_count": 40, "ci_low": 0.1,
            "formal_inference_status": "ok", "fdr_family": "F3_same_layer_specificity",
            "norm_audit_status": "pass",
        }
        for name in names
    ]
    return summaries, contrasts


def test_causal_gate_requires_reverse_specificity_controls():
    summaries, contrasts = _gate_inputs()
    gate = evaluate_causal_gate(
        summaries, contrasts, alpha=0.05, minimum_sign_consistency=0.6
    )
    assert gate["supported"] is False
    for name in ("reverse_r_vs_h", "reverse_r_vs_random", "reverse_r_vs_sham"):
        contrasts.append({
            "layer": 4, "state_key": "clean", "contrast": name,
            "control_layer": 4, "primary_intervention": "reverse_suppression",
            "dose": 1.0, "token_scope": "audio", "contrast_scope": "same_layer_specificity",
            "contrast_effect_mean": 0.1, "fdr_q_value": 0.01,
            "pair_count": 40, "ci_low": 0.1,
            "formal_inference_status": "ok", "fdr_family": "F4_reverse",
            "norm_audit_status": "pass",
        })
    gate = evaluate_causal_gate(
        summaries, contrasts, alpha=0.05, minimum_sign_consistency=0.6
    )
    assert gate["supported"] is True
    assert gate["bidirectional_layers"] == [4]


def test_fractional_replicate_means_are_not_majority_binary_observations():
    rows = []
    for pair in ("a", "b"):
        for replicate, intervention in enumerate((True, True, False)):
            rows.append({
                "pair_id": pair, "layer": 4, "state_key": "fixed:2",
                "intervention": "random_direction_control", "dose": 1.0,
                "token_scope": "audio", "base_refusal": False,
                "intervention_refusal": intervention,
                "base_compliance": False, "intervention_compliance": intervention,
                "replicate": replicate,
            })
    results = paired_binary_tests(rows)
    refusal = next(row for row in results if row["outcome"] == "refusal")
    assert refusal["pair_count"] == 2
    assert refusal["paired_effect_mean"] == pytest.approx(2 / 3)
    assert refusal["baseline_false_to_intervention_true"] is None
    assert refusal["discordant_count"] is None
    assert refusal["inference_status"] == "non_binary_replicate_mean"
    assert refusal["mcnemar_exact_p_value"] is None
    assert refusal["fdr_q_value"] is None


def test_causal_prior_records_locked_doses_and_states():
    summaries, contrasts = _gate_inputs()
    gate = {"supported": True, "bidirectional_layers": [4], "supported_regions": [{"layer": 4}]}
    prior = causal_prior(
        summaries,
        gate=gate,
        protocol_lock_sha256="a" * 64,
        protocol={
            "primary_intervention": "r_direction",
            "primary_restoration_dose": 1.0,
            "primary_suppression_dose": 0.5,
        },
    )
    assert prior["lambda_restore"] == 1.0
    assert prior["eta_suppress"] == 0.5
    assert prior["state_bins"] == []
    assert prior["fixed_state_keys"] == ["fixed:2"]
    assert prior["severity_proxy"]["status"] == "not_provided"


def test_layer_state_model_reports_omnibus_likelihood_ratio_tests(monkeypatch):
    def fake_fit(design, outcome, groups, *, column_names):
        likelihood = {4: -10.0, 3: -14.0, 2: -18.0}[design.shape[1]]
        return {
            "inference_status": "ok",
            "fixed_effect_count": design.shape[1],
            "log_likelihood": likelihood,
            "coefficients": [],
        }

    monkeypatch.setattr("rq2.analysis.fit_random_intercept_model", fake_fit)
    rows = []
    for pair in ("a", "b"):
        for layer in (0, 1):
            for state in ("fixed:2", "fixed:10"):
                rows.append({
                    "pair_id": pair, "layer": layer, "state_key": state,
                    "intervention": "r_direction", "dose": 1.0, "token_scope": "audio",
                    "causal_effect": float(layer) + (state == "fixed:10"),
                })
    model = fit_layer_state_model(rows)
    assert model["interaction_likelihood_ratio_test"]["status"] == "ok"
    assert model["state_profile_likelihood_ratio_test"]["status"] == "ok"
    assert model["interaction_likelihood_ratio_test"]["p_value"] < 0.05


@pytest.mark.parametrize(
    "intervention,base_rubric,patch_rubric,expected_refusal",
    [
        ("r_direction", (0, 4, 4), (0, 1, 1), 0),
        ("r_direction", (0, 4, 4), (1, 1, 1), 1),
        ("reverse_suppression", (1, 1, 1), (0, 4, 4), 1),
    ],
)
def test_paired_endpoints_distinguish_utility_from_refusal(
    intervention, base_rubric, patch_rubric, expected_refusal,
):
    state = "clean" if intervention.startswith("reverse_") else "fixed:2"
    keys = [
        TrialKey("run", "pair", state, None, "baseline"),
        TrialKey("run", "pair", state, 4, intervention, 1.0),
    ]
    trials, labels = [], []
    for index, (key, rubric) in enumerate(zip(keys, (base_rubric, patch_rubric))):
        trials.append(TrialRecord(
            key, "rq2_causal_test", 0 if state == "clean" else 2,
            None if index == 0 else keys[0].trial_id, {}, {}, "a" * 64,
        ).to_record())
        labels.append(label_from_judge_result(
            key.trial_id, "a" * 64,
            dict(zip(("refusal", "convincingness", "specificity"), rubric)),
            judge_fingerprint="b" * 64,
        ).to_record())
    rows, _ = join_effect_rows(trials, labels)
    row = rows[0]
    assert row["utility_effect"] == pytest.approx(0.75)
    assert row["refusal_effect"] == expected_refusal
    assert row["compliance_effect"] == 1
    assert row["causal_effect"] == row["utility_effect"]
    assert row["behavior_recovery_rate"] == row["normalized_utility_effect"]
    summaries = summarize_effects(
        [{**row, "pair_id": f"p{index}"} for index in range(8)],
        replicates=100, confidence=0.95, seed=3,
    )
    assert summaries[0]["refusal_effect_mean"] == expected_refusal
    assert summaries[0]["refusal_pair_count"] == 8
    if expected_refusal == 0:
        gate = evaluate_causal_gate(
            summaries, [], alpha=0.05, minimum_sign_consistency=0.6,
        )
        assert gate["utility_reduction_supported"] is False
        assert gate["refusal_restoration_supported"] is False


@pytest.mark.parametrize("row_index", [0, 1])
@pytest.mark.parametrize(
    "field,value",
    [
        ("refusal_effect_mean", 0.0),
        ("refusal_ci_low", -0.01),
        ("refusal_fdr_q_value", 0.2),
        ("refusal_fdr_q_value", None),
        ("refusal_pair_count", 39),
    ],
)
def test_causal_gate_requires_matched_binary_confirmation(row_index, field, value):
    summaries, contrasts = _gate_inputs()
    contrasts.extend({
        "layer": 4, "state_key": "clean", "contrast": name,
        "control_layer": 4, "primary_intervention": "reverse_suppression",
        "dose": 1.0, "token_scope": "audio", "contrast_scope": "same_layer_specificity",
        "contrast_effect_mean": 0.1, "fdr_q_value": 0.01,
        "pair_count": 40, "ci_low": 0.1,
        "formal_inference_status": "ok", "fdr_family": "F4_reverse",
    } for name in ("reverse_r_vs_h", "reverse_r_vs_random", "reverse_r_vs_sham"))
    summaries[row_index][field] = value
    gate = evaluate_causal_gate(
        summaries, contrasts, alpha=0.05, minimum_sign_consistency=0.6,
    )
    assert gate["utility_reduction_supported"] is True
    assert gate["supported"] is False


def test_specificity_does_not_pool_doses_or_borrow_primary_controls():
    rows = []
    for pair in ("a", "b", "c"):
        for intervention, dose, effect in (
            ("r_direction", 1.0, 0.4),
            ("r_direction", 0.5, 0.9),
            ("h_direction_control", 1.0, 0.1),
        ):
            rows.append({
                "pair_id": pair, "state_key": "fixed:2", "layer": 19,
                "intervention": intervention, "dose": dose,
                "token_scope": "audio", "causal_effect": effect,
            })
    contrasts = paired_specificity_contrasts(
        rows, candidate_layers=(19,), replicates=100, confidence=0.95, seed=3,
    )
    assert len(contrasts) == 1
    assert contrasts[0]["dose"] == 1.0
    assert contrasts[0]["dose_role"] == "primary"
    assert contrasts[0]["contrast_effect_mean"] == pytest.approx(0.3)


def test_layer_profiles_preserve_adjacent_candidate_and_distant_roles():
    rows = [
        {
            "pair_id": pair, "state_key": "fixed:2", "layer": layer,
            "intervention": "r_direction", "dose": 1.0, "token_scope": "audio",
            "causal_effect": effect,
        }
        for pair in ("a", "b")
        for layer, effect in ((26, 0.4), (27, 0.3), (25, 0.2), (2, 0.1))
    ]
    contrasts = paired_specificity_contrasts(
        rows, candidate_layers=(26, 27), neighbor_layers=(25,),
        distant_control_layers=(2,), replicates=100, confidence=0.95, seed=3,
    )
    profile = [row for row in contrasts if row["layer"] == 26]
    assert {(row["control_layer"], row["control_layer_role"]) for row in profile} == {
        (27, "candidate"), (25, "neighbor"), (2, "distant"),
    }
    assert all(row["contrast_scope"] == "layer_profile" for row in contrasts)
    assert all(row["fdr_family"] == "layer_profile" for row in contrasts)


@pytest.mark.parametrize("mismatch", ["dose", "layer_profile", "intervention"])
def test_causal_gate_rejects_mismatched_or_layer_profile_controls(mismatch):
    summaries, contrasts = _gate_inputs()
    for row in contrasts:
        if mismatch == "dose":
            row["dose"] = 0.5
        elif mismatch == "layer_profile":
            row["contrast_scope"] = "layer_profile"
            row["control_layer"] = 2
        else:
            row["primary_intervention"] = "subspace_restoration"
    gate = evaluate_causal_gate(
        summaries, contrasts, alpha=0.05, minimum_sign_consistency=0.6,
    )
    assert gate["refusal_restoration_supported"] is True
    assert gate["restoration_specificity_regions"] == []
    assert gate["supported"] is False


def test_sensitivity_effect_cannot_supply_primary_refusal_claim():
    summaries, contrasts = _gate_inputs()
    for row in summaries:
        row["dose"] = 0.5
    for row in contrasts:
        row["dose"] = 0.5
    gate = evaluate_causal_gate(
        summaries, contrasts, alpha=0.05, minimum_sign_consistency=0.6,
    )
    assert gate["utility_reduction_supported"] is False
    assert gate["refusal_restoration_supported"] is False
    assert gate["supported"] is False
