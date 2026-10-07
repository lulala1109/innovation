"""RQ2 trial planning, prompt caching, and resumable GPU execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch

from core.audio import load_audio
from experiments.collect_safety_states import (
    load_trajectory_checkpoints,
    safe_torch_load,
)
from rq2.artifacts import (
    TRIAL_FORMAT,
    TRIAL_VERSION,
    TrialKey,
    TrialRecord,
    append_jsonl_fsync,
    atomic_jsonl,
    canonical_sha256,
    read_jsonl,
    records_by_id,
    validate_trial_record,
)
from rq2.behavior import make_response_record, validate_response_record
from rq2.data import RQ2Pair
from rq2.interventions import InterventionSpec
from rq2.qwen_runtime import QwenCausalRuntime


class RQ2ExperimentError(RuntimeError):
    """Raised when a trial plan or resumable execution is inconsistent."""


@dataclass(frozen=True)
class PlannedTrial:
    pair_id: str
    rq2_role: str
    state_key: str
    step: Optional[int]
    layer: int
    intervention: str
    dose: float
    token_scope: str = "audio"
    seed: int = 0
    replicate: int = 0
    target_kind: str = "trajectory"

    def to_key(self, run_fingerprint: str) -> TrialKey:
        return TrialKey(
            run_fingerprint=run_fingerprint,
            pair_id=self.pair_id,
            state_key=self.state_key,
            layer=self.layer,
            intervention=self.intervention,
            dose=self.dose,
            token_scope=self.token_scope,
            replicate=self.replicate,
        )

    def to_spec(self) -> InterventionSpec:
        derived_seed = int(
            canonical_sha256(
                {
                    "seed": self.seed,
                    "pair_id": self.pair_id,
                    "state_key": self.state_key,
                    "layer": self.layer,
                    "replicate": self.replicate,
                }
            )[:16],
            16,
        ) % (2**63 - 1)
        return InterventionSpec(
            kind=self.intervention,
            layer=self.layer,
            dose=self.dose,
            token_scope=self.token_scope,
            seed=derived_seed,
            replicate=self.replicate,
        )


@dataclass(frozen=True)
class PreparedTrial:
    source_prompt: Mapping[str, Any]
    target_prompt: Mapping[str, Any]
    harmful_text: str
    source_cache_key: str
    input_provenance: Mapping[str, Any]


def _state_rows(
    state_index: Mapping[str, Any],
    *,
    role: str,
    coordinate: str,
) -> list[Mapping[str, Any]]:
    records = state_index.get("records")
    if not isinstance(records, list):
        raise RQ2ExperimentError("state index lacks records")
    return [
        row
        for row in records
        if isinstance(row, Mapping)
        and row.get("rq2_role") == role
        and row.get("coordinate") == coordinate
        and row.get("available") is True
        and isinstance(row.get("step"), int)
    ]


def _layers(values: Iterable[Any], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in result):
        raise RQ2ExperimentError(f"{name} must contain non-negative layers")
    if len(set(result)) != len(result):
        raise RQ2ExperimentError(f"{name} contains duplicate layers")
    return result


def build_oracle_plan(
    state_index: Mapping[str, Any],
    *,
    candidate_layers: Sequence[int],
    neighbor_layers: Sequence[int] = (),
    coordinate: str = "fixed",
    state_keys: Optional[Sequence[str]] = None,
) -> tuple[PlannedTrial, ...]:
    layers = _layers(candidate_layers, "candidate_layers")
    neighbors = _layers(neighbor_layers, "neighbor_layers") if neighbor_layers else ()
    if set(layers).intersection(neighbors):
        raise RQ2ExperimentError("candidate and neighbor layers must be disjoint")
    trials = []
    if coordinate not in {"fixed", "event"}:
        raise RQ2ExperimentError("unsupported pilot coordinate")
    for row in _state_rows(state_index, role="rq2_dev", coordinate=coordinate):
        if state_keys is not None and row["state_key"] not in state_keys:
            continue
        for layer in layers:
            for intervention, dose in (
                ("full_state", 1.0),
                ("self_patch", 1.0),
                ("sham", 1.0),
                ("r_direction", 0.0),
            ):
                trials.append(
                    PlannedTrial(
                        pair_id=str(row["pair_id"]),
                        rq2_role="rq2_dev",
                        state_key=str(row["state_key"]),
                        step=int(row["step"]),
                        layer=layer,
                        intervention=intervention,
                        dose=dose,
                    )
                )
        for layer in neighbors:
            trials.append(PlannedTrial(
                pair_id=str(row["pair_id"]), rq2_role="rq2_dev",
                state_key=str(row["state_key"]), step=int(row["step"]),
                layer=layer, intervention="full_state", dose=1.0,
            ))
    return tuple(trials)


def build_mechanism_plan(
    state_index: Mapping[str, Any],
    *,
    candidate_layers: Sequence[int],
    restoration_doses: Sequence[float],
    reverse_doses: Sequence[float],
    random_replicates: int,
    seed: int,
    enable_subspace: bool = False,
    primary_restoration_dose: Optional[float] = None,
    neighbor_layers: Sequence[int] = (),
    coordinate: str = "fixed",
    state_keys: Optional[Sequence[str]] = None,
) -> tuple[PlannedTrial, ...]:
    layers = _layers(candidate_layers, "candidate_layers")
    neighbors = _layers(neighbor_layers, "neighbor_layers") if neighbor_layers else ()
    if set(layers).intersection(neighbors):
        raise RQ2ExperimentError("candidate and neighbor layers must be disjoint")
    if random_replicates < 1:
        raise RQ2ExperimentError("random_replicates must be positive")
    doses = tuple(float(value) for value in restoration_doses)
    if not doses:
        raise RQ2ExperimentError("restoration_doses cannot be empty")
    primary = doses[0] if primary_restoration_dose is None else float(primary_restoration_dose)
    if primary not in doses:
        raise RQ2ExperimentError("primary_restoration_dose must be present in restoration_doses")
    trials: list[PlannedTrial] = []
    if coordinate not in {"fixed", "event"}:
        raise RQ2ExperimentError("unsupported pilot coordinate")
    rows = [row for row in _state_rows(state_index, role="rq2_dev", coordinate=coordinate)
            if state_keys is None or row["state_key"] in state_keys]
    for row in rows:
        common = dict(
            pair_id=str(row["pair_id"]),
            rq2_role="rq2_dev",
            state_key=str(row["state_key"]),
            step=int(row["step"]),
        )
        trials.extend(
            PlannedTrial(**common, layer=layer, intervention="r_direction", dose=primary)
            for layer in neighbors
        )
        for layer in layers:
            trials.extend(
                PlannedTrial(**common, layer=layer, intervention="r_direction", dose=dose)
                for dose in doses
            )
            trials.append(PlannedTrial(**common, layer=layer, intervention="h_direction_control", dose=primary))
            trials.append(PlannedTrial(**common, layer=layer, intervention="sham", dose=primary))
            for replicate in range(random_replicates):
                trials.append(
                    PlannedTrial(
                        **common,
                        layer=layer,
                        intervention="random_direction_control",
                        dose=primary,
                        seed=seed,
                        replicate=replicate,
                    )
                )
            if enable_subspace:
                trials.append(PlannedTrial(**common, layer=layer, intervention="subspace_restoration", dose=primary))

    # Clean-state suppression is a pair-level reverse causal test, not one test
    # per adversarial checkpoint.  Deduplicate pair/layer/dose explicitly.
    pair_rows: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        pair_rows.setdefault(str(row["pair_id"]), row)
    for pair_id, row in pair_rows.items():
        for layer in layers:
            for dose in reverse_doses:
                common = dict(
                    pair_id=pair_id,
                    rq2_role="rq2_dev",
                    state_key="clean",
                    step=None,
                    layer=layer,
                    dose=float(dose),
                    target_kind="clean",
                )
                trials.append(PlannedTrial(**common, intervention="reverse_suppression"))
                trials.append(PlannedTrial(**common, intervention="reverse_h_control"))
                trials.append(PlannedTrial(**common, intervention="reverse_sham"))
                for replicate in range(random_replicates):
                    trials.append(
                        PlannedTrial(
                            **common,
                            intervention="reverse_random_control",
                            seed=seed,
                            replicate=replicate,
                        )
                    )
    return tuple(trials)


def build_subspace_plan(
    state_index: Mapping[str, Any],
    *,
    candidate_layers: Sequence[int],
    dose: float = 1.0,
    random_replicates: int = 1,
    seed: int = 42,
    neighbor_layers: Sequence[int] = (),
) -> tuple[PlannedTrial, ...]:
    layers = _layers(candidate_layers, "candidate_layers")
    neighbors = _layers(neighbor_layers, "neighbor_layers") if neighbor_layers else ()
    if set(layers).intersection(neighbors):
        raise RQ2ExperimentError("candidate and neighbor layers must be disjoint")
    if random_replicates < 1:
        raise RQ2ExperimentError("random_replicates must be positive")
    trials: list[PlannedTrial] = []
    for row in _state_rows(state_index, role="rq2_dev", coordinate="fixed"):
        common = dict(
            pair_id=str(row["pair_id"]),
            rq2_role="rq2_dev",
            state_key=str(row["state_key"]),
            step=int(row["step"]),
            dose=float(dose),
        )
        trials.extend(
            PlannedTrial(**common, layer=layer, intervention="subspace_restoration")
            for layer in neighbors
        )
        for layer in layers:
            trials.append(PlannedTrial(**common, layer=layer, intervention="subspace_restoration"))
            trials.append(PlannedTrial(**common, layer=layer, intervention="subspace_h_control"))
            trials.append(PlannedTrial(**common, layer=layer, intervention="sham"))
            for replicate in range(random_replicates):
                trials.append(PlannedTrial(
                    **common,
                    layer=layer,
                    intervention="subspace_random_control",
                    seed=seed,
                    replicate=replicate,
                ))
    return tuple(trials)


def build_formal_plan(
    state_index: Mapping[str, Any],
    protocol_lock: Mapping[str, Any],
    *,
    event: bool = False,
) -> tuple[PlannedTrial, ...]:
    if protocol_lock.get("format") != "rq2-protocol-lock" or protocol_lock.get("version") not in (1, 2):
        raise RQ2ExperimentError("formal execution requires a supported RQ2 protocol lock")
    if protocol_lock["version"] == 2:
        if (
            protocol_lock.get("layer_dose_version") != 2
            or protocol_lock.get("primary_restoration_dose") != 1.0
            or protocol_lock.get("primary_suppression_dose") != 1.0
        ):
            raise RQ2ExperimentError("v2 formal primary restoration/suppression doses must be fixed at 1.0")
    if protocol_lock.get("locked") is not True:
        raise RQ2ExperimentError("protocol lock is not frozen")
    primary_dose = float(protocol_lock["primary_restoration_dose"])
    primary_intervention = str(protocol_lock.get("primary_intervention", "r_direction"))
    if primary_intervention not in {"r_direction", "subspace_restoration"}:
        raise RQ2ExperimentError("unsupported locked primary intervention")
    scan_layers = _layers(protocol_lock["formal_layers"], "formal_layers")
    control_layers = _layers(protocol_lock["control_layers"], "control_layers")
    role = "rq2_causal_test"
    coordinate = "event" if event else "fixed"
    trials: list[PlannedTrial] = []
    for row in _state_rows(state_index, role=role, coordinate=coordinate):
        common = dict(
            pair_id=str(row["pair_id"]),
            rq2_role=role,
            state_key=str(row["state_key"]),
            step=int(row["step"]),
        )
        active_layers = tuple(protocol_lock["candidate_layers"]) if event else scan_layers
        for layer in active_layers:
            trials.append(
                PlannedTrial(**common, layer=int(layer), intervention=primary_intervention, dose=primary_dose)
            )
        if not event:
            for layer in control_layers:
                h_control = (
                    "subspace_h_control"
                    if primary_intervention == "subspace_restoration"
                    else "h_direction_control"
                )
                random_control = (
                    "subspace_random_control"
                    if primary_intervention == "subspace_restoration"
                    else "random_direction_control"
                )
                for intervention in (h_control, "sham"):
                    trials.append(
                        PlannedTrial(**common, layer=layer, intervention=intervention, dose=primary_dose)
                    )
                for replicate in range(int(protocol_lock["random_replicates"])):
                    trials.append(
                        PlannedTrial(
                            **common,
                            layer=layer,
                            intervention=random_control,
                            dose=primary_dose,
                            seed=int(protocol_lock["seed"]),
                            replicate=replicate,
                        )
                    )
                trials.append(
                    PlannedTrial(
                        **common,
                        layer=layer,
                        intervention=primary_intervention,
                        dose=primary_dose,
                        token_scope="position_control",
                    )
                )
    if not event:
        pair_rows: dict[str, Mapping[str, Any]] = {}
        for row in _state_rows(state_index, role=role, coordinate="fixed"):
            pair_rows.setdefault(str(row["pair_id"]), row)
        suppression_dose = float(protocol_lock["primary_suppression_dose"])
        for pair_id in sorted(pair_rows):
            for layer in protocol_lock["candidate_layers"]:
                common = dict(
                    pair_id=pair_id,
                    rq2_role=role,
                    state_key="clean",
                    step=None,
                    layer=int(layer),
                    dose=suppression_dose,
                    target_kind="clean",
                )
                trials.append(PlannedTrial(**common, intervention="reverse_suppression"))
                trials.append(PlannedTrial(**common, intervention="reverse_h_control"))
                trials.append(PlannedTrial(**common, intervention="reverse_sham"))
                for replicate in range(int(protocol_lock["random_replicates"])):
                    trials.append(
                        PlannedTrial(
                            **common,
                            intervention="reverse_random_control",
                            seed=int(protocol_lock["seed"]),
                            replicate=replicate,
                        )
                    )
    return tuple(trials)


class QwenPromptCache:
    """Load each clean/checkpoint waveform and prepared prompt at most once."""

    def __init__(self, model: Any, pairs: Sequence[RQ2Pair]) -> None:
        self.model = model
        self.pairs = {pair.pair_id: pair for pair in pairs}
        self._clean: dict[str, Mapping[str, Any]] = {}
        self._trajectory: dict[tuple[str, int], Mapping[str, Any]] = {}
        self._checkpoints: dict[str, dict[int, Path]] = {}
        self._checkpoint_hashes: dict[tuple[str, int], str] = {}

    def _clean_prompt(self, pair: RQ2Pair) -> Mapping[str, Any]:
        if pair.pair_id not in self._clean:
            waveform = load_audio(str(pair.clean_audio_path), target_sr=self.model.sample_rate)
            self._clean[pair.pair_id] = self.model.prepare_audio_prompt(waveform)
        return self._clean[pair.pair_id]

    def _checkpoint_paths(self, pair: RQ2Pair) -> Mapping[int, Path]:
        if pair.pair_id not in self._checkpoints:
            values = load_trajectory_checkpoints(pair.trajectory_path, pair_id=pair.pair_id)
            self._checkpoints[pair.pair_id] = {item.step: item.path for item in values}
            from rq2.artifacts import file_sha256

            self._checkpoint_hashes.update(
                {(pair.pair_id, item.step): file_sha256(item.path) for item in values}
            )
        return self._checkpoints[pair.pair_id]

    @staticmethod
    def _checkpoint_waveform(path: Path) -> torch.Tensor:
        payload = safe_torch_load(path)
        tensors = payload.get("tensors", payload)
        if not isinstance(tensors, Mapping):
            raise RQ2ExperimentError(f"checkpoint tensors are malformed: {path}")
        for field in ("adversarial_wav", "waveform", "wav", "audio"):
            value = tensors.get(field)
            if isinstance(value, torch.Tensor):
                return value
        raise RQ2ExperimentError(f"checkpoint lacks an audio waveform: {path}")

    def prepare(self, trial: PlannedTrial) -> PreparedTrial:
        if trial.pair_id not in self.pairs:
            raise RQ2ExperimentError(f"unknown pair_id in trial plan: {trial.pair_id}")
        pair = self.pairs[trial.pair_id]
        source = self._clean_prompt(pair)
        if trial.target_kind == "clean":
            target = source
            target_provenance = {
                "kind": "clean",
                "clean_audio_sha256": pair.clean_audio_sha256,
            }
        else:
            if trial.step is None:
                raise RQ2ExperimentError("trajectory trial has no step")
            key = (pair.pair_id, trial.step)
            if key not in self._trajectory:
                checkpoint = self._checkpoint_paths(pair).get(trial.step)
                if checkpoint is None:
                    raise RQ2ExperimentError(
                        f"trajectory step {trial.step} is missing for {pair.pair_id}"
                    )
                waveform = self._checkpoint_waveform(checkpoint)
                self._trajectory[key] = self.model.prepare_audio_prompt(waveform)
            target = self._trajectory[key]
            target_provenance = {
                "kind": "trajectory",
                "step": trial.step,
                "checkpoint_path": str(self._checkpoint_paths(pair)[trial.step]),
                "checkpoint_sha256": self._checkpoint_hashes[key],
            }
        return PreparedTrial(
            source_prompt=source,
            target_prompt=target,
            harmful_text=pair.harmful_text,
            source_cache_key=f"{pair.pair_id}:{pair.clean_audio_sha256}",
            input_provenance={
                "pair_id": pair.pair_id,
                "content_group": pair.content_group,
                "clean_audio_sha256": pair.clean_audio_sha256,
                "target": target_provenance,
            },
        )


def _audio_selection(prompt: Mapping[str, Any]) -> tuple[int, ...]:
    spans = prompt.get("token_spans")
    if not isinstance(spans, Mapping) or "audio" not in spans:
        raise RQ2ExperimentError("prepared prompt lacks audio token span")
    start, end = spans["audio"]
    return tuple(range(int(start), int(end)))


def _position_control(prompt: Mapping[str, Any]) -> tuple[int, ...]:
    embeds = prompt.get("inputs_embeds")
    audio = set(_audio_selection(prompt))
    if not isinstance(embeds, torch.Tensor):
        raise RQ2ExperimentError("prepared prompt lacks inputs_embeds")
    candidates = [index for index in range(embeds.shape[-2]) if index not in audio]
    if not candidates:
        raise RQ2ExperimentError("no non-audio positions are available for token control")
    return tuple(candidates)


def _baseline_key(run_fingerprint: str, trial: PlannedTrial) -> TrialKey:
    return TrialKey(
        run_fingerprint=run_fingerprint,
        pair_id=trial.pair_id,
        state_key=trial.state_key,
        layer=None,
        intervention="baseline",
        dose=0.0,
        token_scope="audio",
        replicate=0,
    )


def run_planned_trials(
    trials: Sequence[PlannedTrial],
    *,
    runtime: QwenCausalRuntime,
    prompt_cache: QwenPromptCache,
    run_fingerprint: str,
    responses_path: str | Path,
    trials_path: str | Path,
    generation: Mapping[str, Any],
    subspaces: Optional[Mapping[int, torch.Tensor]] = None,
) -> dict[str, int]:
    """Execute missing trials, persisting public/private artifacts atomically."""

    if bool(generation.get("do_sample", False)):
        raise RQ2ExperimentError("RQ2 causal generation must be deterministic")

    responses_file = Path(responses_path).expanduser().resolve()
    trials_file = Path(trials_path).expanduser().resolve()
    commits_file = responses_file.parent / "commits.jsonl"
    existing_responses = [
        validate_response_record(row)
        for row in read_jsonl(responses_file, missing_ok=True, recover_truncated=True)
    ]
    response_by_id = records_by_id(existing_responses)
    existing_trials = [
        validate_trial_record(row)
        for row in read_jsonl(trials_file, missing_ok=True, recover_truncated=True)
    ]
    trial_by_id = records_by_id(existing_trials)
    commit_by_id: dict[str, dict[str, Any]] = {}
    for commit in read_jsonl(commits_file, missing_ok=True, recover_truncated=True):
        if commit.get("format") != "rq2-trial-commit" or commit.get("version") != 1:
            raise RQ2ExperimentError("unsupported trial commit format/version")
        response = validate_response_record(commit.get("response", {}))
        trial_record = validate_trial_record(commit.get("trial", {}))
        if (
            response["trial_id"] != trial_record["trial_id"]
            or commit.get("trial_id") != response["trial_id"]
        ):
            raise RQ2ExperimentError("commit response/trial identities differ")
        prior = commit_by_id.get(response["trial_id"])
        if prior is not None and prior != dict(commit):
            raise RQ2ExperimentError("conflicting duplicate trial commit")
        commit_by_id[response["trial_id"]] = dict(commit)
    if commit_by_id:
        response_by_id = {
            trial_id: validate_response_record(commit["response"])
            for trial_id, commit in commit_by_id.items()
        }
        trial_by_id = {
            trial_id: validate_trial_record(commit["trial"])
            for trial_id, commit in commit_by_id.items()
        }
        atomic_jsonl(responses_file, response_by_id.values())
        atomic_jsonl(trials_file, trial_by_id.values())
    elif set(response_by_id) != set(trial_by_id):
        raise RQ2ExperimentError("legacy responses/trials mismatch cannot be recovered without commits")
    elif response_by_id:
        commit_by_id = {
            trial_id: {
                "format": "rq2-trial-commit",
                "version": 1,
                "trial_id": trial_id,
                "response": response_by_id[trial_id],
                "trial": trial_by_id[trial_id],
            }
            for trial_id in response_by_id
        }
        atomic_jsonl(commits_file, commit_by_id.values())
    for record in trial_by_id.values():
        if record["run_fingerprint"] != run_fingerprint:
            raise RQ2ExperimentError("stale trials belong to a different run fingerprint")
        if response_by_id[record["trial_id"]]["response_sha256"] != record["response_sha256"]:
            raise RQ2ExperimentError("trial and response hashes differ")

    completed = 0
    reused = 0

    def persist(response: Mapping[str, Any], trial_record: Mapping[str, Any]) -> None:
        trial_id = str(response["trial_id"])
        if trial_id != trial_record["trial_id"]:
            raise RQ2ExperimentError("response/trial identities differ")
        response_by_id[trial_id] = dict(response)
        trial_by_id[trial_id] = dict(trial_record)
        commit_by_id[trial_id] = {
            "format": "rq2-trial-commit",
            "version": 1,
            "trial_id": trial_id,
            "response": dict(response),
            "trial": dict(trial_record),
        }
        # The private commit is the transaction point. Response and text-free
        # trial projections can always be rebuilt from it after a crash.
        append_jsonl_fsync(commits_file, commit_by_id[trial_id])

    for planned in trials:
        prepared = prompt_cache.prepare(planned)
        baseline_key = _baseline_key(run_fingerprint, planned)
        if baseline_key.trial_id not in trial_by_id:
            response_text = runtime.model.generate_from_prepared_prompt(
                prepared.target_prompt, **dict(generation)
            )
            activations = runtime.capture(prepared.target_prompt)
            profile = runtime.safety_profile(
                activations, _audio_selection(prepared.target_prompt)
            )
            response = make_response_record(
                trial_id=baseline_key.trial_id,
                harmful_text=prepared.harmful_text,
                response=str(response_text),
                pair_id=planned.pair_id,
                run_fingerprint=run_fingerprint,
            )
            record = TrialRecord(
                key=baseline_key,
                rq2_role=planned.rq2_role,
                step=0 if planned.step is None else planned.step,
                baseline_trial_id=None,
                audit={"kind": "baseline", "apply_count": 0},
                diagnostic={"profile": profile, "input_provenance": prepared.input_provenance},
                response_sha256=response["response_sha256"],
            ).to_record()
            persist(response, record)
            completed += 1
        else:
            reused += 1

        key = planned.to_key(run_fingerprint)
        if key.trial_id in trial_by_id:
            reused += 1
            continue
        token_selection: str | Sequence[int] = "audio"
        reference_token_selection: Optional[str | Sequence[int]] = None
        application_scale = 1.0
        if planned.token_scope == "position_control":
            audio_count = len(_audio_selection(prepared.target_prompt))
            token_selection = _position_control(prepared.target_prompt)
            reference_token_selection = "audio"
            application_scale = (audio_count / len(token_selection)) ** 0.5
        result = runtime.run_trial(
            prepared.source_prompt,
            prepared.target_prompt,
            planned.to_spec(),
            token_selection=token_selection,
            reference_token_selection=reference_token_selection,
            application_scale=application_scale,
            subspace=(subspaces or {}).get(planned.layer),
            source_cache_key=prepared.source_cache_key,
            **dict(generation),
        )
        response = make_response_record(
            trial_id=key.trial_id,
            harmful_text=prepared.harmful_text,
            response=result.response,
            pair_id=planned.pair_id,
            run_fingerprint=run_fingerprint,
        )
        audit = result.generation_audit.to_dict()
        audit.update(
            {
                "diagnostic": result.diagnostic_audit.to_dict(),
                "generation_cache_skips": result.generation_cache_skips,
            }
        )
        record = TrialRecord(
            key=key,
            rq2_role=planned.rq2_role,
            step=0 if planned.step is None else planned.step,
            baseline_trial_id=baseline_key.trial_id,
            audit=audit,
            diagnostic={**dict(result.diagnostic), "input_provenance": prepared.input_provenance},
            response_sha256=response["response_sha256"],
        ).to_record()
        persist(response, record)
        completed += 1
    atomic_jsonl(responses_file, response_by_id.values())
    atomic_jsonl(trials_file, trial_by_id.values())
    return {"planned": len(trials), "completed": completed, "reused": reused}


def run_unpatched_trajectory_states(
    scan_index: Mapping[str, Any],
    *,
    runtime: QwenCausalRuntime,
    prompt_cache: QwenPromptCache,
    run_fingerprint: str,
    responses_path: str | Path,
    trials_path: str | Path,
    generation: Mapping[str, Any],
    capture_profiles: bool = True,
) -> dict[str, int]:
    if bool(generation.get("do_sample", False)):
        raise RQ2ExperimentError("RQ2 causal baselines require deterministic generation")
    records = scan_index.get("records")
    if not isinstance(records, list):
        raise RQ2ExperimentError("trajectory scan index lacks records")
    responses_file = Path(responses_path).expanduser().resolve()
    trials_file = Path(trials_path).expanduser().resolve()
    commits_file = responses_file.parent / "commits.jsonl"
    existing_responses = records_by_id(
        validate_response_record(row) for row in read_jsonl(responses_file, missing_ok=True, recover_truncated=True)
    )
    existing_trials = records_by_id(
        validate_trial_record(row) for row in read_jsonl(trials_file, missing_ok=True, recover_truncated=True)
    )
    commits: dict[str, dict[str, Any]] = {}
    for commit in read_jsonl(commits_file, missing_ok=True, recover_truncated=True):
        if commit.get("format") != "rq2-trial-commit" or commit.get("version") != 1:
            raise RQ2ExperimentError("unsupported trajectory scan commit")
        response = validate_response_record(commit.get("response", {}))
        trial = validate_trial_record(commit.get("trial", {}))
        if (
            response["trial_id"] != trial["trial_id"]
            or commit.get("trial_id") != response["trial_id"]
        ):
            raise RQ2ExperimentError("trajectory scan commit identities differ")
        prior = commits.get(response["trial_id"])
        if prior is not None and prior != dict(commit):
            raise RQ2ExperimentError("conflicting duplicate trajectory scan commit")
        commits[response["trial_id"]] = dict(commit)
    if commits:
        existing_responses = {
            key: validate_response_record(value["response"])
            for key, value in commits.items()
        }
        existing_trials = {
            key: validate_trial_record(value["trial"])
            for key, value in commits.items()
        }
        atomic_jsonl(responses_file, existing_responses.values())
        atomic_jsonl(trials_file, existing_trials.values())
    elif set(existing_responses) != set(existing_trials):
        raise RQ2ExperimentError("trajectory scan response/trial sets differ")
    elif existing_responses:
        commits = {
            key: {
                "format": "rq2-trial-commit", "version": 1, "trial_id": key,
                "response": existing_responses[key], "trial": existing_trials[key],
            }
            for key in existing_responses
        }
        atomic_jsonl(commits_file, commits.values())
    for record in existing_trials.values():
        if record["run_fingerprint"] != run_fingerprint:
            raise RQ2ExperimentError("stale trajectory scan belongs to another run fingerprint")
        response = existing_responses.get(record["trial_id"])
        if response is None or response["response_sha256"] != record["response_sha256"]:
            raise RQ2ExperimentError("trajectory scan trial and response hashes differ")
    completed = 0
    for row in records:
        planned = PlannedTrial(
            pair_id=str(row["pair_id"]),
            rq2_role=str(row["rq2_role"]),
            state_key=str(row["state_key"]),
            step=row.get("step"),
            layer=0,
            intervention="sham",
            dose=0.0,
            target_kind=str(row["target_kind"]),
        )
        key = TrialKey(
            run_fingerprint=run_fingerprint,
            pair_id=planned.pair_id,
            state_key=planned.state_key,
            layer=None,
            intervention="trajectory_baseline",
            dose=0.0,
            token_scope="audio",
            replicate=0,
        )
        if key.trial_id in existing_trials:
            continue
        prepared = prompt_cache.prepare(planned)
        text = runtime.model.generate_from_prepared_prompt(prepared.target_prompt, **dict(generation))
        diagnostic: dict[str, Any] = {"input_provenance": prepared.input_provenance}
        if capture_profiles:
            diagnostic["profile"] = runtime.safety_profile(
                runtime.capture(prepared.target_prompt),
                _audio_selection(prepared.target_prompt),
            )
        response = make_response_record(
            trial_id=key.trial_id,
            harmful_text=prepared.harmful_text,
            response=str(text),
            pair_id=planned.pair_id,
            run_fingerprint=run_fingerprint,
        )
        trial_record = TrialRecord(
            key=key,
            rq2_role=planned.rq2_role,
            step=0 if planned.step is None else int(planned.step),
            baseline_trial_id=None,
            audit={"kind": "trajectory_baseline", "apply_count": 0},
            diagnostic=diagnostic,
            response_sha256=response["response_sha256"],
        ).to_record()
        existing_responses[key.trial_id] = response
        existing_trials[key.trial_id] = trial_record
        commits[key.trial_id] = {
            "format": "rq2-trial-commit", "version": 1,
            "trial_id": key.trial_id, "response": response, "trial": trial_record,
        }
        append_jsonl_fsync(commits_file, commits[key.trial_id])
        completed += 1
    atomic_jsonl(responses_file, existing_responses.values())
    atomic_jsonl(trials_file, existing_trials.values())
    return {"planned": len(records), "completed": completed, "reused": len(records) - completed}


__all__ = [
    "PlannedTrial",
    "PreparedTrial",
    "QwenPromptCache",
    "RQ2ExperimentError",
    "build_formal_plan",
    "build_mechanism_plan",
    "build_oracle_plan",
    "build_subspace_plan",
    "run_planned_trials",
    "run_unpatched_trajectory_states",
]
