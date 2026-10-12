"""Explicit recovery of one unreceipted network failure, with a separate ledger."""
from collections import Counter
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, getproxies, proxy_bypass

from evaluation.behavior import StrongRejectEvaluator
from rq2.artifacts import atomic_json, canonical_sha256, file_sha256
from rq2.exploration_config import ROOT, STAGES
from rq2.exploration_pipeline import ExplorationJudge, ExplorationPipeline
from rq2.judge_consistency import scoring_key
from rq2.replication_config import read_object
from rq2.replication_judge import ReplicationStop, _NoRedirect
from rq2.replication_rejudge import MigrationJudge


def forbidden_transport(body):
    raise ReplicationStop("Historical scoring cannot issue requests")


def inspect_failure(prior, attempt):
    rows = prior.state["attempts"]
    unresolved = [row for row in rows if row["status"] != "settled"]
    if (prior.state.get("fatal_error") or len(unresolved) != 1
            or unresolved[0] != rows[-1] or unresolved[0]["number"] != attempt):
        raise ReplicationStop("Recovery requires exactly the selected final unresolved request")
    row = unresolved[0]
    if row["status"] != "uncertain" or row.get("error") != "URLError":
        raise ReplicationStop("This entry only accepts an unreceipted URLError")
    receipt = prior.root / "receipts" / f"{attempt:06d}.json"
    cache = prior.root / "cache" / (row["key"] + ".json")
    if receipt.exists() or cache.exists():
        raise ReplicationStop("A receipt/cache exists; reconcile it instead of paying again")
    expected_cost = (row["prompt_token_reservation"] * prior.budget["input_cny_per_million"]
                     + 800 * prior.budget["output_cny_per_million"])
    if row["cost_micro_cny"] != expected_cost or row.get("reserved_micro_cny") != expected_cost:
        raise ReplicationStop("Uncertain request does not retain its full reservation")
    counts = Counter(a["key"] for a in rows)
    if counts[row["key"]] >= 3:
        raise ReplicationStop("The original three-attempt limit is exhausted for this input")
    expected_receipts = set()
    for settled in rows[:-1]:
        path = prior.root / "receipts" / f"{settled['number']:06d}.json"
        payload = read_object(path)
        usage = payload.get("usage", {})
        p, c = settled.get("prompt_tokens"), settled.get("completion_tokens")
        if (type(p) is not int or type(c) is not int or p < 0 or c < 0 or c > 800
                or p > settled["prompt_token_reservation"]
                or usage.get("prompt_tokens") != p or usage.get("completion_tokens") != c
                or usage.get("total_tokens") != p + c
                or settled["cost_micro_cny"] != p * prior.budget["input_cny_per_million"]
                   + c * prior.budget["output_cny_per_million"]):
            raise ReplicationStop("Historical receipt/usage/accounting mismatch")
        expected_receipts.add(path)
    if set((prior.root / "receipts").glob("*.json")) != expected_receipts:
        raise ReplicationStop("Unexpected historical receipt; reconcile before recovery")
    return row, dict(counts)


def _probe_url(judge):
    url = judge["base_url"].rstrip("/") + "/models"
    parts = urlsplit(url)
    if (parts.scheme != "https" or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ReplicationStop("Endpoint must be plain HTTPS without embedded credentials")
    return url, parts


def configure_direct_connection(judge):
    """Bypass proxies for this endpoint in this process, including later scoring."""
    _, parts = _probe_url(judge)
    entries = [v.strip() for name in ("NO_PROXY", "no_proxy")
               for v in os.environ.get(name, "").split(",") if v.strip()]
    entries.append(parts.hostname)
    value = ",".join(dict.fromkeys(entries))
    # urllib prefers lowercase; set both so the probe and frozen transport agree.
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = value


def endpoint_connection(judge):
    _, parts = _probe_url(judge)
    address = parts.hostname if parts.port is None else f"{parts.hostname}:{parts.port}"
    proxy = getproxies().get("https")
    uses_proxy = bool(proxy) and not proxy_bypass(address)
    result = {"host": parts.hostname, "port": parts.port or 443,
              "connection_mode": "environment_proxy" if uses_proxy else "direct"}
    if uses_proxy:
        try:
            parsed = urlsplit(proxy if "://" in proxy else "http://" + proxy)
            result["proxy"] = {"host": parsed.hostname, "port": parsed.port,
                               "credentials_present": bool(parsed.username or parsed.password)}
        except ValueError:
            result["proxy"] = {"invalid_address": True}
    return result


def check_endpoint(judge):
    """Probe HTTPS without sending credentials, prompts, or completion requests."""
    url, _ = _probe_url(judge)
    connection = endpoint_connection(judge)
    try:
        with build_opener(_NoRedirect()).open(Request(url, method="GET"), timeout=10) as handle:
            status = handle.status
    except HTTPError as exc:
        status = exc.code
        exc.close()
        if status not in {401, 403, 404, 405}:
            raise ReplicationStop(f"Endpoint probe returned HTTP {status}; no scoring request sent") from None
    except (URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        detail = str(reason).strip() or repr(reason)
        raise ReplicationStop(f"Endpoint probe failed ({connection['connection_mode']}): "
                              f"{type(reason).__name__}: {detail}") from None
    return {"https_reachable": True, "http_status": status, "credentials_sent": False,
            "api_key_or_balance_verified": False, "completion_requests": 0,
            "connection": connection}


class NetworkResumeJudge(ExplorationJudge):
    def __init__(self, *args, prior, failed, historical_counts, **kwargs):
        self.prior, self.failed = prior, failed
        self.historical_counts = historical_counts
        super().__init__(*args, **kwargs)

    def summary(self):
        result = super().summary()
        result.update(historical_uncertain_attempts=1,
                      historical_uncertain_reserved_cny=self.failed["reserved_micro_cny"] / 1e6,
                      historical_uncertain_charge_policy="full_reservation_carried_not_provider_settlement")
        return result

    def _reserve(self, key, body):
        count = self.historical_counts.get(key, 0) + sum(a["key"] == key for a in self.state["attempts"])
        if count >= 3:
            raise ReplicationStop("Original per-input three-attempt limit reached across ledgers")
        return super()._reserve(key, body)

    def label(self, response):
        if self.state.get("fatal_error") or any(a["status"] != "settled" for a in self.state["attempts"]):
            raise ReplicationStop("Recovery request unresolved; inspect it before any further retry")
        key = scoring_key(response, self.fingerprint)
        path = self.prior.root / "cache" / (key + ".json")
        if path.is_file():
            label = self.prior.label(response)
            audit = {"key": key, "source_cache": str(path), "source_sha256": file_sha256(path),
                     "policy": "same_contract_content_receipt_verified_no_new_request"}
            target = self.root / "imported_prior_cache" / (key + ".json")
            if target.exists() and read_object(target) != audit:
                raise ReplicationStop("Previously imported scoring cache changed")
            if not target.exists():
                atomic_json(target, audit)
            return label
        if key in self.historical_counts and key != self.failed["key"]:
            raise ReplicationStop("Another historical request lacks cache; do not resubmit it")
        return super().label(response)


class NetworkResumePipeline(ExplorationPipeline):
    def __init__(self, spec, attempt, **kwargs):
        super().__init__(spec, **kwargs)
        self.attempt = attempt
        self.recovery_root = self.config.output_root / "network_recovery" / f"attempt_{attempt:06d}"
        self.recovery_audit = None

    def inspect(self):
        prior = MigrationJudge(self.config.output_root / "scoring", self.config.judge,
                  self.spec.raw["scoring_contract"], self.spec.raw["budget"], self.spec.raw["history"],
                  transport=forbidden_transport)
        failed, counts = inspect_failure(prior, self.attempt)
        # Retrying must address the preserved input, never substitute a new response.
        _, responses, _ = self._checked_phase("window")
        matches = [r for r in responses.values() if scoring_key(r, prior.fingerprint) == failed["key"]]
        if not matches:
            raise ReplicationStop("Failed request is not in the frozen window responses")
        prompt = StrongRejectEvaluator.JUDGE_PROMPT.format(
            target_text=matches[0]["harmful_text"], response_to_evaluate=matches[0]["response"])
        messages = [{"role": "user", "content": prompt}]
        body = {"model": self.config.judge["model"], "messages": messages, "temperature": 0,
                "response_format": {"type": "json_object"}, "max_tokens": 800,
                "thinking": {"type": "disabled"}}
        if canonical_sha256(body) != failed["request_sha256"]:
            raise ReplicationStop("Only the preserved original request can be resumed here")
        files = sorted(p for p in prior.root.rglob("*") if p.is_file())
        if any(p.is_symlink() for p in prior.root.rglob("*")):
            raise ReplicationStop("Historical scoring must not contain symlinks")
        history = {"http_attempts": self.spec.raw["history"]["http_attempts"] + len(prior.state["attempts"]),
                   "cost_micro_cny": self.spec.raw["history"]["cost_micro_cny"]
                      + sum(a["cost_micro_cny"] for a in prior.state["attempts"])}
        audit = {"format": "rq2-explicit-network-recovery", "version": 1,
                 "config_fingerprint": self.spec.fingerprint, "failed_attempt": failed,
                 "history": history, "historical_counts": counts,
                 "original_scoring_files": {str(p): file_sha256(p) for p in files},
                 "implementation": {p: file_sha256(ROOT / p) for p in
                    ("rq2/exploration_network_resume.py", "experiments/rq2_exploration_network_resume.py")},
                 "policy": "explicit_retry_full_uncertain_reservation_carried_original_files_unchanged",
                 "scoring_contract_changed": False, "provider_charge_known": False,
                 "allowed_stages": list(STAGES[5:])}
        if any(p.is_symlink() for p in (self.recovery_root, *self.recovery_root.parents)):
            raise ReplicationStop("Recovery directory cannot be symlinked")
        manifest = self.recovery_root / "recovery_audit.json"
        if manifest.exists() and read_object(manifest) != audit:
            raise ReplicationStop("Recovery inputs changed; do not rebind an active recovery")
        if not manifest.exists() and self.recovery_root.exists() and any(self.recovery_root.iterdir()):
            raise ReplicationStop("Recovery artifacts exist without their audit")
        self.prior, self.failed, self.historical_counts = prior, failed, counts
        self.recovery_audit = audit
        return {"attempt": self.attempt, "error": failed["error"], "original_status": failed["status"],
                "reserved_cny_carried": failed["reserved_micro_cny"] / 1e6,
                "cumulative_accounted_cny_before_recovery": history["cost_micro_cny"] / 1e6,
                "recovery_root": str(self.recovery_root), "gpu_required": False,
                "remaining_attempts_for_failed_input": 3 - counts[failed["key"]],
                "new_ledger_exists": (self.recovery_root / "scoring/ledger.json").exists()}

    def verify(self):
        result = super().verify()
        state = self._state()["stages"]
        for stage in STAGES[:5]:
            if stage not in state or not self._fresh(stage, state[stage]):
                raise ReplicationStop(f"Required completed stage changed/missing: {stage}")
        result["network_recovery"] = self.inspect()
        return result

    def _judge(self):
        if self.recovery_audit is None:
            self.inspect()
        return NetworkResumeJudge(self.recovery_root / "scoring", self.config.judge,
                    self.spec.raw["scoring_contract"], self.spec.raw["budget"],
                    self.recovery_audit["history"], source_judge=self.parent._judge(),
                    prior=self.prior, failed=self.failed, historical_counts=self.historical_counts,
                    transport=self.transport)

    def _score(self, phase):
        if phase != "window":
            raise ReplicationStop("Only window scoring may be resumed")
        # The inherited scheduler holds the original exclusive run lock here.
        self.inspect()
        manifest = self.recovery_root / "recovery_audit.json"
        if not manifest.exists():
            atomic_json(manifest, self.recovery_audit)
        super()._score(phase)

    def run(self, stages, *, accept_uncertain_charge_and_retry=False, **kwargs):
        if tuple(stages) != STAGES[5:] or not accept_uncertain_charge_and_retry:
            raise ReplicationStop("Explicit uncertain-charge acknowledgement required; window_judge through report only")
        return super().run(stages, **kwargs)
