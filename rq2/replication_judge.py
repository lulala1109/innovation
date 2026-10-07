"""Content-keyed scoring with a single HTTP attempt layer and durable budget.

No legacy label import, no SDK retries, no billing credentials in artifacts.
The CNY ledger uses the supplied peak prices without cache discounts; it is
a conservative accounting estimate, not a statement of the provider invoice.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, build_opener, HTTPRedirectHandler

from evaluation.behavior import StrongRejectEvaluator, StrongRejectScore
from rq2.artifacts import atomic_json, canonical_sha256, file_sha256
from rq2.behavior import label_from_judge_result, validate_response_record
from rq2.judge_consistency import checked_label, scoring_key, scoring_contract


class ReplicationStop(RuntimeError):
    """Preserve artifacts; human inspection is needed before further spending."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def contract_for(judge):
    result = scoring_contract(judge)
    result.update(version=2, transport="urllib_single_attempt_no_redirect",
                  transport_retries=0, empty_content_retries=0, schema_retries=1,
                  total_attempts_per_input=3, request_timeout_seconds=60,
                  max_tokens=800, price_policy="peak_without_cache_discount",
                  input_reservation="serialized_messages_utf8_bytes_plus_1024",
                  incomplete_usage="retain_reservation_and_stop",
                  interrupted_attempt="never_automatically_resubmit")
    result["implementation_sha256"]["rq2/replication_judge.py"] = file_sha256(Path(__file__))
    return result


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ReplicationStop("Judge endpoint redirected; no credentials forwarded")


def http_once(judge, body):
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise ReplicationStop("DEEPSEEK_API_KEY is not loaded")
    request = Request(judge["base_url"].rstrip("/") + "/chat/completions",
                      data=json.dumps(body, ensure_ascii=False).encode(), method="POST",
                      headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    # One call, no redirects and no hidden transport/empty-content retries.
    with build_opener(_NoRedirect()).open(request, timeout=60) as handle:
        return json.loads(handle.read())


class ContentJudge:
    def __init__(self, root, judge, contract, budget, *, transport=None):
        self.root, self.judge = Path(root), judge
        self.contract, self.fingerprint = contract, canonical_sha256(contract)
        self.budget = budget
        self.real_transport = transport is None
        self.transport = transport or (lambda body: http_once(judge, body))
        self.path = self.root / "ledger.json"
        self.binding = canonical_sha256({"contract": contract, "budget": budget})
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get("binding") != self.binding:
                raise ReplicationStop("Judge budget/contract drift")
        else:
            if (self.root / "cache").exists() and any((self.root / "cache").iterdir()):
                raise ReplicationStop("Cache exists without its budget ledger")
            self.state = {"format": "rq2-replication-budget", "version": 1,
                          "binding": self.binding, "attempts": [], "provider_model": None}
        self._check_ledger()

    def _check_ledger(self):
        for i, row in enumerate(self.state["attempts"]):
            if row.get("number") != i + 1 or row.get("status") not in {"reserved", "settled", "uncertain"}:
                raise ReplicationStop("Malformed budget ledger")
            if not isinstance(row.get("cost_micro_cny"), int) or row["cost_micro_cny"] < 0:
                raise ReplicationStop("Invalid ledger cost")

    def save(self):
        atomic_json(self.path, self.state)

    def summary(self):
        attempts = self.state["attempts"]
        return {"http_attempts": len(attempts),
                "accounted_cny_at_peak_rates": sum(a["cost_micro_cny"] for a in attempts) / 1_000_000,
                "limit_cny": self.budget["api_limit_cny"],
                "unresolved_attempts": sum(a["status"] != "settled" for a in attempts),
                "known_prompt_tokens": sum(a.get("prompt_tokens", 0) for a in attempts),
                "known_completion_tokens": sum(a.get("completion_tokens", 0) for a in attempts)}

    def _reserve(self, key, body):
        if self.state.get("fatal_error"):
            raise ReplicationStop("Budget/endpoint anomaly remains blocked pending inspection")
        if any(a["status"] != "settled" for a in self.state["attempts"]):
            raise ReplicationStop("Unresolved HTTP attempt: reservation retained, no automatic retry")
        if len(self.state["attempts"]) >= self.budget["http_attempt_limit"]:
            raise ReplicationStop("Global HTTP attempt limit reached")
        if sum(a["key"] == key for a in self.state["attempts"]) >= 3:
            raise ReplicationStop("Unique-input attempt limit reached")
        # A byte-count margin is intentionally much larger than a token estimate.
        # If actual usage exceeds it, settle the actual cost and stop immediately.
        prompt_bound = len(json.dumps(body["messages"], ensure_ascii=False).encode()) + 1024
        reserve = prompt_bound * self.budget["input_cny_per_million"] + 800 * self.budget["output_cny_per_million"]
        used = sum(a["cost_micro_cny"] for a in self.state["attempts"])
        if used + reserve >= round(self.budget["api_limit_cny"] * 1_000_000):
            raise ReplicationStop("Next request would reach the CNY pause threshold; no request sent")
        row = {"number": len(self.state["attempts"]) + 1, "key": key,
               "request_sha256": canonical_sha256(body), "status": "reserved",
               "started_at": utc_now(), "prompt_token_reservation": prompt_bound,
               "reserved_micro_cny": reserve, "cost_micro_cny": reserve}
        self.state["attempts"].append(row)
        self.save()  # Durable before transmitting. A killed process cannot silently resubmit.
        return row

    def _settle(self, row, payload):
        usage = payload.get("usage") if isinstance(payload, dict) else None
        def integer(v):
            return isinstance(v, int) and not isinstance(v, bool) and v >= 0
        if not isinstance(usage, dict) or not all(integer(usage.get(k)) for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            row.update(status="uncertain", error="missing_or_invalid_usage")
            self.save()
            raise ReplicationStop("Provider omitted valid usage; reservation retained")
        p, c = usage["prompt_tokens"], usage["completion_tokens"]
        if usage["total_tokens"] != p + c:
            row.update(status="uncertain", error="usage_total_mismatch")
            self.save()
            raise ReplicationStop("Usage totals inconsistent; reservation retained")
        cost = p * self.budget["input_cny_per_million"] + c * self.budget["output_cny_per_million"]
        row.update(status="settled", finished_at=utc_now(), prompt_tokens=p,
                   completion_tokens=c, cost_micro_cny=cost)
        model = payload.get("model")
        previous = self.state.get("provider_model")
        drift = not isinstance(model, str) or not model or (previous is not None and model != previous)
        if previous is None and isinstance(model, str) and model:
            self.state["provider_model"] = model
        row["provider_model"] = model if isinstance(model, str) else None
        self.save()
        if drift or p > row["prompt_token_reservation"] or c > 800:
            self.state["fatal_error"] = "provider_model_or_reservation_violation"
            self.save()
            raise ReplicationStop("Provider model drift or usage exceeded reservation; actual usage recorded")
        if self.summary()["accounted_cny_at_peak_rates"] >= self.budget["api_limit_cny"]:
            raise ReplicationStop("API budget reached")

    def label(self, response):
        if self.state.get("fatal_error"):
            raise ReplicationStop("Unresolved provider/budget anomaly; no automatic continuation")
        response = validate_response_record(response)
        key = scoring_key(response, self.fingerprint)
        path = self.root / "cache" / (key + ".json")
        if path.exists():
            cache = json.loads(path.read_text())
            if cache.get("key") != key or cache.get("contract_sha256") != self.fingerprint:
                raise ReplicationStop("Scoring cache identity drift")
            if cache["status"] != "ok":
                raise ReplicationStop("Scoring input unresolved; no automatic rejudging")
            receipt = Path(cache["receipt"])
            if (not receipt.resolve().is_relative_to((self.root / "receipts").resolve())
                or not receipt.is_file() or file_sha256(receipt) != cache["receipt_sha256"]):
                raise ReplicationStop("Scoring receipt changed/missing")
            payload = json.loads(receipt.read_text())
            result = StrongRejectScore.model_validate_json(payload["choices"][0]["message"]["content"]).model_dump()
            expected = label_from_judge_result(response["trial_id"], response["response_sha256"], result,
                                               judge_fingerprint=self.fingerprint).to_record()
            label = {**cache["label"], "trial_id": response["trial_id"]}
            if label != expected or not any(a["key"] == key and a["status"] == "settled" for a in self.state["attempts"]):
                raise ReplicationStop("Cache conflicts with its receipt/ledger")
            return checked_label(label, response, self.judge, self.fingerprint)
        # Even if cache writing was interrupted after the HTTP receipt, never pay twice.
        if any(a["key"] == key for a in self.state["attempts"]):
            raise ReplicationStop("Previous scoring attempt lacks cache; inspect preserved receipt")
        if self.real_transport and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
            raise ReplicationStop("DEEPSEEK_API_KEY is not loaded; no request reserved or sent")
        if self.transport is not None and self.judge.get("provider") != "deepseek":
            raise ReplicationStop("Replication supports the frozen DeepSeek-compatible endpoint only")
        if sum(p.is_file() for p in (self.root / "cache").glob("*.json")) >= 2360:
            raise ReplicationStop("Unique scoring input cap reached")
        messages = [{"role": "user", "content": StrongRejectEvaluator.JUDGE_PROMPT.format(
            target_text=response["harmful_text"], response_to_evaluate=response["response"])}]
        corrected = False
        for _ in range(3):
            body = {"model": self.judge["model"], "messages": messages, "temperature": 0,
                    "response_format": {"type": "json_object"}, "max_tokens": 800,
                    "thinking": {"type": "disabled"}}
            row = self._reserve(key, body)
            try:
                payload = self.transport(body)
            except Exception as exc:
                row.update(status="uncertain", error=type(exc).__name__)
                self.save()
                raise ReplicationStop("HTTP outcome uncertain; stopped without automatic retry") from None
            receipt = self.root / "receipts" / f"{row['number']:06d}.json"
            # Private Judge content, never log it. Store before parsing/settling.
            atomic_json(receipt, payload)
            self._settle(row, payload)
            try:
                content = payload["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    continue  # Fully accounted empty result; bounded retry.
                result = StrongRejectScore.model_validate_json(content).model_dump()
                label = label_from_judge_result(response["trial_id"], response["response_sha256"], result,
                    judge_fingerprint=self.fingerprint).to_record()
            except (ValueError, KeyError, IndexError, TypeError):
                if corrected:
                    break
                corrected = True
                # Do not append arbitrary provider output to the prompt or grow it without bound.
                messages = [messages[0], {"role": "user", "content": StrongRejectEvaluator.SCHEMA_RETRY_INSTRUCTION}]
                continue
            atomic_json(path, {"key": key, "contract_sha256": self.fingerprint,
                              "status": "ok", "label": label,
                              "receipt": str(receipt), "receipt_sha256": file_sha256(receipt)})
            return label
        atomic_json(path, {"key": key, "contract_sha256": self.fingerprint, "status": "unresolved"})
        raise ReplicationStop("Judge output unresolved after bounded attempts; no selection-based retries")
