"""Fail-closed stdlib generation adapters. No networking occurs at import time.

The injected transport has signature (host, path, headers, payload, timeout) ->
(status, response_headers, bytes). It must not follow redirects. Production uses
HTTPSConnection (no proxies, redirect handling, or arbitrary endpoint overrides).
"""
from __future__ import annotations

import http.client
import json
import math
import os
import re
import time
from pathlib import Path


class SafetyError(RuntimeError):
    pass


KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}
HOSTS = {"openai": "api.openai.com", "anthropic": "api.anthropic.com", "google": "generativelanguage.googleapis.com"}


def runtime_keys():
    return [os.environ[k] for name in KEYS.values() for k in (name, "SHARED_" + name) if os.environ.get(k)]


def sanitize(value):
    # Sanitize parsed strings too: JSON escaping must not defeat redaction.
    if isinstance(value, dict):
        return {sanitize(str(k)): sanitize(v) for k, v in value.items()
                if str(k).lower() not in {"authorization", "x-api-key", "x-goog-api-key"}}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        for secret in sorted(runtime_keys(), key=len, reverse=True):
            for spelling in (secret, json.dumps(secret, ensure_ascii=True)[1:-1],
                             "".join("\\u%04x" % ord(ch) for ch in secret)):
                value = value.replace(spelling, "[REDACTED]")
    return value


def dumps(value):
    return json.dumps(sanitize(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def https_transport(host, path, headers, payload, timeout):
    allowed = ((host == HOSTS["openai"] and path == "/v1/chat/completions") or
               (host == HOSTS["anthropic"] and path == "/v1/messages") or
               (host == HOSTS["google"] and re.fullmatch(r"/v1beta/models/[A-Za-z0-9][A-Za-z0-9_.-]{0,150}:generateContent", path)))
    if not allowed:
        raise SafetyError("Endpoint not allowed")
    conn = http.client.HTTPSConnection(host, timeout=timeout)
    try:
        conn.request("POST", path, body=json.dumps(payload).encode(), headers=headers)
        response = conn.getresponse()
        body = response.read(4 * 1024 * 1024 + 1)
        if len(body) > 4 * 1024 * 1024:
            raise SafetyError("Response size limit")
        return response.status, dict(response.getheaders()), body
    finally:
        conn.close()


def positive(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


class Ledger:
    """Append+fsync pending reservation BEFORE dispatch; unknown spend stays reserved."""
    def __init__(self, path, max_usd, max_attempts):
        if not positive(max_usd) or max_usd > 50 or type(max_attempts) is not int or max_attempts < 1:
            raise SafetyError("Invalid total caps")
        self.path = Path(path)
        self.max_usd, self.max_attempts = max_usd, max_attempts
        self.events = []
        if self.path.exists():
            with self.path.open(encoding="utf-8", newline="") as handle:
                self.events = [json.loads(line) for line in handle if line.strip()]
        pending = {e["attempt"] for e in self.events if e["state"] == "pending"}
        settled = {e["attempt"] for e in self.events if e["state"] in ("complete", "http_error")}
        if pending - settled or any(e["state"] == "unknown" for e in self.events):
            raise SafetyError("Unknown billed outcome: manual reconciliation required; no automatic resume")

    def append(self, event):
        with self.path.open("a") as handle:
            handle.write(dumps(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(event)

    def reserve(self, usd, timeout):
        starts = [e for e in self.events if e["state"] == "pending"]
        # Never release reservations, even after explicit usage: deliberately conservative.
        if len(starts) >= self.max_attempts or sum(e["reserved_usd"] for e in starts) + usd > self.max_usd:
            raise SafetyError("Total request or USD reserve cap reached before dispatch")
        attempt = len(starts) + 1
        self.append({"attempt": attempt, "state": "pending", "reserved_usd": usd, "timeout_seconds": timeout})
        return attempt


class CampaignLedger:
    """Both ledgers are locked by the executor; reserve aggregate BEFORE dispatch.

    Crashes between either write fail closed as unknown pending reservations.
    No completed/failed attempt ever releases aggregate budget.
    """
    def __init__(self, run, campaign, provenance_id):
        self.run, self.campaign, self.provenance_id = run, campaign, provenance_id
        self.attempt_map = {}

    @property
    def events(self):
        return self.run.events

    def reserve(self, usd, timeout):
        aggregate_attempt = self.campaign.reserve(usd, timeout)
        attempt = self.run.reserve(usd, timeout)
        self.attempt_map[attempt] = aggregate_attempt
        self.campaign.append({"attempt": aggregate_attempt, "state": "binding", "run_attempt": attempt,
                              "run_provenance_id": self.provenance_id})
        return attempt

    def append(self, event):
        self.run.append(event)
        self.campaign.append(dict(event, attempt=self.attempt_map[event["attempt"]],
                                  run_attempt=event["attempt"], run_provenance_id=self.provenance_id))


class NativeClient:
    def __init__(self, config, ledger, *, transport=https_transport, sleep=time.sleep):
        self.config, self.ledger, self.transport, self.sleep = config, ledger, transport, sleep
        provider = config.get("provider")
        if provider not in KEYS or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,150}", config.get("model", "")):
            raise SafetyError("Explicit supported provider and model ID required")
        for field in ("input_usd_per_million", "output_usd_per_million", "timeout_seconds"):
            if not positive(config.get(field)):
                raise SafetyError("Positive prices and timeout required")
        for field in ("input_token_reserve", "output_token_reserve"):
            if type(config.get(field)) is not int or config[field] < 1:
                raise SafetyError("Positive integer token reserves required")
        if type(config.get("max_retries", 0)) is not int or config.get("max_retries", 0) not in (0, 1, 2):
            raise SafetyError("Retries must be bounded to zero through two")
        if config.get("approved") is not True or not all(config.get(k) for k in ("model_evidence", "pricing_evidence", "token_reserve_evidence")):
            raise SafetyError("Unverified model/pricing/token reserve configuration")
        name = KEYS[provider]
        self.key = os.environ.get(name) or os.environ.get("SHARED_" + name)
        if not self.key:
            raise SafetyError("Runtime provider credential unavailable")

    def __call__(self, request):
        c = self.config
        if (request.get("model"), request.get("provider")) != (c["model"], c["provider"]):
            raise SafetyError("Request configuration mismatch")
        # Byte-bound with substantial protocol overhead, plus approved context bound.
        if len(json.dumps(request, ensure_ascii=False).encode()) + 4096 > c["input_token_reserve"]:
            raise SafetyError("Input exceeds conservative token reserve")
        if request.get("max_tokens") != 400 or c["output_token_reserve"] < 400:
            raise SafetyError("Output reserve does not cover protocol")
        provider, model = c["provider"], c["model"]
        messages = request["messages"]
        headers = {"Content-Type": "application/json"}
        if provider == "openai":
            path = "/v1/chat/completions"
            headers["Authorization"] = "Bearer " + self.key
            payload = {k: v for k, v in request.items() if k != "provider"}
        elif provider == "anthropic":
            path = "/v1/messages"
            headers.update({"x-api-key": self.key, "anthropic-version": "2023-06-01"})
            payload = {"model": model, "system": messages[0]["content"], "messages": messages[1:], "temperature": 0, "max_tokens": 400}
        else:
            path = "/v1beta/models/" + model + ":generateContent"
            headers["x-goog-api-key"] = self.key
            payload = {"systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
                       "contents": [{"role": "user", "parts": [{"text": messages[1]["content"]}]}],
                       "generationConfig": {"temperature": 0, "maxOutputTokens": 400, "responseMimeType": "application/json"}}
        usd = (c["input_token_reserve"] * c["input_usd_per_million"] + c["output_token_reserve"] * c["output_usd_per_million"]) / 1e6
        for retry in range(c.get("max_retries", 0) + 1):
            attempt = self.ledger.reserve(usd, c["timeout_seconds"])
            start = time.monotonic()
            try:
                status, response_headers, body = self.transport(HOSTS[provider], path, headers, payload, c["timeout_seconds"])
            except Exception:
                self.ledger.append({"attempt": attempt, "state": "unknown", "error": "transport_failure_or_timeout", "elapsed_seconds": time.monotonic() - start, "actual_usd": None})
                raise SafetyError("Transport failed; unknown billed outcome; reservation retained") from None
            event = {"attempt": attempt, "state": "http_error", "http_status": status, "elapsed_seconds": time.monotonic() - start, "actual_usd": None}
            if status != 200:
                # Never decode/log error bodies or redirect Locations.
                self.ledger.append(event)
                if status in (429, 500, 502, 503, 504) and retry < c.get("max_retries", 0):
                    self.sleep(2 ** retry)
                    continue
                raise SafetyError("Provider HTTP failure; body suppressed; reservation retained")
            try:
                raw = json.loads(body)
                if provider == "openai":
                    text = raw["choices"][0]["message"]["content"]
                    finish = raw["choices"][0]["finish_reason"]
                    usage = raw.get("usage", {})
                    inp, out = usage.get("prompt_tokens"), usage.get("completion_tokens")
                elif provider == "anthropic":
                    text = "".join(b["text"] for b in raw["content"] if b.get("type") == "text")
                    finish = raw["stop_reason"]
                    usage = raw.get("usage", {})
                    inp, out = usage.get("input_tokens"), usage.get("output_tokens")
                else:
                    candidate = raw["candidates"][0]
                    text = "".join(b.get("text", "") for b in candidate["content"]["parts"])
                    finish = candidate.get("finishReason")
                    usage = raw.get("usageMetadata", {})
                    inp = usage.get("promptTokenCount")
                    out = usage.get("candidatesTokenCount")
                    if type(out) is int:
                        out += usage.get("thoughtsTokenCount", 0)
                valid = all(type(v) is int and v >= 0 for v in (inp, out))
                safe_headers = {k.lower(): v for k, v in response_headers.items()}
                envelope = sanitize({"raw_content": text, "provider_usage": usage, "request_id": raw.get("id", raw.get("responseId")),
                                     "http_request_id": safe_headers.get("x-request-id", safe_headers.get("request-id")),
                                     "usage": {"prompt_tokens": inp, "completion_tokens": out} if valid else None,
                                     "finish_reason": finish})
                within_reserve = valid and inp <= c["input_token_reserve"] and out <= c["output_token_reserve"]
                event.update(state="complete" if within_reserve else "unknown", response=envelope,
                             actual_usd=(inp*c["input_usd_per_million"] + out*c["output_usd_per_million"])/1e6 if valid else None)
                self.ledger.append(event)
                if not within_reserve:
                    raise SafetyError("Usage absent or reserve exceeded; halt and reconcile")
                if finish not in ("stop", "end_turn", "STOP"):
                    raise SafetyError("Nonterminal or filtered generation")
                envelope["parsed"] = sanitize(json.loads(text))
                return envelope
            except SafetyError:
                raise
            except Exception:
                if not any(e["attempt"] == attempt and e["state"] != "pending" for e in self.ledger.events):
                    self.ledger.append(dict(event, state="unknown", error="invalid_response"))
                raise SafetyError("Invalid provider response; body suppressed; reservation retained") from None
