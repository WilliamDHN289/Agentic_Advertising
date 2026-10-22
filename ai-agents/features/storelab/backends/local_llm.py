"""Local LLM backend: a model served by ollama, the original StoreLab path (SL-04).

Ported from agents.py l.41-94 with the failure handling fixed (SL-00 D-02): the
original swallowed every exception and returned "" after three attempts, which the
regime runners then turned into a substituted first item. Here a call that never
succeeds returns a ``Completion`` with an error code (tickets.md section 3.5):

| Condition | error_code | Retried |
|---|---|---|
| connection refused, timeout, 5xx, malformed response | provider_unavailable | yes |
| HTTP 429 | rate_limited | yes |
| HTTP 404 (model not pulled) | model_unavailable | no |
| other 4xx | bad_request (401/403: auth) | no |

``decide`` parses the reply with the SL-03 parser, so a failure is a ``TrialOutcome``
with status ``error`` and never a choice. ``model_digest`` gives the weights' digest
for the manifest (``record_model_digest``).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Collection

from ..core.protocol import Completion, Invalid, Leave, Select, TrialOutcome, parse_reply

DEFAULT_URL = "http://127.0.0.1:11434"


def _classify(status: int) -> tuple[str, bool]:
    """HTTP status -> (error_code, retry)."""
    if status == 429:
        return "rate_limited", True
    if status >= 500:
        return "provider_unavailable", True
    if status == 404:
        return "model_unavailable", False
    if status in (401, 403):
        return "auth", False
    return "bad_request", False


class LocalLLM:
    def __init__(self, model: str, *, temperature: float | None = None, top_p: float | None = None,
                 num_predict: int = 256, timeout_s: float = 180.0, max_attempts: int = 3,
                 backoff_s: float = 2.0, base_url: str = DEFAULT_URL,
                 sleep: Callable[[float], None] = time.sleep):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.model, self.temperature, self.top_p = model, temperature, top_p
        self.num_predict, self.timeout_s = num_predict, timeout_s
        self.max_attempts, self.backoff_s = max_attempts, backoff_s
        self.base_url, self.sleep = base_url.rstrip("/"), sleep

    def _post(self, path: str, payload: dict) -> Any:
        request = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode())

    def complete(self, prompt: str, seed: int | None = None) -> Completion:
        options: dict[str, Any] = {"num_predict": self.num_predict}
        if self.temperature is not None:  # null: ollama's default
            options["temperature"] = self.temperature
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if seed is not None:
            options["seed"] = seed
        payload = {"model": self.model, "prompt": prompt, "stream": False, "options": options}
        code = "provider_unavailable"
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            try:
                out = self._post("/api/generate", payload)
                if not isinstance(out, dict) or not isinstance(out.get("response"), str):
                    raise ValueError("no 'response' text in the reply")
                return Completion(text=out["response"], attempts=attempt, latency_s=time.monotonic() - started,
                                  returned_model=out.get("model") if isinstance(out.get("model"), str) else None)
            except urllib.error.HTTPError as error:
                code, retry = _classify(error.code)
                if not retry:
                    return Completion(error_code=code, attempts=attempt)
            except (OSError, ValueError):  # URLError, refused, timeout; malformed JSON or body
                code = "provider_unavailable"
            if attempt < self.max_attempts:
                self.sleep(self.backoff_s * attempt)
        return Completion(error_code=code, attempts=self.max_attempts)

    def decide(self, prompt: str, allowed_ids: Collection[str], seed: int | None = None) -> TrialOutcome:
        """One trial for the text and structured interfaces: call, then parse strictly."""
        reply = self.complete(prompt, seed)
        common = {"backend": "local_llm", "requested_model": self.model, "attempts": reply.attempts,
                  "latency_s": reply.latency_s, "returned_model": reply.returned_model}
        if reply.error_code is not None:
            return TrialOutcome(status="error", error_code=reply.error_code, **common)
        action = parse_reply(reply.text, allowed_ids)
        if isinstance(action, Select):
            return TrialOutcome(status="choice", chosen_id=action.product_id, **common)
        if isinstance(action, Leave):
            return TrialOutcome(status="outside", **common)
        assert isinstance(action, Invalid)
        return TrialOutcome(status="invalid", invalid_reason=action.reason, **common)

    def model_digest(self) -> str | None:
        """The served model's digest, from /api/show or, if it does not report one, from
        the /api/tags listing; None if the server is down or the model is absent."""
        try:
            shown = self._post("/api/show", {"model": self.model})
            if isinstance(shown, dict) and isinstance(shown.get("digest"), str):
                return shown["digest"]
            request = urllib.request.Request(self.base_url + "/api/tags")
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                listing = json.loads(response.read().decode())
        except (OSError, ValueError):
            return None
        for entry in listing.get("models", []) if isinstance(listing, dict) else []:
            if self.model in (entry.get("name"), entry.get("model")) and isinstance(entry.get("digest"), str):
                return entry["digest"]
        return None
