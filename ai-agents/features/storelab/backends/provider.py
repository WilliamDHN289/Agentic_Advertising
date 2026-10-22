"""Provider adapters (SL-09): the interface, the call loop and the governance hooks
shared by ``openai_adapter``, ``google_adapter`` and ``anthropic_adapter``.

An adapter sends one ``LLMRequest`` to the model it names and nothing else: there is
no fallback chain, no alias resolution and no second model. The SDK's own retries
are off (``max_retries=0`` or one attempt), so ``ProviderAdapter.complete`` owns the
retries and its attempt count is exact (tickets.md section 3.5):

| Failure | Code | Retried |
|---|---|---|
| HTTP 429 | ``rate_limited`` | yes, exponential backoff from ``backoff_initial_s``, at least ``Retry-After``, at most ``backoff_max_s`` |
| HTTP 408, 409, 5xx, timeout, connection error | ``provider_unavailable`` | yes, same backoff |
| HTTP 400, 422 and other 4xx | ``bad_request`` | no |
| HTTP 401, 403 | ``auth`` | no |
| HTTP 404 | ``model_unavailable`` | no |

After ``max_attempts`` the call ends with its last code; any other exception is a
defect and propagates (the runner fails the run). A safety refusal is a response
(``refusal``), not an error.

Every attempt passes the ``CallGuard`` first: the kill switch (no attempt is made once
it is engaged; ``KillSwitchEngaged`` if none was made), and on the platform the
circuit breaker, provider slots and rate limiter (``governance.PlatformGovernance``).
``LLMBackend`` writes one ``log_call`` per call that reached the provider.

Each response records the returned model id, the provider request id, the OpenAI
``system_fingerprint`` when present, the parameters actually sent (a null temperature
is omitted and recorded as ``provider_default``) and whether a seed was sent.
"""
from __future__ import annotations

import email.utils
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, NamedTuple, Protocol

from ..core.cache import LLMRequest, ProviderResponse
from ..core.config import FailurePolicyConfig
from ..core.protocol import ErrorCode, Usage

RETRYABLE = ("rate_limited", "provider_unavailable")
# Exceptions the SDKs raise for a timeout or a dropped connection, by class name, so
# that classification does not depend on which HTTP library an SDK release uses.
TRANSPORT_ERRORS = ("APITimeoutError", "APIConnectionError", "TimeoutException", "ConnectTimeout", "ReadTimeout",
                    "WriteTimeout", "PoolTimeout", "ConnectError", "ReadError", "WriteError", "RemoteProtocolError",
                    "NetworkError", "TransportError")


class KillSwitchEngaged(RuntimeError):
    """The kill switch was engaged before the first attempt: no call was made."""


class CallGuard(Protocol):
    def before_attempt(self, provider: str, model: str) -> str | None:
        """None to make the attempt; "kill_switch" to stop before it; an ErrorCode to end
        the call with that code without calling (for example an open circuit breaker)."""

    def after_attempt(self, provider: str, model: str, error: BaseException | None) -> None: ...

    def log_call(self, call: Mapping[str, Any]) -> None:
        """One durable line per call that reached the provider (``LLMBackend``)."""


class NoGuard:
    def before_attempt(self, provider: str, model: str) -> str | None:
        return None

    def after_attempt(self, provider: str, model: str, error: BaseException | None) -> None:
        return None

    def log_call(self, call: Mapping[str, Any]) -> None:
        return None


class Classified(NamedTuple):
    code: ErrorCode
    retry_after_s: float | None = None


@dataclass
class Sent:
    """What one successful SDK call returned, in provider-neutral form."""

    text: str | None
    returned_model: str | None
    usage: Usage
    raw: Any
    refusal: bool = False
    request_id: str | None = None
    system_fingerprint: str | None = None
    extra: dict = field(default_factory=dict)


def code_for_status(status: int) -> ErrorCode:
    if status == 429:
        return "rate_limited"
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "model_unavailable"
    if status in (408, 409) or status >= 500:
        return "provider_unavailable"
    return "bad_request"


def retry_after_s(headers: Mapping[str, str] | None, now: Callable[[], float] = time.time) -> float | None:
    """``retry-after-ms`` or ``retry-after`` (seconds or an HTTP date), if present."""
    if not headers:
        return None
    try:
        if headers.get("retry-after-ms") is not None:
            return max(0.0, float(headers["retry-after-ms"]) / 1000.0)
        value = headers.get("retry-after")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            parsed = email.utils.parsedate_to_datetime(value)
            return max(0.0, parsed.timestamp() - now())
    except (TypeError, ValueError, OverflowError):
        return None


def classify_http(error: BaseException, status: int | None, headers: Mapping[str, str] | None) -> Classified | None:
    """An HTTP status or transport failure as an error code; None if neither applies."""
    if status is not None:
        return Classified(code_for_status(status), retry_after_s(headers))
    if isinstance(error, (TimeoutError, ConnectionError)) or any(
            cls.__name__ in TRANSPORT_ERRORS for cls in type(error).__mro__):
        return Classified("provider_unavailable")
    return None


def sent_params(request: LLMRequest) -> dict:
    """The sampling parameters an adapter sends, in neutral names."""
    params = request.params
    sent = {"temperature": "provider_default" if params.get("temperature") is None else params["temperature"],
            "max_output_tokens": params["max_output_tokens"]}
    if params.get("top_p") is not None:
        sent["top_p"] = params["top_p"]
    if params.get("seed") is not None:
        sent["seed"] = params["seed"]
    return sent


class ProviderAdapter:
    """Base adapter: subclasses implement ``send`` (one SDK call) and ``classify``."""

    name: str
    adapter_version: str
    api_version: str | None = None

    def __init__(self, policy: FailurePolicyConfig, guard: CallGuard | None = None, *,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.policy, self.guard = policy, guard or NoGuard()
        self.sleep, self.clock = sleep, clock

    def send(self, request: LLMRequest) -> Sent:
        raise NotImplementedError

    def classify(self, error: BaseException) -> Classified | None:
        raise NotImplementedError

    def backoff_s(self, attempt: int, retry_after: float | None) -> float:
        """Delay after failed attempt ``attempt`` (1-based)."""
        delay = self.policy.backoff_initial_s * 2 ** (attempt - 1)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return min(delay, self.policy.backoff_max_s)

    def complete(self, request: LLMRequest) -> ProviderResponse:
        if request.provider != self.name:
            raise ValueError("adapter %s got a request for provider %s" % (self.name, request.provider))
        model, started = request.requested_model, self.clock()
        attempts, last = 0, None
        while attempts < self.policy.max_attempts:
            verdict = self.guard.before_attempt(self.name, model)
            if verdict == "kill_switch":
                if attempts == 0:
                    raise KillSwitchEngaged("kill switch engaged before calling %s" % model)
                break  # the call ends with the error of its last attempt
            if verdict is not None:
                last = Classified(verdict)
                break
            attempts += 1
            try:
                sent = self.send(request)
            except Exception as error:
                classified = self.classify(error)
                self.guard.after_attempt(self.name, model, error)
                if classified is None:
                    raise
                last = classified
                if classified.code not in RETRYABLE or attempts >= self.policy.max_attempts:
                    break
                self.sleep(self.backoff_s(attempts, classified.retry_after_s))
                continue
            self.guard.after_attempt(self.name, model, None)
            return self.response(request, sent, attempts, self.clock() - started)
        return ProviderResponse(error_code=last.code if last else "provider_unavailable", attempts=attempts,
                                latency_s=self.clock() - started, sent_params=sent_params(request),
                                seed_supported="seed" in request.params, api_version=self.api_version)

    def response(self, request: LLMRequest, sent: Sent, attempts: int, latency_s: float) -> ProviderResponse:
        return ProviderResponse(text=sent.text, refusal=sent.refusal, attempts=attempts, latency_s=latency_s,
                                returned_model=sent.returned_model, usage=sent.usage, raw=sent.raw,
                                request_id=sent.request_id, system_fingerprint=sent.system_fingerprint,
                                sent_params=sent_params(request), seed_supported="seed" in request.params,
                                api_version=self.api_version)


def adapters_for(config, settings, guard: CallGuard | None = None,
                 http_clients: Mapping[str, Any] | None = None) -> dict[str, ProviderAdapter]:
    """One adapter per configured API provider, with the settings' keys; ``http_clients``
    optionally gives an SDK HTTP client per provider (recording, tests)."""
    from .anthropic_adapter import AnthropicAdapter
    from .google_adapter import GoogleAdapter
    from .openai_adapter import OpenAIAdapter

    classes = {"openai": OpenAIAdapter, "anthropic": AnthropicAdapter, "google": GoogleAdapter}
    adapters = {}
    for provider in sorted({m.provider for m in config.agents.models} & set(classes)):
        key = settings.api_key(provider)
        if key is None:
            raise ValueError("no API key for %s" % provider)
        adapters[provider] = classes[provider].from_key(key, config.failure_policy, guard,
                                                        http_client=(http_clients or {}).get(provider))
    return adapters
