"""``PlatformGovernance`` (SL-09): the runner's governance and the adapters' call guard
on the ai-agents platform, built from the gateway's sub-components rather than the
``AdgraphClient`` facade (whose failover chain switches models):

| Hook | Platform function | Effect here |
|---|---|---|
| before each trial and each attempt | ``enforce_llm_kill_switch`` (``features/adgraph/utils/llm_guard.py:183``) | ``LoadTestAborted`` → ``kill_switch`` |
| before each attempt | ``CircuitBreakerRegistry.allow`` (``utils/llm_circuit_breaker.py:141``) | open breaker → ``provider_unavailable``, no call |
| before each attempt | ``acquire_provider_slot`` / ``acquire_model_capacity`` (``features/adgraph/utils/rate_limiter.py:597, 765``) | no slot or capacity in time → ``rate_limited``, no call |
| after each attempt | ``release_provider_slot`` (``rate_limiter.py:648``), ``record_success`` / ``record_failure`` | |
| once per provider call | ``create_llm_invocation_log_or_raise`` (``utils/llm_invocation_log_client.py:553``) | ``invocation_kind="storelab"``, ``job_id`` = run id, ``task_ref`` = trial key; a failed write fails the run |

The platform modules are loaded by ``Platform.load`` only on the platform; tests and
other callers pass their own ``Platform``. Budget and cancel work as in
``LocalGovernance``: a ``core.cost.Meter`` reserves each trial's maximum cost and
charges each provider call (SL-11). Each call's line, with its USD, is also appended
to the run's local ``invocations.jsonl`` and each cache hit to ``cache_hits.jsonl``,
so the spend survives a resume and the cost report can be built (SL-12 adds the run API).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .core.cost import Meter, call_usd, logged_spend
from .core.runner import Trial, append_jsonl, hold_until
from .core.protocol import TrialOutcome
from .settings import StorelabSettings

CALL_SITE = "storelab.trial"


@dataclass
class Platform:
    enforce_kill_switch: Callable[..., None]
    kill_switch_error: type[BaseException]
    breaker: Any  # allow(provider, model), record_success(provider, model), record_failure(provider, model, exc)
    breaker_open_error: type[BaseException]
    acquire_provider_slot: Callable[[str], None]
    release_provider_slot: Callable[[str], None]
    slot_error: type[BaseException]
    acquire_model_capacity: Callable[..., Any]
    write_invocation_log: Callable[..., None]

    @classmethod
    def load(cls, settings: StorelabSettings) -> Platform:
        """The ai-agents implementations (importable only inside that repository)."""
        from features.adgraph.utils.llm_guard import LoadTestAborted, enforce_llm_kill_switch
        from features.adgraph.utils.rate_limiter import (ConcurrencyLimitExceeded, acquire_model_capacity,
                                                         acquire_provider_slot, release_provider_slot)
        from utils.llm_circuit_breaker import CircuitBreakerRegistry, LLMCircuitOpenError
        from utils.llm_invocation_log_client import create_llm_invocation_log_or_raise

        breaker = CircuitBreakerRegistry(failure_threshold=settings.breaker_failure_threshold,
                                         window_seconds=settings.breaker_window_s,
                                         cooldown_seconds=settings.breaker_cooldown_s)
        return cls(enforce_llm_kill_switch, LoadTestAborted, breaker, LLMCircuitOpenError, acquire_provider_slot,
                   release_provider_slot, ConcurrencyLimitExceeded, acquire_model_capacity,
                   create_llm_invocation_log_or_raise)


class PlatformGovernance:
    kind = "platform"

    def __init__(self, run_id: str, max_usd: float, platform: Platform, *, cancel_flag: Path | None = None,
                 invocation_log: Path | None = None, cache_log: Path | None = None):
        self.run_id, self.max_usd, self.platform = run_id, max_usd, platform
        self.cancel_flag, self.invocation_log, self.cache_log = cancel_flag, invocation_log, cache_log
        self.meter = Meter(max_usd, logged_spend(invocation_log))
        self.lock = threading.Lock()
        self.held = threading.local()  # the provider slot this thread holds during an attempt

    @property
    def spent_usd(self) -> float:
        return self.meter.spent_usd

    @property
    def reserved_usd(self) -> float:
        return self.meter.reserved_usd

    def _killed(self) -> bool:
        try:
            self.platform.enforce_kill_switch(call_site=CALL_SITE)
        except self.platform.kill_switch_error:
            return True
        return False

    # Governance (core/runner.py)
    def authorize(self, trial: Trial, estimated_usd: float) -> str | None:
        if self._killed():
            return "kill_switch"
        if not self.meter.reserve(trial.trial_key, estimated_usd):
            return "budget"
        return None

    def record(self, trial: Trial, outcome: TrialOutcome, cost_usd: float) -> None:
        self.meter.settle(trial.trial_key, cost_usd)

    def hold(self, trial: Trial, until) -> None:
        hold_until(self.meter, trial.trial_key, until)

    def cancel_requested(self) -> bool:
        return self.cancel_flag is not None and Path(self.cancel_flag).exists()

    # CallGuard (backends/provider.py)
    def before_attempt(self, provider: str, model: str) -> str | None:
        p = self.platform
        if self._killed():
            return "kill_switch"
        try:
            p.breaker.allow(provider, model)
        except p.breaker_open_error:
            return "provider_unavailable"
        try:
            p.acquire_provider_slot(provider)
        except p.slot_error:
            return "rate_limited"
        try:
            p.acquire_model_capacity(model=model, provider=provider)
        except TimeoutError:
            p.release_provider_slot(provider)
            return "rate_limited"
        except BaseException:  # never leak the slot, whatever the limiter raises
            p.release_provider_slot(provider)
            raise
        self.held.provider = provider
        return None

    def after_attempt(self, provider: str, model: str, error: BaseException | None) -> None:
        if getattr(self.held, "provider", None) is not None:
            self.platform.release_provider_slot(self.held.provider)
            self.held.provider = None
        if error is None:
            self.platform.breaker.record_success(provider, model)
        else:
            self.platform.breaker.record_failure(provider, model, error)

    def log_call(self, call: Mapping[str, Any]) -> None:
        self.meter.charge(call.get("trial_key"), call_usd(call))
        append_jsonl(self.invocation_log, call, self.lock)
        latency = call.get("latency_s")
        self.platform.write_invocation_log(
            model_provider=call["provider"], model_name=call["requested_model"], invocation_kind="storelab",
            status="FAILED" if call["error_code"] else "SUCCEEDED",
            latency_ms=None if latency is None else int(round(latency * 1000)),
            prompt_tokens=call.get("input_tokens"), completion_tokens=call.get("output_tokens"),
            error_message=call["error_code"], job_id=call["run_id"], task_type="storelab_trial",
            task_ref=call["trial_key"], attempt=max(1, call["attempts"]),
            details={"call_site": CALL_SITE, "provider": call["provider"], "request_id": call.get("request_id")})

    def log_cache_hit(self, call: Mapping[str, Any]) -> None:
        append_jsonl(self.cache_log, call, self.lock)
