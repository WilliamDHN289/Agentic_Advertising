"""LLM backends behind the response cache (SL-08).

``LLMBackend`` turns a trial into an ``LLMRequest`` (the rendered view as one user
message, the model's parameters, the template digest, the repetition and the cache
namespace), consults the ``ResponseCache`` under the run's policy, calls its
``Provider`` when the policy allows, and maps the response to a ``TrialOutcome``
with the SL-03 parser. ``raw_ref`` is ``cache:<call_key>`` whenever the response is
in the cache. A failed call is returned as its error and not cached; under
``read_only`` a miss is ``error/cache_miss`` and no provider is called.

Providers implement ``complete(request) -> ProviderResponse``: ``OllamaProvider``
here (the local model, SL-04 client), the API adapters in SL-09 (``provider.py``). The
call seed is derived from the master seed and the trial's coordinates, not its key, so
a replay (whose config_hash differs) asks for the same call.

Every call that reaches the provider (not a cache hit, not a call the kill switch
stopped before its first attempt) is written once through the run's ``CallGuard``
``log_call`` with ``run_id`` and ``trial_key`` (SL-09): an invocation-log write under
``PlatformGovernance``, a line in ``invocations.jsonl`` under ``LocalGovernance``. A
kill switch engaged before the first attempt gives ``not_run``/``kill_switch``.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Protocol

from ..core.cache import (READS, WRITES, LLMRequest, Message, Policy, ProviderResponse, ResponseCache, call_key,
                          new_entry)
from ..core.config import RunConfig, canonical_json
from ..core.model_registry import rule_for
from ..core.protocol import Invalid, Leave, Observation, Select, TrialOutcome, parse_reply
from ..core.render import template_sha256
from ..core.runner import Trial
from ..pricing import UnknownModelError, call_cost_usd
from .local_llm import DEFAULT_URL, LocalLLM
from .provider import CallGuard, KillSwitchEngaged, NoGuard

# provider -> TrialOutcome.backend
BACKEND_NAMES = {"ollama": "local_llm", "openai": "openai", "anthropic": "anthropic", "google": "google"}


class Provider(Protocol):
    name: str  # openai, anthropic, google or ollama
    adapter_version: str

    def complete(self, request: LLMRequest) -> ProviderResponse: ...


def call_seed(config: RunConfig, trial: Trial) -> int:
    coordinates = [config.seed, trial.task_id, trial.set_id, trial.presentation_id, trial.interface, trial.model_key,
                   trial.temperature, trial.rep]
    return int(hashlib.sha256(canonical_json(coordinates).encode("utf-8")).hexdigest()[:8], 16)


def build_request(config: RunConfig, model_key: str, trial: Trial, observation: Observation) -> LLMRequest:
    if trial.interface == "browser_tool":
        raise NotImplementedError("browser_tool calls need the tool loop (SL-10)")
    spec = next(m for m in config.agents.models if m.key == model_key)
    params = {"temperature": trial.temperature, "top_p": spec.top_p, "max_output_tokens": spec.max_output_tokens}
    if rule_for(spec.provider, spec.model).seed_supported:
        params["seed"] = call_seed(config, trial)
    return LLMRequest(provider=spec.provider, requested_model=spec.model, params=params,
                      prompt_template_sha=template_sha256(config.protocol.prompt_templates[trial.interface]),
                      messages=[Message(role="user", content=observation.views[trial.interface])], rep=trial.rep,
                      sample_namespace=config.cache.namespace)


class LLMBackend:
    def __init__(self, config: RunConfig, model_key: str, *, provider: Provider | None, cache: ResponseCache | None,
                 policy: Policy, run_id: str | None = None, guard: CallGuard | None = None):
        if policy != "off" and cache is None:
            raise ValueError("cache policy %s needs a cache" % policy)
        if policy != "read_only" and provider is None:
            raise ValueError("cache policy %s needs a provider for model %s" % (policy, model_key))
        self.config, self.model_key, self.provider, self.cache, self.policy = config, model_key, provider, cache, policy
        self.run_id, self.guard = run_id, guard or NoGuard()
        spec = next(m for m in config.agents.models if m.key == model_key)
        self.backend, self.model = BACKEND_NAMES[spec.provider], spec.model

    def decide(self, trial: Trial, observation: Observation) -> TrialOutcome:
        request = build_request(self.config, self.model_key, trial, observation)
        key = call_key(request)
        common = {"backend": self.backend, "requested_model": self.model}
        entry = self.cache.get(request.provider, request.requested_model, key) if self.policy in READS else None
        if entry is not None:
            response, action, raw_ref = entry.response, entry.parsed_action, "cache:" + key
        elif self.policy == "read_only":
            return TrialOutcome(status="error", error_code="cache_miss", attempts=0, raw_ref="cache:" + key, **common)
        else:
            try:
                response = self.provider.complete(request)
            except KillSwitchEngaged:
                return TrialOutcome(status="not_run", error_code="kill_switch", **common)
            if response.attempts:
                self.log_call(trial, request, key, response)
            if response.error_code is not None:  # failed after the adapter's retries: not cached
                return TrialOutcome(status="error", error_code=response.error_code, attempts=response.attempts,
                                    latency_s=response.latency_s, returned_model=response.returned_model,
                                    usage=response.usage, **common)
            action = parse_reply(response.text, observation.candidate_ids)
            raw_ref = None
            if self.policy in WRITES:
                self.cache.put(new_entry(request, response, action, self.provider.adapter_version,
                                         run_id=self.run_id, trial_key=trial.trial_key))
                raw_ref = "cache:" + key
        return outcome_from(response, action, raw_ref, **common)

    def log_call(self, trial: Trial, request: LLMRequest, key: str, response: ProviderResponse) -> None:
        try:
            cost = 0.0 if request.provider == "ollama" else call_cost_usd(request.requested_model, response.usage)
        except UnknownModelError:
            cost = None
        self.guard.log_call({
            "run_id": self.run_id, "trial_key": trial.trial_key, "call_key": key, "provider": request.provider,
            "requested_model": request.requested_model, "returned_model": response.returned_model,
            "status": "failed" if response.error_code else "succeeded", "error_code": response.error_code,
            "refusal": response.refusal, "attempts": response.attempts, "latency_s": response.latency_s,
            "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
            "request_id": response.request_id, "cost_usd": cost})


def outcome_from(response: ProviderResponse, action, raw_ref: str | None, **common) -> TrialOutcome:
    fields = dict(attempts=response.attempts, latency_s=response.latency_s, returned_model=response.returned_model,
                  usage=response.usage, raw_ref=raw_ref, turns=1, **common)
    if response.refusal:
        return TrialOutcome(status="refusal", **fields)
    if isinstance(action, Select):
        return TrialOutcome(status="choice", chosen_id=action.product_id, **fields)
    if isinstance(action, Leave):
        return TrialOutcome(status="outside", **fields)
    assert isinstance(action, Invalid)
    return TrialOutcome(status="invalid", invalid_reason=action.reason, **fields)


class OllamaProvider:
    """The local model through the SL-04 client; its retries are the attempts."""

    name = "ollama"
    adapter_version = "ollama-generate@1"

    def __init__(self, config: RunConfig, base_url: str = DEFAULT_URL,
                 client_factory: Callable[..., LocalLLM] = LocalLLM):
        self.policy, self.base_url, self.client_factory = config.failure_policy, base_url, client_factory

    def complete(self, request: LLMRequest) -> ProviderResponse:
        params = request.params
        client = self.client_factory(request.requested_model, temperature=params.get("temperature"),
                                     top_p=params.get("top_p"), num_predict=params.get("max_output_tokens", 256),
                                     timeout_s=self.policy.trial_timeout_s, max_attempts=self.policy.max_attempts,
                                     backoff_s=self.policy.backoff_initial_s, base_url=self.base_url)
        completion = client.complete(request.messages[-1].content, params.get("seed"))
        return ProviderResponse(text=completion.text, error_code=completion.error_code, attempts=completion.attempts,
                                latency_s=completion.latency_s, returned_model=completion.returned_model)
