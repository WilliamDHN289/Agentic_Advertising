# Provider adapters (sandbox mode)

SL-09. Code: `backends/provider.py` (interface, call loop, `CallGuard`),
`backends/openai_adapter.py`, `backends/anthropic_adapter.py`,
`backends/google_adapter.py`, `governance.py` (`PlatformGovernance`), `settings.py`
(`StorelabSettings`), `pricing.py` (prices), `live_smoke.py`. Tests:
`tests/features/storelab/test_providers.py`.

## No fallback, exact attribution

An adapter sends a request to the model the trial's config names and to nothing else.
There is no failover chain, alias resolution or second model, which is why the
`AdgraphClient` facade is not used (its failover at `model_gateway.py:1068`, the Google
`CompatResponse` that drops the model version, the OpenAI/Google-only target validator,
no tool calls). An import test keeps `get_adgraph_client`, `AdgraphClient`,
`AdgraphFallbackSettings` and `model_gateway` out of every StoreLab module.

| Provider | SDK (pinned) | Call | Returned id | Request id | Refusal |
|---|---|---|---|---|---|
| OpenAI | `openai==2.8.0`, `max_retries=0` | Chat Completions | `model` | `x-request-id` header; `system_fingerprint` kept | `message.refusal`, `finish_reason` `content_filter` |
| Anthropic | `anthropic==1.17.0`, `max_retries=0`, API version `2023-06-01` | Messages | `model` | `request-id` header | `stop_reason` `refusal` |
| Google | `google-genai==1.56.0`, `HttpRetryOptions(attempts=1)` | `generateContent` | `modelVersion` | `responseId` | `promptFeedback.blockReason`; finish `SAFETY`, `PROHIBITED_CONTENT`, `BLOCKLIST`, `SPII`, `IMAGE_SAFETY` |

The Anthropic adapter pins API version 2023-06-01 because 2026-10-21 (the default from
SDK 2.0) rejects `temperature` on every model, and E1 runs each model at its default
temperature and at 0. SDK 1.x does not type `temperature` and `top_p`, so they go in the
request body. Moving to SDK 2.x means removing the T = 0 arm for Anthropic.

Parameters sent: the prompt as one user message; the output-token cap; `temperature`
and `top_p` only when set (a null temperature is omitted and recorded as
`provider_default`); `seed` where the model registry says the model takes one (OpenAI
non-reasoning models and Gemini; not Anthropic). Each `ProviderResponse` records
`sent_params`, `seed_supported`, `request_id`, `system_fingerprint`, `api_version`,
the returned id, usage and latency; the trial record carries the requested and
returned ids, usage and latency, and the manifest the returned ids and access times
per model. Parsing stays with the SL-03 parser.

## Retries and errors (tickets.md section 3.5)

The SDKs do not retry. `ProviderAdapter.complete` makes at most
`failure_policy.max_attempts` attempts and waits `backoff_initial_s × 2^(n−1)`, at
least `Retry-After` (or `retry-after-ms`), at most `backoff_max_s` between them.

| Failure | Code | Retried |
|---|---|---|
| 429 | `rate_limited` | yes |
| 408, 409, 5xx (incl. 529), timeout, connection error | `provider_unavailable` | yes |
| 401, 403 | `auth` | no |
| 404 | `model_unavailable` | no |
| 400, 422, other 4xx | `bad_request` | no |
| any other exception | a defect: propagates, the runner fails the run | no |

The run effects (FAILED, breaker PAUSED, rate-limit storm, model drift) are the
runner's (docs/cli.md).

## Governance hooks

Every attempt passes the run's `CallGuard`, and every call that reached the provider is
logged once:

| Hook | `LocalGovernance` (CLI, tests) | `PlatformGovernance` (ai-agents) |
|---|---|---|
| kill switch, before each trial and attempt | `LOADTEST_LLM_KILL_SWITCH_FORCE` set to anything but `0` | `enforce_llm_kill_switch` (`llm_guard.py:183`) |
| breaker, slots, rate limit, before each attempt | none (the runner's breaker counts consecutive `provider_unavailable`) | `CircuitBreakerRegistry.allow`, `acquire_provider_slot`, `acquire_model_capacity`; refusal → `provider_unavailable` or `rate_limited` without a call |
| after each attempt | none | `release_provider_slot`; `record_success` / `record_failure` |
| once per provider call | a line in `<run>/invocations.jsonl` (fsynced): `run_id`, `trial_key`, `call_key`, provider, requested and returned id, status, error code, attempts, latency, tokens, request id, `cost_usd` | `create_llm_invocation_log_or_raise`, `invocation_kind="storelab"`, `job_id` = run id, `task_ref` = trial key; a failed write fails the run |

A cache hit is not a call and is not logged. The kill switch is checked immediately
before each attempt, so once it is engaged no new attempt starts; a call stopped before
its first attempt is `not_run`/`kill_switch` and the run ends CANCELLED with reason
`kill_switch`. With `max_concurrency` > 1, attempts that passed the check before the
switch was engaged complete.

`PlatformGovernance` takes the platform functions as a `Platform`;
`Platform.load(settings)` imports them inside ai-agents (this subtree does not contain
them; tests pass fakes).

## Settings, models and prices

`StorelabSettings` (`settings.py`; a section of `config/settings.py` on the platform)
reads the keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` or
`GEMINI_API_KEY`), `STORELAB_ALLOWED_MODELS`, breaker thresholds and the queue name.
Planning a sandbox run fails unless every API model is a pinned snapshot, allowed and
priced (`pricing.py`, the rows SL-09 adds to `evals/pricing.py`). The prices are the
providers' list rates for the A5 placeholder models:

| Model | Input $/1M | Output $/1M |
|---|---|---|
| gpt-4o-2024-08-06 | 2.50 | 10.00 |
| gpt-4.1-2025-04-14 | 2.00 | 8.00 |
| claude-sonnet-4-20250514 | 3.00 | 15.00 |
| claude-sonnet-4-5-20250929 | 3.00 | 15.00 |
| gemini-2.0-flash-001 | 0.10 | 0.40 |
| gemini-2.0-flash-lite-001 | 0.075 | 0.30 |

Budget metering against these prices is SL-11; until then the runner's cost per trial
is 0 and the invocation log carries each call's `cost_usd`.

## Live smoke

```
python -m features.storelab.live_smoke features/storelab/configs/e1_pilot.yaml            # plan and cost bound only
python -m features.storelab.live_smoke features/storelab/configs/e1_pilot.yaml --live \
    [--record-fixtures tests/features/storelab/fixtures/providers]
```

It runs every configured model at each of its temperatures on 6 two-product sets (one
task, one repetition, text interface, cache off) through the runner, refuses to start
if any model's cost bound (all attempts billed, full output) exceeds
`--max-usd-per-model` (default $1), and writes `report.json` and `report.md` (returned
ids, access date, calls, outcomes, tokens, cost per model; SDK and API versions) to
`artifacts/smoke/sl09/<UTC time>/`. Its `runs/` directory stays local (it holds catalog
text, A7). Recorded fixtures replace prompt text by its sha256.

## Fixtures

`tests/features/storelab/fixtures/providers/*_ok.json` and `*_refusal.json` are
authored from each API's documented response format (`"source": "authored"`).
`live_smoke --record-fixtures` adds `recorded_<provider>_NN.json` exchanges, which the
contract tests read the same way.
