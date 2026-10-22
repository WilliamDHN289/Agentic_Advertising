# Browser interface (`browser_tool`)

SL-10. Code: `core/storefront.py` (store, tools, events), `backends/browser_tool.py`
(tool loop, simulated agents), tool calling in the three provider adapters. Tests:
`tests/features/storelab/test_browser_tool.py`, transcripts in
`tests/features/storelab/fixtures/browser/`.

## Scope and limitation

`browser_tool` is a synthetic storefront navigated through four tool calls, in process.
It is the cheapest interface that exposes page requests for products the agent does not
choose (H1e). It is a tool-driven browser, not a GUI agent: there is no pixel
rendering, no HTML shown to the model, no Comet or computer-use agent (tickets.md
section 2).

**H1e conclusions apply to `browser_tool` only unless SL-10b runs.** SL-10b (a
rendered-page GUI agent driving a Playwright storefront in an isolated container,
tickets.md section 7) is deferred; until it runs, no H1e result may be stated for
browser-based agents in general.

## Storefront

One observation is one store. It holds the results page and one detail page per
candidate, both from the SL-05 page model (`render.page_model`, template
`e1_browser_tool@1`): `Observation.views["browser_tool"]` is the results-page text (K
items in their assigned positions, sponsored labels, links `/product/<ID>`) and
`Observation.pages` the detail-page text of each candidate (catalog fields; the
description cut to `environment.description_chars`). Pages are part of the observation
and of `observation_hash()`, so every mode and backend is shown, and served, the same
bytes (S2). The store has nothing else: no other page, no search, no URL fetch, no
network tool.

| Tool | Effect | Event (`kind`, `product_id`) |
|---|---|---|
| `open_product(product_id)` | serves the product's detail page | `page_request`, the product |
| `back()` | serves the results page | `back`, the product whose page was left (none on the results page) |
| `select(product_id)` | the effective choice; ends the trial | `select`, the product |
| `leave()` | the outside option; ends the trial | `leave`, none |

`open_product` and `select` work from any page. The tools and their descriptions are
versioned by `TOOL_SCHEMA_VERSION` (`storelab-browser-tools@1`), which is in every
call key; `tool_schema_sha256()` is pinned by a test, so a change needs a new version.

## Turns, events and outcomes

A turn is one response of the decision-maker. Its tool calls run in order and each is
exactly one `EventRecord`: `turn` (1-based), `seq` (order in the trial), `tool`,
`product_id`, the product's `position` and `sponsored` label, `module` (`serp` or
`detail`, the page the call was made from), `t_s` (seconds since the trial started; 0
for simulated agents), `simulated` (true for simulated agents), `human_viewport` and
`viewport_telemetry` false.

Never executed, recorded as an `invalid` event with `invalid_reason`, and the trial
ends `invalid` with that reason (no re-ask, as under the text interface):

| Call | Reason |
|---|---|
| an id outside the observation | `unknown_id` |
| a blank id | `empty` |
| another tool, arguments that are not a JSON object, missing or extra arguments, a non-string id | `schema_violation` |
| `select` or `leave` followed by another call in the same response (it and every later call) | `multiple_choices` |
| a call after a trial-ending invalid call in the same response | that call's reason |

A response with no tool call ends the trial `invalid` (`empty`, or `schema_violation`
if it has text). If `protocol.max_turns` (default 8) turns pass without `select` or
`leave`, the trial is `invalid`/`max_turns`, never a choice; a `select` on the last
allowed turn counts, and no call is made after it.

The outcome: `choice` with `chosen_id`, `outside`, or `invalid` with the reason;
`page_requests` lists the products whose pages were served, in order (the same product
can appear twice); `turns` the turns taken. Page requests and the effective choice are
recorded separately, per product, in the outcome and as events.

## Language models

Each turn is one provider call through the response cache and the run's guard
(SL-08/SL-09: kill switch, invocation log, retries). The conversation is the
storefront's system prompt, the results page as the user message, then for each turn
the assistant's tool calls and the pages served as tool results. The call forces a tool
call:

| Provider | Tools | Forced call | One call per response |
|---|---|---|---|
| OpenAI | `tools` (functions) | `tool_choice: required` | `parallel_tool_calls: false` |
| Anthropic | `tools` with `input_schema`; system prompt in `system` | `tool_choice: {type: any}` | `disable_parallel_tool_use: true` |
| Google | `function_declarations`; system prompt in `system_instruction` | `function_calling_config.mode: ANY`, automatic calling off | not available: several calls run in order |

| A turn ends the trial early | Outcome |
|---|---|
| provider error after retries, replay cache miss | `error` with the code; the events so far are kept |
| safety refusal | `refusal`; the events so far are kept |
| kill switch before the turn's first attempt | `not_run`/`kill_switch` without events; the trial runs again on resume (earlier turns are cache hits under `read_write`; their calls are in the invocation log) |
| `failure_policy.trial_timeout_s` passed since the trial started | `error`/`provider_unavailable` before the next call (the runner's per-trial timeout, which also bounds the whole trial) |

Attempts, tokens and latency are summed over turns; the returned model id is the last
turn's. Each turn is a cache entry (no parsed action); `raw_ref` names the last turn's
entry, whose request holds the whole transcript, so a replay serves every turn from the
cache and reproduces the events. The local model (`local_llm`) has no tool calling: a
run with `browser_tool` and that backend is refused.

## Simulated agents

A simulated agent acts through the same storefront: each action of
`SimulatedAgent.act` (χ-driven page requests in position order, then `select` or
`leave`; docs/simulated_backend.md) is one turn with one tool call. Its outcome equals
`SimulatedAgent.decide`, and its events have the same form as a model's.

## Storage

The runner writes a trial's events immediately before its `TrialRecord`, in the same
gzip member of the run's `records/part-*.jsonl.gz` shard, so they are kept or lost
together and never doubled on resume. `LocalStore.events()` returns the events of each
trial's effective record.

## Fixtures

`fixtures/browser/*.json` are per-provider transcripts in each API's response format,
authored from the documented tool-calling formats (`"source": "authored"`), one
response per turn, with the expected events and outcome: OpenAI open–back–open–select
and an unknown id, Anthropic open–open–leave with text before a call, Google two calls
in one response. Tests serve them through the real pinned SDKs.
