# Cost estimation, metering and the budget (SL-11)

Spend is predicted before a run (the plan's estimate) and bounded during it (the
budget meter). This validates S5: spend ≤ `max_usd` + `max_concurrency` × the maximum
per-call cost. Code: `core/cost.py` (formulas, meter, reports), `costs.py` (prices,
tokens and run directories), `tokens.py` (token counts).

## Estimate

A trial makes one call under `text` and `structured`, and E[turns] calls under
`browser_tool`. E[turns] is `budget.expected_turns[<model key>]`, calibrated on the
pilot (SL-17). A model that is not listed is estimated at `protocol.max_turns`, the
upper bound. The number of calls is

    N_calls = Σ_tasks Σ_K n_sets(K) × r × |interfaces| × |temperatures| × |model-versions| × E[turns]

The estimator evaluates this trial by trial over the planned trials, with E[turns] = 1
for the single-call interfaces.

**Tokens.**
- Input tokens are counted from the rendered prompts.
  - OpenAI models use tiktoken 0.14.0 with `o200k_base`, the encoding of every pinned
    OpenAI model. The encoding file is read from
    `artifacts/tokenizers/o200k_base/<sha256>/`, with the hash tiktoken expects, never
    from the network.
  - Anthropic and Google publish no offline tokenizer, so their count is ⌈characters ÷
    3.5 × 1.2⌉, computed in integers as ⌈12 × characters / 35⌉.
- Output tokens are counted at their cap, `max_output_tokens`.
- A `browser_tool` conversation grows with each turn:
  - Turn t sends `base + (t − 1) × step`. Here `base` is the system prompt, the tool
    definitions and the results page, and `step` is the longest product page plus the
    output cap: each earlier turn adds the model's reply and a tool result.
  - Over T turns that is `T × base + step × T(T − 1)/2` input tokens and `T × cap`
    output tokens.
  - T = E[turns] gives the expectation, and T = `max_turns` gives the trial's maximum.

**Prices** come from `pricing.py`, the StoreLab rows of `evals/pricing.py`, in USD per
1M input and output tokens. Each call is assumed to be billed once, because providers
do not bill failed attempts (429, 5xx). A model with no price refuses a sandbox plan
(`PlanError`), and no run directory is written. Offline and replay plans call no paid
provider: they are estimated at $0 and reserve nothing.

**Illustrative E1 count** (to be replaced by the SL-17 design):
- 10 tasks with v = 20, so each task has 190 + ~35 + ~8 = 233 sets (K = 2, 4, 8),
  and r = 10.
- 3 interfaces, 2 temperatures and 6 model-versions.
- Trials: 10 × 233 × 10 × 3 × 2 × 6 = 838,800 ≈ 0.84 M.
- Calls, with 4 turns per `browser_tool` trial: 10 × 233 × 10 × 2 × 6 × (1 + 1 + 4) =
  1,677,600 ≈ 1.7 M.

This is why the pilot sizes the main run.

**Report.** `plan` writes `reports/cost_estimate.json` and `reports/cost_estimate.md`
for a sandbox run. Per model they give:
- trials and expected calls;
- E[turns];
- input and output tokens;
- USD;
- the maximum cost of one call and of one trial.

They also give the totals, the token-counting rules and the spend bound `max_usd` +
`max_concurrency` × the maximum per-call cost. The plan is `PENDING_APPROVAL` when the
estimate is above `budget.approval_usd`.

## Metering and the budget

`core.cost.Meter` holds two amounts:
- the **committed** spend: the USD of every provider call made, charged as its usage
  arrives (`log_call`, from the call's `usage` and `pricing.py`);
- the **reservations** of the trials in flight.

The rules:
1. Before a trial starts, its maximum cost is reserved. The trial starts only if
   committed + reserved + its maximum ≤ `max_usd`. Otherwise the trial is
   `not_run`/budget and the run ends `LIMIT_REACHED`.
2. A trial's reservation shrinks as its calls are charged and is released when the
   trial is recorded.
3. Trials already in flight finish. With single-call trials, the spend is therefore at
   most `max_usd` + `max_concurrency` × the maximum per-call cost, even when calls cost
   more than estimated. Within the estimate it stays at most `max_usd`.

Each provider call is one fsynced line, with its `cost_usd`, in the run's
`invocations.jsonl`. Under `PlatformGovernance` this local copy is written next to the
platform invocation log. On `resume` the spend is read back from `invocations.jsonl`,
plus any charge recorded in `calls.jsonl`, so every call is counted once:
- A trial whose record a kill lost is rerun. Under `read_write` its calls are served
  from the cache and are not charged again.
- `resume --max-usd` raises the budget of a `LIMIT_REACHED` run.

A call with no `cost_usd` is never metered as free: metering it fails the run.

Each call served from the response cache is one line in `cache_hits.jsonl`, with what
it would have cost. These lines are the savings.

## Estimate vs actual and the SL-18 gate

    python -m features.storelab.cli cost RUN [--tolerance 0.25] [--root DIR]

This writes `reports/cost.json` and `reports/cost.md`. It considers only the trials
with a final record. Per model it reports:
- the estimate;
- the billed USD (`invocations.jsonl`);
- cache hits and saved USD (`cache_hits.jsonl`);
- **actual** = billed + saved, the cost of the calls the trials made;
- `relative_error` = (estimate − actual) / actual;
- the mean `browser_tool` turns, which are the value to register as
  `budget.expected_turns` after the pilot.

The gate passes when |relative_error| ≤ 25% for every model. A model with no actual
cost fails the gate. The command exits 1 when the gate fails. SL-18 may start only
after the pilot passes the gate (SL-17 calibrates E[turns] and re-plans).
