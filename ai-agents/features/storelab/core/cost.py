"""Cost estimation, metering and the budget (SL-11).

**Estimate.** A trial makes one call under ``text`` and ``structured``, and E[turns]
calls under ``browser_tool``: ``budget.expected_turns[model_key]`` (calibrated on the
pilot, SL-17), else ``protocol.max_turns``, the upper bound. Over a plan::

    N_calls = Σ_tasks Σ_K n_sets(K) × r × |interfaces| × |temperatures| × |model-versions| × E[turns]

which ``Estimator`` evaluates trial by trial (E[turns] is 1 for the single-call
interfaces). Input tokens are counted from the rendered prompts by a ``TokenCounter``
(tiktoken for OpenAI, characters ÷ 3.5 × 1.2 otherwise; ``tokens.py``), output tokens
at their cap ``max_output_tokens``, and priced per million tokens by a ``PriceFn``
(``pricing.py``); a model without a price raises ``CostError``, so the plan is refused.

A ``browser_tool`` turn t (1-based) sends the system prompt, the tool definitions and
the results page (``base`` tokens) plus, for each earlier turn, the model's reply (at
most the output cap) and a tool result (at most the longest product page): ``base +
(t − 1) × step`` with ``step = page + cap``. Over T turns that is ``T × base + step ×
T(T − 1)/2`` input and ``T × cap`` output tokens; with T = E[turns] it is the
expectation, with T = ``max_turns`` the trial's maximum. Billing assumes one billed
call per call: a failed attempt (429, 5xx) is not billed by the providers.

**Metering.** ``Meter`` holds the committed spend (the USD of every provider call
made, charged as its usage arrives) and the reservations of the trials in flight. A
trial is started only if committed + reserved + its own maximum cost ≤ ``max_usd``;
otherwise it is ``not_run``/budget and the run is LIMIT_REACHED (``runner.py``). A
trial's reservation shrinks as its calls are charged and is released when it is
recorded, or, for a trial abandoned after its timeout, when its call returns. In-flight
trials finish, so spend ≤ ``max_usd`` + ``max_concurrency`` × the maximum per-call
cost even when a call costs more than its estimate.

**Report.** ``estimate_report`` (JSON) and ``estimate_markdown`` give the estimate by
model; ``compare`` sets it against the metered spend (and the savings from cache
hits) of a finished run, and ``gate`` applies the SL-18 rule |estimate − actual| /
actual ≤ 25% for every model.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Mapping

from .config import RunConfig, canonical_json
from .protocol import Observation
from .storefront import PAGE_INTERFACE, SYSTEM_PROMPT, TOOL_SCHEMA_VERSION, tool_definitions

if TYPE_CHECKING:
    from .records import TrialRecord
    from .runner import Trial

REPORT_VERSION = 1
GATE_TOLERANCE = 0.25  # SL-18 engineering gate
FORMULA = ("N_calls = sum_tasks sum_K n_sets(K) x r x |interfaces| x |temperatures| x |model-versions| x E[turns]")
HEURISTIC = "ceil(characters / 3.5 x 1.2)"

TokenCounter = Callable[[str, str, str], int]  # (provider, model, text) -> tokens
PriceFn = Callable[[str], tuple[float, float]]  # model -> USD per 1M input and output tokens


class CostError(ValueError):
    """A cost that cannot be estimated (no price, no tokenizer): never guessed."""


def heuristic_tokens(text: str) -> int:
    """characters ÷ 3.5 × 1.2 rounded up, in integers: ceil(12 × characters / 35)."""
    return -(-12 * len(text) // 35)


# ------------------------------------------------------------------- prompts
@dataclass(frozen=True)
class TrialPrompt:
    """The text a trial's calls carry: ``first`` is the first request (the rendered view,
    or for ``browser_tool`` the system prompt, tool definitions and results page);
    ``pages`` are the product pages a later ``browser_tool`` turn may get back."""

    first: str
    pages: tuple[str, ...] = ()


def trial_prompt(observation: Observation, interface: str) -> TrialPrompt:
    view = observation.views[interface]
    if interface != PAGE_INTERFACE:
        return TrialPrompt(view)
    tools = canonical_json(tool_definitions(TOOL_SCHEMA_VERSION))
    return TrialPrompt("\n".join((SYSTEM_PROMPT, tools, view)), tuple((observation.pages or {}).values()))


@dataclass(frozen=True)
class TrialCost:
    """Expected and maximum calls, tokens and USD of one trial."""

    calls: float
    input_tokens: float
    output_tokens: float
    usd: float
    max_calls: int
    max_usd: float
    max_call_usd: float


def expected_turns(config: RunConfig, model_key: str) -> float:
    """E[turns] of a browser_tool trial: the calibrated value, else the upper bound."""
    return config.budget.expected_turns.get(model_key, float(config.protocol.max_turns))


def _turns_cost(base: int, step: int, cap: int, turns: float) -> tuple[float, float]:
    """(input, output) tokens of ``turns`` turns of a growing conversation."""
    return turns * base + step * turns * (turns - 1) / 2, turns * cap


class Estimator:
    """Per-trial costs of a config's models; ``prompt`` gives a trial's ``TrialPrompt``."""

    def __init__(self, config: RunConfig, prompt: Callable[[Trial], TrialPrompt], tokens: TokenCounter,
                 price: PriceFn):
        self.config, self.prompt, self.tokens = config, prompt, tokens
        self.specs = {m.key: m for m in config.agents.models}
        self.rates: dict[str, tuple[float, float]] = {}
        for m in config.agents.models:
            if m.provider == "ollama":  # the local model bills nothing
                self.rates[m.key] = (0.0, 0.0)
                continue
            try:
                self.rates[m.key] = tuple(price(m.model))
            except (KeyError, ValueError) as error:
                raise CostError("model %s (%s): no price, so the cost cannot be estimated: %s"
                                % (m.key, m.model, error)) from None
        self._memo: dict[tuple, TrialCost] = {}

    def _usd(self, model_key: str, input_tokens: float, output_tokens: float) -> float:
        rate_in, rate_out = self.rates[model_key]
        return input_tokens / 1_000_000 * rate_in + output_tokens / 1_000_000 * rate_out

    def trial(self, trial: Trial) -> TrialCost:
        key = (trial.presentation_id, trial.interface, trial.model_key)
        if key not in self._memo:
            self._memo[key] = self._cost(trial)
        return self._memo[key]

    def _cost(self, trial: Trial) -> TrialCost:
        spec = self.specs[trial.model_key]
        prompt, cap = self.prompt(trial), spec.max_output_tokens
        base = self.tokens(spec.provider, spec.model, prompt.first)
        if trial.interface != PAGE_INTERFACE:
            usd = self._usd(trial.model_key, base, cap)
            return TrialCost(1.0, float(base), float(cap), usd, 1, usd, usd)
        page = max((self.tokens(spec.provider, spec.model, p) for p in prompt.pages), default=0)
        step, top = page + cap, self.config.protocol.max_turns
        turns = expected_turns(self.config, trial.model_key)
        expected_in, expected_out = _turns_cost(base, step, cap, turns)
        max_in, max_out = _turns_cost(base, step, cap, top)
        return TrialCost(turns, expected_in, expected_out, self._usd(trial.model_key, expected_in, expected_out), top,
                         self._usd(trial.model_key, max_in, max_out),
                         self._usd(trial.model_key, base + (top - 1) * step, cap))


@dataclass
class ModelEstimate:
    model_key: str
    provider: str
    model: str
    input_usd_per_1m: float
    output_usd_per_1m: float
    trials: int = 0
    calls: float = 0.0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    usd: float = 0.0
    max_call_usd: float = 0.0
    max_trial_usd: float = 0.0


def estimate(estimator: Estimator, trials: Iterable[Trial]) -> list[ModelEstimate]:
    """Per model, in config order: trials, expected calls and tokens, USD and maxima."""
    rows = {m.key: ModelEstimate(m.key, m.provider, m.model, *estimator.rates[m.key])
            for m in estimator.config.agents.models}
    for trial in trials:
        cost, row = estimator.trial(trial), rows[trial.model_key]
        row.trials += 1
        row.calls += cost.calls
        row.input_tokens += cost.input_tokens
        row.output_tokens += cost.output_tokens
        row.usd += cost.usd
        row.max_call_usd = max(row.max_call_usd, cost.max_call_usd)
        row.max_trial_usd = max(row.max_trial_usd, cost.max_usd)
    return list(rows.values())


def estimate_report(config: RunConfig, rows: list[ModelEstimate], *, mode: str, run_id: str | None,
                    token_counting: Mapping[str, str]) -> dict:
    """The estimate as JSON data (``reports/cost_estimate.json``)."""
    budget = config.budget
    max_call = max((r.max_call_usd for r in rows), default=0.0)
    return {
        "report_version": REPORT_VERSION, "kind": "cost_estimate", "run_id": run_id,
        "config_hash": config.config_hash, "mode": mode, "formula": FORMULA,
        # browser_tool calls per trial; the other interfaces make one
        "expected_turns": ({m.key: expected_turns(config, m.key) for m in config.agents.models}
                           if PAGE_INTERFACE in config.interfaces else {}),
        "token_counting": dict(token_counting), "output_tokens": "max_output_tokens per call",
        "models": [vars(r).copy() for r in rows],
        "totals": {"trials": sum(r.trials for r in rows), "calls": sum(r.calls for r in rows),
                   "input_tokens": sum(r.input_tokens for r in rows),
                   "output_tokens": sum(r.output_tokens for r in rows), "usd": sum(r.usd for r in rows)},
        "max_call_usd": max_call, "max_trial_usd": max((r.max_trial_usd for r in rows), default=0.0),
        "max_usd": budget.max_usd, "approval_usd": budget.approval_usd, "max_concurrency": budget.max_concurrency,
        "spend_bound_usd": budget.max_usd + budget.max_concurrency * max_call,
    }


def _number(value: float) -> str:
    return "%d" % value if float(value).is_integer() else "%.2f" % value


def estimate_markdown(report: Mapping) -> str:
    lines = ["# Cost estimate", "", "Run `%s`, mode %s, config `%s`." % (report["run_id"], report["mode"],
                                                                        report["config_hash"][:12]),
             "", "`%s`" % report["formula"], "",
             "| Model | Trials | Calls | E[turns] | Input tokens | Output tokens | USD | Max per call | Max per trial |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["models"]:
        lines.append("| %s (`%s`) | %d | %s | %s | %s | %s | %.4f | %.6f | %.6f |" % (
            r["model_key"], r["model"], r["trials"], _number(r["calls"]),
            _number(report["expected_turns"].get(r["model_key"], 1)), _number(r["input_tokens"]),
            _number(r["output_tokens"]), r["usd"], r["max_call_usd"], r["max_trial_usd"]))
    t = report["totals"]
    lines += ["| **Total** | %d | %s | | %s | %s | %.4f | %.6f | %.6f |" % (
        t["trials"], _number(t["calls"]), _number(t["input_tokens"]), _number(t["output_tokens"]), t["usd"],
        report["max_call_usd"], report["max_trial_usd"]), "",
        "Token counting: %s. Output tokens at the cap." % "; ".join(
            "%s: %s" % item for item in sorted(report["token_counting"].items())),
        "Budget: max_usd $%.2f, approval above $%s, max_concurrency %d; spend bound max_usd + max_concurrency "
        "x max per-call cost = $%.4f." % (report["max_usd"], "%.2f" % report["approval_usd"]
                                          if report["approval_usd"] is not None else "-",
                                          report["max_concurrency"], report["spend_bound_usd"]), ""]
    return "\n".join(lines)


# ------------------------------------------------------------------- metering
class Meter:
    """Committed spend and in-flight reservations against ``max_usd`` (thread-safe)."""

    def __init__(self, max_usd: float, spent_usd: float = 0.0):
        self.max_usd, self.spent_usd = max_usd, spent_usd
        self._held: dict[str, list[float]] = {}  # trial key -> [reserved, charged so far]
        self._kept: set[str] = set()  # trials recorded while a call may still bill (``hold``)
        self._lock = threading.Lock()

    @property
    def reserved_usd(self) -> float:
        with self._lock:
            return sum(max(0.0, reserved - charged) for reserved, charged in self._held.values())

    def reserve(self, key: str, usd: float) -> bool:
        """Hold ``usd`` for trial ``key`` if committed + reserved + usd ≤ max_usd."""
        with self._lock:
            held = sum(max(0.0, reserved - charged) for reserved, charged in self._held.values())
            if self.spent_usd + held + usd > self.max_usd + 1e-12:
                return False
            self._held[key] = [usd, 0.0]
            return True

    def charge(self, key: str | None, usd: float) -> None:
        """A provider call's cost: committed at once, and taken off the trial's reservation."""
        with self._lock:
            self.spent_usd += usd
            if key in self._held:
                self._held[key][1] += usd

    def settle(self, key: str, usd: float = 0.0) -> float:
        """The trial is recorded: release its reservation (unless held) and commit ``usd``
        charged for it outside the metered calls; returns what its calls were charged."""
        with self._lock:
            self.spent_usd += usd
            if key in self._kept:
                return self._held.get(key, [0.0, 0.0])[1]
            return self._held.pop(key, [0.0, 0.0])[1]

    def hold(self, key: str) -> None:
        """Keep the trial's reservation past ``settle`` until ``release``: the trial was
        abandoned after its timeout and its call may still bill."""
        with self._lock:
            if key in self._held:
                self._kept.add(key)

    def release(self, key: str) -> None:
        with self._lock:
            self._kept.discard(key)
            self._held.pop(key, None)


def jsonl(path: Path | None) -> Iterator[dict]:
    """The complete JSON lines of ``path``; a line cut short by a kill is skipped."""
    if path is None or not Path(path).is_file():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            yield json.loads(line)
        except ValueError:
            continue


def call_usd(call: Mapping) -> float:
    """The USD of one provider call; a call without a price is never metered as free."""
    if call.get("cost_usd") is None:
        raise CostError("provider call of trial %s (model %s) has no cost_usd: spend cannot be metered"
                        % (call.get("trial_key"), call.get("requested_model")))
    return float(call["cost_usd"])


def logged_spend(invocation_log: Path | None, call_log: Path | None = None) -> float:
    """The spend a run's logs hold: every provider call's ``cost_usd`` (invocation log,
    one fsynced line per call) plus the charges recorded with trials (call log)."""
    total = 0.0
    for line in jsonl(invocation_log):
        total += call_usd(line)
    return total + sum(float(line.get("cost_usd") or 0.0) for line in jsonl(call_log))


# ------------------------------------------------------------- estimate vs actual
def compare(estimator: Estimator, trials: Iterable[Trial], records: Mapping[str, TrialRecord],
            invocations: Iterable[Mapping], cache_hits: Iterable[Mapping], *,
            tolerance: float = GATE_TOLERANCE) -> dict:
    """Estimate against actual per model, over the trials with a final record.

    ``actual_usd`` is the cost of the calls those trials made: billed (the invocation
    log) plus those served from the cache, priced at their recorded usage
    (``cache_saved_usd``, the savings). ``relative_error`` is (estimate − actual) / actual
    and ``within_tolerance`` |relative_error| ≤ ``tolerance``.
    """
    trials = list(trials)
    finished = {t.trial_key: t for t in trials if t.trial_key in records
                and records[t.trial_key].outcome.status != "not_run"}
    model_of = {t.trial_key: t.model_key for t in trials}
    rows: dict[str, dict] = {}
    for m in estimator.config.agents.models:
        rows[m.key] = {"model_key": m.key, "model": m.model, "planned_trials": 0, "finished_trials": 0,
                       "estimate_usd": 0.0, "planned_estimate_usd": 0.0, "estimated_calls": 0.0, "calls": 0,
                       "billed_usd": 0.0, "cache_hits": 0, "cache_saved_usd": 0.0, "turns": []}
    for trial in trials:
        cost, row = estimator.trial(trial), rows[trial.model_key]
        row["planned_trials"] += 1
        row["planned_estimate_usd"] += cost.usd
        if trial.trial_key in finished:
            row["finished_trials"] += 1
            row["estimate_usd"] += cost.usd
            row["estimated_calls"] += cost.calls
            if trial.interface == PAGE_INTERFACE:
                row["turns"].append(records[trial.trial_key].outcome.turns)
    for line in invocations:
        key = line.get("trial_key")
        if key in finished:
            row = rows[model_of[key]]
            row["calls"] += 1
            row["billed_usd"] += float(line.get("cost_usd") or 0.0)
    for line in cache_hits:
        key = line.get("trial_key")
        if key in finished:
            row = rows[model_of[key]]
            row["cache_hits"] += 1
            row["cache_saved_usd"] += float(line.get("cost_usd") or 0.0)
    out = []
    for row in rows.values():
        turns = row.pop("turns")
        row["mean_browser_turns"] = sum(turns) / len(turns) if turns else None
        row["actual_usd"] = row["billed_usd"] + row["cache_saved_usd"]
        actual = row["actual_usd"]
        row["relative_error"] = (row["estimate_usd"] - actual) / actual if actual > 0 else None
        row["within_tolerance"] = row["relative_error"] is not None and abs(row["relative_error"]) <= tolerance
        out.append(row)
    return {"report_version": REPORT_VERSION, "kind": "cost_actual", "tolerance": tolerance, "models": out,
            "totals": {k: sum(r[k] for r in out) for k in ("estimate_usd", "planned_estimate_usd", "billed_usd",
                                                           "cache_saved_usd", "actual_usd", "calls", "cache_hits")},
            "gate": gate(out)}


def gate(rows: Iterable[Mapping]) -> dict:
    """The SL-18 gate: every model's estimate within the tolerance of its actual cost (a
    model with no actual cost fails: there is nothing to calibrate against)."""
    failing = [r["model_key"] for r in rows if not r["within_tolerance"]]
    return {"passed": not failing, "failing_models": failing}


def actual_markdown(report: Mapping) -> str:
    lines = ["# Cost: estimate vs actual", "",
             "Over the trials with a final record. Actual = billed calls + calls served from the cache (savings), "
             "priced at their usage. Gate (SL-18): |estimate - actual| / actual <= %g for every model: **%s**."
             % (report["tolerance"], "passed" if report["gate"]["passed"] else "failed"), "",
             "| Model | Trials | Estimate USD | Billed USD | Cache hits | Saved USD | Actual USD | Error | Mean turns |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["models"]:
        error = "-" if r["relative_error"] is None else "%+.1f%%" % (100 * r["relative_error"])
        turns = "-" if r["mean_browser_turns"] is None else "%.2f" % r["mean_browser_turns"]
        lines.append("| %s | %d of %d | %.4f | %.4f | %d | %.4f | %.4f | %s | %s |" % (
            r["model_key"], r["finished_trials"], r["planned_trials"], r["estimate_usd"], r["billed_usd"],
            r["cache_hits"], r["cache_saved_usd"], r["actual_usd"], error, turns))
    t = report["totals"]
    lines += ["| **Total** | | %.4f | %.4f | %d | %.4f | %.4f | | |" % (
        t["estimate_usd"], t["billed_usd"], t["cache_hits"], t["cache_saved_usd"], t["actual_usd"]), ""]
    return "\n".join(lines)
