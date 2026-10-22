"""Live smoke run of the provider adapters (SL-09), behind ``--live``.

    python -m features.storelab.live_smoke CONFIG [--live] [--set KEY=VALUE ...] [--out DIR]
        [--max-usd-per-model USD] [--record-fixtures DIR]

Shrinks CONFIG to one category, one task, K = 2 over 4 eligible products (6 sets), one
repetition and the text interface, then runs every configured model at each of its
temperatures in mode sandbox with the cache off, so each trial is a live call. Before
calling it bounds each model's cost from the rendered prompts (characters ÷ 3.5 × 1.2
input tokens, ``max_output_tokens`` output tokens, ``pricing.py`` rates) and refuses to
start if any bound exceeds ``--max-usd-per-model`` (default 1.0). Without ``--live`` it
prints that plan and calls nothing.

The run goes through the runner (``api.run``), so its records, manifest and
``invocations.jsonl`` are the evidence. ``report.json`` and ``report.md`` list, per
model: requested id, returned ids, access date (UTC), calls, outcomes, tokens and cost,
plus the SDK and API versions. Output goes to ``artifacts/smoke/sl09/<UTC time>/`` by
default. ``--record-fixtures`` also saves each HTTP exchange (no credentials; prompt
text replaced by its sha256, since the prompts quote catalog text, A7) for the
contract tests.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from . import api
from .config_files import parse_set, read_config_data
from .core.config import RunConfig, apply_overrides, resolve_config
from .core.render import build_observation
from .core.runner import plan_trials
from .core.store import LocalStore
from .legacy_baseline import default_artifact_root
from .pricing import call_cost_usd
from .core.protocol import Usage
from .settings import ANTHROPIC_API_VERSION, StorelabSettings

SMOKE = {"environment.tasks_per_category": 1, "design.eligible_size": 4, "design.set_sizes": [2],
         "repetitions": 1, "interfaces": ["text"], "mode": "sandbox", "cache.policy": "off",
         "cache.namespace": "sl09_live_smoke", "budget.max_concurrency": 2}
SDKS = ("openai", "anthropic", "google-genai")
KEPT_HEADERS = ("content-type", "request-id", "x-request-id", "retry-after", "retry-after-ms")


def smoke_config(path: Path, overrides: dict) -> RunConfig:
    data = read_config_data(path)
    first = data["environment"]["categories"][:1]
    text = {"text": data.get("protocol", {}).get("prompt_templates", {}).get("text", "e1_text@1")}
    return resolve_config(apply_overrides(data, {**SMOKE, "environment.categories": first,
                                                 "protocol.prompt_templates": text, **overrides}))


def worst_case_usd(config: RunConfig, snapshot: Path | None = None) -> dict[str, dict]:
    """Per model key: trials and the cost bound from the rendered prompts."""
    stimuli = api.build_stimuli(config, snapshot)
    specs = {t.task.task_id: t.task for t in stimuli.tasks}
    designs = {s.set_id: d for d in stimuli.designs for s in d.sets}
    bounds: dict[str, dict] = {}
    for trial in plan_trials(config, stimuli):
        spec = next(m for m in config.agents.models if m.key == trial.model_key)
        observation = build_observation(config, specs[trial.task_id],
                                        designs[trial.set_id].presentation_for(trial.set_id, trial.rep),
                                        stimuli.products)
        tokens = int(len(observation.views[trial.interface]) / 3.5 * 1.2) + 1
        cost = call_cost_usd(spec.model, Usage(input_tokens=tokens, output_tokens=spec.max_output_tokens))
        bound = bounds.setdefault(trial.model_key, {"model": spec.model, "trials": 0, "usd": 0.0})
        bound["trials"] += 1
        bound["usd"] += cost * config.failure_policy.max_attempts  # every attempt may bill
    return bounds


def _recording_client(module, provider: str, sink: list):
    class Recording(module.HTTPTransport):
        def handle_request(self, request):
            response = super().handle_request(request)
            body = response.read()
            sent = json.loads(request.content or b"null")
            for message in (sent or {}).get("messages", []) + (sent or {}).get("contents", []):
                _redact(message)
            sink.append({"source": "recorded", "provider": provider,
                         "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         "request": {"method": request.method, "path": request.url.path, "json": sent},
                         "response": {"status": response.status_code,
                                      "headers": {k: v for k, v in response.headers.items() if k in KEPT_HEADERS},
                                      "json": json.loads(body or b"null")}})
            headers = [(k, v) for k, v in response.headers.items() if k not in ("content-encoding", "content-length")]
            return module.Response(response.status_code, headers=headers, content=body, request=request)

    return module.Client(transport=Recording())


def _redact(message: dict) -> None:
    def digest(text: str) -> str:
        return "<prompt sha256:%s>" % hashlib.sha256(text.encode("utf-8")).hexdigest()

    if isinstance(message.get("content"), str):
        message["content"] = digest(message["content"])
    for part in message.get("parts", []) or []:
        if isinstance(part.get("text"), str):
            part["text"] = digest(part["text"])


def recording_clients(sink: list) -> dict:
    """HTTP clients per provider that copy each exchange into ``sink``."""
    import httpx
    import httpx2

    return {"openai": _recording_client(httpx, "openai", sink), "anthropic": _recording_client(httpx2, "anthropic", sink),
            "google": _recording_client(httpx, "google", sink)}


def report(run_dir: Path, bounds: dict) -> dict:
    store = LocalStore(run_dir)
    manifest = store.read_manifest()
    calls = [json.loads(line) for line in store.invocation_log.read_text(encoding="utf-8").splitlines()] \
        if store.invocation_log.is_file() else []
    outcomes = collections.defaultdict(collections.Counter)
    for record in store.records().values():
        outcomes[record.model_key][record.outcome.status + ("/" + record.outcome.error_code
                                                            if record.outcome.error_code else "")] += 1
    models = []
    for m in manifest.config.agents.models:
        mine = [c for c in calls if c["requested_model"] == m.model]
        record = next(r for r in manifest.models if r.key == m.key)
        models.append({
            "key": m.key, "provider": m.provider, "requested_model": m.model, "returned_ids": record.returned_ids,
            "access_date_utc": record.last_access_utc.date().isoformat() if record.last_access_utc else None,
            "temperatures": m.temperatures, "calls": len(mine), "attempts": sum(c["attempts"] for c in mine),
            "outcomes": dict(outcomes[m.key]),
            "input_tokens": sum(c["input_tokens"] or 0 for c in mine),
            "output_tokens": sum(c["output_tokens"] or 0 for c in mine),
            "cost_usd": round(sum(c["cost_usd"] or 0.0 for c in mine), 6),
            "bound_usd": round(bounds[m.key]["usd"], 6), "request_ids": [c["request_id"] for c in mine][:3]})
    return {"run_id": manifest.run_id, "run_dir": str(run_dir), "status": manifest.status_history[-1].status,
            "reason": manifest.status_history[-1].reason, "config_hash": manifest.config_hash,
            "sdk_versions": {name: _version(name) for name in SDKS}, "anthropic_api_version": ANTHROPIC_API_VERSION,
            "models": models}


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def markdown(data: dict) -> str:
    lines = ["# SL-09 live smoke", "",
             "Run `%s`: %s%s. SDKs: %s; Anthropic API version %s." % (
                 data["run_id"], data["status"], " (%s)" % data["reason"] if data["reason"] else "",
                 ", ".join("%s %s" % kv for kv in data["sdk_versions"].items()), data["anthropic_api_version"]), "",
             "| Model key | Requested id | Returned ids | Access date (UTC) | Calls | Outcomes | Tokens in/out | "
             "Cost USD | Bound USD |", "|---|---|---|---|---|---|---|---|---|"]
    for m in data["models"]:
        lines.append("| %s | %s | %s | %s | %d | %s | %d / %d | %.6f | %.4f |" % (
            m["key"], m["requested_model"], ", ".join(m["returned_ids"]) or "-", m["access_date_utc"] or "-",
            m["calls"], ", ".join("%s %d" % kv for kv in sorted(m["outcomes"].items())), m["input_tokens"],
            m["output_tokens"], m["cost_usd"], m["bound_usd"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.storelab.live_smoke")
    parser.add_argument("config", type=Path)
    parser.add_argument("--live", action="store_true", help="make the live calls (otherwise print the plan only)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-usd-per-model", type=float, default=1.0)
    parser.add_argument("--record-fixtures", type=Path, help="save the HTTP exchanges here as JSON fixtures")
    args = parser.parse_args(argv)

    try:
        config = smoke_config(args.config, parse_set(args.set))
        bounds = worst_case_usd(config, args.snapshot)
    except (ValidationError, OSError, ValueError, KeyError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    over = {k: round(b["usd"], 4) for k, b in bounds.items() if b["usd"] > args.max_usd_per_model}
    print(json.dumps({"trials": {k: b["trials"] for k, b in bounds.items()},
                      "bound_usd": {k: round(b["usd"], 4) for k, b in bounds.items()}}, indent=1))
    if over:
        print("error: cost bound above $%g per model: %s" % (args.max_usd_per_model, over), file=sys.stderr)
        return 2
    if not args.live:
        print("dry run: pass --live to call the providers", file=sys.stderr)
        return 0
    settings = StorelabSettings.from_env()
    missing = sorted({m.provider for m in config.agents.models if settings.api_key(m.provider) is None})
    if missing:
        print("error: no API key for %s" % ", ".join(missing), file=sys.stderr)
        return 2
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or default_artifact_root() / "smoke" / "sl09" / stamp
    sink = [] if args.record_fixtures else None
    total = sum(b["usd"] for b in bounds.values())
    run_dir, _ = api.run(config.model_copy(update={"budget": config.budget.model_copy(update={"max_usd": total})}),
                         snapshot=args.snapshot, root=out / "runs", estimate_cost=lambda c, t, m: total,
                         http_clients=recording_clients(sink) if sink is not None else None)
    data = report(run_dir, bounds)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")
    (out / "report.md").write_text(markdown(data), encoding="utf-8")
    if sink is not None:
        args.record_fixtures.mkdir(parents=True, exist_ok=True)
        counts: collections.Counter = collections.Counter()
        for exchange in sink:
            counts[exchange["provider"]] += 1
            name = "recorded_%s_%02d.json" % (exchange["provider"], counts[exchange["provider"]])
            (args.record_fixtures / name).write_text(json.dumps(exchange, indent=1) + "\n", encoding="utf-8")
    print(markdown(data))
    return 0 if data["status"] in ("COMPLETED", "COMPLETED_WITH_GAPS") else 3


if __name__ == "__main__":
    sys.exit(main())
