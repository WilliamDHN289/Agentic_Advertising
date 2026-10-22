"""StoreLab command line: SL-02 ``validate`` and ``schemas``, SL-05 ``e1-stimuli``,
SL-07 the runner commands (docs/cli.md; the Python API is ``api.py``).

    python -m features.storelab.cli validate CONFIG [--set KEY=VALUE ...] [--format yaml|json]
    python -m features.storelab.cli schemas [--check]
    python -m features.storelab.cli e1-stimuli CONFIG --snapshot DIR --out DIR [--set KEY=VALUE ...]
    python -m features.storelab.cli plan|run CONFIG [--set KEY=VALUE ...] [--snapshot DIR] [--root DIR]
    python -m features.storelab.cli status|cancel RUN [--root DIR]
    python -m features.storelab.cli resume RUN [--max-usd USD] [--root DIR]
    python -m features.storelab.cli export RUN [--format parquet|csv|jsonl] [--root DIR]
    python -m features.storelab.cli replay RUN [--root DIR] [--cache-root DIR]
    python -m features.storelab.cli cache list|inspect|verify|purge|export [KEY] [--namespace N] [--provider P]
        [--model M] [--before YYYY-MM-DD] [--reason TEXT] [--out TAR] [--cache-root DIR]

``validate`` prints the resolved config (every default written out) on stdout and
its config_hash on stderr; it exits 2 on an invalid config. ``schemas`` regenerates
the JSON Schemas in contracts/storelab/, or with ``--check`` fails if they are stale.
``e1-stimuli`` builds the E1 tasks, designs and design report (e1_stimuli.py); it
exits 2 on bad input and 1 if the label rate is outside its 99% bound.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from .config_files import dump_config, load_config, parse_set
from .core.agent_params import AgentParams
from .core.config import format_errors, run_config_schema
from .core.manifest import manifest_schema
from .core.protocol import ACTION, ChoiceReply, Observation
from .core.records import EventRecord, TrialRecord

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "storelab"
DRAFT = "https://json-schema.org/draft/2020-12/schema"


def _model_schema(model):
    return lambda: {"$schema": DRAFT, **model.model_json_schema()}


SCHEMAS = {
    "run_config.schema.json": run_config_schema,
    "manifest.schema.json": manifest_schema,
    "observation.schema.json": _model_schema(Observation),
    "action.schema.json": lambda: {"$schema": DRAFT, "title": "Action", **ACTION.json_schema()},
    "choice_reply.schema.json": _model_schema(ChoiceReply),
    "trial_record.schema.json": _model_schema(TrialRecord),
    "event_record.schema.json": _model_schema(EventRecord),
    "agent_params.schema.json": _model_schema(AgentParams),
}


def schema_text(name: str) -> str:
    return json.dumps(SCHEMAS[name](), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


RUNNER_COMMANDS = ("plan", "run", "status", "resume", "cancel", "export", "replay")
# Run status -> exit code of run and resume: 0 finished, 3 needs an operator, 4 stopped for good.
RUN_EXIT = {"COMPLETED": 0, "COMPLETED_WITH_GAPS": 0, "PAUSED": 3, "LIMIT_REACHED": 3, "PENDING_APPROVAL": 3,
            "QUEUED": 3, "FAILED": 4, "CANCELLED": 4}


def _print(value) -> None:
    print(json.dumps(value, indent=1, ensure_ascii=False, default=str))


def _runner(args) -> int:
    from . import api
    from .core.runner import PlanError
    from .core.store import RunLocked

    try:
        if args.command in ("plan", "run"):
            config = load_config(args.config, parse_set(args.set))
            if args.command == "plan":
                run_dir, planned = api.plan_run(config, snapshot=args.snapshot, root=args.root)
                last = planned.manifest.status_history[-1]
                _print({"run_dir": str(run_dir), "run_id": planned.manifest.run_id,
                        "mode": planned.manifest.mode.model_dump(), "trials": len(planned.trials),
                        "estimate_usd": planned.estimate_usd, "status": last.status, "reason": last.reason})
                return 0
            run_dir, result = api.run(config, snapshot=args.snapshot, root=args.root, cache_root=args.cache_root)
        else:
            run_dir = api.resolve_run_dir(args.run, args.root)
            if args.command == "replay":
                run_dir, result = api.replay(run_dir, root=args.root, cache_root=args.cache_root)
            if args.command == "status":
                _print(api.status(run_dir))
                return 0
            if args.command == "cancel":
                _print({"run_dir": str(run_dir), "cancel": api.cancel(run_dir)})
                return 0
            if args.command == "export":
                _print({"run_dir": str(run_dir), "export": str(api.export(run_dir, args.format))})
                return 0
            if args.command == "resume":
                result = api.resume(run_dir, max_usd=args.max_usd, cache_root=args.cache_root)
    except ValidationError as error:
        print("\n".join(format_errors(error)), file=sys.stderr)
        return 2
    except (PlanError, RunLocked, NotImplementedError, ImportError, OSError, ValueError, KeyError,
            yaml.YAMLError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    summary = api.status(run_dir)
    _print({"run_dir": str(run_dir), **summary})
    return RUN_EXIT.get(summary["status"], 3) if result is not None else 3


def _cache(args) -> int:
    import datetime

    from . import api
    from .core.cache import Filters

    cache = api.open_cache(args.cache_root)
    try:
        filters = Filters(namespace=args.namespace, provider=args.provider, model=args.model,
                          before=datetime.date.fromisoformat(args.before) if args.before else None)
        if args.action == "list":
            _print([{"call_key": e.call_key, "provider": e.request.provider, "model": e.request.requested_model,
                     "namespace": e.request.sample_namespace, "rep": e.request.rep,
                     "created_utc": e.created_utc.isoformat(), "run_id": e.run_id}
                    for e in cache.entries(filters)])
            return 0
        if args.action == "inspect":
            if not args.key:
                raise ValueError("inspect needs a call_key")
            entry = next((e for e in cache.entries() if e.call_key == args.key), None)
            if entry is None:
                raise KeyError("no cache entry %s" % args.key)
            _print(entry.model_dump(mode="json"))
            return 0
        if args.action == "verify":
            bad = cache.verify()
            _print({"entries": sum(1 for _ in cache.store.paths()), "tampered": bad})
            return 1 if bad else 0
        if args.action == "purge":
            _print(cache.purge(filters, args.reason or ""))
            return 0
        if args.out is None:
            raise ValueError("export needs --out")
        _print(cache.export(filters, args.reason or "", args.out))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.storelab.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate a run config and print it resolved")
    validate.add_argument("config", type=Path)
    validate.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                          help="override a field by dotted path, e.g. design.set_sizes=[2,4]")
    validate.add_argument("--format", choices=("yaml", "json"), default="yaml")
    schemas = sub.add_parser("schemas", help="write the JSON Schemas to contracts/storelab/")
    schemas.add_argument("--check", action="store_true", help="fail if a committed schema is stale")
    stimuli = sub.add_parser("e1-stimuli", help="build E1 tasks, designs and the design report from a snapshot")
    stimuli.add_argument("config", type=Path)
    stimuli.add_argument("--snapshot", type=Path, required=True, help="dataset snapshot directory")
    stimuli.add_argument("--out", type=Path, required=True, help="output directory")
    stimuli.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    cache_root = {"type": Path, "help": "response cache directory (default: artifacts/cache)"}
    for name, text in (("plan", "plan a run and write its directory"), ("run", "plan a run and execute it")):
        command = sub.add_parser(name, help=text)
        command.add_argument("config", type=Path)
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        command.add_argument("--snapshot", type=Path, help="dataset snapshot (default: the configured one under "
                                                           "artifacts/datasets)")
        command.add_argument("--root", type=Path, help="runs directory (default: outputs.root)")
        if name == "run":
            command.add_argument("--cache-root", **cache_root)
    for name, text in (("status", "summarise a run"), ("resume", "continue a paused, limited or interrupted run"),
                       ("cancel", "cancel a run"), ("export", "write the run's analysis table"),
                       ("replay", "replay a finished run from the response cache (no provider calls)")):
        command = sub.add_parser(name, help=text)
        command.add_argument("run", help="run directory, or run id under --root")
        command.add_argument("--root", type=Path, help="runs directory (default: runs)")
        if name == "resume":
            command.add_argument("--max-usd", type=float, help="raise the budget of a LIMIT_REACHED run")
        if name in ("resume", "replay"):
            command.add_argument("--cache-root", **cache_root)
        if name == "export":
            command.add_argument("--format", choices=("parquet", "csv", "jsonl"), default="parquet")
    cache = sub.add_parser("cache", help="list, inspect, verify, purge or export response-cache entries")
    cache.add_argument("action", choices=("list", "inspect", "verify", "purge", "export"))
    cache.add_argument("key", nargs="?", help="call_key (inspect)")
    cache.add_argument("--cache-root", **cache_root)
    for option in ("namespace", "provider", "model"):
        cache.add_argument("--" + option)
    cache.add_argument("--before", help="entries created before this UTC date, YYYY-MM-DD")
    cache.add_argument("--reason", help="why (required by purge and export; written to the audit log)")
    cache.add_argument("--out", type=Path, help="tarball path (export)")
    args = parser.parse_args(argv)

    if args.command in RUNNER_COMMANDS:
        return _runner(args)
    if args.command == "cache":
        return _cache(args)

    if args.command == "e1-stimuli":
        from . import e1_stimuli

        try:
            config = load_config(args.config, parse_set(args.set))
        except ValidationError as error:
            print("\n".join(format_errors(error)), file=sys.stderr)
            return 2
        except (OSError, ValueError, yaml.YAMLError) as error:
            print("error: %s" % error, file=sys.stderr)
            return 2
        try:
            built = e1_stimuli.build(config, args.snapshot)
        except (OSError, ValueError, KeyError) as error:
            print("error: %s" % error, file=sys.stderr)
            return 2
        manifest = e1_stimuli.write(built, args.out)
        print(json.dumps({k: manifest[k] for k in ("designs_hash", "observations_sha256",
                                                   "label_rate_within_bound")}))
        return 0 if manifest["label_rate_within_bound"] else 1

    if args.command == "validate":
        try:
            config = load_config(args.config, parse_set(args.set))
        except ValidationError as error:
            print("\n".join(format_errors(error)), file=sys.stderr)
            return 2
        except (OSError, ValueError, yaml.YAMLError) as error:
            print("error: %s" % error, file=sys.stderr)
            return 2
        sys.stdout.write(dump_config(config, args.format))
        print("config_hash: %s" % config.config_hash, file=sys.stderr)
        return 0

    stale = []
    for name in SCHEMAS:
        path, text = CONTRACTS / name, schema_text(name)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(str(path))
        else:
            CONTRACTS.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(path)
    if stale:
        print("stale schemas (run without --check): %s" % ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
