"""StoreLab command line. SL-02 provides ``validate`` and ``schemas``, SL-05
``e1-stimuli``; SL-07 adds plan, run, status, resume, cancel and export.

    python -m features.storelab.cli validate CONFIG [--set KEY=VALUE ...] [--format yaml|json]
    python -m features.storelab.cli schemas [--check]
    python -m features.storelab.cli e1-stimuli CONFIG --snapshot DIR --out DIR [--set KEY=VALUE ...]

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
}


def schema_text(name: str) -> str:
    return json.dumps(SCHEMAS[name](), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


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
    args = parser.parse_args(argv)

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
