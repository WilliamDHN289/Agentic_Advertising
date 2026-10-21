"""StoreLab command line. SL-02 provides ``validate`` and ``schemas``; SL-07 adds plan,
run, status, resume, cancel and export.

    python -m features.storelab.cli validate CONFIG [--set KEY=VALUE ...] [--format yaml|json]
    python -m features.storelab.cli schemas [--check]

``validate`` prints the resolved config (every default written out) on stdout and
its config_hash on stderr; it exits 2 on an invalid config. ``schemas`` regenerates
the JSON Schemas in contracts/storelab/, or with ``--check`` fails if they are stale.
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

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts" / "storelab"
SCHEMAS = {"run_config.schema.json": run_config_schema, "manifest.schema.json": manifest_schema}


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
    args = parser.parse_args(argv)

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
