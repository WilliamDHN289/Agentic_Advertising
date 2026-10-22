"""Registration receipts and the E1 timing check (SL-01, SL-18).

    python -m features.storelab.analysis.registration check-e1 RUN_DIR [RUN_DIR ...] [--receipts PATH]

``docs/registrations.yaml`` records each OSF registration: status, URL and the
registration time (UTC). ``check_e1`` passes only if the E1 registration is filed and
its time is earlier than the first trial of every given run, read from the run
manifests: the first RUNNING entry of the status history, which precedes every trial.
Exit codes: 0 passed, 1 failed (including not filed yet), 2 bad input.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AwareDatetime, Field, ValidationError, model_validator

from ..core.manifest import Manifest, read_manifest
from .register import CLAIM_IDS, Strict, Text

RECEIPTS_PATH = Path(__file__).resolve().parents[1] / "docs" / "registrations.yaml"


class Receipt(Strict):
    id: Literal["E1", "E5"]
    venue: Text
    template: Text
    covers: list[Literal[CLAIM_IDS]] = Field(min_length=1)
    status: Literal["not_filed", "filed"]
    url: Text | None = None
    registered_utc: AwareDatetime | None = None
    embargo_until: datetime.date | None = None

    @model_validator(mode="after")
    def _filed_has_proof(self) -> Receipt:
        if (self.status == "filed") != (self.url is not None and self.registered_utc is not None):
            raise ValueError("%s: a filed registration has a url and registered_utc, and only it" % self.id)
        return self


class Receipts(Strict):
    receipts_version: Literal[1]
    registrations: list[Receipt]

    def get(self, registration: str) -> Receipt:
        found = [r for r in self.registrations if r.id == registration]
        if len(found) != 1:
            raise ValueError("receipts: need exactly one %s entry" % registration)
        return found[0]


def load_receipts(path: Path | None = None) -> Receipts:
    return Receipts.model_validate(yaml.safe_load(Path(path or RECEIPTS_PATH).read_text(encoding="utf-8")))


def first_trial_utc(manifest: Manifest) -> datetime.datetime | None:
    """When the run first started executing trials (its first RUNNING status)."""
    return next((e.at_utc for e in manifest.status_history if e.status == "RUNNING"), None)


def check_e1(receipts: Receipts, manifests: list[Manifest]) -> dict:
    receipt = receipts.get("E1")
    starts = [(first_trial_utc(m), m.run_id) for m in manifests]
    started = sorted((t, run) for t, run in starts if t is not None)
    first = started[0] if started else None
    out = {"registration": "E1", "status": receipt.status, "url": receipt.url,
           "registered_utc": receipt.registered_utc.isoformat() if receipt.registered_utc else None,
           "runs": len(manifests), "first_trial_utc": first[0].isoformat() if first else None,
           "first_run_id": first[1] if first else None}
    if not manifests:
        return {**out, "passed": False, "reason": "no SL-18 run manifests given"}
    if receipt.status != "filed":
        return {**out, "passed": False, "reason": "the E1 registration is not filed"}
    if first is None:
        return {**out, "passed": True, "reason": "registered; no run has started a trial yet"}
    passed = receipt.registered_utc < first[0]
    return {**out, "passed": passed, "reason": "registered before the first trial" if passed else
            "registered at or after the first trial of run %s" % first[1]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.storelab.analysis.registration")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-e1", help="E1 registered before the first trial of the given runs")
    check.add_argument("runs", nargs="*", type=Path, help="SL-18 run directories")
    check.add_argument("--receipts", type=Path, help="registration receipts (default docs/registrations.yaml)")
    args = parser.parse_args(argv)
    try:
        receipts = load_receipts(args.receipts)
        manifests = [read_manifest(Path(run) / "manifest.json") for run in args.runs]
    except (ValidationError, ValueError, OSError, yaml.YAMLError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    result = check_e1(receipts, manifests)
    print(json.dumps(result, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
