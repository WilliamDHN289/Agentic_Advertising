"""SL-03 record tests: JSONL.gz and Parquet round trips, and the legacy event mapping.

Parquet tests need pyarrow and are skipped, with that reason, where it is not
installed; the file-path test needs a writable temporary directory:

    cd ai-agents && python -B -m unittest discover -s tests/features/storelab -t . -v
"""
from __future__ import annotations

import gzip
import importlib.util
import io
import json
import os
import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from features.storelab import legacy_baseline
from features.storelab.core.legacy_events import (FIELD_MAP, REGIME_TO_INTERFACE, event_from_legacy,
                                                  event_to_legacy, events_from_legacy, legacy_trial_key)
from features.storelab.core.records import (RECORD_SCHEMA_VERSION, EventRecord, TrialRecord, read_jsonl_gz,
                                            read_parquet, write_jsonl_gz, write_parquet)

HAVE_PYARROW = importlib.util.find_spec("pyarrow") is not None
NO_PYARROW = "pyarrow is not installed in this interpreter"
DOC = Path(__file__).resolve().parents[3] / "features" / "storelab" / "docs" / "protocol.md"
LEGACY_EVENTS = json.loads((legacy_baseline.snapshot_dir() / "logs" / "events.json").read_text(encoding="utf-8"))
T0 = datetime(2026, 10, 21, 12, 0, 0, 123456, tzinfo=timezone.utc)
KEY = "ab" * 32
RUN = "1b4e28ba-2fa1-4d2e-883f-0016d3cca427"


def trial(status: str, **outcome) -> TrialRecord:
    return TrialRecord(run_id=RUN, trial_key=KEY, config_hash="cd" * 32, task_id="task_0", set_id="set_00017",
                       presentation_id="p0", interface="browser_tool", model_key="gpt_4o_20240806",
                       temperature=None if status == "outside" else 0.1, rep=9,
                       candidate_ids=["B07ZPKBL9V", "B08N5WRWNW", "B0C1H26C46"],
                       outcome={"status": status, "backend": "openai", **outcome},
                       started_utc=T0, finished_utc=T0.replace(second=7) if status != "not_run" else None)


TRIALS = [
    trial("choice", chosen_id="B08N5WRWNW", page_requests=["B07ZPKBL9V", "B08N5WRWNW"], turns=3, attempts=2,
          usage={"input_tokens": 1234, "output_tokens": 56}, latency_s=1e-300, requested_model="gpt-4o-2024-08-06",
          returned_model="gpt-4o-2024-08-06", raw_ref="cache/openai/gpt-4o-2024-08-06/" + "e" * 64 + ".json"),
    trial("outside", turns=1, attempts=1, latency_s=0.1, usage={"input_tokens": 0}),
    trial("invalid", invalid_reason="max_turns", turns=8, attempts=1, raw_ref="réponse 中"),
    trial("refusal", attempts=1, returned_model="gpt-4o-2024-11-20"),
    trial("error", error_code="rate_limited", attempts=4, latency_s=123456789.25),
    trial("not_run", error_code="budget"),
]
EVENTS = [
    EventRecord(trial_key=KEY, seq=i, interface="browser_tool", t_s=0.5 * i, module=module, human_viewport=False,
                viewport_telemetry=True, **fields)
    for i, (module, fields) in enumerate([
        ("serp", {"kind": "impression", "product_id": "B07ZPKBL9V", "position": 1, "sponsored": True, "cpc": 0.371}),
        ("detail", {"kind": "page_request", "product_id": "B07ZPKBL9V", "turn": 0}),
        ("serp", {"kind": "click", "product_id": "B08N5WRWNW", "position": 2, "cpc": 0.0, "turn": 1}),
        ("detail", {"kind": "select", "product_id": "B08N5WRWNW", "turn": 2, "trial_label": "trial é"}),
        ("crosssell", {"kind": "cart", "product_id": "B0C1H26C46", "simulated": True}),
        ("detail", {"kind": "purchase", "product_id": "B08N5WRWNW", "simulated": True}),
        ("serp", {"kind": "leave", "turn": 3}),
        ("serp", {"kind": "examine", "product_id": "B0C1H26C46", "position": 3}),
    ])
]


def to_stream(write, records, record_type) -> io.BytesIO:
    buffer = io.BytesIO()
    write(buffer, records, record_type)
    buffer.seek(0)
    return buffer


class JsonlRoundTripTest(unittest.TestCase):
    """AC-3 (JSONL.gz): round trips preserve every field; files carry schema_version."""

    def test_round_trip_preserves_every_field(self):
        for records, record_type in ((TRIALS, TrialRecord), (EVENTS, EventRecord)):
            with self.subTest(record_type=record_type.__name__):
                back = read_jsonl_gz(to_stream(write_jsonl_gz, records, record_type), record_type)
                self.assertEqual(back, records)
                self.assertEqual([r.model_dump() for r in back], [r.model_dump() for r in records])

    def test_every_line_carries_every_field_and_schema_version(self):
        lines = gzip.decompress(to_stream(write_jsonl_gz, TRIALS, TrialRecord).getvalue()).decode().splitlines()
        self.assertEqual(len(lines), len(TRIALS))
        for line in lines:
            data = json.loads(line)
            self.assertEqual(list(data), list(TrialRecord.model_fields))
            self.assertEqual(data["schema_version"], RECORD_SCHEMA_VERSION)

    def test_output_is_byte_deterministic(self):
        first = to_stream(write_jsonl_gz, EVENTS, EventRecord).getvalue()
        self.assertEqual(to_stream(write_jsonl_gz, EVENTS, EventRecord).getvalue(), first)

    def test_other_schema_versions_and_mixed_types_are_rejected(self):
        line = json.dumps({**json.loads(EVENTS[0].model_dump_json()), "schema_version": 2})
        with self.assertRaisesRegex(ValueError, "schema_version 2"):
            read_jsonl_gz(io.BytesIO(gzip.compress(line.encode())), EventRecord)
        with self.assertRaises(TypeError):
            write_jsonl_gz(io.BytesIO(), TRIALS[:1] + EVENTS[:1], TrialRecord)

    def test_write_to_a_path_is_atomic(self):
        try:
            directory = tempfile.TemporaryDirectory()
        except OSError as exc:
            self.skipTest("no writable temporary directory: %s" % exc)
        with directory:
            path = Path(directory.name) / "part-00000.jsonl.gz"
            write_jsonl_gz(path, EVENTS, EventRecord)
            write_jsonl_gz(path, EVENTS[:2], EventRecord)
            self.assertEqual(read_jsonl_gz(path, EventRecord), EVENTS[:2])
            self.assertEqual(os.listdir(directory.name), [path.name])
            if HAVE_PYARROW:
                table = Path(directory.name) / "trials.parquet"
                write_parquet(table, TRIALS, TrialRecord)
                self.assertEqual(read_parquet(table, TrialRecord), TRIALS)


@unittest.skipUnless(HAVE_PYARROW, NO_PYARROW)
class ParquetRoundTripTest(unittest.TestCase):
    """AC-3 (Parquet): round trips preserve every field; files carry schema_version."""

    def test_round_trip_preserves_every_field(self):
        for records, record_type in ((TRIALS, TrialRecord), (EVENTS, EventRecord)):
            with self.subTest(record_type=record_type.__name__):
                back = read_parquet(to_stream(write_parquet, records, record_type), record_type)
                self.assertEqual(back, records)
                self.assertEqual([r.model_dump() for r in back], [r.model_dump() for r in records])

    def test_file_carries_every_field_and_schema_version(self):
        import pyarrow.parquet as pq

        table = pq.read_table(to_stream(write_parquet, TRIALS, TrialRecord))
        self.assertEqual(table.column_names, list(TrialRecord.model_fields))
        self.assertEqual(table.schema.metadata[b"storelab.schema_version"], b"1")
        self.assertEqual(table.schema.metadata[b"storelab.record_type"], b"trial_record")
        self.assertEqual(set(table.column("schema_version").to_pylist()), {RECORD_SCHEMA_VERSION})

    def test_wrong_record_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a event_record file"):
            read_parquet(to_stream(write_parquet, TRIALS, TrialRecord), EventRecord)

    def test_an_empty_table_keeps_its_schema(self):
        self.assertEqual(read_parquet(to_stream(write_parquet, [], EventRecord), EventRecord), [])


class RecordValidationTest(unittest.TestCase):
    def test_outcome_must_stay_within_the_candidates(self):
        with self.assertRaisesRegex(ValidationError, "chosen_id"):
            trial("choice", chosen_id="B000000000", attempts=1)
        with self.assertRaisesRegex(ValidationError, "page_requests"):
            trial("outside", page_requests=["B000000000"], attempts=1)

    def test_event_kind_rules(self):
        with self.assertRaisesRegex(ValidationError, "simulated"):
            EVENTS[5].model_validate({**EVENTS[5].model_dump(), "simulated": False})
        with self.assertRaisesRegex(ValidationError, "product_id"):
            EVENTS[6].model_validate({**EVENTS[6].model_dump(), "product_id": "B07ZPKBL9V"})
        with self.assertRaisesRegex(ValidationError, "product_id"):
            EVENTS[0].model_validate({**EVENTS[0].model_dump(), "product_id": None})


class LegacyEventMappingTest(unittest.TestCase):
    """AC-4: every legacy event field maps one-to-one (over the vendored events.json)."""

    def test_the_vendored_events_have_exactly_the_mapped_fields(self):
        self.assertEqual(len(LEGACY_EVENTS), 6584)
        for event in LEGACY_EVENTS:
            self.assertEqual(set(event), set(FIELD_MAP))

    def test_the_field_map_is_one_to_one_onto_event_record_fields(self):
        self.assertEqual(len(set(FIELD_MAP.values())), len(FIELD_MAP))
        self.assertLessEqual(set(FIELD_MAP.values()), set(EventRecord.model_fields))
        self.assertEqual(len(set(REGIME_TO_INTERFACE.values())), len(REGIME_TO_INTERFACE))
        self.assertEqual({e["regime"] for e in LEGACY_EVENTS}, set(REGIME_TO_INTERFACE))

    def test_every_event_converts_and_converts_back_unchanged(self):
        records = events_from_legacy(LEGACY_EVENTS)
        back = [event_to_legacy(r) for r in records]
        # json.dumps also compares types: True and 1, or 0 and 0.0, would differ.
        self.assertEqual(json.dumps(back), json.dumps(LEGACY_EVENTS))
        for event, record in zip(LEGACY_EVENTS, records):
            for legacy_field, v2_field in FIELD_MAP.items():
                if legacy_field != "regime":
                    self.assertEqual(getattr(record, v2_field), event[legacy_field])

    def test_derived_fields(self):
        records = events_from_legacy(LEGACY_EVENTS)
        episodes = {}
        for record in records:
            episodes.setdefault(record.trial_label, []).append(record)
        self.assertEqual(len(episodes), 520)
        self.assertEqual(len({r.trial_key for r in records}), 520)
        for label, items in episodes.items():
            self.assertEqual([r.seq for r in items], list(range(len(items))))
            self.assertEqual({r.trial_key for r in items}, {legacy_trial_key(label)})
        self.assertEqual([r.kind for r in records if r.simulated], [e["kind"] for e in LEGACY_EVENTS if e["kind"] == "purchase"])

    def test_converted_events_round_trip_through_both_formats(self):
        records = events_from_legacy(LEGACY_EVENTS)
        self.assertEqual(read_jsonl_gz(to_stream(write_jsonl_gz, records, EventRecord), EventRecord), records)
        if HAVE_PYARROW:
            self.assertEqual(read_parquet(to_stream(write_parquet, records, EventRecord), EventRecord), records)

    def test_malformed_legacy_events_are_rejected(self):
        event = dict(LEGACY_EVENTS[0])
        with self.assertRaises(ValueError):
            event_from_legacy({**event, "extra": 1}, 0)
        del event["cpc"]
        with self.assertRaises(ValueError):
            event_from_legacy(event, 0)

    def test_the_documented_tables_match_the_code(self):
        text = DOC.read_text(encoding="utf-8")
        rows = dict(re.findall(r"^\| `(\w+)` \| `(\w+)` \|$", text, flags=re.M))
        self.assertEqual({k: v for k, v in rows.items() if k in FIELD_MAP}, FIELD_MAP)
        self.assertEqual({k: v for k, v in rows.items() if k in REGIME_TO_INTERFACE}, REGIME_TO_INTERFACE)


if __name__ == "__main__":
    unittest.main()
