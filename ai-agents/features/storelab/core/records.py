"""Trial and event records (SL-03) and their two file formats.

``TrialRecord`` is one per trial: its design coordinates and its ``TrialOutcome``.
``EventRecord`` is one storefront event; the kinds are a superset of the original
StoreLab events, and ``human_viewport`` (ground truth) is kept apart from
``viewport_telemetry`` (what a platform observes), as in storelab.py l.219-240.

Raw shards are JSONL.gz and analysis tables Parquet. Every record carries
``schema_version``; a Parquet file also carries it, with the record type, in its
metadata. Writers take a path (written atomically) or a binary stream, and gzip
output is byte-deterministic (mtime 0), so equal records give equal files.
"""
from __future__ import annotations

import datetime
import gzip
import io
import json
import os
import types
import typing
import uuid
from pathlib import Path
from typing import IO, Iterable, Literal, Union

from pydantic import AwareDatetime, BaseModel, Field, NonNegativeFloat, NonNegativeInt, model_validator

from .config import Ident, Interface, NonEmptyStr, Section, Sha256, Uuid4
from .manifest import UtcDatetime
from .protocol import ProductId, TrialOutcome

RECORD_SCHEMA_VERSION = 1

EventKind = Literal["impression", "examine", "click", "page_request", "select", "cart", "purchase", "leave"]
Module = Literal["serp", "crosssell", "banner", "detail", "feed"]
# The surface the actor saw; legacy_human_sim is the original simulated-human arm.
EventInterface = Literal[Interface.__args__ + ("legacy_human_sim",)]


class TrialRecord(Section):
    schema_version: Literal[1] = RECORD_SCHEMA_VERSION
    run_id: Uuid4
    trial_key: Sha256
    config_hash: Sha256
    task_id: NonEmptyStr
    set_id: NonEmptyStr
    presentation_id: NonEmptyStr
    interface: Interface
    model_key: Ident
    temperature: float | None  # null: the provider default
    rep: NonNegativeInt
    candidate_ids: list[ProductId] = Field(min_length=1)
    outcome: TrialOutcome
    started_utc: UtcDatetime | None = None
    finished_utc: UtcDatetime | None = None

    @model_validator(mode="after")
    def _outcome_within_candidates(self) -> TrialRecord:
        ids, outcome = set(self.candidate_ids), self.outcome
        if outcome.chosen_id is not None and outcome.chosen_id not in ids:
            raise ValueError("outcome.chosen_id: %r is not a candidate" % outcome.chosen_id)
        if not set(outcome.page_requests) <= ids:
            raise ValueError("outcome.page_requests: ids outside the candidates")
        if self.started_utc and self.finished_utc and self.finished_utc < self.started_utc:
            raise ValueError("finished_utc: earlier than started_utc")
        return self


class EventRecord(Section):
    schema_version: Literal[1] = RECORD_SCHEMA_VERSION
    trial_key: Sha256
    trial_label: str | None = None  # readable trial name; the episode id for legacy events
    seq: NonNegativeInt  # order within the trial
    interface: EventInterface
    kind: EventKind
    product_id: ProductId | None = None
    position: NonNegativeInt = 0  # display position; 0 = not a ranked result
    sponsored: bool = False
    cpc: NonNegativeFloat = 0.0  # USD charged for a click on this sponsored slot
    t_s: NonNegativeFloat  # seconds since the trial started
    module: Module
    human_viewport: bool  # ground truth: did a human eye see it?
    viewport_telemetry: bool  # observable: did the client emit viewport events?
    simulated: bool = False  # the event did not happen in a real store
    turn: NonNegativeInt | None = None  # browser_tool turn index

    @model_validator(mode="after")
    def _kind_rules(self) -> EventRecord:
        if self.kind == "purchase" and not self.simulated:
            raise ValueError("simulated: a purchase is always simulated")
        if (self.kind == "leave") != (self.product_id is None):
            raise ValueError("product_id: required for every kind except leave")
        return self


RECORD_TYPES = {"trial_record": TrialRecord, "event_record": EventRecord}


def _type_name(record_type: type[BaseModel]) -> str:
    return next(name for name, cls in RECORD_TYPES.items() if cls is record_type)


def _check(records: list[BaseModel], record_type: type[BaseModel]) -> None:
    for record in records:
        if type(record) is not record_type:
            raise TypeError("a file holds one record type; got %s among %s" % (type(record).__name__,
                                                                                record_type.__name__))


def _check_version(value: object, where: str) -> None:
    if value != RECORD_SCHEMA_VERSION:
        raise ValueError("%s: schema_version %r, expected %d" % (where, value, RECORD_SCHEMA_VERSION))


def _write(target: Path | IO[bytes], data: bytes) -> None:
    if not isinstance(target, (str, os.PathLike)):
        target.write(data)
        return
    path = Path(target)
    temporary = path.with_name(".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
    try:
        with open(temporary, "xb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _read(source: Path | IO[bytes]) -> bytes:
    return Path(source).read_bytes() if isinstance(source, (str, os.PathLike)) else source.read()


# --------------------------------------------------------------------- JSONL.gz
def write_jsonl_gz(target: Path | IO[bytes], records: Iterable[BaseModel], record_type: type[BaseModel]) -> None:
    records = list(records)
    _check(records, record_type)
    lines = "".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False, allow_nan=False,
                               separators=(",", ":")) + "\n" for r in records)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as out:
        out.write(lines.encode("utf-8"))
    _write(target, buffer.getvalue())


def read_jsonl_gz(source: Path | IO[bytes], record_type: type[BaseModel]) -> list:
    text = gzip.decompress(_read(source)).decode("utf-8")
    records = []
    for number, line in enumerate(text.splitlines(), 1):
        data = json.loads(line)
        _check_version(data.get("schema_version"), "line %d" % number)
        records.append(record_type.model_validate(data))
    return records


# ---------------------------------------------------------------------- Parquet
def _arrow_type(annotation):
    import pyarrow as pa

    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Annotated:
        return _arrow_type(args[0])
    if origin in (Union, types.UnionType):
        options = {str(_arrow_type(a)): _arrow_type(a) for a in args if a is not type(None)}
        if len(options) != 1:
            raise TypeError("no single Arrow type for %r" % annotation)
        return next(iter(options.values()))
    if origin is Literal:
        kinds = {type(a) for a in args}
        return pa.string() if kinds == {str} else pa.int64()
    if origin is list:
        return pa.list_(_arrow_type(args[0]))
    if annotation is AwareDatetime or annotation is datetime.datetime:
        return pa.timestamp("us", tz="UTC")
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return pa.struct([pa.field(name, _arrow_type(f.annotation)) for name, f in annotation.model_fields.items()])
    for python_type, arrow_type in ((bool, pa.bool_()), (int, pa.int64()), (float, pa.float64()), (str, pa.string())):
        if annotation is python_type:
            return arrow_type
    raise TypeError("no Arrow type for %r" % annotation)


def arrow_schema(record_type: type[BaseModel]):
    """The Parquet schema of ``record_type``: one column per field, record type and
    schema_version in the metadata."""
    import pyarrow as pa

    fields = [pa.field(name, _arrow_type(f.annotation)) for name, f in record_type.model_fields.items()]
    return pa.schema(fields, metadata={b"storelab.record_type": _type_name(record_type).encode(),
                                       b"storelab.schema_version": str(RECORD_SCHEMA_VERSION).encode()})


def write_parquet(target: Path | IO[bytes], records: Iterable[BaseModel], record_type: type[BaseModel]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = list(records)
    _check(records, record_type)
    table = pa.Table.from_pylist([r.model_dump() for r in records], schema=arrow_schema(record_type))
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    _write(target, buffer.getvalue())


def read_parquet(source: Path | IO[bytes], record_type: type[BaseModel]) -> list:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(pa.BufferReader(_read(source)))
    metadata = table.schema.metadata or {}
    if metadata.get(b"storelab.record_type") != _type_name(record_type).encode():
        raise ValueError("not a %s file: record type %r" % (_type_name(record_type),
                                                             metadata.get(b"storelab.record_type")))
    _check_version(int(metadata.get(b"storelab.schema_version", b"0")), "file metadata")
    return [record_type.model_validate(row) for row in table.to_pylist()]
