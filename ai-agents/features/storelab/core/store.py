"""Run artifacts (SL-07): the ``ArtifactStore`` interface and ``LocalStore``.

Layout of a run directory (tickets.md section 3.7)::

    manifest.json                 resolved config, hashes, versions, mode, status history
    plan/trials.jsonl.gz          planned trials (trial_key and design coordinates)
    plan/stimuli.json             the tasks, designs and products the trials show
    records/part-NNNNN-<id>.jsonl.gz   append-only TrialRecord shards, one per session
    export/trials.<format>        analysis tables (one row per trial_key)
    calls.jsonl                   local governance call log
    .lock                         pid of the process executing the run

A shard is a sequence of gzip members, each a batch of complete JSON lines appended
and fsynced together, so a process killed at any instant leaves complete members
plus at most one truncated member, which readers skip; its trials are executed again
on resume. Records are deduplicated on ``trial_key``: a trial with a final record
(any status but ``not_run``) is never recorded again, and a final record supersedes
a ``not_run`` one (a trial blocked by the budget or the kill switch runs on resume).
"""
from __future__ import annotations

import errno
import gzip
import json
import os
import uuid
import zlib
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from .manifest import Manifest, read_manifest, write_manifest
from .records import TrialRecord, _check_version


def effective(records: Iterable[TrialRecord]) -> dict[str, TrialRecord]:
    """One record per trial_key: the first final record, else the last not_run."""
    out: dict[str, TrialRecord] = {}
    for record in records:
        held = out.get(record.trial_key)
        if held is None or held.outcome.status == "not_run":
            out[record.trial_key] = record
    return out


def gzip_members(data: bytes) -> Iterator[bytes]:
    """The decompressed complete members of a multi-member gzip stream; a truncated or
    corrupt tail is skipped."""
    position = 0
    while position < len(data):
        inflater = zlib.decompressobj(wbits=31)
        try:
            chunk = inflater.decompress(data[position:])
        except zlib.error:
            return
        if not inflater.eof:
            return
        yield chunk
        position = len(data) - len(inflater.unused_data)


class ArtifactStore(Protocol):
    def write_manifest(self, manifest: Manifest) -> None: ...
    def read_manifest(self) -> Manifest: ...
    def write_plan(self, trials: list[dict], stimuli: dict) -> None: ...
    def read_plan(self) -> tuple[list[dict], dict]: ...
    def records(self) -> dict[str, TrialRecord]: ...
    def open_writer(self) -> RecordWriter: ...


class RecordWriter:
    """Appends records to one shard in fsynced gzip members, never twice per trial."""

    def __init__(self, path: Path, known: dict[str, TrialRecord], flush_every: int = 50):
        self.path, self.known, self.flush_every = Path(path), known, flush_every
        self._pending: list[bytes] = []
        self.written = 0

    def append(self, record: TrialRecord) -> bool:
        """Queue ``record``; False (and nothing written) if its trial already has a final record."""
        held = self.known.get(record.trial_key)
        if held is not None and held.outcome.status != "not_run":
            return False
        self.known[record.trial_key] = record
        line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self._pending.append((line + "\n").encode("utf-8"))
        self.written += 1
        if len(self._pending) >= self.flush_every:
            self.flush()
        return True

    def flush(self) -> None:
        if not self._pending:
            return
        member = gzip.compress(b"".join(self._pending), mtime=0)
        with open(self.path, "ab") as out:
            out.write(member)
            out.flush()
            os.fsync(out.fileno())
        self._pending.clear()


class RunLocked(RuntimeError):
    pass


class LocalStore:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / "manifest.json"
        self.plan_dir = self.run_dir / "plan"
        self.records_dir = self.run_dir / "records"
        self.export_dir = self.run_dir / "export"
        self.call_log = self.run_dir / "calls.jsonl"
        self.invocation_log = self.run_dir / "invocations.jsonl"  # one line per provider call (SL-09)
        self.cancel_flag = self.run_dir / "CANCEL"
        self.lock_path = self.run_dir / ".lock"

    # manifest and plan
    def write_manifest(self, manifest: Manifest) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(manifest, self.manifest_path)

    def read_manifest(self) -> Manifest:
        return read_manifest(self.manifest_path)

    def write_plan(self, trials: list[dict], stimuli: dict) -> None:
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        lines = "".join(json.dumps(t, ensure_ascii=False, separators=(",", ":")) + "\n" for t in trials)
        _atomic(self.plan_dir / "trials.jsonl.gz", gzip.compress(lines.encode("utf-8"), mtime=0))
        _atomic(self.plan_dir / "stimuli.json", (json.dumps(stimuli, ensure_ascii=False, sort_keys=True,
                                                            allow_nan=False) + "\n").encode("utf-8"))

    def read_plan(self) -> tuple[list[dict], dict]:
        lines = gzip.decompress((self.plan_dir / "trials.jsonl.gz").read_bytes()).decode("utf-8").splitlines()
        stimuli = json.loads((self.plan_dir / "stimuli.json").read_text(encoding="utf-8"))
        return [json.loads(line) for line in lines], stimuli

    # records
    def shards(self) -> list[Path]:
        return sorted(self.records_dir.glob("part-*.jsonl.gz")) if self.records_dir.is_dir() else []

    def iter_records(self) -> Iterator[TrialRecord]:
        for shard in self.shards():
            for member in gzip_members(shard.read_bytes()):
                for line in member.decode("utf-8").splitlines():
                    data = json.loads(line)
                    _check_version(data.get("schema_version"), "%s" % shard.name)
                    yield TrialRecord.model_validate(data)

    def records(self) -> dict[str, TrialRecord]:
        return effective(self.iter_records())

    def open_writer(self, flush_every: int = 50) -> RecordWriter:
        self.records_dir.mkdir(parents=True, exist_ok=True)
        number = len(self.shards()) + 1
        path = self.records_dir / ("part-%05d-%s.jsonl.gz" % (number, uuid.uuid4().hex[:8]))
        return RecordWriter(path, self.records(), flush_every)

    # one executing process per run
    def acquire(self) -> None:
        """Take the run lock; a lock left by a dead process is taken over."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    pid = int(self.lock_path.read_text().strip() or "0")
                except (OSError, ValueError):
                    pid = 0
                if pid and _alive(pid):
                    raise RunLocked("run %s is being executed by process %d" % (self.run_dir.name, pid))
                self.lock_path.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w") as out:
                out.write(str(os.getpid()))
            return
        raise RunLocked("could not take the lock of run %s" % self.run_dir.name)

    def release(self) -> None:
        self.lock_path.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as error:
        return error.errno == errno.EPERM
    return True


def _atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
    with open(temporary, "xb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def write_bytes(path: Path, data: bytes) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _atomic(Path(path), data)
