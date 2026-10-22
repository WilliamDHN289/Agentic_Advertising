"""Response cache (SL-08): content-addressed provider responses, policies and audit.

An entry is keyed by ``call_key`` (tickets.md section 3.6): the sha256 of the
canonical JSON of the ``LLMRequest`` (provider, requested model, parameters, tool
schema version, prompt template digest, rendered messages, repetition and sample
namespace). Because the repetition and the namespace are part of the key, intended
repeated samples are never served as one cached answer. An entry holds the request,
the raw response, the parsed action, usage, the returned model id, timestamps, the
adapter version and the run and trial that produced it, plus ``content_sha256`` over
everything else, so ``verify`` detects a changed request (key mismatch) or response
(digest mismatch). Only successful responses are stored; a failed call is retried
live by the next run.

Policies (``cache.policy``): ``off`` (no reads or writes), ``read_write`` (serve a
hit, else call and store; the sandbox default), ``write_only`` (always call, store
the fresh sample, replacing an older entry), ``read_only`` (replay: a miss is
``error/cache_miss`` and never a live call).

Entries live at ``<provider>/<model>/<call_key>.json`` in a ``CacheStore``: a local
directory (``artifacts/cache`` by default, tickets.md section 3.7) or S3 under
``storelab/cache/`` (``cache_s3.py``). ``purge`` and ``export`` need a reason and
append an audit event (a local JSONL log here; ``storelab_run_event`` on the
platform, SL-12).
"""
from __future__ import annotations

import datetime
import getpass
import hashlib
import io
import json
import os
import tarfile
import uuid
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol

from pydantic import Field, NonNegativeFloat, NonNegativeInt

from .config import Ident, NonEmptyStr, Section, Sha256, Uuid4, canonical_json
from .manifest import UtcDatetime
from .protocol import Action, ErrorCode, Usage

CACHE_VERSION = 1
Policy = Literal["off", "read_write", "write_only", "read_only"]
READS = ("read_write", "read_only")
WRITES = ("read_write", "write_only")


class Message(Section):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class LLMRequest(Section):
    """Everything that determines a provider call (section 3.6 call_key inputs)."""

    provider: NonEmptyStr
    requested_model: NonEmptyStr
    params: dict[str, Any]  # temperature (null: provider default), top_p, max_output_tokens, seed
    tool_schema_version: str | None = None
    prompt_template_sha: Sha256
    messages: list[Message] = Field(min_length=1)
    rep: NonNegativeInt
    sample_namespace: Ident


def call_key(request: LLMRequest) -> str:
    return hashlib.sha256(canonical_json(request.model_dump(mode="json")).encode("utf-8")).hexdigest()


def _unset(value: Any) -> bool:
    return value is None


class ProviderResponse(Section):
    """One provider call after the adapter's retries: text, or an error code."""

    text: str | None = None
    error_code: ErrorCode | None = None
    refusal: bool = False  # the provider signalled a safety refusal (SL-09)
    attempts: NonNegativeInt = 1
    latency_s: NonNegativeFloat | None = None
    returned_model: str | None = None
    usage: Usage = Field(default_factory=Usage)
    raw: Any = None  # the provider's response body as JSON, when available
    # SL-09 call details. Left out of the JSON form while unset, so entries written
    # before they existed keep their content_sha256.
    request_id: str | None = Field(default=None, exclude_if=_unset)
    system_fingerprint: str | None = Field(default=None, exclude_if=_unset)
    sent_params: dict[str, Any] | None = Field(default=None, exclude_if=_unset)  # null temperature: "provider_default"
    seed_supported: bool | None = Field(default=None, exclude_if=_unset)
    api_version: str | None = Field(default=None, exclude_if=_unset)


class CacheEntry(Section):
    cache_version: Literal[1] = CACHE_VERSION
    call_key: Sha256
    request: LLMRequest
    response: ProviderResponse
    parsed_action: Action
    adapter_version: NonEmptyStr
    created_utc: UtcDatetime
    run_id: Uuid4 | None = None
    trial_key: Sha256 | None = None
    content_sha256: Sha256

    @property
    def path(self) -> str:
        return entry_path(self.request.provider, self.request.requested_model, self.call_key)


def entry_path(provider: str, model: str, key: str) -> str:
    return "%s/%s/%s.json" % (provider, model, key)


def content_digest(data: dict) -> str:
    body = {k: v for k, v in data.items() if k != "content_sha256"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def new_entry(request: LLMRequest, response: ProviderResponse, action: Action, adapter_version: str, *,
              run_id: str | None = None, trial_key: str | None = None,
              now: datetime.datetime | None = None) -> CacheEntry:
    provisional = CacheEntry(call_key=call_key(request), request=request, response=response, parsed_action=action,
                             adapter_version=adapter_version,
                             created_utc=now or datetime.datetime.now(datetime.timezone.utc),
                             run_id=run_id, trial_key=trial_key, content_sha256="0" * 64)
    # The digest is over the JSON form, as verify() recomputes it from a stored entry.
    return provisional.model_copy(update={"content_sha256": content_digest(provisional.model_dump(mode="json"))})


def entry_problems(data: dict) -> list[str]:
    """Why a stored entry cannot be trusted: schema, key or content digest."""
    try:
        entry = CacheEntry.model_validate(data)
    except ValueError as error:
        return ["not a cache entry: %s" % str(error).splitlines()[0]]
    problems = []
    if call_key(entry.request) != entry.call_key:
        problems.append("call_key does not match the stored request")
    if content_digest(entry.model_dump(mode="json")) != entry.content_sha256:
        problems.append("content_sha256 does not match the stored content")
    return problems


# ---------------------------------------------------------------------- stores
class CacheStore(Protocol):
    def get(self, path: str) -> bytes | None: ...
    def put(self, path: str, data: bytes) -> None: ...
    def delete(self, path: str) -> None: ...
    def paths(self) -> Iterator[str]: ...


class LocalCacheStore:
    """A directory: ``<root>/<provider>/<model>/<call_key>.json``; writes are atomic."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def get(self, path: str) -> bytes | None:
        target = self.root / path
        return target.read_bytes() if target.is_file() else None

    def put(self, path: str, data: bytes) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(".%s.%s.tmp" % (target.name, uuid.uuid4().hex))
        with open(temporary, "xb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, target)

    def delete(self, path: str) -> None:
        (self.root / path).unlink(missing_ok=True)

    def paths(self) -> Iterator[str]:
        if self.root.is_dir():
            for path in sorted(self.root.rglob("*.json")):
                relative = path.relative_to(self.root).as_posix()
                if relative.count("/") >= 2:  # provider/model/key.json; skips the audit log and others
                    yield relative


# --------------------------------------------------------------------- audit
class AuditSink(Protocol):
    def write(self, event: dict) -> None: ...


class LocalAuditLog:
    """Append-only JSONL audit log (the CLI's; storelab_run_event on the platform)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, event: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as out:
            out.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())

    def events(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------- cache
class Filters(Section):
    namespace: str | None = None
    provider: str | None = None
    model: str | None = None
    before: datetime.date | None = None  # entries created before this UTC date

    def match(self, entry: CacheEntry) -> bool:
        return ((self.namespace is None or entry.request.sample_namespace == self.namespace)
                and (self.provider is None or entry.request.provider == self.provider)
                and (self.model is None or entry.request.requested_model == self.model)
                and (self.before is None or entry.created_utc.date() < self.before))

    def is_empty(self) -> bool:
        return all(v is None for v in self.model_dump().values())


class TamperedEntry(ValueError):
    pass


class ResponseCache:
    def __init__(self, store: CacheStore, audit: AuditSink | None = None):
        self.store, self.audit = store, audit

    def get(self, provider: str, model: str, key: str) -> CacheEntry | None:
        """The entry for ``key``; TamperedEntry if it fails verification."""
        data = self.store.get(entry_path(provider, model, key))
        if data is None:
            return None
        raw = json.loads(data)
        problems = entry_problems(raw)
        if problems or raw.get("call_key") != key:
            raise TamperedEntry("cache entry %s: %s" % (key, "; ".join(problems) or "stored under another key"))
        return CacheEntry.model_validate(raw)

    def put(self, entry: CacheEntry) -> None:
        text = json.dumps(entry.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=1) + "\n"
        self.store.put(entry.path, text.encode("utf-8"))

    def entries(self, filters: Filters | None = None) -> Iterator[CacheEntry]:
        for path in self.store.paths():
            data = self.store.get(path)
            if data is None:
                continue
            try:
                entry = CacheEntry.model_validate_json(data)
            except ValueError:
                continue  # verify() reports it
            if filters is None or filters.match(entry):
                yield entry

    def verify(self) -> dict[str, list[str]]:
        """path -> problems, for every stored entry that fails re-hashing."""
        bad = {}
        for path in self.store.paths():
            data = self.store.get(path)
            try:
                problems = entry_problems(json.loads(data))
            except ValueError as error:
                problems = ["unreadable JSON: %s" % error]
            if not problems:
                entry = CacheEntry.model_validate_json(data)
                if path != entry.path:
                    problems = ["stored at %s, expected %s" % (path, entry.path)]
            if problems:
                bad[path] = problems
        return bad

    def _audit(self, action: str, reason: str, filters: Filters, paths: list[str], **extra) -> dict:
        event = {"event": "cache_%s" % action, "reason": reason, "filters": filters.model_dump(mode="json"),
                 "entries": len(paths), "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "user": _user(), **extra}
        if self.audit is not None:
            self.audit.write(event)
        return event

    def purge(self, filters: Filters, reason: str) -> dict:
        """Delete the matching entries; needs a reason and at least one filter; audited."""
        if not reason or not reason.strip():
            raise ValueError("purge needs a reason")
        if filters.is_empty():
            raise ValueError("purge needs a filter (namespace, provider, model or before)")
        paths = [e.path for e in self.entries(filters)]
        event = self._audit("purge", reason, filters, paths)
        for path in paths:
            self.store.delete(path)
        return event

    def export(self, filters: Filters, reason: str, target: Path) -> dict:
        """Write the matching entries to a tar.gz (for a release); needs a reason; audited."""
        if not reason or not reason.strip():
            raise ValueError("export needs a reason")
        paths = sorted(e.path for e in self.entries(filters))
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for path in paths:
                data = self.store.get(path)
                info = tarfile.TarInfo("cache/" + path)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(buffer.getvalue())
        return self._audit("export", reason, filters, paths, target=str(target),
                           sha256=hashlib.sha256(buffer.getvalue()).hexdigest())


def _user() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return "unknown"
