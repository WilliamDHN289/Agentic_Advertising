"""Run manifest (SL-02): everything needed to trace a run back to what produced it.

It records the resolved config with its config_hash and the run_id; the requested
and resolved mode with the reason; git SHA and dirty flag, interpreter and package
versions; env_version, prompt template ids with sha256 and protocol_version; the
dataset snapshot; per model the requested id, observed returned ids, first and last
access, temperature semantics and seed support; cache policy and namespace; the
governance kind; and the status history. ``manifest_schema()`` is the published
contract, ``contracts/storelab/manifest.schema.json``.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping

from pydantic import AfterValidator, AwareDatetime, Field, StringConstraints, model_validator

from .config import (DatasetRef, Ident, NonEmptyStr, RunConfig, Section, Sha256, TemplateId, Uuid4,
                     compute_config_hash)
from .model_registry import rule_for
from .versioning import CORE_DIR, env_version, git_state, package_versions, python_info, text_sha256

RunStatus = Literal["PLANNED", "PENDING_APPROVAL", "QUEUED", "RUNNING", "PAUSED", "STALLED", "LIMIT_REACHED",
                    "CANCELLED", "FAILED", "COMPLETED", "COMPLETED_WITH_GAPS"]
# Requested mode -> modes it may resolve to (tickets.md section 3.4).
RESOLVES_TO = {"auto": ("offline", "sandbox"), "offline": ("offline",), "sandbox": ("sandbox",),
               "replay": ("replay",)}


def _utc(value: datetime) -> datetime:
    if value.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("timestamps must be in UTC")
    return value


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_utc)]


class ModeRecord(Section):
    requested: Literal["auto", "offline", "sandbox", "replay"]
    resolved: Literal["offline", "sandbox", "replay"]
    reason: NonEmptyStr


class GitRecord(Section):
    sha: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")] | None
    dirty: bool | None


class PythonRecord(Section):
    version: NonEmptyStr
    implementation: NonEmptyStr
    platform: NonEmptyStr


class VersionsRecord(Section):
    git: GitRecord
    python: PythonRecord
    packages: dict[str, str | None]  # null: not installed
    env_version: NonEmptyStr
    protocol_version: NonEmptyStr


class PromptTemplateRecord(Section):
    id: TemplateId
    sha256: Sha256


class ModelRecord(Section):
    key: Ident
    provider: NonEmptyStr
    requested_id: NonEmptyStr
    returned_ids: list[NonEmptyStr] = []
    first_access_utc: UtcDatetime | None = None
    last_access_utc: UtcDatetime | None = None
    temperatures: list[Literal["provider_default"] | float] = Field(min_length=1)
    seed_supported: bool
    digest: str | None = None  # local model digest (ollama /api/show), filled by SL-04


class CacheRecord(Section):
    policy: Literal["off", "read_write", "write_only", "read_only"]
    namespace: Ident


class StatusEvent(Section):
    status: RunStatus
    at_utc: UtcDatetime
    reason: str | None = None


class Manifest(Section):
    manifest_version: Literal[1] = 1
    run_id: Uuid4
    created_utc: UtcDatetime
    config: RunConfig
    config_hash: Sha256
    overrides: dict[str, Any] = {}
    mode: ModeRecord
    versions: VersionsRecord
    prompt_templates: list[PromptTemplateRecord]
    dataset_snapshot: DatasetRef | None
    models: list[ModelRecord]
    cache: CacheRecord
    governance: Literal["local", "platform"]
    status_history: list[StatusEvent] = Field(min_length=1)

    @model_validator(mode="after")
    def _agrees_with_config(self) -> Manifest:
        cfg, problems = self.config, []
        if self.config_hash != compute_config_hash(cfg):
            problems.append("config_hash: does not match config")
        if self.mode.requested != cfg.mode:
            problems.append("mode.requested: %s differs from config.mode %s" % (self.mode.requested, cfg.mode))
        if self.mode.resolved not in RESOLVES_TO[self.mode.requested]:
            problems.append("mode.resolved: %s cannot result from mode %s" % (self.mode.resolved, self.mode.requested))
        if [(m.key, m.provider, m.requested_id) for m in self.models] != \
                [(m.key, m.provider, m.model) for m in cfg.agents.models]:
            problems.append("models: do not match config.agents.models")
        if (self.cache.policy, self.cache.namespace) != (cfg.cache.policy, cfg.cache.namespace):
            problems.append("cache: does not match config.cache")
        if self.dataset_snapshot != cfg.environment.dataset:
            problems.append("dataset_snapshot: does not match config.environment.dataset")
        if sorted(t.id for t in self.prompt_templates) != sorted(cfg.protocol.prompt_templates.values()):
            problems.append("prompt_templates: ids do not match config.protocol.prompt_templates")
        if self.status_history[0].status != "PLANNED":
            problems.append("status_history: must start with PLANNED")
        for m in self.models:
            if (m.first_access_utc is None) != (m.last_access_utc is None) or (
                    m.first_access_utc is not None and m.first_access_utc > m.last_access_utc):
                problems.append("models.%s: first_access_utc and last_access_utc are inconsistent" % m.key)
        if problems:
            raise ValueError("; ".join(problems))
        return self


def new_manifest(config: RunConfig, *, resolved_mode: str, mode_reason: str, governance: str,
                 protocol_version: str, prompt_templates: Mapping[str, str] | None = None,
                 overrides: Mapping[str, Any] | None = None, run_id: str | None = None,
                 now: datetime | None = None, repo_dir: Path = CORE_DIR) -> Manifest:
    """A PLANNED manifest for ``config``; ``prompt_templates`` maps template id to its text."""
    now = now or datetime.now(timezone.utc)
    models = [ModelRecord(key=m.key, provider=m.provider, requested_id=m.model,
                          temperatures=["provider_default" if t is None else t for t in m.temperatures],
                          seed_supported=rule_for(m.provider, m.model).seed_supported)
              for m in config.agents.models]
    return Manifest(
        run_id=run_id or str(uuid.uuid4()),
        created_utc=now,
        config=config,
        config_hash=config.config_hash,
        overrides=dict(overrides or {}),
        mode=ModeRecord(requested=config.mode, resolved=resolved_mode, reason=mode_reason),
        versions=VersionsRecord(git=GitRecord(**git_state(repo_dir)), python=PythonRecord(**python_info()),
                                packages=package_versions(), env_version=env_version(),
                                protocol_version=protocol_version),
        prompt_templates=[PromptTemplateRecord(id=template_id, sha256=text_sha256(text))
                          for template_id, text in sorted((prompt_templates or {}).items())],
        dataset_snapshot=config.environment.dataset,
        models=models,
        cache=CacheRecord(policy=config.cache.policy, namespace=config.cache.namespace),
        governance=governance,
        status_history=[StatusEvent(status="PLANNED", at_utc=now, reason=mode_reason)],
    )


def record_model_access(manifest: Manifest, key: str, returned_id: str, at: datetime) -> None:
    """Note one provider response: its returned model id and when it arrived."""
    at = _utc(at)
    record = next((m for m in manifest.models if m.key == key), None)
    if record is None:
        raise KeyError("no model %r in the manifest" % key)
    if returned_id not in record.returned_ids:
        record.returned_ids.append(returned_id)
    if record.first_access_utc is None or at < record.first_access_utc:
        record.first_access_utc = at
    if record.last_access_utc is None or at > record.last_access_utc:
        record.last_access_utc = at


def append_status(manifest: Manifest, status: str, at: datetime, reason: str | None = None) -> None:
    manifest.status_history.append(StatusEvent(status=status, at_utc=at, reason=reason))


def manifest_json(manifest: Manifest) -> str:
    return json.dumps(manifest.model_dump(mode="json"), indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def write_manifest(manifest: Manifest, path: Path) -> Path:
    """Write ``manifest`` as JSON, atomically: readers see the old file or the new one."""
    path = Path(path)
    text = manifest_json(Manifest.model_validate(manifest.model_dump()))
    # open(..., "x") honours the umask like a plain write; mkstemp would leave the file 0600.
    temporary = path.with_name(".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
    try:
        with open(temporary, "x", encoding="utf-8") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return path


def read_manifest(path: Path) -> Manifest:
    return Manifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def manifest_schema() -> dict[str, Any]:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **Manifest.model_json_schema()}
