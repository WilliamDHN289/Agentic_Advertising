"""Run configs as YAML or JSON files, and ``--set key=value`` overrides (SL-02).

Outside core/ because PyYAML is not on core's import allow-list. Configs load with
YAML 1.2 booleans: only true and false are booleans, so ``policy: off`` stays the
string the schema expects instead of becoming False as in YAML 1.1.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .core.config import RunConfig, resolve_config


class _Loader(yaml.SafeLoader):
    pass


_BOOL = "tag:yaml.org,2002:bool"
_Loader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != _BOOL]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(_BOOL, re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))


def parse_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)


def parse_set(items: Iterable[str]) -> dict[str, Any]:
    """``key=value`` strings to overrides; a value is YAML ("7", "[2, 4]", "null", "off")."""
    overrides: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError("--set %r: expected KEY=VALUE" % item)
        overrides[key.strip()] = parse_yaml(raw) if raw.strip() else ""
    return overrides


def read_config_data(path: Path) -> dict[str, Any]:
    data = parse_yaml(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("%s: a run config must be a mapping" % path)
    return data


def load_config(path: Path, overrides: Mapping[str, Any] | None = None) -> RunConfig:
    """Read a YAML or JSON config, apply overrides and validate it."""
    return resolve_config(read_config_data(path), overrides)


def dump_config(config: RunConfig, fmt: str = "yaml") -> str:
    """The resolved config, every default written out, as YAML or JSON text."""
    data = config.model_dump(mode="json")
    if fmt == "json":
        return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if fmt != "yaml":
        raise ValueError("format must be yaml or json")
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
