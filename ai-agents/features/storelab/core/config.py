"""Run configuration: one language for every mode (SL-02).

A run is fully specified by (config, seed, code version). ``RunConfig`` is the
resolved form of a config: overrides applied, defaults filled in, cross-field rules
checked. Every section rejects unknown fields.

``config_hash`` is the sha256 of the canonical JSON of the trial-affecting fields:
every field except those under HASH_EXCLUDED (``outputs``, ``budget``,
``failure_policy.breaker``). Canonical JSON has sorted keys, no whitespace and UTF-8
text; floats use Python's shortest round-trip repr with -0.0 folded to 0.0, and
NaN and infinities are rejected at validation.

core/ may import only the standard library and numpy, scipy, pandas, pyarrow,
pydantic and cvxpy, so YAML I/O and ``--set`` string parsing live in
``features/storelab/config_files.py``; ``apply_overrides`` is the typed API.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Annotated, Any, Literal, Mapping

from pydantic import (BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat,
                      PositiveInt, StringConstraints, ValidationError, model_validator)

from .model_registry import API_PROVIDERS, is_pinned, rule_for

LEGACY_EXPERIMENT = "legacy_storelab_2026_08"
HASH_EXCLUDED = ("outputs", "budget", "failure_policy.breaker")

Ident = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_]*$", max_length=64)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
TemplateId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_]*@[1-9][0-9]*$")]
Uuid4 = Annotated[str, StringConstraints(
    pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]

Interface = Literal["text", "structured", "browser_tool", "legacy_chatbot", "legacy_gui_agent", "legacy_api"]
LEGACY_INTERFACES = ("legacy_chatbot", "legacy_gui_agent", "legacy_api")
Mechanism = Literal["M1", "M2", "M3_set_blind", "M3", "M4", "profit_rule"]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DatasetRef(Section):
    name: Ident
    sha256: Sha256


class FileRef(Section):
    path: NonEmptyStr
    sha256: Sha256


class LegacyEnvironment(Section):
    """Constants of the 2026-08-14 run, from the vendored legacy_2026_08/ code."""

    catalog_seed: int  # storelab.py l.73
    products_per_category: PositiveInt  # storelab.py l.73
    store_seed: int  # storelab.py l.174; also seeds the auction (l.179)
    ivt_seed: int  # storelab.py l.270
    ivt_threshold: Probability  # storelab.py l.270
    task_seed_main: int  # run_experiment.py l.52
    task_seed_secondary: int  # run_experiment.py l.197
    episodes_main: PositiveInt  # run_experiment.py l.43
    episodes_secondary: PositiveInt  # run_experiment.py l.44
    ad_positions: list[PositiveInt] = Field(min_length=1)  # storelab.py l.149
    bottom_positions: list[PositiveInt] = Field(min_length=1)  # run_experiment.py l.49


class EnvironmentConfig(Section):
    catalog: Literal["amazon_reviews_2023", "simulated_market", "legacy_synthetic"]
    dataset: DatasetRef | None = None
    categories: list[NonEmptyStr] = []
    tasks_per_category: PositiveInt | None = None
    description_chars: PositiveInt = 300
    legacy: LegacyEnvironment | None = None


class DesignConfig(Section):
    eligible_size: PositiveInt | None = None  # v: eligible products per task or context
    set_sizes: list[PositiveInt] = []  # K
    sponsored_slots: list[NonNegativeInt] = []  # K^s
    randomisation: Literal["per_set", "per_repetition"] = "per_set"  # A15
    label_rate: Probability = 0.3


class ModelConfig(Section):
    key: Ident
    provider: Literal["openai", "anthropic", "google", "ollama"]
    model: Annotated[str, StringConstraints(pattern=r"^\S+$", max_length=128)]
    # null: omit the parameter and use the provider default.
    temperatures: list[float | None] = Field(default_factory=lambda: [None], min_length=1)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_output_tokens: PositiveInt = 256

    @model_validator(mode="after")
    def _accepted_by_model(self) -> ModelConfig:
        name = "model %r (%s)" % (self.model, self.provider)
        rule = rule_for(self.provider, self.model)
        if rule is None:
            raise ValueError("%s matches no rule in core/model_registry.py" % name)
        if len(set(self.temperatures)) != len(self.temperatures):
            raise ValueError("temperatures: duplicate values for %s" % name)
        for t in self.temperatures:
            if t is None:
                continue
            if rule.temperature is None:
                raise ValueError("temperatures: %s does not accept a temperature; got %s, "
                                 "use null for the provider default" % (name, t))
            low, high = rule.temperature
            if t < low or (high is not None and t > high):
                raise ValueError("temperatures: %s accepts temperatures in [%s, %s]; got %s"
                                 % (name, low, high if high is not None else "inf", t))
        return self


class SimulatedAgents(Section):
    parameters: Literal["stipulated", "estimated"]
    params_file: FileRef | None = None  # estimated: an SL-15 agent_params.json, pinned by hash
    sigma: PositiveFloat | None = None
    label_effect: float | None = None
    position_effects: list[float] = []
    page_request_prob: Probability | None = None
    outside_value: float | None = None

    @model_validator(mode="after")
    def _one_source(self) -> SimulatedAgents:
        values = (self.sigma, self.label_effect, self.page_request_prob, self.outside_value)
        if self.parameters == "estimated":
            if self.params_file is None or any(v is not None for v in values) or self.position_effects:
                raise ValueError("estimated parameters come from params_file only; set params_file "
                                 "and leave sigma, label_effect, position_effects, page_request_prob "
                                 "and outside_value unset")
        elif self.params_file is not None or any(v is None for v in values) or not self.position_effects:
            raise ValueError("stipulated parameters need sigma, label_effect, position_effects, "
                             "page_request_prob and outside_value, and no params_file")
        return self


class AgentsConfig(Section):
    offline_backend: Literal["simulated", "local_llm"] = "simulated"
    models: list[ModelConfig] = []
    simulated: SimulatedAgents | None = None


class HumansConfig(Section):
    model: Literal["none", "position_weights", "legacy_cascade"] = "none"
    position_weights: list[Probability] = []  # A11


class MechanismConfig(Section):
    mechanisms: list[Mechanism] = Field(min_length=1)
    markets_per_configuration: PositiveInt
    agent_shares: list[Probability] = Field(min_length=1)  # alpha
    organic_regimes: list[Literal["human_optimised", "agent_aware", "gated"]] = Field(min_length=1)
    organic_variation: list[Literal["none", "randomised"]] = Field(min_length=1)
    value_distributions: list[Literal["uniform", "lognormal"]] = Field(min_length=1)
    bidding_rules: list[Literal["fixed_point_value_per_click", "human_conversion_rate"]] = Field(min_length=1)


class ProtocolConfig(Section):
    prompt_templates: dict[Interface, TemplateId] = {}
    max_turns: PositiveInt = 8
    reask_once: bool = False
    legacy_parse_fallback: Literal["none", "first_item"] = "none"


class CacheConfig(Section):
    # null resolves to read_only under mode replay and read_write otherwise.
    policy: Literal["off", "read_write", "write_only", "read_only"] | None = None
    namespace: Ident = "default"


class BreakerConfig(Section):
    failure_threshold: PositiveInt = 5
    cooldown_s: PositiveFloat = 60.0
    max_auto_resumes: NonNegativeInt = 3


class FailurePolicyConfig(Section):
    max_attempts: PositiveInt = 4
    backoff_initial_s: PositiveFloat = 1.0
    backoff_max_s: PositiveFloat = 60.0
    trial_timeout_s: PositiveFloat = 120.0
    rate_limit_window_s: PositiveFloat = 60.0
    rate_limit_pause_share: Probability = 0.5
    missingness_threshold: Probability = 0.05
    breaker: BreakerConfig = Field(default_factory=BreakerConfig)


class BudgetConfig(Section):
    max_usd: NonNegativeFloat = 0.0
    approval_usd: NonNegativeFloat | None = None
    max_concurrency: PositiveInt = 4


class OutputsConfig(Section):
    root: NonEmptyStr = "runs"
    formats: list[Literal["jsonl", "parquet", "csv"]] = ["jsonl", "parquet"]


class RunConfig(Section):
    schema_version: Literal[1] = 1
    experiment: Ident
    mode: Literal["auto", "offline", "sandbox", "replay"] = "auto"
    source_run_id: Uuid4 | None = None
    seed: Annotated[int, Field(ge=0, lt=2 ** 63)]
    environment: EnvironmentConfig
    design: DesignConfig = Field(default_factory=DesignConfig)
    interfaces: list[Interface] = []
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    humans: HumansConfig = Field(default_factory=HumansConfig)
    mechanism: MechanismConfig | None = None
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    repetitions: PositiveInt = 1
    cache: CacheConfig = Field(default_factory=CacheConfig)
    failure_policy: FailurePolicyConfig = Field(default_factory=FailurePolicyConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)

    @model_validator(mode="after")
    def _cross_field_rules(self) -> RunConfig:
        problems = _problems(self)
        if problems:
            raise ValueError("; ".join(problems))
        if self.cache.policy is None:
            policy = "read_only" if self.mode == "replay" else "read_write"
            self.cache = self.cache.model_copy(update={"policy": policy})
        return self

    @property
    def config_hash(self) -> str:
        return compute_config_hash(self)


def _duplicates(values) -> list:
    seen, repeated = set(), []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return repeated


def _problems(cfg: RunConfig) -> list[str]:
    """Cross-field rules; each message names the fields or the model involved."""
    out = []
    legacy = cfg.experiment == LEGACY_EXPERIMENT
    env, design, agents = cfg.environment, cfg.design, cfg.agents

    # The legacy experiment and its catalog, interfaces, human model and parse fallback.
    if legacy != (env.catalog == "legacy_synthetic"):
        out.append("environment.catalog: legacy_synthetic is required by, and only allowed for, "
                   "experiment %s (got experiment %s, catalog %s)" % (LEGACY_EXPERIMENT, cfg.experiment, env.catalog))
    if (env.legacy is not None) != (env.catalog == "legacy_synthetic"):
        out.append("environment.legacy: required with catalog legacy_synthetic and not allowed otherwise")
    for interface in cfg.interfaces:
        if (interface in LEGACY_INTERFACES) != legacy:
            out.append("interfaces: %s is not allowed in experiment %s" % (interface, cfg.experiment))
    if cfg.humans.model == "legacy_cascade" and not legacy:
        out.append("humans.model: legacy_cascade is allowed only for experiment %s" % LEGACY_EXPERIMENT)
    if cfg.protocol.legacy_parse_fallback != "none" and not legacy:
        out.append("protocol.legacy_parse_fallback: %s is allowed only for experiment %s (got experiment %s)"
                   % (cfg.protocol.legacy_parse_fallback, LEGACY_EXPERIMENT, cfg.experiment))

    # What each catalog needs.
    if env.catalog == "amazon_reviews_2023":
        if not env.categories:
            out.append("environment.categories: required with catalog amazon_reviews_2023")
        if env.tasks_per_category is None:
            out.append("environment.tasks_per_category: required with catalog amazon_reviews_2023")
    if (cfg.mechanism is not None) != (env.catalog == "simulated_market"):
        out.append("mechanism: required with catalog simulated_market and not allowed otherwise")
    if design.eligible_size is None or not design.set_sizes:
        out.append("design.eligible_size and design.set_sizes: required")

    # Set sizes.
    for name, values in (("design.set_sizes", design.set_sizes), ("design.sponsored_slots", design.sponsored_slots),
                         ("interfaces", cfg.interfaces)):
        if _duplicates(values):
            out.append("%s: duplicate entries %s" % (name, _duplicates(values)))
    if design.eligible_size is not None:
        for k in design.set_sizes:
            if k > design.eligible_size:
                out.append("design.set_sizes: set size %d exceeds design.eligible_size %d" % (k, design.eligible_size))
    if design.sponsored_slots and (not design.set_sizes or max(design.sponsored_slots) > min(design.set_sizes)):
        out.append("design.sponsored_slots: %s exceed the smallest set size in design.set_sizes %s"
                   % (design.sponsored_slots, design.set_sizes))
    largest = max(design.set_sizes, default=0)
    simulated = agents.simulated
    if simulated is not None and simulated.parameters == "stipulated" and len(simulated.position_effects) < largest:
        out.append("agents.simulated.position_effects: %d values do not cover set size %d"
                   % (len(simulated.position_effects), largest))
    if cfg.humans.model == "position_weights" and len(cfg.humans.position_weights) < max(largest, 1):
        out.append("humans.position_weights: %d values do not cover set size %d"
                   % (len(cfg.humans.position_weights), largest))
    if cfg.humans.model != "position_weights" and cfg.humans.position_weights:
        out.append("humans.position_weights: allowed only with humans.model position_weights")
    if cfg.mechanism is not None:
        for field, values in cfg.mechanism.model_dump().items():
            if isinstance(values, list) and _duplicates(values):
                out.append("mechanism.%s: duplicate entries %s" % (field, _duplicates(values)))

    # Prompt templates: one per interface the agents see.
    for interface in cfg.protocol.prompt_templates:
        if interface not in cfg.interfaces:
            out.append("protocol.prompt_templates: %s is not in interfaces" % interface)
    if env.catalog != "simulated_market":
        for interface in cfg.interfaces:
            if interface not in cfg.protocol.prompt_templates:
                out.append("protocol.prompt_templates: no template for interface %s" % interface)

    # Models, mode and backends.
    for key in _duplicates([m.key for m in agents.models]):
        out.append("agents.models: duplicate key %s" % key)
    for i, m in enumerate(agents.models):
        where = "agents.models[%d] (%s)" % (i, m.key)
        if m.provider in API_PROVIDERS and cfg.mode in ("sandbox", "auto") and not is_pinned(m.provider, m.model):
            out.append("%s: model %r is an alias, not a pinned dated snapshot; mode %s requires pinned snapshots"
                       % (where, m.model, cfg.mode))
        if m.provider == "ollama" and (cfg.mode == "sandbox" or agents.offline_backend != "local_llm"):
            out.append("%s: provider ollama runs only through agents.offline_backend local_llm, "
                       "never in mode sandbox" % where)
        if m.provider != "ollama" and agents.offline_backend == "local_llm":
            out.append("%s: agents.offline_backend local_llm runs ollama models only; got provider %s"
                       % (where, m.provider))
    if agents.offline_backend == "simulated" and cfg.mode in ("offline", "auto") and simulated is None:
        out.append("agents.simulated: required when agents.offline_backend is simulated in mode %s" % cfg.mode)

    # Replay.
    if cfg.mode == "replay":
        if cfg.source_run_id is None:
            out.append("source_run_id: required with mode replay")
        if cfg.cache.policy not in (None, "read_only"):
            out.append("cache.policy: mode replay requires read_only; got %s" % cfg.cache.policy)
    elif cfg.source_run_id is not None:
        out.append("source_run_id: allowed only with mode replay")
    return out


def _canonical(value: Any) -> Any:
    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def trial_affecting(config: RunConfig) -> dict[str, Any]:
    """The resolved config as JSON data, without the HASH_EXCLUDED fields."""
    data = config.model_dump(mode="json")
    for path in HASH_EXCLUDED:
        *parents, leaf = path.split(".")
        node = data
        for part in parents:
            node = node[part]
        del node[leaf]
    return data


def compute_config_hash(config: RunConfig) -> str:
    return hashlib.sha256(canonical_json(trial_affecting(config)).encode("utf-8")).hexdigest()


def apply_overrides(data: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """A deep copy of ``data`` with each dotted ``path: value`` override set, in order.

    Segments name mapping keys or, inside lists, 0-based indices. Missing or null
    mappings along the path are created. A path the schema does not know is not an
    error here; validation rejects it and names it.
    """
    result = copy.deepcopy(dict(data))
    for path, value in overrides.items():
        parts = path.split(".")
        if not all(parts):
            raise ValueError("override %r: empty path segment" % path)
        node: Any = result
        for depth, part in enumerate(parts):
            where = ".".join(parts[:depth]) or "the config root"
            if isinstance(node, list):
                if not part.isdigit() or int(part) >= len(node):
                    raise ValueError("override %r: %r is not an index of %s (length %d)" % (path, part, where, len(node)))
                key: Any = int(part)
            elif isinstance(node, dict):
                key = part
            else:
                raise ValueError("override %r: %s is not a mapping or a list" % (path, where))
            if depth == len(parts) - 1:
                node[key] = copy.deepcopy(value)
            else:
                if isinstance(node, dict) and node.get(key) is None:
                    node[key] = {}
                node = node[key]
    return result


def resolve_config(data: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> RunConfig:
    """Validate ``data`` with ``overrides`` applied; the API form of ``--set``."""
    return RunConfig.model_validate(apply_overrides(data, overrides or {}))


def format_errors(error: ValidationError) -> list[str]:
    """One ``location: message`` line per validation error."""
    return ["%s: %s" % (".".join(str(part) for part in e["loc"]) or "<config>", e["msg"]) for e in error.errors()]


def run_config_schema() -> dict[str, Any]:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **RunConfig.model_json_schema()}
