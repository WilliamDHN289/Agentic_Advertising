"""SL-02 run config tests: round trips, config_hash coverage, validation and overrides.

Read-only except one test that writes a temporary file (skipped if none is writable):

    cd ai-agents && python -B -m unittest discover -s tests/features/storelab -t . -v
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import typing
import unittest
from pathlib import Path

import jsonschema
from pydantic import BaseModel, ValidationError

from features.storelab import cli
from features.storelab.config_files import dump_config, load_config, parse_set, parse_yaml, read_config_data
from features.storelab.core.config import (HASH_EXCLUDED, RunConfig, apply_overrides, canonical_json,
                                           format_errors, resolve_config, trial_affecting)

CONFIGS = Path(cli.__file__).resolve().parent / "configs"
EXAMPLES = sorted(CONFIGS.glob("*.yaml"))
UUID_A = "1b4e28ba-2fa1-4d2e-883f-0016d3cca427"
UUID_B = "6f1a2c3d-4b5e-4f60-9a1b-2c3d4e5f6a7b"


def raw(name: str) -> dict:
    return read_config_data(CONFIGS / (name + ".yaml"))


BASE_DATA = {
    "e1": raw("e1_pilot"),
    "e2": raw("e2_core"),
    "legacy": raw("legacy_2026_08"),
    "replay": apply_overrides(raw("e1_pilot"), {"mode": "replay", "source_run_id": UUID_A, "cache.policy": "read_only"}),
    "estimated": apply_overrides(raw("smoke_offline"), {"agents.simulated": {
        "parameters": "estimated", "params_file": {"path": "agent_params.json", "sha256": "a" * 64}}}),
}
BASES = {name: resolve_config(data) for name, data in BASE_DATA.items()}


def errors_of(data: dict) -> str:
    with unittest.TestCase().assertRaises(ValidationError) as caught:
        resolve_config(data)
    return "\n".join(format_errors(caught.exception))


# --------------------------------------------------------------------- helpers
def _nested_model(annotation) -> tuple[type[BaseModel], bool] | None:
    """The BaseModel inside ``X``, ``X | None`` or ``list[X]``, and whether it is a list."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation, False
    if typing.get_origin(annotation) is list:
        inner = _nested_model(typing.get_args(annotation)[0])
        return (inner[0], True) if inner else None
    for arg in typing.get_args(annotation):
        if arg is not type(None):
            inner = _nested_model(arg)
            if inner:
                return inner
    return None


def leaf_paths(model: type[BaseModel], prefix: str = "") -> set[str]:
    """Every scalar, list-of-scalar or mapping field of ``model``; list items as index 0."""
    paths = set()
    for name, field in model.model_fields.items():
        nested = _nested_model(field.annotation)
        if nested is None:
            paths.add(prefix + name)
        else:
            paths |= leaf_paths(nested[0], prefix + name + (".0." if nested[1] else "."))
    return paths


def value_at(data, path: str):
    for part in path.split("."):
        if data is None:
            return None
        data = data[int(part)] if isinstance(data, list) else data.get(part)
    return data


def unvalidated(config: BaseModel, path: str, value) -> BaseModel:
    """``config`` with one field replaced, bypassing validation (for fields whose every
    alternative value also requires other fields to change)."""
    head, _, rest = path.partition(".")
    if not rest:
        return config.model_copy(update={head: value})
    child = getattr(config, head)
    if isinstance(child, list):
        index, _, rest = rest.partition(".")
        items = list(child)
        items[int(index)] = unvalidated(items[int(index)], rest, value)
        return config.model_copy(update={head: items})
    return config.model_copy(update={head: unvalidated(child, rest, value)})


# One row per leaf field: (base, path, new value, validate). validate=False rows
# change the field alone where the cross-field rules would also demand other changes.
ROWS = [
    ("e1", "schema_version", 2, False),
    ("e1", "experiment", "e1_pilot_variant", True),
    ("e1", "mode", "offline", True),
    ("replay", "source_run_id", UUID_B, True),
    ("e1", "seed", 7, True),
    ("e1", "environment.catalog", "simulated_market", False),
    ("legacy", "environment.dataset.name", "legacy_catalog_variant", True),
    ("legacy", "environment.dataset.sha256", "b" * 64, True),
    ("e1", "environment.categories", ["Home_and_Kitchen"], True),
    ("e1", "environment.tasks_per_category", 3, True),
    ("e1", "environment.description_chars", 200, True),
    ("legacy", "environment.legacy.catalog_seed", 1, True),
    ("legacy", "environment.legacy.products_per_category", 300, True),
    ("legacy", "environment.legacy.store_seed", 12, True),
    ("legacy", "environment.legacy.ivt_seed", 4, True),
    ("legacy", "environment.legacy.ivt_threshold", 0.5, True),
    ("legacy", "environment.legacy.task_seed_main", 100, True),
    ("legacy", "environment.legacy.task_seed_secondary", 556, True),
    ("legacy", "environment.legacy.episodes_main", 91, True),
    ("legacy", "environment.legacy.episodes_secondary", 41, True),
    ("legacy", "environment.legacy.ad_positions", [1, 2, 5, 10], True),
    ("legacy", "environment.legacy.bottom_positions", [8, 10, 11, 12], True),
    ("legacy", "environment.legacy.rng_mode", "keyed", False),
    ("e1", "design.eligible_size", 12, True),
    ("e1", "design.set_sizes", [2, 4], True),
    ("e2", "design.sponsored_slots", [2], True),
    ("e1", "design.randomisation", "per_repetition", True),
    ("e1", "design.label_rate", 0.25, True),
    ("e2", "interfaces", ["text", "structured"], True),
    ("e2", "agents.offline_backend", "local_llm", False),
    ("e1", "agents.models.0.key", "gpt_4o_main", True),
    ("e1", "agents.models.0.provider", "anthropic", False),
    ("e1", "agents.models.0.model", "gpt-4o-2024-11-20", True),
    ("e1", "agents.models.0.temperatures", [None], True),
    ("e1", "agents.models.0.top_p", 0.9, True),
    ("e1", "agents.models.0.max_output_tokens", 512, True),
    ("e1", "agents.simulated.parameters", "estimated", False),
    ("estimated", "agents.simulated.params_file.path", "agent_params_v2.json", True),
    ("estimated", "agents.simulated.params_file.sha256", "c" * 64, True),
    ("e1", "agents.simulated.sigma", 0.5, True),
    ("e1", "agents.simulated.label_effect", -0.4, True),
    ("e1", "agents.simulated.position_effects", [0.0, -0.1, -0.2, -0.3, -0.35, -0.4, -0.45, -0.55], True),
    ("e1", "agents.simulated.page_request_prob", 0.2, True),
    ("e1", "agents.simulated.outside_value", -2.0, True),
    ("e2", "humans.model", "none", False),
    ("e2", "humans.position_weights", [0.30, 0.20, 0.15, 0.10], True),
    ("e2", "mechanism.mechanisms", ["M3", "M4"], True),
    ("e2", "mechanism.markets_per_configuration", 100, True),
    ("e2", "mechanism.agent_shares", [0.0, 1.0], True),
    ("e2", "mechanism.organic_regimes", ["gated"], True),
    ("e2", "mechanism.organic_variation", ["none"], True),
    ("e2", "mechanism.value_distributions", ["lognormal"], True),
    ("e2", "mechanism.bidding_rules", ["human_conversion_rate"], True),
    ("e1", "protocol.prompt_templates",
     {"text": "e1_text@2", "structured": "e1_structured@1", "browser_tool": "e1_browser_tool@1"}, True),
    ("e1", "protocol.max_turns", 6, True),
    ("e1", "protocol.reask_once", True, True),
    ("legacy", "protocol.legacy_parse_fallback", "none", True),
    ("e1", "repetitions", 5, True),
    ("e1", "cache.policy", "write_only", True),
    ("e1", "cache.namespace", "e1_pilot_variant", True),
    ("e1", "failure_policy.max_attempts", 5, True),
    ("e1", "failure_policy.backoff_initial_s", 2.0, True),
    ("e1", "failure_policy.backoff_max_s", 30.0, True),
    ("e1", "failure_policy.trial_timeout_s", 60.0, True),
    ("e1", "failure_policy.rate_limit_window_s", 30.0, True),
    ("e1", "failure_policy.rate_limit_pause_share", 0.4, True),
    ("e1", "failure_policy.missingness_threshold", 0.1, True),
    ("e1", "failure_policy.breaker.failure_threshold", 10, True),
    ("e1", "failure_policy.breaker.cooldown_s", 120.0, True),
    ("e1", "failure_policy.breaker.max_auto_resumes", 1, True),
    ("e1", "budget.max_usd", 10.0, True),
    ("e1", "budget.approval_usd", 5.0, True),
    ("e1", "budget.max_concurrency", 2, True),
    ("e1", "budget.expected_turns", {"gpt_4o_20240806": 3.0}, True),
    ("e1", "outputs.root", "elsewhere", True),
    ("e1", "outputs.formats", ["csv"], True),
]
ALL_LEAVES = leaf_paths(RunConfig)


def excluded(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") for prefix in HASH_EXCLUDED)


# ----------------------------------------------------------------------- AC-1
class RoundTripTest(unittest.TestCase):
    """AC-1: for every example config, load -> dump -> load gives the same config_hash."""

    def test_the_example_configs_exist(self):
        self.assertEqual([p.stem for p in EXAMPLES], ["e1_pilot", "e2_core", "e2_stipulated", "legacy_2026_08",
                                                      "smoke_offline"])

    def test_load_dump_load_keeps_the_hash(self):
        for path in EXAMPLES:
            for fmt in ("yaml", "json"):
                with self.subTest(config=path.name, format=fmt):
                    first = load_config(path)
                    second = resolve_config(parse_yaml(dump_config(first, fmt)))
                    self.assertEqual(second.config_hash, first.config_hash)
                    self.assertEqual(second, first)
                    self.assertEqual(dump_config(second, fmt), dump_config(first, fmt))

    def test_round_trip_through_a_file(self):
        try:
            directory = tempfile.TemporaryDirectory()
        except OSError as exc:
            self.skipTest("no writable temporary directory: %s" % exc)
        with directory:
            for path in EXAMPLES:
                first = load_config(path)
                target = Path(directory.name) / path.name
                target.write_text(dump_config(first), encoding="utf-8")
                self.assertEqual(load_config(target).config_hash, first.config_hash, path.name)

    def test_raw_examples_validate_against_the_published_schema(self):
        schema = json.loads((cli.CONTRACTS / "run_config.schema.json").read_text(encoding="utf-8"))
        for path in EXAMPLES:
            with self.subTest(config=path.name):
                jsonschema.Draft202012Validator(schema).validate(read_config_data(path))

    def test_hash_is_the_sha256_of_canonical_json(self):
        config = BASES["e1"]
        data = config.model_dump(mode="json")
        for path in HASH_EXCLUDED:
            *parents, leaf = path.split(".")
            del (value_at(data, ".".join(parents)) if parents else data)[leaf]
        text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.assertEqual(config.config_hash, hashlib.sha256(text.encode()).hexdigest())

    def test_hash_ignores_key_order_and_number_spelling(self):
        data = raw("e1_pilot")
        reordered = dict(reversed(list(data.items())))
        respelled = apply_overrides(data, {"agents.models.0.temperatures": [None, 0], "design.label_rate": 0.30,
                                           "agents.simulated.sigma": 1})
        negative_zero = apply_overrides(data, {"agents.models.0.temperatures": [None, -0.0]})
        hashes = {resolve_config(d).config_hash for d in (data, reordered, respelled, negative_zero)}
        self.assertEqual(len(hashes), 1)
        self.assertEqual(canonical_json({"b": [1.0, -0.0], "a": "é"}), '{"a":"é","b":[1.0,0.0]}')


# ----------------------------------------------------------------------- AC-2
class HashCoverageTest(unittest.TestCase):
    """AC-2: changing any trial-affecting field changes config_hash; changing an
    excluded field does not (table-driven)."""

    def test_excluded_fields_are_the_ones_the_ticket_names(self):
        self.assertEqual(HASH_EXCLUDED, ("outputs", "budget", "failure_policy.breaker"))

    def test_the_table_covers_every_field(self):
        self.assertEqual(sorted(path for _, path, _, _ in ROWS), sorted(ALL_LEAVES))

    def test_each_field_change(self):
        for base, path, value, validate in ROWS:
            with self.subTest(field=path):
                before = BASES[base]
                after = (resolve_config(apply_overrides(BASE_DATA[base], {path: value})) if validate
                         else unvalidated(before, path, value))
                old, new = before.model_dump(mode="json"), after.model_dump(mode="json")
                changed = {leaf for leaf in ALL_LEAVES if value_at(old, leaf) != value_at(new, leaf)}
                self.assertEqual(changed, {path}, "the row must change exactly its own field")
                if excluded(path):
                    self.assertEqual(after.config_hash, before.config_hash)
                else:
                    self.assertNotEqual(after.config_hash, before.config_hash)

    def test_excluded_fields_are_absent_from_the_hashed_data(self):
        data = trial_affecting(BASES["e1"])
        self.assertNotIn("outputs", data)
        self.assertNotIn("budget", data)
        self.assertNotIn("breaker", data["failure_policy"])
        self.assertIn("max_attempts", data["failure_policy"])


# ----------------------------------------------------------------------- AC-3
class RejectionTest(unittest.TestCase):
    """AC-3: unknown fields, unpinned aliases in sandbox mode and unsupported
    temperatures are rejected with messages naming the field or model."""

    SECTIONS = [("e1", ""), ("e1", "environment."), ("legacy", "environment.dataset."),
                ("legacy", "environment.legacy."), ("e1", "design."), ("e1", "agents."),
                ("e1", "agents.models.0."), ("e1", "agents.simulated."), ("estimated", "agents.simulated.params_file."),
                ("e2", "humans."), ("e2", "mechanism."), ("e1", "protocol."), ("e1", "cache."),
                ("e1", "failure_policy."), ("e1", "failure_policy.breaker."), ("e1", "budget."), ("e1", "outputs.")]

    def test_unknown_fields_are_rejected_by_name_in_every_section(self):
        for base, prefix in self.SECTIONS:
            with self.subTest(section=prefix or "<root>"):
                message = errors_of(apply_overrides(BASE_DATA[base], {prefix + "colour": "blue"}))
                self.assertIn(prefix + "colour: Extra inputs are not permitted", message)

    def test_aliases_are_rejected_in_sandbox_and_auto_by_model(self):
        for provider_index, alias in ((0, "gpt-4o"), (2, "claude-sonnet-4-5"), (2, "claude-3-5-sonnet-latest"),
                                      (4, "gemini-2.0-flash")):
            for mode in ("sandbox", "auto"):
                with self.subTest(model=alias, mode=mode):
                    message = errors_of(apply_overrides(BASE_DATA["e1"], {
                        "mode": mode, "agents.models.%d.model" % provider_index: alias}))
                    self.assertIn("model %r is an alias, not a pinned dated snapshot" % alias, message)
                    self.assertIn("agents.models[%d]" % provider_index, message)

    def test_pinned_snapshots_pass_in_sandbox_and_aliases_pass_offline(self):
        self.assertEqual(resolve_config(apply_overrides(BASE_DATA["e1"], {"mode": "sandbox"})).mode, "sandbox")
        offline = apply_overrides(BASE_DATA["e1"], {"mode": "offline", "agents.models.0.model": "gpt-4o"})
        self.assertEqual(resolve_config(offline).agents.models[0].model, "gpt-4o")

    def test_unsupported_temperatures_are_rejected_by_model(self):
        cases = [(0, "o3-2025-04-16", [0.0], "does not accept a temperature; got 0.0"),
                 (0, "gpt-5-2025-08-07", [None, 0.0], "does not accept a temperature; got 0.0"),
                 (2, "claude-sonnet-4-20250514", [1.5], "accepts temperatures in [0.0, 1.0]; got 1.5"),
                 (4, "gemini-2.0-flash-001", [2.5], "accepts temperatures in [0.0, 2.0]; got 2.5"),
                 (0, "gpt-4o-2024-08-06", [-0.5], "accepts temperatures in [0.0, 2.0]; got -0.5")]
        for index, model, temperatures, expected in cases:
            with self.subTest(model=model, temperatures=temperatures):
                message = errors_of(apply_overrides(BASE_DATA["e1"], {
                    "agents.models.%d.model" % index: model, "agents.models.%d.temperatures" % index: temperatures}))
                self.assertIn("agents.models.%d: Value error, temperatures: model %r" % (index, model), message)
                self.assertIn(expected, message)

    def test_reasoning_models_run_at_the_provider_default(self):
        data = apply_overrides(BASE_DATA["e1"], {"agents.models.0.model": "o3-2025-04-16",
                                                 "agents.models.0.temperatures": [None]})
        self.assertEqual(resolve_config(data).agents.models[0].temperatures, [None])

    def test_models_no_rule_covers_are_rejected_by_name(self):
        message = errors_of(apply_overrides(BASE_DATA["e1"], {"agents.models.0.model": "llama-3-70b"}))
        self.assertIn("model 'llama-3-70b' (openai) matches no rule", message)


class CrossFieldRuleTest(unittest.TestCase):
    """The remaining SL-02 validation rules, and the mode and backend rules."""

    CASES = [
        ("e1", {"design.set_sizes": [2, 4, 12]}, "design.set_sizes: set size 12 exceeds design.eligible_size 10"),
        ("e1", {"protocol.legacy_parse_fallback": "first_item"},
         "protocol.legacy_parse_fallback: first_item is allowed only for experiment legacy_storelab_2026_08"),
        ("legacy", {"environment.legacy.rng_mode": "keyed"},
         "protocol.legacy_parse_fallback: first_item is allowed only with environment.legacy.rng_mode legacy_shared"),
        ("e1", {"mode": "replay", "cache.policy": None}, "source_run_id: required with mode replay"),
        ("replay", {"cache.policy": "read_write"}, "cache.policy: mode replay requires read_only"),
        ("e1", {"source_run_id": UUID_A}, "source_run_id: allowed only with mode replay"),
        ("legacy", {"mode": "sandbox"}, "provider ollama runs only through agents.offline_backend local_llm"),
        ("e1", {"agents.offline_backend": "local_llm"}, "local_llm runs ollama models only"),
        ("e1", {"interfaces": ["text", "legacy_api"], "protocol.prompt_templates.legacy_api": "legacy_api@1"},
         "interfaces: legacy_api is not allowed in experiment e1_pilot"),
        ("e1", {"humans.model": "legacy_cascade"}, "humans.model: legacy_cascade is allowed only"),
        ("e1", {"experiment": "legacy_storelab_2026_08"}, "environment.catalog: legacy_synthetic is required by"),
        ("e2", {"mechanism": None}, "mechanism: required with catalog simulated_market"),
        ("e1", {"protocol.prompt_templates": {"text": "e1_text@1", "structured": "e1_structured@1"}},
         "protocol.prompt_templates: no template for interface browser_tool"),
        ("e1", {"agents.models.1.key": "gpt_4o_20240806"}, "agents.models: duplicate key gpt_4o_20240806"),
        ("e1", {"agents.simulated": None}, "agents.simulated: required when agents.offline_backend is simulated"),
        ("e1", {"agents.simulated.position_effects": [0.0, -0.1]},
         "agents.simulated.position_effects: 2 values do not cover set size 8"),
        ("e1", {"agents.simulated.params_file": {"path": "p.json", "sha256": "a" * 64}},
         "stipulated parameters need"),
        ("e2", {"humans.position_weights": [0.3]}, "humans.position_weights: 1 values do not cover set size 4"),
        ("e2", {"design.sponsored_slots": [5]}, "design.sponsored_slots: [5] exceed the smallest set size"),
        ("e1", {"environment.categories": []}, "environment.categories: required with catalog amazon_reviews_2023"),
        ("e1", {"design.label_rate": float("nan")}, "design.label_rate"),
    ]

    def test_each_rule_rejects_with_a_named_field(self):
        for base, overrides, expected in self.CASES:
            with self.subTest(overrides=overrides):
                self.assertIn(expected, errors_of(apply_overrides(BASE_DATA[base], overrides)))

    def test_legacy_parse_fallback_is_accepted_for_the_legacy_experiment(self):
        self.assertEqual(BASES["legacy"].protocol.legacy_parse_fallback, "first_item")
        self.assertEqual(BASES["legacy"].environment.legacy.rng_mode, "legacy_shared")
        keyed = resolve_config(apply_overrides(BASE_DATA["legacy"], {
            "environment.legacy.rng_mode": "keyed", "protocol.legacy_parse_fallback": "none"}))
        self.assertEqual(keyed.environment.legacy.rng_mode, "keyed")
        default = apply_overrides(BASE_DATA["legacy"], {"protocol.legacy_parse_fallback": "none"})
        del default["environment"]["legacy"]["rng_mode"]
        self.assertEqual(resolve_config(default).environment.legacy.rng_mode, "keyed")  # keyed is the default

    def test_cache_policy_resolves_from_the_mode(self):
        unset = {"cache.policy": None}
        self.assertEqual(resolve_config(apply_overrides(BASE_DATA["e1"], unset)).cache.policy, "read_write")
        replay = apply_overrides(BASE_DATA["replay"], unset)
        self.assertEqual(resolve_config(replay).cache.policy, "read_only")


# ------------------------------------------------------------------ overrides
class OverrideTest(unittest.TestCase):
    """``--set key=value`` and its API form; overrides enter the resolved config and the hash."""

    def test_set_values_are_parsed_as_yaml_with_yaml_12_booleans(self):
        self.assertEqual(parse_set(["seed=7", "design.set_sizes=[2, 4]", "environment.dataset=null",
                                    "cache.policy=off", "protocol.reask_once=true", "experiment=e1_b", "outputs.root="]),
                         {"seed": 7, "design.set_sizes": [2, 4], "environment.dataset": None, "cache.policy": "off",
                          "protocol.reask_once": True, "experiment": "e1_b", "outputs.root": ""})
        for bad in ("seed", "=7", " =7"):
            with self.subTest(item=bad), self.assertRaises(ValueError):
                parse_set([bad])

    def test_overrides_copy_index_lists_and_create_mappings(self):
        data = raw("e1_pilot")
        before = json.dumps(data, sort_keys=True)
        out = apply_overrides(data, {"agents.models.1.top_p": 0.5, "failure_policy.breaker.cooldown_s": 5.0})
        self.assertEqual(json.dumps(data, sort_keys=True), before)
        self.assertEqual(out["agents"]["models"][1]["top_p"], 0.5)
        self.assertEqual(out["failure_policy"], {"breaker": {"cooldown_s": 5.0}})
        for path in ("agents.models.9.top_p", "agents.models.x.top_p", "seed.x", "design..set_sizes"):
            with self.subTest(path=path), self.assertRaises(ValueError) as caught:
                apply_overrides(data, {path: 1})
            self.assertIn(repr(path), str(caught.exception))

    def test_overrides_enter_the_resolved_config_and_its_hash(self):
        data = raw("e1_pilot")
        overridden = resolve_config(data, {"seed": 7})
        self.assertEqual(overridden.seed, 7)
        self.assertEqual(overridden.config_hash, resolve_config({**data, "seed": 7}).config_hash)
        self.assertNotEqual(overridden.config_hash, resolve_config(data).config_hash)
        self.assertEqual(load_config(CONFIGS / "e1_pilot.yaml", parse_set(["seed=7"])).config_hash,
                         overridden.config_hash)

    def test_an_override_naming_an_unknown_field_is_rejected_by_name(self):
        self.assertIn("design.colour: Extra inputs are not permitted",
                      errors_of(apply_overrides(raw("e1_pilot"), parse_set(["design.colour=blue"]))))

    def test_cli_validate_applies_set_and_reports_the_hash(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["validate", str(CONFIGS / "e1_pilot.yaml"), "--set", "seed=7", "--format", "json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["seed"], 7)
        self.assertEqual(err.getvalue().strip(), "config_hash: " + resolve_config(raw("e1_pilot"), {"seed": 7}).config_hash)

    def test_cli_validate_exits_2_and_names_the_field(self):
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = cli.main(["validate", str(CONFIGS / "e1_pilot.yaml"), "--set", "colour=blue"])
        self.assertEqual(code, 2)
        self.assertIn("colour: Extra inputs are not permitted", err.getvalue())


if __name__ == "__main__":
    unittest.main()
