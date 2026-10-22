# StoreLab behaviour protocol and result format

Protocol version `storelab-protocol@1`, record `schema_version` 1 (SL-03). Code:
`core/protocol.py` (types and parser), `core/records.py` (records and file formats),
`core/legacy_events.py` (original events). JSON Schemas: `contracts/storelab/`
(`observation`, `action`, `choice_reply`, `trial_record`, `event_record`), regenerated
with `python -m features.storelab.cli schemas`.

## One protocol for every choice source

A simulated agent, a human model, a local LLM, a provider API and a replay all receive
the same `Observation` and answer with `Action` values. Only the choice source differs
between modes (S2).

**Observation**: the `task` (id and instruction), the `candidates` (`product_id`,
1-based `position`, `sponsored_label`), and `views`, the rendered text per interface.
Ids are unique and positions run 1..K. `observation_hash()` is the sha256 of its
canonical JSON, views included, so two modes can be checked to have shown the same
thing.

**Actions**

| Action | Meaning |
|---|---|
| `RequestPage(product_id)` | open a product's detail page (browser interface) |
| `Select(product_id)` | the effective choice: a confirmed selection or add-to-cart (`v1.tex` l.316) |
| `Leave` | the outside option |
| `Invalid(reason)` | the reply could not be read as one of the above |

**TrialOutcome**: `status`, `chosen_id`, `page_requests`, `turns`, `attempts`,
`error_code`, `invalid_reason`, `usage`, `latency_s`, `backend`, `requested_model`,
`returned_model`, `raw_ref`. Every status other than `choice` carries its explanation,
and validation enforces it (S3):

| Status | Required | Forbidden |
|---|---|---|
| `choice` | `chosen_id` | `error_code`, `invalid_reason` |
| `outside` | | `chosen_id`, `error_code`, `invalid_reason` |
| `invalid` | `invalid_reason` | `chosen_id`, `error_code` |
| `refusal` | | `chosen_id`, `error_code`, `invalid_reason` |
| `error` | `error_code`: `rate_limited`, `provider_unavailable`, `bad_request`, `auth`, `model_unavailable`, `cache_miss` | `chosen_id`, `invalid_reason` |
| `not_run` | `error_code`: `budget` or `kill_switch`; `attempts` = 0 | `chosen_id`, `invalid_reason` |

A `TrialRecord` also requires `chosen_id` and every page request to be among its
`candidate_ids`. Refusals are detected by the adapters (SL-09) from provider refusal
signals; the parser never infers one from text.

## Parser (text and structured interfaces)

The reply asked for is `{"choice": "<id>" | null, "reason": "<=20 words"}`
(`choice_reply.schema.json`). `parse_reply(text, allowed_ids)` applies these rules in
order; the first that fires decides.

1. Empty or whitespace-only output → `Invalid(empty)`.
2. The top-level JSON values are extracted; prose, Markdown fences and reasoning traces
   between them are ignored. Decoding starts at each `{` or `[` that is not inside an
   earlier value. The contents of a JSON array are never replies. When a bracketed span
   fails to decode, all of it is skipped, to its matching close bracket or to the end
   of the text if it never closes, so a truncated or malformed outer object cannot expose
   a reply nested inside it. Strict JSON: single quotes, `NaN`, `Infinity` and numbers
   that overflow to infinity (`1e999`) are not JSON. No top-level object: if a top-level
   array was found → `Invalid(schema_violation)`, otherwise → `Invalid(no_json)`.
3. No top-level object has a `choice` key → `Invalid(schema_violation)`. A reply nested
   under another key counts as no `choice` key.
4. Any of these → `Invalid(multiple_choices)`: an object repeats `choice` with
   different values; more than one distinct object has a `choice` key; `choice` is a
   list of more than one distinct value. Restating the identical object is tolerated.
5. Any of these → `Invalid(schema_violation)`: `choice` is a one-item list, or neither
   a string nor null; a key is repeated; a key other than `choice` and `reason` is
   present (the 2026-08 `extras` list included); `reason` is missing, null, or not a
   string of at most 20 whitespace-separated words.
6. `choice` null → `Leave`.
7. `choice` empty after stripping whitespace → `Invalid(empty)`.
8. `choice` not exactly one of `allowed_ids` after stripping surrounding whitespace
   (case-sensitive; titles, positions, annotated ids and off-list ids such as
   cross-sell items all fail) → `Invalid(unknown_id)`.
9. Otherwise → `Select(choice)`.

The parser never raises: any unexpected decoding failure is `Invalid(schema_violation)`.
There is no re-ask and no substitution. `max_turns` is the sixth invalid reason; the
browser tool loop (SL-10) sets it. The 63 golden fixtures are in
`tests/features/storelab/fixtures/parser_cases.json`. The original raw replies were not
saved (SL-00 D-07), so each fixture names the reply shape it models.

## Records and files

`TrialRecord` (one per trial): `schema_version`, `run_id`, `trial_key`, `config_hash`,
`task_id`, `set_id`, `presentation_id`, `interface`, `model_key`, `temperature` (null
means the provider default), `rep`, `candidate_ids`, `outcome`, `started_utc`,
`finished_utc`.

`EventRecord` (one per storefront event): `schema_version`, `trial_key`,
`trial_label`, `seq`, `interface`, `kind`, `product_id`, `position` (0 means not a
ranked result), `sponsored`, `cpc`, `t_s`, `module`, `human_viewport` (ground truth),
`viewport_telemetry` (observable), `simulated`, `turn`. The kinds are `impression`,
`examine`, `click`, `page_request`, `select`, `cart`, `purchase` and `leave`: a
superset of the original five. `click` is a click on a rendered element, billable when
sponsored; `select` is the effective choice. A purchase is always `simulated`, and
only `leave` has no product.

| Format | Use | Writer / reader | Where `schema_version` is |
|---|---|---|---|
| JSONL.gz | raw, append-only shards | `write_jsonl_gz` / `read_jsonl_gz` | every line |
| Parquet | analysis tables | `write_parquet` / `read_parquet` | every row, and file metadata `storelab.schema_version` with `storelab.record_type` |

Writers take a path, written atomically, or a binary stream. Gzip output uses mtime 0,
so equal records give byte-identical files. The Parquet schema is derived from the
record model, one column per field and nested models as structs, so an all-null
column keeps its type. Readers reject another record type or `schema_version`.
Parquet needs `pyarrow`, which is on core's import allow-list but not yet a pinned
ai-agents dependency (SL-03 open item).

## Legacy events

`core/legacy_events.py` converts the original events (`storelab.py` l.219–240; the
vendored `events.json` holds 6,584) to `EventRecord` and back without loss. Each legacy
field maps to exactly one v2 field:

| Legacy field | v2 field |
|---|---|
| `episode` | `trial_label` |
| `regime` | `interface` |
| `kind` | `kind` |
| `sku` | `product_id` |
| `sponsored` | `sponsored` |
| `rank` | `position` |
| `cpc` | `cpc` |
| `t` | `t_s` |
| `human_viewport` | `human_viewport` |
| `module` | `module` |
| `viewport_telemetry` | `viewport_telemetry` |

| Legacy `regime` | v2 `interface` |
|---|---|
| `human_sim` | `legacy_human_sim` |
| `chatbot` | `legacy_chatbot` |
| `gui_agent` | `legacy_gui_agent` |
| `api` | `legacy_api` |

Kinds (`impression`, `examine`, `click`, `cart`, `purchase`) and modules (`serp`,
`crosssell`, `banner`, `detail`, `feed`) keep their values. The v2-only fields are
derived: `trial_key` = sha256 of `legacy_storelab_2026_08`, a NUL byte and the
episode id; `seq` = the event's index within its episode; `simulated` = true for
purchases. In the LLM regimes the legacy `click` was also the agent's choice; it stays
a `click`, because v2 records the effective choice as a separate `select` event.
