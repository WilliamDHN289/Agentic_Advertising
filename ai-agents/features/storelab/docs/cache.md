# Response cache and replay

SL-08. Code: `core/cache.py` (keys, entries, policies, verify, purge, export, audit),
`backends/llm.py` (`LLMBackend` and the `Provider` interface; `OllamaProvider`),
`cache_s3.py` (S3 store), `api.py` / `cli.py` (`replay`, `cache`). It makes sandbox
runs re-analysable and resumable without paying again (S4).

## Keys and entries

`call_key` = sha256 of the canonical JSON of the `LLMRequest` (tickets.md section 3.6):

| Field | Value |
|---|---|
| `provider`, `requested_model` | from the model config |
| `params` | `temperature` (null: provider default), `top_p`, `max_output_tokens`, and `seed` where the model supports one |
| `tool_schema_version` | null for `text` and `structured` (the `browser_tool` loop is SL-10) |
| `prompt_template_sha` | `template_sha256` of the interface's template |
| `messages` | the rendered view as one user message |
| `rep` | the repetition |
| `sample_namespace` | `cache.namespace` |

`rep` and the namespace make every intended repetition a separate sample: a run with
r = 10 on a cold cache makes 10 calls per presentation. The seed comes from the master
seed and the trial's coordinates, not from its `trial_key`, so a replay (whose
`config_hash`, and so trial keys, differ) builds the same requests.

An entry, stored at `<provider>/<model>/<call_key>.json`, holds the request, the raw
response (text, refusal flag, attempts, latency, returned model id, usage, raw body),
the parsed action, the adapter version, `created_utc`, the producing `run_id` and
`trial_key`, and `content_sha256` over all of it. A trial's outcome is rebuilt from
the entry's stored action, so a replay reproduces the original outcome even if the
parser changes later; `raw_ref` is `cache:<call_key>`. Only successful responses are
stored. Reading an entry that fails verification raises `TamperedEntry`, which fails
the run instead of serving the entry.

## Policies (`cache.policy`)

| Policy | Hit | Miss | Writes |
|---|---|---|---|
| `off` | not read | live call | none |
| `read_write` (sandbox default) | served | live call | the new response |
| `write_only` | not read | live call | the fresh sample, replacing the entry |
| `read_only` (replay) | served | `error/cache_miss`, no call | none |

## Replay

`python -m features.storelab.cli replay RUN [--cache-root DIR]` (or `api.replay`)
plans a new run with the source config, `mode: replay`, `source_run_id` = the source
run and `cache.policy: read_only`. A replay config may differ from its source only in
`mode`, `source_run_id` and `cache.policy`; anything else is a plan error. The replay
reuses the source run's stimuli (`plan/stimuli.json`) and calls no provider. The
manifest links the parent through `config.source_run_id` and the mode reason. A
replay of a complete source with a complete cache has the same `record_set_hash` as
the source. The hash leaves out the run identity (`run_id`, `config_hash`,
`trial_key`, which differ by design) and the volatile fields `started_utc`,
`finished_utc` and `outcome.latency_s` (`core/runner.py` `RUN_IDENTITY_FIELDS`,
`VOLATILE_FIELDS`). A missing entry gives `error/cache_miss` and the run ends
COMPLETED_WITH_GAPS.

## Management

```
python -m features.storelab.cli cache list    [--namespace N] [--provider P] [--model M] [--before YYYY-MM-DD]
python -m features.storelab.cli cache inspect CALL_KEY
python -m features.storelab.cli cache verify                      # exit 1 if any entry fails re-hashing
python -m features.storelab.cli cache purge   --reason TEXT FILTER...   # at least one filter
python -m features.storelab.cli cache export  --reason TEXT --out cache.tar.gz [FILTER...]
```

`verify` recomputes each entry's `call_key` from its request and its
`content_sha256`, and checks it is stored at its own path. `purge` and `export` need a
reason and append an audit event (event, reason, filters, entry count, user, time;
export also the tarball's path and sha256) to `<cache root>/audit.jsonl`; on the
platform the same events go to `storelab_run_event` (SL-12). The cache root defaults
to `artifacts/cache` (`--cache-root` to change it).

## Stores

`LocalCacheStore` (a directory; atomic writes) and `S3CacheStore` (keys under
`storelab/cache/`; any client with boto3's `get_object`, `put_object`,
`delete_object` and `list_objects_v2`). boto3 is not yet a StoreLab dependency; the
S3 store is tested with a fake client only.
