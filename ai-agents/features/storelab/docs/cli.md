# StoreLab runner: CLI and Python API

SL-07. Code: `core/runner.py` (mode, plan, execution, error policy, status),
`core/store.py` (run directory, record shards), `api.py` (Python API), `cli.py`.
Run from `ai-agents/`:

```
python -m features.storelab.cli plan    CONFIG [--set K=V ...] [--snapshot DIR] [--root DIR]
python -m features.storelab.cli run     CONFIG [--set K=V ...] [--snapshot DIR] [--root DIR]
python -m features.storelab.cli resume  RUN [--max-usd USD]
python -m features.storelab.cli status  RUN
python -m features.storelab.cli cancel  RUN
python -m features.storelab.cli export  RUN [--format parquet|csv|jsonl]
python -m features.storelab.cli replay  RUN [--cache-root DIR]
python -m features.storelab.cli cache   list|inspect|verify|purge|export ...   (docs/cache.md)
```

`RUN` is a run directory, or a run id under `--root` (default `runs`). `plan` and
`run` read the dataset snapshot named by `environment.dataset` from
`artifacts/datasets/` unless `--snapshot` is given. The Python API mirrors the
commands: `api.plan_run`, `api.run`, `api.resume`, `api.status`, `api.cancel`,
`api.export`, `api.replay` (and `api.execute_run`).

| Command | Does | Exit code |
|---|---|---|
| `plan` | resolve the mode, build the E1 stimuli, enumerate the trials, write `<root>/<run_id>/`; prints run id, mode, trial count, estimate, status | 0; 2 on a config or plan error |
| `run` | `plan`, then execute if the plan is QUEUED | 0 COMPLETED or COMPLETED_WITH_GAPS; 3 PAUSED, LIMIT_REACHED or awaiting approval; 4 FAILED or CANCELLED; 2 error |
| `resume` | continue a QUEUED, PAUSED, LIMIT_REACHED or interrupted run; `--max-usd` raises the budget | as `run` |
| `status` | status, counts by outcome, pending trials, cells above the missingness threshold, refusal rate, record-set hash | 0 |
| `cancel` | if a process is executing the run, ask it to stop (it finishes in-flight trials); otherwise mark the run CANCELLED | 0 |
| `export` | `export/trials.<parquet\|csv\|jsonl.gz>`: one row per planned trial with a record, in plan order | 0 |
| `replay` | a new run replaying RUN from the response cache, no provider calls (docs/cache.md) | as `run` |

## Mode resolution (once, at plan time)

| `mode` | Credentials of the configured API providers | Result |
|---|---|---|
| `offline` | any | offline, backend `agents.offline_backend` |
| `auto` | none | offline; reason "no credentials for …" |
| `auto` | all | sandbox |
| `auto` | some | plan error naming the providers without credentials |
| `sandbox` | not all | plan error; never falls back to offline |
| `replay` | any | replay of `source_run_id` |

Credentials are read from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and
`GOOGLE_API_KEY` or `GEMINI_API_KEY`. The resolved mode and its reason are in the
manifest (`mode`) and the first status event. Executing sandbox runs needs a provider
adapter per configured provider (SL-09; the API accepts `providers=`); until then the
CLI refuses them. Replay runs read the response cache (SL-08, docs/cache.md). Offline
backends: the simulated agents (SL-06) and the local LLM (SL-04, through the
response cache; text and structured interfaces only, since `browser_tool` needs the
SL-10 tool loop).

## Trials

`plan` enumerates, in design order, every candidate set × repetition × interface ×
model × temperature, so the trial count is Σ_designs |sets| × r × |interfaces| ×
Σ_models |temperatures| (90,720 for `e1_pilot`). Each trial has
`trial_key = sha256(config_hash, task_id, set_id, presentation_id, interface,
model_key, temperature, rep)`.

## Execution and the error policy

At most `budget.max_concurrency` trials are in flight, each on its own thread with
the timeout `failure_policy.trial_timeout_s`. Before each trial the governance
checks the cancel flag, the kill switch (`LOADTEST_LLM_KILL_SWITCH_FORCE=1`) and the
budget; every finished trial is appended to `calls.jsonl` with `run_id` and
`trial_key`. Run effects (tickets.md section 3.5):

| Trial outcome | Run |
|---|---|
| error/bad_request, error/auth, error/model_unavailable | FAILED |
| error/provider_unavailable or timeout, `breaker.failure_threshold` in a row | PAUSED (`breaker_open`), auto-resume after `cooldown_s`, at most `max_auto_resumes` times, then FAILED |
| error/rate_limited above `rate_limit_pause_share` of the trials finished within `rate_limit_window_s` (at least `failure_threshold` of them) | PAUSED (`rate_limit_storm`), auto-resumed as above |
| a different returned model id | outcome kept, ids in the manifest, PAUSED (`model_drift`); `resume` by an operator |
| refusal, invalid, error/cache_miss | recorded; missing in their cell |
| budget would be exceeded | `not_run`/budget, LIMIT_REACHED; `resume --max-usd` |
| kill switch | `not_run`/kill_switch, CANCELLED |
| the backend raises | FAILED; the trial stays pending |

Nothing is substituted: an outcome is recorded as the backend returned it, and a
choice outside the candidate set fails the run instead of being recorded. When
every planned trial has a final record the run is COMPLETED, or
COMPLETED_WITH_GAPS if a cell (task, K, interface, model, temperature) has more than
`missingness_threshold` of invalid, refusal or error trials, or a replay cache miss
occurred.

## Run directory, records and resume

```
<root>/<run_id>/manifest.json         config, hashes, versions, mode, status history
               plan/trials.jsonl.gz   planned trials
               plan/stimuli.json      tasks, designs and products the trials show
               records/part-*.jsonl.gz  append-only TrialRecord shards, one per session
               export/                analysis tables
               calls.jsonl            call log
```

Records are appended in gzip members of up to 50 records, each fsynced; a process
killed at any point loses at most its unflushed trials, and a truncated last member
is ignored when reading. Records are deduplicated on `trial_key`: a trial with a
final record is never recorded or executed again; a `not_run` record is superseded
when the trial later runs. Because every random draw is keyed by the trial, a run
killed and resumed any number of times produces the same record set (compare
`status` → `record_set_hash`, which leaves out the run identity `run_id`,
`config_hash` and `trial_key`, and the volatile `started_utc`, `finished_utc` and
`outcome.latency_s`). A lock
file keeps a second process from executing the same run; a stale lock left by a
killed process is taken over, and the run's last RUNNING status is followed by
STALLED and QUEUED.

Not yet: Celery workers, heartbeats and the stall reaper (SL-13), the approval flow
for PENDING_APPROVAL (SL-12), cost estimation (SL-11; offline runs cost 0), event
records (SL-10, SL-14).
