# The original StoreLab offline: `legacy_shared` and `keyed`

`core/legacy_env.py` ports the 2026-08-14 StoreLab (`legacy_2026_08/`) so it runs
offline. Choose the mode with `environment.legacy.rng_mode`; `configs/legacy_2026_08.yaml`
uses `legacy_shared` with `legacy_parse_fallback: first_item` to reproduce the run.

## What each mode draws from

| Random draw | `legacy_shared` (reproduction) | `keyed` (default) |
|---|---|---|
| Catalog (`build_catalog`) | `Random(catalog_seed)` | same |
| Task lists (`make_tasks`) | `Random(task_seed_main)`, `Random(task_seed_secondary)` | same |
| Auction bid shading | one `Random(store_seed)` for the whole run, in search order | `py_random(seed, "legacy", "auction", task_id, arm)` per episode |
| IVT detector jitter | one `Random(ivt_seed)` for the whole run, in judging order | `py_random(seed, "legacy", "ivt", task_id, arm)` per episode |
| `HumanSim` | `Random(task seed)` per episode | `py_random(seed, "legacy", "human", task_id, arm)` |
| Chatbot timing and impulse | `Random(task seed)` per episode | `py_random(seed, "legacy", "chatbot", task_id, arm)` |
| Model call | ollama `seed` = task seed | same |

`seed` is the config's master seed, and `arm` includes the 2x2 cell. The catalog,
the tasks, ranking, renderers, prompts and metrics are the same in both modes; only
the auction, IVT, simulated-human and chatbot draws differ.

## Consequences

- **`legacy_shared` depends on order.** An episode's bids, CPCs and detector score
  depend on how many searches and judgements ran before it. Permuting the episodes,
  or resuming a run, changes every later episode (SL-00 defect D-01). Use it only to
  reproduce the original run, in the original order, uninterrupted.
- **`keyed` does not.** Each episode draws from its own streams, so any order, a
  parallel run or a resume after a crash gives the same records. The tests check
  three random orders and a run killed mid-episode and resumed (AC-2).
- **Neither mode gives every arm the same auction for a task.** In the original the
  shared generator gave each arm different bids for the same task. In `keyed` mode
  the streams are keyed by `(task_id, arm)`, as the ticket specifies, so arms still
  draw separately: 0 of 90 tasks get identical sponsored slots and CPCs in
  `human_sim` and `gui_agent`, in either mode. Pages identical across arms would need
  the auction keyed by task alone. That is an experiment-design decision left open
  (SL-04 open item).
- **The legacy parse fallback needs `legacy_shared`.** `first_item` reproduces the
  original substitution of the first listed item for an unusable reply, and is
  allowed only with `experiment: legacy_storelab_2026_08` and `rng_mode:
  legacy_shared`. Each substituted episode records `substituted: true`, and the
  manifest is flagged `contains_substituted_outcomes`. Without the fallback, such an
  episode is `invalid` and buys nothing.
- **A failed model call is always an error.** In both modes and under either
  fallback, it ends the episode with `status: error` and its code (for example
  `provider_unavailable` when ollama is stopped), never a substitution. The original
  returned `""` here and substituted.

## Reproducing the 2026-08-14 run

In `legacy_shared` mode, with each LLM episode's saved choice and extras replayed,
the port reproduces the saved data:

- all 520 rows of `episodes.json`, byte for byte;
- all 520 saved IVT scores and filter flags;
- `events.json` with every field identical except that, on arm64, some event
  timestamps (`t`) differ in the last bit (below 1e-14 relative).

The timestamp difference comes from the platform's maths library behind
`random.gauss`, not from the port. The vendored original shows the same pattern:
CPython 3.9.13 (Anaconda x86_64, the original run's interpreter) and CPython 3.11.1
run as x86_64 both reproduce the saved events exactly, while 3.11.1 on arm64 differs
in 21 of the 90 `human_sim` episodes. The metrics rows round these values, so they
match on both. The port and the vendored original are byte-identical to each other on
the same platform (AC-1; `tests/features/storelab/legacy_equivalence.py`); this was
verified on arm64. Running the port itself as x86_64 needs x86_64 builds of its
dependencies, which this machine does not have; there the test asserts exact equality
with `events.json`.

The LLM arms cannot be re-run: the run's raw replies were not saved (SL-00 D-07), and
ollama generation is not reproducible. The port keeps each new episode's raw reply.
