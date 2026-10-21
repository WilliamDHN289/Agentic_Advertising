# Defects in the original StoreLab (2026-08-14 run)

Line numbers refer to the vendored files in this directory, which are byte-identical
to `~/Desktop/agentic_amazon/5_storelab/` (see `PROVENANCE.md`). "Target" is the
StoreLab v2 ticket that removes the defect; "legacy only" means v2 does not carry
the behaviour forward and it survives only in the SL-04 `legacy_shared` reproduction.
`tests/features/storelab/test_legacy_baseline.py` checks that every location below
exists and contains the quoted code.

## Defect list

| ID | Defect | Locations | Target |
|---|---|---|---|
| D-01 | **Shared, order-dependent RNG streams.** `Auction` (one per `Store`, seeded 11 by `storelab.py:179`) and `IVTClassifier` (seed 3) each hold one `random.Random` that every episode of every arm draws from, so an episode's bids, CPCs and detector jitter depend on how many searches and judgements ran before it. Resuming makes it worse: a resumed arm skips its searches and judgements, so every later arm draws a different stream than in an uninterrupted run. Measured on the saved data: for the same task, the sponsored (rank, SKU) list differs between `human_sim` and `gui_agent` in 83 of 90 tasks and the CPCs in 90 of 90, contrary to "the catalog, ranking, auction, and prices are identical across regimes" (`agents.py` l.5–8). The saved run was uninterrupted (`run.log` has no `RESUMED` line). The ticket cites l.271 for the IVT stream; the RNG is created on l.272 (l.271 is `self.threshold = threshold`). | `storelab.py:123`, `storelab.py:131`, `storelab.py:272`, `storelab.py:325`, `run_experiment.py:165-170`, `run_experiment.py:201-205` | SL-04 |
| D-02 | **Failed LLM calls become `""`.** `LocalLLM.generate` retries 3 times, swallows every exception (connection, HTTP, JSON decoding) and returns `""`, which the regime runners treat like unparseable output (D-03). Attempts and errors are not recorded, and `n_calls` counts successful calls only. In the saved run `llm_calls` = 430 = 3 × 90 + 4 × 40 LLM episodes, so no call exhausted its retries. | `agents.py:61-73` | SL-03, SL-07 (ollama client port: SL-04) |
| D-03 | **Unparseable or unknown choices are replaced by the first listed item.** When the reply has no JSON object or names an id that is not on the page, the runner substitutes `page.items[0]` and logs it as a real choice and click; only the aggregate `n_parse_fail` counter records it. In the saved run `llm_parse_failures` = 9, so 9 of 430 LLM episodes hold a substituted choice, and `episodes.json` does not mark which. | `agents.py:212-214`, `agents.py:264-266`, `agents.py:307-309` | SL-03, SL-07 |
| D-04 | **No prompt version, model digest or code version in `run_meta`.** The manifest records the model tag (`qwen2:7b`) only: no ollama model digest, prompt template text or hash, sampling parameters, seeds, interpreter or package versions, or code hash. | `run_experiment.py:228-236` | SL-02 |
| D-05 | **Hard-coded sampling parameters.** `temperature=0.7` (constructor default, never overridden: `run_experiment.py` l.156) and `top_p=0.9` are fixed in code and not written to `run_meta`. | `agents.py:44`, `agents.py:56` | SL-02 |
| D-06 | **Prompt-induced extras channel.** The shared task prompt invites "up to 2 extra items", which the LLM arms add to the basket, while `human_sim` draws extras from a fixed 0.19 hazard (`storelab.py` l.334). The impulse channel therefore measures the prompt, not behaviour (`PROGRESS.md` §3.2(c); `findings.md` §3, "Ignore the last two columns"). | `agents.py:150-152`, `agents.py:224`, `agents.py:288`, `agents.py:317` | legacy only: kept solely in the SL-04 `legacy_shared` reproduction; the SL-03 protocol has no extras action |
| D-07 | **Raw model outputs are discarded.** Each regime runner returns `{"raw": raw[:200], "parsed": ...}`, already truncated, and `run_experiment.py` drops it (`basket, _ = ...`), so neither the replies nor the per-episode parse status were saved. D-03 substitutions cannot be audited or replayed. | `agents.py:237`, `run_experiment.py:181`, `run_experiment.py:183`, `run_experiment.py:185`, `run_experiment.py:214` | SL-03 (`raw_ref`), SL-08 (raw responses in the cache) |
| D-08 | **Checkpoints are written in place and read errors are swallowed.** A kill during `save_checkpoint` leaves a truncated file, which `load_checkpoint` silently treats as absent; the arm is then re-run on a shifted RNG stream (D-01) and nothing is logged. | `run_experiment.py:133-134`, `run_experiment.py:142-145` | SL-07 |
| D-09 | **One module-level RNG drives every resampling step of the analysis.** The evasive-agent re-timing, every bootstrap and every permutation test draw from the same stream in code order, so adding, removing or reordering any analysis changes all later CIs, p-values and the evasive counterfactual. The output is deterministic as written, which is why SL-00 expects a zero diff. `make_figures.py` repeats the pattern. | `analyze.py:46`, `analyze.py:97`, `make_figures.py:29-30` | SL-15 (keyed `("bootstrap", analysis_id)` streams, tickets §3.6) |
| D-10 | **Numbers hard-coded in the findings prose.** Several sentences state results as literals instead of computing them (17.8%, 46.7%, n=160, p<0.0001, 13.3 / 6.0 / 0.0, 18.9% / 11.5% / 9.2%). They match the tables computed from the saved data, but a zero diff on these lines verifies nothing, and they would be wrong on any other data. | `analyze.py:316-324`, `analyze.py:400-407` | legacy only: the v2 analysis (SL-15) does not port this script |

## Location check

Each location above, with code that must appear in that line range. The test reads
this table.

| Location | Code at location |
|---|---|
| `storelab.py:123` | `self.rng = random.Random(seed)` |
| `storelab.py:131` | `self.rng.uniform(0.55, 0.95)` |
| `storelab.py:272` | `self.rng = random.Random(seed)` |
| `storelab.py:325` | `self.rng.gauss(0, 0.06)` |
| `run_experiment.py:165-170` | `ck = load_checkpoint(regime)` |
| `run_experiment.py:201-205` | `ck = load_checkpoint("sec_"` |
| `agents.py:61-73` | `return ""` |
| `agents.py:212-214` | `chosen = page.items[0].product` |
| `agents.py:264-266` | `chosen = page.items[0].product` |
| `agents.py:307-309` | `chosen = page.items[0].product` |
| `run_experiment.py:228-236` | `meta = {"model": MODEL` |
| `agents.py:44` | `temperature: float = 0.7` |
| `agents.py:56` | `"top_p": 0.9` |
| `agents.py:150-152` | `optionally add up to 2 extra` |
| `agents.py:224` | `obj.get("extras")` |
| `agents.py:288` | `obj.get("extras")` |
| `agents.py:317` | `obj.get("extras")` |
| `agents.py:237` | `{"raw": raw[:200]` |
| `run_experiment.py:181` | `basket, _ = run_chatbot(` |
| `run_experiment.py:183` | `basket, _ = run_gui_agent(` |
| `run_experiment.py:185` | `basket, _ = run_api(` |
| `run_experiment.py:214` | `basket, _ = run_gui_agent(` |
| `run_experiment.py:133-134` | `with open(checkpoint_path(name), "w")` |
| `run_experiment.py:142-145` | `except Exception:` |
| `analyze.py:46` | `RNG = random.Random(20260813)` |
| `analyze.py:97` | `RNG.choice(human_gaps)` |
| `make_figures.py:29-30` | `RNG = random.Random(20260827)` |
| `analyze.py:316-324` | `wrongly filters **17.8%**` |
| `analyze.py:400-407` | `n=160, p<0.0001` |

## Reviewed and not counted as defects

- `HumanSim` clicks `page.items[0]` when the simulated shopper clicked nothing (`storelab.py` l.369–380). This is a documented modelling assumption of the human baseline (a directed purchase), not a substituted agent outcome; SL-06 keeps the legacy cascade.
- `Store.rng` (`storelab.py` l.180) is never used, and `Auction`'s default seed 7 never applies because `Store` passes its own seed 11 (l.179). Neither changes behaviour; both matter to the SL-04 port.
- `PROGRESS.md` §1.7 reports a 3.95 s mean latency; `run_meta.json` and `findings.md` both say 3.83 s. The saved data is authoritative.
