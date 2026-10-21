# Provenance of the original StoreLab baseline

| | |
|---|---|
| Source | `~/Desktop/agentic_amazon/5_storelab/` (not under version control) |
| Captured | 2026-10-21, ticket SL-00 |
| Run | `run_meta.json`: model `qwen2:7b` via ollama, `run_utc` 2026-08-14T17:51:08Z (end), `runtime_s` 1648.7, 430 LLM calls, 9 parse failures, catalog 2,004 SKUs |
| Run start | 2026-08-14T17:23:39Z (end − runtime); the first checkpoint (`_ckpt_human_sim.json`) is stamped 17:23:40Z |
| Records | 520 episodes, 6,584 events (`run.log`; matches `episodes.json` and `events.json`) |
| Interpreter | CPython 3.9: the `__pycache__/*.cpython-39.pyc` files carry the 3.9 magic number `610d0d0a`, and the source mtime and size in each header equal those of the current `storelab.py`, `agents.py` and `analyze.py` |

**Consistency checks.** The three modules the run imported or executed (`storelab.py`, `agents.py`, `run_experiment.py`) were last modified before the run started. `run.log` has no `RESUMED` line, so every arm ran in one uninterrupted process and the order-dependent RNG streams (`DEFECTS.md` D-01) were consumed in the original order. `analyze.py` was last edited at 19:32:00Z and `findings.md` written at 19:32:56Z. `make_figures.py` and the figures date from 2026-08-27.

**Dispositions.**
- `vendored`: copied byte-for-byte to this directory.
- `stored`: copied to the local artifact store as dataset snapshot `legacy_storelab_2026_08`, address `65d6137645b7585a04de285892ab5da52910d90cf77efed3028f9b11f68f1228`, at `artifacts/datasets/legacy_storelab_2026_08/<address>/<path>`, with a `MANIFEST.json`. The address is the sha256 of the compact JSON of the sorted `[path, sha256, size]` rows (`legacy_baseline.snapshot_digest`).
- `hash only`: recorded here, not copied. The figures can be regenerated with `make_figures.py`, which SL-00 does not run; the bytecode is evidence of the interpreter only.

`.DS_Store` (Finder metadata) is excluded.

**PR size (A14).** The vendored code (1,688 lines) and the data snapshot form one labelled commit, exempt from the 400-added-line convention. Tooling, documents and tests are in a separate commit.

## Files

mtime is the source file's modification time in UTC.

| Path | sha256 | Bytes | mtime (UTC) | Disposition |
|---|---|---|---|---|
| `storelab.py` | `e246c13e057e47b34a18ec977e164aa7f01358d8e01b7471d73f7bc5b9264e3e` | 18424 | 2026-08-14T17:18:51Z | vendored |
| `agents.py` | `2c800fbb5390adffb257819275e62c6a3284677e7a15de67eee8cd2de2998e1f` | 14249 | 2026-08-14T17:19:27Z | vendored |
| `run_experiment.py` | `209859ed391bacaa355fb4e45072686650a101886421fbee3e58d6c97bc862a1` | 11645 | 2026-08-14T17:22:24Z | vendored |
| `analyze.py` | `87915f2a4105c6ba88821b0e9749a4ca628487649c4e8f0b7a197f67f4c7ddca` | 23854 | 2026-08-14T19:32:00Z | vendored |
| `make_figures.py` | `b140d9547c2bae6d2dbd5c22a854d46bc05c032b6fb486a5336952f0947db032` | 12022 | 2026-08-27T20:19:50Z | vendored |
| `logs/events.json` | `9aa7448c97cab624f31694b7f900a5f5bec44e2d6bdfb54027e6de15e7cbba71` | 1556421 | 2026-08-14T17:51:08Z | stored |
| `logs/run.log` | `e4b25791e0534c3f3a13c0774c9d88f9a6b7af7460f7df9959aa5316de0f2a5c` | 1305 | 2026-08-14T17:51:08Z | stored |
| `results/_ckpt_api.json` | `ebe46f0d8556082b3d010df259730910b92b741df23538eee6c0fdf0cb129c65` | 100967 | 2026-08-14T17:40:47Z | stored |
| `results/_ckpt_chatbot.json` | `7be575ac7fd6dd596bf4500882bf8a6d0341e9665c0ab996e3efd264aaca7b6c` | 241547 | 2026-08-14T17:28:10Z | stored |
| `results/_ckpt_gui_agent.json` | `10a0f48806b916f1d25e78fc393e4bed06924bf274710b7b6709624dc5ea398f` | 389253 | 2026-08-14T17:34:26Z | stored |
| `results/_ckpt_human_sim.json` | `7873a9321d04dbc12d7e0dee1b9e47cd5b19158e01045f1dd1c4aadf8950e8cc` | 468451 | 2026-08-14T17:23:40Z | stored |
| `results/_ckpt_sec_labeloff_posbottom.json` | `7e9781381644d0110a4aad0ea3fb849c907058f847f91a4ca013bfa2ad6063d3` | 182576 | 2026-08-14T17:51:08Z | stored |
| `results/_ckpt_sec_labeloff_postop.json` | `19f7802872c087d67ab707b2e76eac1aa94ef8c2732228d7116c64cf19cd91f5` | 182005 | 2026-08-14T17:48:57Z | stored |
| `results/_ckpt_sec_labelon_posbottom.json` | `cfda6dacae819b333a2f63f07fcf7e4d7efc00113285b99210159d062aca0a21` | 182375 | 2026-08-14T17:46:07Z | stored |
| `results/_ckpt_sec_labelon_postop.json` | `49a3310da2f83dd7935c5f57fece983ff38fb68f8ee098036cae23d926105ee6` | 180856 | 2026-08-14T17:43:23Z | stored |
| `results/episodes.json` | `b8f2436310e25464ba29da8e880c91457da06101b66270fd7405bf0562b56aec` | 407835 | 2026-08-14T17:51:08Z | stored |
| `results/findings.md` | `da76128e6bb6017747f76f9d89146fce5e4979555bffd40b7b0e620425ecd9db` | 13349 | 2026-08-14T19:32:56Z | stored |
| `results/run_meta.json` | `428dcaa60ec8ec2159647f247bf977826836d55425c32fa608b2f676d13bacba` | 252 | 2026-08-14T17:51:08Z | stored |
| `figures/fig1_attention_funnel.pdf` | `8241ee0dd71203e6ac486e6ba1ac5cfb6f16d17580ec7bd818cfbdc998bfffe4` | 20302 | 2026-08-27T20:19:54Z | hash only |
| `figures/fig1_attention_funnel.png` | `0d438fb8c2bbc3c1f722d5a42e8c451fc666ff04b851342a4f94f9ae6065b866` | 70557 | 2026-08-27T20:19:53Z | hash only |
| `figures/fig2_detection_frontier.pdf` | `b99d624feee41285bf0d94df853340fe676aa1053834f48eb244a51c1571300a` | 16561 | 2026-08-27T20:19:55Z | hash only |
| `figures/fig2_detection_frontier.png` | `57278eb077e9e1dcbdd4f4e806a2d322e0eb669b6756382f89d8db2da42a8867` | 112866 | 2026-08-27T20:19:55Z | hash only |
| `figures/fig3_displacement_contamination.pdf` | `94a2fd4fc2c6e5c944cda41ba4ec6c39eb8f8effa1ee9c2d6537103cab52ceb6` | 16890 | 2026-08-27T20:19:55Z | hash only |
| `figures/fig3_displacement_contamination.png` | `c60752f321d193aec100aab3fb273813c44b47c6e5c73b067188c47ad0ca604b` | 61414 | 2026-08-27T20:19:55Z | hash only |
| `figures/fig4_position_vs_label.pdf` | `f791c6f313b148461a7b9c43df45f3e36c448a70dca7da46a98a92189cc3a8d2` | 14194 | 2026-08-27T20:19:56Z | hash only |
| `figures/fig4_position_vs_label.png` | `918b1ce24334394ed204aef0c3523809cfd8d774ad87efb879b4c1817c200288` | 46744 | 2026-08-27T20:19:56Z | hash only |
| `figures/fig5_contrasts.pdf` | `09843117d4933502239c1ffc7124fc6ac79af75f7ed860c2ac047ba617c50e74` | 19204 | 2026-08-27T20:20:00Z | hash only |
| `figures/fig5_contrasts.png` | `9dab84f430f2b69e4c284533348b198ba0da9fe8c8e7c8b4efe7858f1cf38bbb` | 52125 | 2026-08-27T20:19:59Z | hash only |
| `__pycache__/agents.cpython-39.pyc` | `f367c3c60028dea4a0a7af66c1b07076235865f010256f11b5edd5e2d7769a27` | 11599 | 2026-08-14T17:21:23Z | hash only |
| `__pycache__/analyze.cpython-39.pyc` | `2ff360436a29e0418cb93f8f451c846f627493835805946e33af6ac4dc869987` | 26961 | 2026-08-27T16:48:24Z | hash only |
| `__pycache__/storelab.cpython-39.pyc` | `1eacc64c6ae30091244fcfbf594b14dc20df92f4b8b8b39f2e329dbd2834ebd0` | 15666 | 2026-08-14T17:20:18Z | hash only |
