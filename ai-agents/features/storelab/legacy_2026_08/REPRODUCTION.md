# Reproducing the original StoreLab analysis

SL-00 checks that the August 2026 analysis is reproducible from the saved data and
that the vendored code is the code that ran. The LLM arms are not re-run: ollama
output is not reproducible (tickets.md SL-00, "Out").

All commands run from `ai-agents/`. `-B` stops Python from writing bytecode; every
check is read-only except `reproduce --mode copy`, which writes to a temporary
directory.

## Interpreters

| Role | Interpreter | Platform | numpy | scipy | pandas | matplotlib |
|---|---|---|---|---|---|---|
| ai-agents line (`requires-python >=3.11,<3.12`) | CPython 3.11.1, `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3` | macOS 26.5 arm64 | 2.4.3 | 1.17.1 | 3.0.1 | 3.10.8 |
| Original run (cross-check) | CPython 3.9.13 (Anaconda), `~/opt/anaconda3/bin/python3.9` | macOS 26.5 x86_64 | 1.23.5 | 1.9.1 | 1.4.4 | 3.5.2 |

The 3.9 interpreter is identified as the original by the `cpython-39` bytecode in the
source directory (`PROVENANCE.md`). `storelab.py` and `analyze.py` import only the
standard library; numpy and matplotlib are used only by `make_figures.py`, which SL-00
does not run. The package versions therefore cannot affect the results below.

## Commands

```bash
# Hashes: vendored code against the source, the snapshot against its manifest and the source
python -B -m features.storelab.legacy_baseline verify --source ~/Desktop/agentic_amazon/5_storelab

# analyze.py on the saved data, diffed against the saved findings.md
python -B -m features.storelab.legacy_baseline reproduce --python PYTHON --mode in-memory
python -B -m features.storelab.legacy_baseline reproduce --python PYTHON --mode copy

# build_catalog() against the catalog line in run.log
python -B -m features.storelab.legacy_baseline catalog --python PYTHON

# Acceptance tests (also collected by pytest)
python -B -m unittest discover -s tests/features/storelab -t . -v
```

`--mode in-memory` runs `analyze.main()` from the unmodified vendored file, reading the
snapshot and capturing the `findings.md` write in memory; it writes nothing, so it
also works in a read-only sandbox. `--mode copy` rebuilds the original layout (code
beside `results/` and `logs/`) in a temporary directory and runs `python analyze.py`
there unchanged. The same check by hand, which is the ticket's procedure literally:

```bash
SNAP=../artifacts/datasets/legacy_storelab_2026_08/65d6137645b7585a04de285892ab5da52910d90cf77efed3028f9b11f68f1228
W=$(mktemp -d) && mkdir -p $W/results $W/logs
cp features/storelab/legacy_2026_08/*.py $W/
cp $SNAP/results/episodes.json $SNAP/results/run_meta.json $W/results/ && cp $SNAP/logs/events.json $W/logs/
(cd $W && PYTHON -B analyze.py > /dev/null) && diff $SNAP/results/findings.md $W/results/findings.md
```

## Results (2026-10-21)

| Check | 3.11.1 | 3.9.13 |
|---|---|---|
| Vendored code sha256 = source sha256 (5 files) | equal | equal |
| Snapshot sha256 = manifest = source (13 files) | equal | equal |
| `findings.md`, `--mode in-memory` | zero diff | zero diff |
| `findings.md`, `--mode copy` | zero diff | zero diff |
| `findings.md`, by hand (`diff`, `cmp`) | zero diff, byte-identical | zero diff, byte-identical |
| `build_catalog()` | 2,004 SKUs, 6 categories, 712 ad-eligible | 2,004 SKUs, 6 categories, 712 ad-eligible |
| `run.log` catalog line | 2004 SKUs across 6 categories, 712 ad-eligible | same file |
| Acceptance tests | 17 passed | 17 passed |

**Diff result.** The regenerated `findings.md` is byte-identical to the saved one
under both interpreters and all three methods: sha256
`da76128e6bb6017747f76f9d89146fce5e4979555bffd40b7b0e620425ecd9db`, 13,349 bytes.
There is no diff to root-cause. `analyze.py` draws every bootstrap and permutation
from `random.Random(20260813)` (l.46), and Python's `random` algorithms are the same
in 3.9 and 3.11, so an exact match is expected.

**Catalog fingerprint.** sha256 of the sorted-key JSON of every field of every SKU:
`9491773fa9aa2601d95f6396b39a08c6887f77a7bd053dff9e756b91872bbe07`, identical under
both interpreters. SL-04's port can compare all fields against this, not just counts.

**What the zero diff does not cover.** Some sentences of `findings.md` are literals in
`analyze.py`, not computed values (`DEFECTS.md` D-10), so they match trivially. They
were checked by hand against the computed tables of the regenerated file:

| Literal | Computed value |
|---|---|
| 17.8% human FP and 28% evasive catch at threshold 0.20 | 17.8%, 27.8% |
| 46.7% human FP and 53% evasive catch at 0.16 | 46.7%, 53.3% |
| no evasive agent caught above 0.30 | 0.0% at 0.30, 0.50, 0.75, 0.90 |
| n=160, position p<0.0001, label p=0.72 (§8) | 4 × 40, p = 0.0000, p = 0.7200 |
| human-viewed impressions 13.3 / 6.0 / 0.0 | 13.3, 6.0, 0.0, 0.0 |
| sponsored CTR 18.9% / 11.5% / 9.2% / 0% | 18.89%, 11.48%, 9.17%, 0.00% |

## Lint and coverage exclusion

`legacy_2026_08/` is excluded in `ai-agents/pyproject.toml` (ruff `extend-exclude`, with
`force-exclude` so explicitly passed paths are skipped too) and `ai-agents/.coveragerc`
(`omit`). Checked with ruff 0.17.1 and coverage 7.16.2:

- `ruff check .` passes. With the configuration bypassed (`--isolated`, same rule set
  E4, E7, E9, F), the vendored code has 40 findings, so the exclusion is what keeps it out.
- `ruff check features/storelab/legacy_2026_08/storelab.py` lints no files.
- A driver that runs the vendored `build_catalog()` in-process: under `.coveragerc` the
  report lists no `legacy_2026_08` file; under a control rcfile without the omit list it
  reports `legacy_2026_08/storelab.py` at 42%.

`.gitattributes` marks the vendored `.py` files `-text`, so no checkout converts line
endings and changes their hashes, and the snapshot `binary`, so diffs do not print 3.9 MB
of JSON; its integrity is checked by hash.

## Not done in SL-00

- The LLM arms were not re-run (out of scope; not reproducible).
- `make_figures.py` was not run; the figures are recorded by hash only (`PROVENANCE.md`).
