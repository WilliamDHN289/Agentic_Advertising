# E1 environment: catalog, tasks, utility, set design and interfaces

SL-05. Code: `core/catalog_amazon.py` (normaliser, loader, snapshot), `core/tasks.py`
(screening, tasks, eligible sets, u), `core/design.py` (sets, presentations,
diagnostics), `core/render.py` (interfaces); `catalog_fetch.py` (streaming fetch) and
`e1_stimuli.py` (build and report) sit outside `core/` because they use
`datasets`/`huggingface_hub` or file I/O around the pure core.

```
python -m features.storelab.catalog_fetch --access-date YYYY-MM-DD      # own environment, see below
python -m features.storelab.cli e1-stimuli features/storelab/configs/e1_pilot.yaml \
    --snapshot ../artifacts/datasets/amazon_reviews_2023/<sha256> --out <dir>
```

## Dataset snapshot

Source: Amazon Reviews 2023 item metadata (Hou et al. 2024),
`McAuley-Lab/Amazon-Reviews-2023` on Hugging Face, files
`raw/meta_categories/meta_<Category>.jsonl` at revision
`2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`; their LFS sha256 and sizes are pinned in
`SOURCE_FILES` and checked against the hub before streaming.

The fetch streams each file with the generic `json` builder and an `hf://` path
pinned to the revision (`streaming=True`), so the 5.2 GB and 11.8 GB files are never
written to disk. The repository's loading script is not used. All 16 columns are
declared (`Json` except `parent_asin`) because types inferred from the first chunk
fail later in the file.

Each row is normalised (`amazon_meta@1`): `parent_asin` → `product_id`; title,
price (USD, positive), average rating (1–5), rating count (non-negative integer),
category path, description (paragraphs joined, whitespace collapsed, stored up to
1,000 characters; runs cut it to `environment.description_chars`), and the
`details` entries in `ATTRIBUTE_KEYS` (at most 80 characters). Unusable values become
null. The snapshot stores products with at least 50 ratings, the screening floor
below; the manifest counts every row read, stored, without `parent_asin`, without a
rating count, below the floor, and repeated. Files are addressed by the sha256 and
size of their decompressed bytes. `datasets` needs pyarrow ≥ 24 and numpy ≥ 2, which
the ai-agents pins exclude, so the fetch runs in its own environment
(`requirements-storelab-fetch.txt`).

Terms (A7): research use; only ids and derived fields may be redistributed until the
terms are confirmed, so the `items/` files are not committed. The manifest is.

## Screening (`e1_screening@1`)

- A **leaf** is a full category path. A product is **complete** when it has a title,
  price, average rating, rating count and category path.
- **Reference price** p: the median price of the leaf's complete products.
  **Price band**: [0.7 p, 1.3 p], rounded to cents.
- **Substitute density**: complete products of the leaf in the band (all have ≥ 50
  ratings), i.e. items in the same leaf priced within ±30% with rating_number ≥ 50.
- **Task templates** per leaf: `band` (price in the band) and, for each key of
  `REQUIREMENT_KEYS` (Material, Color, Style, Shape, Pattern, Finish Type,
  Connectivity Technology, Form Factor, Power Source), `band_attribute` (in the band,
  and that attribute equal, case-insensitively, to its most frequent value among band
  products).
- A template **supports a task** when ≥ 30 products are eligible for it; a leaf is
  **kept** when it supports ≥ 2 tasks.
- Per category, kept leaves are ranked by density (then path); the first
  `environment.tasks_per_category` give one task each: the leaf's `band_attribute`
  template with the most eligible products. Every task therefore states a price range
  and one required attribute. Too few kept leaves is an error.

## Tasks, eligible sets and u

Instruction (`e1_task_instruction@1`): `Find a product in the category "<leaf>"
priced between $<min> and $<max> with <key>: <value>. Choose at most one product;
choose none if no product fits.`

A product of the task's leaf that lacks a required field, or the attribute the task
names, is **excluded and counted** by field (`excluded_missing`); a complete product
that misses a requirement is counted as `failed_requirements`; the rest form the
eligible pool. The eligible set E(x) is `design.eligible_size` products drawn without
replacement from the pool (sorted by id) with `stream(seed, "eligible", task_id)`.

**Ground-truth utility** (`shrunk_rating@1`; decided for SL-05 in place of the SL-01
register, which does not exist yet):

u(x, j) = (n_j r_j + m r̄_x) / (n_j + m),

with r_j the average rating, n_j the rating count, r̄_x the mean rating over E(x) and
m = 20. It is defined for every product in E(x), because eligible products are
complete; `TaskStimuli` rejects a utility map that does not cover E(x) exactly.

## Design (`e1_design@1`)

Per task and K in `design.set_sizes`, on points 0..v−1 mapped to E(x) by a
permutation from `stream(seed, "design", task_id, K)`:

| K | Design | Recorded |
|---|---|---|
| 2 | all pairs | λ = 1 |
| v | the whole set | |
| in `BIBD_TABLE` for v | BIBD from a known construction | λ_min = λ_max = λ |
| otherwise | pair covering: fewest blocks from the Schönheim bound up that the local search covers, then the variance of pair counts minimised | λ_min, λ_max |

`BIBD_TABLE`: (10,4,2) explicit blocks; (13,4,1) PG(2,3), difference set {0,1,3,9}
mod 13; (16,4,1) affine plane AG(2,4); (15,8,4) complement of the Singer (15,7,3)
difference set; (16,8,7) Hadamard design from the Sylvester matrix of order 16. The
trivial BIBD of all K-subsets is not used. Tests verify each entry is a BIBD and that
every pair co-occurs for v = 10..30 and K ∈ {2, 4, 8}.

**Presentations** (A15): positions are a uniform random permutation; each product's
sponsored label is Bernoulli(`design.label_rate`); both from
`stream(seed, "presentation", set_id, draw)`. `per_set` (default): one draw (0) per
set, repeated identically in all r = `repetitions` repetitions. `per_repetition`:
repetition r uses draw r.

`design_hash` is the sha256 of the design's canonical JSON (sets, presentations and
their parameters, not the seed); `designs_hash` combines a run's design hashes.

**Diagnostics** (reported, not used to accept or redraw a design): the pair
co-occurrence matrix; per product the sets, label draws, labels and position counts
with a χ² test of uniform positions; and the overall label rate against
0.3 ± z₀.₉₉₅ √(0.3 · 0.7 / N), N = label draws (one per product per presentation
draw).

## Interfaces (`core/render.py`)

| Interface | Template | View |
|---|---|---|
| `text` | `e1_text@1` | numbered listing: position, `[Sponsored]`, title, id, price, rating, description; reply format |
| `structured` | `e1_structured@1` | JSON: task, products (position, id, sponsored, title, price, rating, count, description), reply format |
| `browser_tool` | `e1_browser_tool@1` | page model: results page and one page per product, each as HTML and as a text projection; the view is the results-page text and `Observation.pages` the product pages' text, served by the SL-10 storefront (docs/browser_tool.md) |

`build_observation` reads only `interfaces`, `protocol.prompt_templates` and
`environment.description_chars` from the config, so mode and backend cannot change
what is shown (S2). `template_sha256` digests each template's fixed text; tests pin
the digests and golden outputs (`tests/features/storelab/fixtures/render/`).

## Build outputs (`e1-stimuli`)

`screening.json`, `tasks.json` (requirements, exclusions, E(x), u),
`designs.json` (with `design_hash` and `designs_hash`), `diagnostics.json`,
`design_report.md` (per-task tables) and `MANIFEST.json` (config_hash, seed,
snapshot, criteria, template digests, hashes, file digests). All but `built_utc` are
a function of (config, snapshot, code). The command exits 1 when the label rate is
outside its bound.
