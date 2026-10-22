"""Build the E1 stimuli of a run config from a dataset snapshot (SL-05).

    python -m features.storelab.cli e1-stimuli CONFIG --snapshot DIR --out DIR [--set KEY=VALUE ...]

Steps: verify the snapshot named by environment.dataset; screen its categories and
choose the tasks (core/tasks.py); draw each task's eligible set and compute u; build
a design per task and set size (core/design.py); diagnose it; render every
presentation once (core/render.py). Writes to ``--out``:

    screening.json      registered criteria, screened leaves, chosen tasks
    tasks.json          per task: requirements, exclusions, eligible set, u
    designs.json        designs, with design_hash per design and designs_hash overall
    diagnostics.json    co-occurrence, label and position balance, label-rate check
    design_report.md    the per-task design report
    MANIFEST.json       config_hash, seed, snapshot, criteria, hashes, file digests

Everything but MANIFEST.json's ``built_utc`` is a function of (config, snapshot, code).
"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .core.catalog_amazon import DATASET, Product, check_snapshot, iter_products, load_products, read_manifest
from .core.config import RunConfig, canonical_json
from .core.design import (Design, DesignDiagnostics, LabelRateCheck, build_design, designs_hash, diagnose,
                          label_rate_check)
from .core.render import build_observation, template_sha256
from .core.tasks import Screening, TaskStimuli, screen, task_stimuli
from .core.versioning import env_version


@dataclass
class E1Stimuli:
    config: RunConfig
    snapshot_sha256: str
    screening: Screening
    tasks: list[TaskStimuli]
    designs: list[Design]
    diagnostics: list[DesignDiagnostics]
    label_rate: LabelRateCheck
    products: dict[str, Product]  # the eligible products, descriptions cut to description_chars
    observations_sha256: str  # over the observation hashes of every presentation, in order


def _check_config(config: RunConfig) -> None:
    env = config.environment
    if env.catalog != "amazon_reviews_2023":
        raise ValueError("e1-stimuli needs environment.catalog amazon_reviews_2023, got %s" % env.catalog)
    if env.dataset is None or env.dataset.name != DATASET:
        raise ValueError("environment.dataset must name the %s snapshot (name and sha256)" % DATASET)


def build(config: RunConfig, snapshot: Path) -> E1Stimuli:
    _check_config(config)
    snapshot = Path(snapshot)
    manifest = read_manifest(snapshot)
    if manifest.snapshot_sha256 != config.environment.dataset.sha256:
        raise ValueError("snapshot %s is not environment.dataset %s" % (manifest.snapshot_sha256,
                                                                         config.environment.dataset.sha256))
    problems = check_snapshot(snapshot)
    if problems:
        raise ValueError("snapshot %s: %s" % (snapshot, "; ".join(problems)))
    env, design = config.environment, config.design
    # Screening needs no descriptions; dropping them keeps a full category in memory.
    catalog = [p.model_copy(update={"description": ""}) for c in env.categories for p in iter_products(snapshot, c)]
    screening = screen(catalog, env.categories, env.tasks_per_category)
    tasks = [task_stimuli(t, catalog, design.eligible_size, config.seed) for t in screening.tasks]
    del catalog
    products = load_products(snapshot, env.categories, env.description_chars,
                             ids={pid for t in tasks for pid in t.eligible_ids})
    designs = [build_design(t.task.task_id, t.eligible_ids, k, config.repetitions, design.label_rate,
                            design.randomisation, config.seed) for t in tasks for k in design.set_sizes]
    specs = {t.task.task_id: t.task for t in tasks}
    observation_hashes = [build_observation(config, specs[d.task_id], p, products).observation_hash()
                          for d in designs for p in d.presentations]
    return E1Stimuli(config=config, snapshot_sha256=manifest.snapshot_sha256, screening=screening, tasks=tasks,
                     designs=designs, diagnostics=[diagnose(d) for d in designs],
                     label_rate=label_rate_check(designs), products=products,
                     observations_sha256=hashlib.sha256(canonical_json(observation_hashes).encode()).hexdigest())


# ---------------------------------------------------------------------- report
def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _p(value: float | None) -> str:
    return "n/a" if value is None else "%.3f" % value


def report(stimuli: E1Stimuli) -> str:
    """The per-task design report (Markdown)."""
    cfg, out = stimuli.config, []
    lr = stimuli.label_rate
    out += ["# E1 design report", "",
            "Experiment `%s`, config_hash `%s`, seed %d, snapshot `%s`." % (
                cfg.experiment, cfg.config_hash, cfg.seed, stimuli.snapshot_sha256),
            "Screening `%s`; utility `%s` (prior weight m = %g ratings); randomisation `%s`; r = %d repetitions." % (
                stimuli.screening.criteria["screening"], stimuli.screening.criteria["utility"],
                stimuli.screening.criteria["prior_ratings"], cfg.design.randomisation, cfg.repetitions),
            "", "## Overall", "",
            _table(["designs", "sets", "label draws N", "labelled", "label rate", "target", "99% bound", "within"],
                   [[len(stimuli.designs), sum(len(d.sets) for d in stimuli.designs), lr.draws, lr.labelled,
                     "%.4f" % lr.rate, lr.target, "±%.4f" % lr.bound, "yes" if lr.within else "**no**"]]),
            "", "## Screening", ""]
    for category in stimuli.screening.categories:
        out += ["### %s" % category.category, "",
                "%d products in %d leaves; %d leaves reach density %d; %d kept." % (
                    category.products, category.leaves, len(category.candidates),
                    stimuli.screening.criteria["min_eligible"], category.leaves_kept), "",
                _table(["leaf", "products", "complete", "reference price", "density", "tasks supported", "kept"],
                       [[" > ".join(c.leaf_path), c.products, c.complete, "$%.2f" % c.reference_price, c.density,
                         c.supported, "yes" if c.kept else "no"] for c in category.candidates[:15]]), ""]
    by_task = {}
    for d, diag in zip(stimuli.designs, stimuli.diagnostics):
        by_task.setdefault(d.task_id, []).append((d, diag))
    for t in stimuli.tasks:
        spec = t.task
        out += ["## Task `%s`" % spec.task_id, "", "- Leaf: %s" % " > ".join(spec.leaf_path),
                "- Instruction: %s" % spec.instruction,
                "- Leaf products %d; excluded for a missing field %d %s; outside the requirements %d; "
                "eligible pool %d; eligible set |E(x)| = %d." % (
                    t.leaf_products, t.excluded_products, json.dumps(t.excluded_missing), t.failed_requirements,
                    t.pool_size, len(t.eligible_ids)),
                "- u defined for %d of %d eligible products; r̄ = %.4f." % (
                    len(t.utility), len(t.eligible_ids), t.mean_rating), "",
                _table(["#", "product", "price", "rating", "ratings", "u"],
                       [[i + 1, pid, "$%.2f" % stimuli.products[pid].price, stimuli.products[pid].average_rating,
                         stimuli.products[pid].rating_number, "%.4f" % t.utility[pid]]
                        for i, pid in enumerate(t.eligible_ids)]), ""]
        for d, diag in by_task[spec.task_id]:
            draws = diag.label_draws
            out += ["### `%s`, K = %d" % (spec.task_id, d.k), "",
                    "%s (%s): %d sets, λ_min = %d, λ_max = %d; %d presentation draws, label rate %.3f." % (
                        d.design_type, d.construction, len(d.sets), d.lambda_min, d.lambda_max,
                        len(d.presentations), diag.labelled / draws if draws else 0.0), "",
                    "Pair co-occurrence (rows and columns are the products above, in order):", "",
                    _table([""] + [str(i + 1) for i in range(len(diag.eligible_ids))],
                           [[i + 1] + ["–" if i == j else n for j, n in enumerate(row)]
                            for i, row in enumerate(diag.cooccurrence)]), "",
                    "Label and position balance per product (χ² test of uniform positions, reported only):", "",
                    _table(["product", "sets", "draws", "labelled", "label rate", "position counts", "χ²", "p"],
                           [[b.product_id, b.sets, b.draws, b.labelled,
                             "%.3f" % (b.labelled / b.draws) if b.draws else "n/a",
                             " ".join(map(str, b.position_counts)), _p(b.chi2), _p(b.p_value)]
                            for b in diag.products]), ""]
    return "\n".join(out)


# ----------------------------------------------------------------------- write
def _dump(value) -> bytes:
    return (json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write(stimuli: E1Stimuli, out: Path) -> dict:
    """Write the outputs listed in the module docstring; returns the manifest."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "screening.json": _dump(stimuli.screening.model_dump(mode="json")),
        "tasks.json": _dump([t.model_dump(mode="json") for t in stimuli.tasks]),
        "designs.json": _dump({"designs_hash": designs_hash(stimuli.designs),
                               "designs": [{"design_hash": d.design_hash(), **d.model_dump(mode="json")}
                                           for d in stimuli.designs]}),
        "diagnostics.json": _dump({"label_rate": stimuli.label_rate.model_dump(mode="json"),
                                   "designs": [d.model_dump(mode="json") for d in stimuli.diagnostics]}),
        "design_report.md": report(stimuli).encode("utf-8"),
    }
    for name, data in files.items():
        (out / name).write_bytes(data)
    cfg = stimuli.config
    manifest = {
        "experiment": cfg.experiment, "config_hash": cfg.config_hash, "seed": cfg.seed,
        "snapshot_sha256": stimuli.snapshot_sha256, "env_version": env_version(),
        "criteria": stimuli.screening.criteria,
        "templates": {t: template_sha256(t) for t in cfg.protocol.prompt_templates.values()},
        "designs_hash": designs_hash(stimuli.designs), "observations_sha256": stimuli.observations_sha256,
        "label_rate_within_bound": stimuli.label_rate.within,
        "files": [{"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                  for name, data in sorted(files.items())],
        "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out / "MANIFEST.json").write_bytes(_dump(manifest))
    return manifest
