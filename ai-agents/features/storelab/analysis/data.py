"""Choice data for the E1 estimators (SL-15): one row per trial with a recorded choice.

``ChoiceData`` holds, per trial, the displayed products (as indices of the cell's
products, one score per (task, product)), their positions and sponsored labels, and
the chosen slot (-1 for the outside option). Rows are padded to the largest set
(product index -1). ``from_records`` builds it from SL-03 trial records and the
SL-05 designs: complete-case (``invalid``, ``refusal``, ``error`` and ``not_run`` are
dropped, the primary analysis) or with them coded as the outside option (the
sensitivity analysis, A12 §3.4). ``set_ids`` and ``reps`` keep the candidate set and
repetition of each trial for the cluster bootstrap and the consistency statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np

from ..core.design import Design, Presentation
from ..core.records import TrialRecord

MISSING = ("invalid", "refusal", "error", "not_run")


@dataclass
class ChoiceData:
    products: list[tuple[str, str]]  # score index -> (task_id, product_id)
    prod: np.ndarray  # (T, S) int, -1 = padding
    pos: np.ndarray  # (T, S) int, 1-based positions, 0 = padding
    label: np.ndarray  # (T, S) float, 1 = sponsored
    chosen: np.ndarray  # (T,) int slot, -1 = outside option
    set_ids: list[str]
    reps: np.ndarray  # (T,) int
    max_position: int = field(init=False)

    def __post_init__(self) -> None:
        self.prod = np.asarray(self.prod, dtype=np.int64)
        self.pos = np.asarray(self.pos, dtype=np.int64)
        self.label = np.asarray(self.label, dtype=np.float64)
        self.chosen = np.asarray(self.chosen, dtype=np.int64)
        self.reps = np.asarray(self.reps, dtype=np.int64)
        t, s = self.prod.shape
        if self.pos.shape != (t, s) or self.label.shape != (t, s) or self.chosen.shape != (t,) or len(self.set_ids) != t:
            raise ValueError("ChoiceData: inconsistent shapes")
        if ((self.chosen < -1) | (self.chosen >= s)).any():
            raise ValueError("ChoiceData: chosen slot out of range")
        rows = np.arange(t)
        picked = self.chosen >= 0
        if (self.prod[rows[picked], self.chosen[picked]] < 0).any():
            raise ValueError("ChoiceData: a chosen slot is padding")
        self.max_position = int(self.pos.max()) if t else 0

    @property
    def n_trials(self) -> int:
        return len(self.chosen)

    @property
    def tasks(self) -> list[str]:
        return list(dict.fromkeys(task for task, _ in self.products))

    def subset(self, rows: np.ndarray) -> ChoiceData:
        rows = np.asarray(rows)
        return ChoiceData(self.products, self.prod[rows], self.pos[rows], self.label[rows], self.chosen[rows],
                          [self.set_ids[i] for i in rows], self.reps[rows])

    def chosen_product(self) -> np.ndarray:
        """(T,) score index of the chosen product, -1 for the outside option."""
        out = np.full(self.n_trials, -1)
        picked = self.chosen >= 0
        out[picked] = self.prod[np.arange(self.n_trials)[picked], self.chosen[picked]]
        return out


def from_presentations(rows: Sequence[tuple[str, Presentation, int, str | None]]) -> ChoiceData:
    """``rows``: (task_id, presentation, rep, chosen product id or None for the outside option)."""
    index: dict[tuple[str, str], int] = {}
    for task, presentation, _, _ in rows:
        for c in presentation.candidates:
            index.setdefault((task, c.product_id), len(index))
    width = max((len(p.candidates) for _, p, _, _ in rows), default=1)
    t = len(rows)
    prod, pos = np.full((t, width), -1), np.zeros((t, width), dtype=np.int64)
    label, chosen = np.zeros((t, width)), np.full(t, -1)
    for i, (task, presentation, _, choice) in enumerate(rows):
        for slot, c in enumerate(presentation.candidates):
            prod[i, slot], pos[i, slot], label[i, slot] = index[task, c.product_id], c.position, c.sponsored_label
            if c.product_id == choice:
                chosen[i] = slot
        if choice is not None and chosen[i] < 0:
            raise ValueError("trial %d: chosen %s is not displayed" % (i, choice))
    return ChoiceData(list(index), prod, pos, label, chosen, [p.set_id for _, p, _, _ in rows],
                      np.asarray([rep for _, _, rep, _ in rows]))


def from_records(records: Iterable[TrialRecord], designs: Sequence[Design],
                 missing: Literal["drop", "outside"] = "drop") -> ChoiceData:
    """Trials of one cell from their records and designs; non-choices dropped
    (primary) or coded as the outside option (sensitivity)."""
    by_set = {s.set_id: d for d in designs for s in d.sets}
    rows = []
    for record in records:
        status = record.outcome.status
        if status in MISSING:
            if missing == "drop":
                continue
            choice = None
        else:
            choice = record.outcome.chosen_id if status == "choice" else None
        design = by_set[record.set_id]
        rows.append((record.task_id, design.presentation_for(record.set_id, record.rep), record.rep, choice))
    return from_presentations(rows)
