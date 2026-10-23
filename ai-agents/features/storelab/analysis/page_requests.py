"""Page requests for products agents do not choose (SL-15, H1e; χ_m for E2).

Over the cell's ``browser_tool`` trials with a recorded choice (``choice`` or
``outside``), every displayed product other than the chosen one is a unit; it is
requested if its page was opened (``TrialOutcome.page_requests``). The rate has a
Wilson interval, and ``decide_input`` gives the counts the registered rule reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..core.records import TrialRecord
from .intervals import wilson


@dataclass
class PageRequestRate:
    requested: int
    non_chosen: int
    trials: int

    @property
    def rate(self) -> float:
        return self.requested / self.non_chosen if self.non_chosen else float("nan")

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        return wilson(self.requested, self.non_chosen, level)

    def decide_input(self) -> dict:
        return {"n": self.non_chosen, "counts": {"requested": self.requested}, "estimate": self.rate,
                "ci": self.ci()}


def page_request_rate(records: Iterable[TrialRecord]) -> PageRequestRate:
    requested = non_chosen = trials = 0
    for record in records:
        outcome = record.outcome
        if record.interface != "browser_tool" or outcome.status not in ("choice", "outside"):
            continue
        trials += 1
        others = [p for p in record.candidate_ids if p != outcome.chosen_id]
        opened = set(outcome.page_requests)
        non_chosen += len(others)
        requested += sum(p in opened for p in others)
    return PageRequestRate(requested, non_chosen, trials)
