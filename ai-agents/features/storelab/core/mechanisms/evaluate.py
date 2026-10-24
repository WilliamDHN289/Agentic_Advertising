"""Run the mechanisms on a market (SL-16): one entry point, one outcome record.

``evaluate(market, mechanism, values)`` gives the allocation, payments and outcomes
of M1, M2 (GSP, with a bidding rule), M3_set_blind, M3, M4 (truthful bids, which
are dominant, Prop ic) and profit_rule (truthful bids, coefficients φ̄_j(θ_j)).
Payments: externality form for the surplus rules, eq. payment for the profit rule;
with ``by_type`` the surplus rules also get eq. payment, which splits payments into
human and agent parts (A1 §E5) and checks the identity of Prop ic(c).

``evaluate_markets`` runs several mechanisms on several markets. Any solver status
other than ``optimal`` in any solve of a market (allocation or payment) excludes that
market from every mechanism, so comparisons stay paired, and adds it to the
``Exclusions`` count; an excluded market contributes no outcome.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import gsp
from . import payments as pay
from .allocation import RULES, Allocation, Program
from .market import ActionTable, Market, action_table, match_reference
from .solver import Exclusions, SolverFailure

MECHANISMS = ("M1", "M2", "M3_set_blind", "M3", "M4", "profit_rule")


@dataclass
class Outcome:
    market_id: str
    market_hash: str
    mechanism: str
    bids: dict[str, float] | dict[str, dict[str, float]]  # value bids, or click bids per context (GSP)
    X: dict[str, float]
    X_human: dict[str, float]
    X_agent: dict[str, float]
    payments: dict[str, float]
    payments_human: dict[str, float] | None
    payments_agent: dict[str, float] | None
    contribution: float
    cost: dict[str, float]
    match: float
    m_ref: float | None
    guardrail_slack: float | None  # M − (1 − κ) M^ref
    seconds: float
    converged: bool = True  # GSP fixed-point bidding
    iterations: int = 0
    identity_gap: dict[str, float] = field(default_factory=dict)  # |externality − integral| (by_type)
    cost_model_error: float = 0.0  # overflow tangents: exact − modelled cost at the solution
    payment_error_bound: float = 0.0  # eq. payment: bound on |error| from unresolved profile intervals

    @property
    def revenue(self) -> float:
        return math.fsum(self.payments.values())

    @property
    def operating_value(self) -> float:
        return self.contribution - math.fsum(self.cost.values())

    @property
    def profit(self) -> float:
        return self.revenue + self.operating_value


class MarketRunner:
    """Caches the action table and the programs of one market."""

    def __init__(self, market: Market):
        self.market = market
        self.table: ActionTable = action_table(market)
        self.programs: dict[str, Program] = {}
        self._m_ref = None

    def program(self, rule: str) -> Program:
        if rule not in self.programs:
            self.programs[rule] = Program(self.market, rule, self.table, self.m_ref)
        return self.programs[rule]

    @property
    def m_ref(self) -> float | None:
        if self.market.guardrail is None:
            return None
        if self._m_ref is None:
            self._m_ref = match_reference(self.market)
        return self._m_ref

    def evaluate(self, mechanism: str, values: dict[str, float], bidding_rule: str = "fixed_point_value_per_click",
                 by_type: bool = False) -> Outcome:
        start = time.perf_counter()
        market = self.market
        theta = np.array([values[j] for j in market.advertisers], dtype=float)
        human = agent = None
        gaps: dict[str, float] = {}
        converged, iterations, model_error, bound = True, 0, 0.0, 0.0
        if mechanism in ("M1", "M2"):
            if bidding_rule == "human_conversion_rate":
                bids = gsp.human_rate_bids(market, values)
            elif bidding_rule == "fixed_point_value_per_click":
                bids, converged, iterations = gsp.fixed_point_bids(self.table, mechanism, values)
            else:
                raise ValueError("unknown bidding rule %s" % bidding_rule)
            result = gsp.auction(market, mechanism, bids)
            prog = self.program("M3")  # only to evaluate the outcome of a fixed policy
            allocation = prog.allocation(gsp.policy(self.table, result.sponsored))
            paid = gsp.payments(market, result)
        elif mechanism in ("M3_set_blind", "M3", "M4"):
            prog = self.program(mechanism)
            out, allocation = pay.externality_payments(prog, theta)
            paid = dict(zip(market.advertisers, map(float, out)))
            bids = dict(zip(market.advertisers, map(float, theta)))
            model_error = allocation.cost_model_error
            if by_type:
                human, agent = {}, {}
                for k, j in enumerate(market.advertisers):
                    p = pay.integral_payment(prog, theta, k, pay.BidMap())
                    human[j], agent[j] = p.payment_human, p.payment_agent
                    bound += p.error_bound
                    gaps[j] = abs(p.payment - paid[j])
        elif mechanism == "profit_rule":
            prog = self.program("profit_rule")
            allocation = prog.solve(pay.coefficients(prog, theta))
            paid, human, agent = {}, {}, {}
            for k, j in enumerate(market.advertisers):
                p = pay.integral_payment(prog, theta, k, pay.bid_map(prog, k))
                paid[j], human[j], agent[j] = p.payment, p.payment_human, p.payment_agent
                bound += p.error_bound
            bids = dict(zip(market.advertisers, map(float, theta)))
            model_error = allocation.cost_model_error
        else:
            raise ValueError("unknown mechanism %s" % mechanism)
        outcome = self._outcome(mechanism, bids, allocation, paid, human, agent, time.perf_counter() - start,
                                converged, iterations, gaps, model_error)
        outcome.payment_error_bound = bound
        return outcome

    def _outcome(self, mechanism, bids, allocation: Allocation, paid, human, agent, seconds, converged, iterations,
                 gaps, model_error) -> Outcome:
        m_ref = self.m_ref
        slack = None if m_ref is None else allocation.match - (1 - self.market.guardrail.kappa) * m_ref
        return Outcome(market_id=self.market.market_id, market_hash=self.market.market_hash(), mechanism=mechanism,
                       bids=bids, X=allocation.X, X_human=allocation.X_human, X_agent=allocation.X_agent,
                       payments=paid, payments_human=human, payments_agent=agent,
                       contribution=allocation.contribution, cost=allocation.cost, match=allocation.match,
                       m_ref=m_ref, guardrail_slack=slack, seconds=seconds, converged=converged,
                       iterations=iterations, identity_gap=gaps, cost_model_error=model_error)


def evaluate(market: Market, mechanism: str, values: dict[str, float], **options) -> Outcome:
    return MarketRunner(market).evaluate(mechanism, values, **options)


@dataclass
class Batch:
    outcomes: list[Outcome]
    exclusions: Exclusions


def evaluate_markets(markets: list[tuple[Market, dict[str, float]]], mechanisms=MECHANISMS, **options) -> Batch:
    """Every mechanism on every (market, values); a solver failure excludes the market."""
    outcomes, exclusions = [], Exclusions()
    for market, values in markets:
        runner = MarketRunner(market)
        done = []
        try:
            for mechanism in mechanisms:
                current = mechanism
                done.append(runner.evaluate(mechanism, values, **options))
        except SolverFailure as failure:
            exclusions.add(market.market_id, current, failure)
            continue
        outcomes += done
    return Batch(outcomes, exclusions)


__all__ = ["MECHANISMS", "RULES", "Batch", "MarketRunner", "Outcome", "evaluate", "evaluate_markets"]
