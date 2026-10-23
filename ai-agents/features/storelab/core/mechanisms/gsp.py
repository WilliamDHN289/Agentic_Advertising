"""Click-based benchmarks M1 and M2: GSP position auctions (SL-16; l.487–512, Lemma click).

Both auctions are set-blind (A1 §1.7): in context x they fix one ordered sponsored
list S^s(x), whatever the organic ranking. With click bids b^c_j(x), sponsored slot k
goes to the remaining eligible advertiser with the highest b^c_j r_k(j), where

- M1 (CTR from pooled logs): r_k(j) = (1 − α(x)) γ_k ρ_j(x, 1) + α(x) χ̄(x), and every
  page request is charged;
- M2 (oracle, composition-corrected CTR): r_k(j) = γ_k ρ_j(x, 1), and only human page
  requests are charged (agent requests are discarded like filtered clicks);

χ̄(x) = Σ_m π_m(x) χ_m. A product with b^c_j r_k(j) = 0 is not allocated; ties go to the
smaller product id. The slot-k winner pays per charged click the next price: the
smallest bid that keeps the slot, max over the advertisers still unassigned of
b^c_i r_k(i), divided by its own r_k (0 if none).

Expected charged clicks per request in slot k: M1 (1 − α) γ_k ρ_j(x, 1) + α χ̄(x); M2
(1 − α) γ_k ρ_j(x, 1) (Ass. retrieval). Expected payment P_j = n Σ_x μ(x) price × clicks.

**Bidding rules** (config ``bidding_rules``), per advertiser and context:

- ``human_conversion_rate``: b^c_j(x) = θ_j ζ_j(x, 1), the value of a human click.
- ``fixed_point_value_per_click``: b^c_j(x) = θ_j E[choices of j at x | sponsored] /
  E[charged clicks of j at x], the value of a charged click under the auction's own
  outcome. A winner uses its slot; a loser the slot it would take as the last
  sponsored product (replacing the last winner when the list is full). Iterated from
  the human conversion rate bids until no bid moves by more than 1e-12 (relative);
  after ``MAX_ITERATIONS`` the outcome is reported with ``converged`` false. Where
  a slot has no charged clicks (M2 with α = 1) the bid stays at the human rate.

The standard auctions ignore the guardrail; the outcome reports whether they meet it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .market import ActionTable, Context, Market

MAX_ITERATIONS = 200
BID_TOL = 1e-12


@dataclass
class Auction:
    sponsored: dict[str, tuple[str, ...]]  # context -> S^s(x)
    prices: dict[str, dict[str, float]]  # context -> winner -> price per charged click
    clicks: dict[str, dict[str, float]]  # context -> winner -> charged clicks per request


def _chi_bar(market: Market, context: Context) -> float:
    return math.fsum(context.model_shares.get(m.key, 0.0) * m.page_request_prob for m in market.models)


def rates(market: Market, context: Context, mechanism: str, j: str, k: int) -> tuple[float, float]:
    """(ranking rate r_k(j), charged clicks per request) of j in sponsored slot k (1-based)."""
    alpha = context.agent_share
    human = market.position_weights[k - 1] * context.offers[j].sponsored.request
    if mechanism == "M1":
        pooled = (1 - alpha) * human + alpha * _chi_bar(market, context)
        return pooled, pooled
    if mechanism == "M2":
        return human, (1 - alpha) * human
    raise ValueError("GSP mechanisms are M1 and M2")


def auction(market: Market, mechanism: str, click_bids: dict[str, dict[str, float]]) -> Auction:
    sponsored, prices, clicks = {}, {}, {}
    advertisers = set(market.advertisers)
    for context in market.contexts:
        x = context.context_id
        remaining = sorted(j for j in context.offers if j in advertisers)
        chosen: list[str] = []
        prices[x], clicks[x] = {}, {}
        for k in range(1, market.max_sponsored + 1):
            scored = sorted(((click_bids[x].get(j, 0.0) * rates(market, context, mechanism, j, k)[0], j)
                             for j in remaining), key=lambda s: (-s[0], s[1]))
            if not scored or scored[0][0] <= 0.0:
                break
            winner = scored[0][1]
            rate, charged = rates(market, context, mechanism, winner, k)
            prices[x][winner] = (scored[1][0] / rate) if len(scored) > 1 else 0.0
            clicks[x][winner] = charged
            chosen.append(winner)
            remaining.remove(winner)
        sponsored[x] = tuple(chosen)
    return Auction(sponsored, prices, clicks)


def policy(table: ActionTable, sponsored: dict[str, tuple[str, ...]]) -> np.ndarray:
    """The row policy that plays S^s(x) under every organic ranking."""
    y = np.zeros(len(table.rows))
    for g, (ci, _) in enumerate(table.groups):
        y[table.find(g, sponsored[table.market.contexts[ci].context_id])] = 1.0
    return y


def _choices_when_sponsored(table: ActionTable, ci: int, sponsored: tuple[str, ...], j: str) -> float:
    """E over Q^o of q̄_j(x, (R, S^s)) per request in context x."""
    context = table.market.contexts[ci]
    total = 0.0
    for g, (c, oi) in enumerate(table.groups):
        if c == ci:
            total += context.organic[oi].probability * table.q_bar[table.find(g, sponsored), table.index[j]]
    return total


def human_rate_bids(market: Market, values: dict[str, float]) -> dict[str, dict[str, float]]:
    return {c.context_id: {j: values[j] * c.offers[j].sponsored.conversion for j in c.offers if j in values}
            for c in market.contexts}


def fixed_point_bids(table: ActionTable, mechanism: str, values: dict[str, float]) -> tuple[dict, bool, int]:
    """(click bids, converged, iterations) of the fixed-point rule."""
    market = table.market
    bids = human_rate_bids(market, values)
    for iteration in range(1, MAX_ITERATIONS + 1):
        result = auction(market, mechanism, bids)
        new = {}
        for ci, context in enumerate(market.contexts):
            x, winners = context.context_id, result.sponsored[context.context_id]
            new[x] = {}
            for j in bids[x]:
                if j in winners:
                    s, k = winners, winners.index(j) + 1
                elif market.max_sponsored == 0:
                    new[x][j] = bids[x][j]
                    continue
                else:
                    s = (winners[:market.max_sponsored - 1] if len(winners) == market.max_sponsored else winners) + (j,)
                    k = len(s)
                charged = rates(market, context, mechanism, j, k)[1]
                if charged <= 0.0:
                    new[x][j] = values[j] * context.offers[j].sponsored.conversion
                else:
                    new[x][j] = values[j] * _choices_when_sponsored(table, ci, s, j) / charged
        moved = max((abs(new[x][j] - bids[x][j]) / max(1.0, abs(bids[x][j])) for x in bids for j in bids[x]),
                    default=0.0)
        bids = new
        if moved <= BID_TOL:
            return bids, True, iteration
    return bids, False, MAX_ITERATIONS


def payments(market: Market, result: Auction) -> dict[str, float]:
    out = {j: 0.0 for j in market.advertisers}
    for context in market.contexts:
        x = context.context_id
        for j, price in result.prices[x].items():
            out[j] += market.requests * context.probability * price * result.clicks[x][j]
    return out
