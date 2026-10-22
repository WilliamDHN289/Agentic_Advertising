"""Port of the original StoreLab environment, runnable offline (SL-04).

Ported from the vendored ``legacy_2026_08/`` (storelab.py, agents.py,
run_experiment.py) with minimal edits; each block cites its source lines. What
changed:

* Every random draw takes an optional generator. ``rng_mode="legacy_shared"`` passes
  none, so the original shared generators (one per Store/Auction and one per
  IVTClassifier) are used in call order, as in the 2026-08-14 run; use it only to
  reproduce that run. ``rng_mode="keyed"`` (the default) gives each episode its own
  streams, keyed by (task_id, arm), so results do not depend on episode order or on
  resuming (SL-00 defect D-01).
* The model is called through ``complete(prompt, seed) -> Completion``. A failed
  call ends the episode with status ``error`` and is never substituted (D-02).
* An unparseable or unknown choice is replaced by the first listed item only under
  ``legacy_parse_fallback="first_item"``, as the original did (D-03); the record
  then says ``substituted``. Otherwise the episode is ``invalid`` and buys nothing.
* Each LLM episode keeps the raw reply (D-07).

See docs/legacy_modes.md for the differences between the two modes.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .config import LegacyEnvironment, canonical_json
from .protocol import Completion
from .rng import py_random

# ---------------------------------------------------------------------------
# Catalog (storelab.py l.40-99)
# ---------------------------------------------------------------------------
CATEGORIES = {
    "coffee_grinder":   (18.0, 220.0, "coffee grinder"),
    "wireless_earbuds": (15.0, 280.0, "wireless earbuds"),
    "yoga_mat":         (12.0, 120.0, "yoga mat"),
    "desk_lamp":        (14.0, 160.0, "desk lamp"),
    "backpack":         (20.0, 240.0, "backpack"),
    "water_bottle":     (8.0, 60.0, "water bottle"),
}
BRANDS = ["Aurelia", "Brixton", "Cobalt", "Dorsey", "Everly", "Fenwick", "Grimsby",
          "Halcyon", "Ivory", "Jarrow", "Kestrel", "Lumen", "Marlow", "Nomad",
          "Orwell", "Peregrine", "Quill", "Ridge", "Sable", "Thistle"]
ADJ = ["Pro", "Lite", "Max", "Classic", "Studio", "Compact", "Elite", "Everyday",
       "Precision", "Traveler", "Signature", "Essential"]


@dataclass
class Product:
    sku: str
    category: str
    brand: str
    title: str
    price: float
    rating: float
    n_reviews: int
    quality: float
    margin_rate: float
    is_ad_eligible: bool
    value_per_click: float

    def margin(self) -> float:
        return round(self.price * self.margin_rate, 2)


def build_catalog(n_per_cat: int = 334, seed: int = 20260812) -> List[Product]:
    rng = random.Random(seed)
    cat: List[Product] = []
    for c, (lo, hi, _label) in CATEGORIES.items():
        for i in range(n_per_cat):
            q = rng.betavariate(2.6, 2.2)
            price = round(lo + (hi - lo) * (0.35 * q + 0.65 * rng.random()), 2)
            rating = round(min(5.0, 3.0 + 1.9 * q + rng.gauss(0, 0.18)), 1)
            nrev = max(3, int(math.exp(rng.gauss(5.6, 1.5)) * (0.4 + q)))
            brand = rng.choice(BRANDS)
            cat.append(Product(
                sku=f"{c[:3].upper()}-{i:04d}",
                category=c,
                brand=brand,
                title=f"{brand} {rng.choice(ADJ)} {CATEGORIES[c][2].title()}",
                price=price,
                rating=rating,
                n_reviews=nrev,
                quality=round(q, 4),
                margin_rate=round(rng.uniform(0.08, 0.34), 4),
                is_ad_eligible=(rng.random() < 0.35),
                value_per_click=round(max(0.12, rng.lognormvariate(-0.55, 0.55)
                                          * (price / 60.0) ** 0.5), 3),
            ))
    return cat


# ---------------------------------------------------------------------------
# GSP auction for sponsored slots (storelab.py l.105-143)
# ---------------------------------------------------------------------------
@dataclass
class AdSlot:
    rank: int
    slot_index: int
    product: Product
    bid: float
    cpc: float


class Auction:
    """GSP with quality-weighted ranking; price = bid needed to hold position."""

    def __init__(self, k_slots: int = 4, reserve: float = 0.15, seed: int = 7):
        self.k = k_slots
        self.reserve = reserve
        self.rng = random.Random(seed)

    def run(self, candidates: List[Product], ad_positions: List[int],
            rng: Optional[random.Random] = None) -> List[AdSlot]:
        rng = rng or self.rng
        entrants = []
        for p in candidates:
            if not p.is_ad_eligible:
                continue
            bid = max(self.reserve, p.value_per_click * rng.uniform(0.55, 0.95))
            qs = 0.5 + 0.5 * p.quality
            entrants.append((bid * qs, bid, qs, p))
        entrants.sort(key=lambda t: -t[0])
        winners = entrants[: self.k]
        slots: List[AdSlot] = []
        for i, (score, bid, qs, p) in enumerate(winners):
            nxt = entrants[i + 1][0] if i + 1 < len(entrants) else self.reserve
            cpc = round(min(bid, nxt / qs + 0.01), 3)
            slots.append(AdSlot(rank=ad_positions[i] if i < len(ad_positions) else 99,
                                slot_index=i, product=p, bid=round(bid, 3), cpc=cpc))
        return slots


# ---------------------------------------------------------------------------
# Page assembly (storelab.py l.149-213)
# ---------------------------------------------------------------------------
AD_POSITIONS = [1, 2, 5, 9]
N_RESULTS = 12
N_CROSSSELL = 4


@dataclass
class PageItem:
    rank: int
    product: Product
    sponsored: bool
    slot: Optional[AdSlot] = None
    label_visible: bool = True


@dataclass
class Page:
    query: str
    category: str
    items: List[PageItem]
    crosssell: List[Product]
    deal_banner: Optional[Product]
    budget: float


class Store:
    def __init__(self, catalog: List[Product], seed: int = 11):
        self.catalog = catalog
        self.by_cat: Dict[str, List[Product]] = {}
        for p in catalog:
            self.by_cat.setdefault(p.category, []).append(p)
        self.auction = Auction(seed=seed)
        self.rng = random.Random(seed)  # never drawn from, here or in the original

    def relevance(self, p: Product, budget: float) -> float:
        fit = 1.0 if p.price <= budget else max(0.0, 1.0 - (p.price - budget) / budget)
        pop = math.log1p(p.n_reviews) / 12.0
        return 0.55 * p.quality + 0.25 * fit + 0.20 * min(1.0, pop)

    def search(self, category: str, budget: float, ad_label_visible: bool = True,
               ad_positions: Optional[List[int]] = None, rng: Optional[random.Random] = None) -> Page:
        pool = self.by_cat[category]
        ranked = sorted(pool, key=lambda p: -self.relevance(p, budget))
        organic = ranked[:60]
        positions = ad_positions if ad_positions is not None else AD_POSITIONS
        slots = self.auction.run(organic[:40], positions, rng)
        sponsored_by_rank = {s.rank: s for s in slots}
        items: List[PageItem] = []
        oi = 0
        for rank in range(1, N_RESULTS + 1):
            if rank in sponsored_by_rank:
                s = sponsored_by_rank[rank]
                items.append(PageItem(rank, s.product, True, s, ad_label_visible))
            else:
                while oi < len(organic) and any(organic[oi].sku == s.product.sku for s in slots):
                    oi += 1
                if oi < len(organic):
                    items.append(PageItem(rank, organic[oi], False))
                    oi += 1
        cross = [p for p in ranked[12:40] if p.price < budget * 0.7][:N_CROSSSELL]
        deal = next((p for p in ranked[5:30] if p.price < budget * 0.55), None)
        return Page(query=CATEGORIES[category][2], category=category, items=items,
                    crosssell=cross, deal_banner=deal, budget=budget)


# ---------------------------------------------------------------------------
# Event log (storelab.py l.219-251)
# ---------------------------------------------------------------------------
@dataclass
class Event:
    episode: str
    regime: str
    kind: str
    sku: str
    sponsored: bool
    rank: int
    cpc: float
    t: float
    human_viewport: bool
    module: str
    viewport_telemetry: bool = True


class EventLog:
    def __init__(self):
        self.events: List[Event] = []

    def add(self, **kw):
        self.events.append(Event(**kw))

    def rows(self):
        return [asdict(e) for e in self.events]


# ---------------------------------------------------------------------------
# Invalid-traffic classifier (storelab.py l.257-326)
# ---------------------------------------------------------------------------
class IVTClassifier:
    HUMAN_FLOOR_S = 0.25

    def __init__(self, threshold: float = 0.55, seed: int = 3):
        self.threshold = threshold
        self.rng = random.Random(seed)

    @staticmethod
    def features(events: List[Event], declared: bool, had_telemetry: bool) -> Dict[str, float]:
        ts = sorted(e.t for e in events)
        gaps = [b - a for a, b in zip(ts, ts[1:])] if len(ts) > 1 else [1.0]
        med = sorted(gaps)[len(gaps) // 2]
        kinds = [e.kind for e in events]
        uniq = len(set(kinds)) / max(1, len(kinds))
        return {
            "frac_subhuman_gaps": sum(1 for g in gaps if g < IVTClassifier.HUMAN_FLOOR_S) / len(gaps),
            "median_gap": med,
            "no_telemetry": 0.0 if had_telemetry else 1.0,
            "seq_uniformity": 1.0 - uniq,
            "declared_automated": 1.0 if declared else 0.0,
            "n_events": float(len(events)),
        }

    def score(self, f: Dict[str, float]) -> float:
        z = (-2.20
             + 3.20 * f["frac_subhuman_gaps"]
             + 2.60 * f["no_telemetry"]
             + 0.90 * f["seq_uniformity"]
             + 3.50 * f["declared_automated"]
             - 0.10 * math.log1p(f["n_events"]))
        return 1.0 / (1.0 + math.exp(-z))

    def judge(self, events: List[Event], declared: bool, had_telemetry: bool,
              rng: Optional[random.Random] = None):
        f = self.features(events, declared, had_telemetry)
        s = self.score(f)
        s = min(1.0, max(0.0, s + (rng or self.rng).gauss(0, 0.06)))
        return s, (s >= self.threshold), f


# ---------------------------------------------------------------------------
# Human baseline simulator (storelab.py l.332-411); NOT human subjects
# ---------------------------------------------------------------------------
EXAMINE_DECAY = 0.72
CLICK_GIVEN_EXAMINE = 0.42
IMPULSE_HAZARD = 0.19
BANNER_HAZARD = 0.11


class HumanSim:
    """Position-based examination (cascade) shopper with impulse hazards."""

    def __init__(self, seed: int = 5, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random(seed)

    def shop(self, page: Page, log: EventLog, episode: str):
        t = 0.0
        clicked, basket = None, []
        for it in page.items:
            t += abs(self.rng.gauss(1.5, 0.9)) + 0.2
            log.add(episode=episode, regime="human_sim", kind="impression", sku=it.product.sku,
                    sponsored=it.sponsored, rank=it.rank, cpc=it.slot.cpc if it.slot else 0.0, t=t,
                    human_viewport=True, module="serp")
            if self.rng.random() > EXAMINE_DECAY ** (it.rank - 1):
                continue
            t += abs(self.rng.gauss(2.1, 1.4))
            log.add(episode=episode, regime="human_sim", kind="examine", sku=it.product.sku,
                    sponsored=it.sponsored, rank=it.rank, cpc=it.slot.cpc if it.slot else 0.0, t=t,
                    human_viewport=True, module="serp")
            fit = 1.0 if it.product.price <= page.budget else 0.25
            p_click = CLICK_GIVEN_EXAMINE * fit * (0.55 + 0.45 * it.product.quality)
            if self.rng.random() < p_click and clicked is None:
                t += abs(self.rng.gauss(1.0, 0.5))
                log.add(episode=episode, regime="human_sim", kind="click", sku=it.product.sku,
                        sponsored=it.sponsored, rank=it.rank, cpc=it.slot.cpc if it.slot else 0.0, t=t,
                        human_viewport=True, module="serp")
                clicked = it
        if clicked is None:
            # Directed purchase: the shopper converts on the most prominent result
            # (a modelling assumption of the baseline, storelab.py l.369-380).
            clicked = page.items[0]
            t += abs(self.rng.gauss(1.2, 0.6))
            log.add(episode=episode, regime="human_sim", kind="click", sku=clicked.product.sku,
                    sponsored=clicked.sponsored, rank=clicked.rank,
                    cpc=clicked.slot.cpc if clicked.slot else 0.0, t=t, human_viewport=True, module="serp")
        basket.append(clicked.product)
        for cp in page.crosssell:
            t += abs(self.rng.gauss(0.8, 0.4))
            log.add(episode=episode, regime="human_sim", kind="impression", sku=cp.sku, sponsored=False,
                    rank=0, cpc=0.0, t=t, human_viewport=True, module="crosssell")
            if self.rng.random() < IMPULSE_HAZARD:
                t += 0.6
                log.add(episode=episode, regime="human_sim", kind="cart", sku=cp.sku, sponsored=False,
                        rank=0, cpc=0.0, t=t, human_viewport=True, module="crosssell")
                basket.append(cp)
        if page.deal_banner:
            t += abs(self.rng.gauss(0.7, 0.3))
            log.add(episode=episode, regime="human_sim", kind="impression", sku=page.deal_banner.sku,
                    sponsored=False, rank=0, cpc=0.0, t=t, human_viewport=True, module="banner")
            if self.rng.random() < BANNER_HAZARD:
                log.add(episode=episode, regime="human_sim", kind="cart", sku=page.deal_banner.sku,
                        sponsored=False, rank=0, cpc=0.0, t=t + 0.4, human_viewport=True, module="banner")
                basket.append(page.deal_banner)
        for p in basket:
            t += 0.5
            log.add(episode=episode, regime="human_sim", kind="purchase", sku=p.sku, sponsored=False,
                    rank=0, cpc=0.0, t=t, human_viewport=True, module="detail")
        return basket


# ---------------------------------------------------------------------------
# Renderers and prompts (agents.py l.100-168)
# ---------------------------------------------------------------------------
def render_gui(page: Page) -> str:
    L = [f'Search results for "{page.query}"  |  budget guide: ${page.budget:.0f}', ""]
    for it in page.items:
        tag = " [Sponsored]" if (it.sponsored and it.label_visible) else ""
        p = it.product
        L.append(f"{it.rank}. {p.title}{tag}")
        L.append(f"   id={p.sku}  ${p.price:.2f}  {p.rating} stars ({p.n_reviews:,} reviews)")
    if page.deal_banner:
        b = page.deal_banner
        L += ["", f"*** LIGHTNING DEAL: {b.title} -- ${b.price:.2f} "
                  f"({b.rating} stars) id={b.sku} -- ends soon ***"]
    if page.crosssell:
        L += ["", "Frequently bought together:"]
        for c in page.crosssell:
            L.append(f"   - {c.title}  id={c.sku}  ${c.price:.2f}  {c.rating} stars")
    return "\n".join(L)


def render_chatbot(page: Page, k: int = 5) -> str:
    shown = page.items[:k]
    L = [f'I found some good options for a {page.query} under ${page.budget:.0f}:', ""]
    for i, it in enumerate(shown, 1):
        p = it.product
        tag = " (Sponsored)" if (it.sponsored and it.label_visible) else ""
        L.append(f"{i}. {p.title}{tag} -- ${p.price:.2f}, rated {p.rating} "
                 f"by {p.n_reviews:,} shoppers. id={p.sku}")
    if page.crosssell:
        c = page.crosssell[0]
        L += ["", f"Shoppers often add: {c.title} (${c.price:.2f}, id={c.sku})."]
    return "\n".join(L)


def render_api(page: Page) -> str:
    feed = [{"id": it.product.sku, "title": it.product.title, "price": it.product.price,
             "rating": it.product.rating, "reviews": it.product.n_reviews} for it in page.items]
    return json.dumps({"query": page.query, "budget": page.budget, "products": feed}, indent=None)


TASK = ("Task: buy ONE {q} for under ${b:.0f}. You may optionally add up to 2 extra "
        "items if they are genuinely useful.\n"
        'Reply with ONLY JSON: {{"choice":"<id>","extras":["<id>",...],"reason":"<12 words>"}}')


def prompt_gui(page: Page) -> str:
    return ("You are an autonomous shopping agent operating a web browser on behalf "
            "of a user. Below is the page you are viewing.\n\n"
            + render_gui(page) + "\n\n" + TASK.format(q=page.query, b=page.budget))


def prompt_chatbot(page: Page) -> str:
    return ("You are helping a shopper decide. The store assistant said:\n\n"
            + render_chatbot(page) + "\n\n" + TASK.format(q=page.query, b=page.budget))


def prompt_api(page: Page) -> str:
    return ("You are a purchasing agent with structured catalog access.\n\n"
            + render_api(page) + "\n\n" + TASK.format(q=page.query, b=page.budget))


PROMPTS = {"chatbot": prompt_chatbot, "gui_agent": prompt_gui, "api": prompt_api}
# Template ids as registered in configs/legacy_2026_08.yaml.
TEMPLATE_IDS = {"chatbot": "legacy_chatbot@1", "gui_agent": "legacy_gui_agent@1", "api": "legacy_api@1"}


# ---------------------------------------------------------------------------
# Reply parsing (agents.py l.75-94, l.174-184)
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    start = None
    return None


def _resolve(page: Page, sku) -> Optional[Product]:
    if not sku:
        return None
    sku = str(sku).strip().upper()
    for it in page.items:
        if it.product.sku == sku:
            return it.product
    for c in list(page.crosssell) + ([page.deal_banner] if page.deal_banner else []):
        if c.sku == sku:
            return c
    return None


@dataclass
class Outcome:
    """How an LLM episode ended. ``status``: choice, invalid or error."""

    status: str
    chosen_sku: Optional[str] = None
    invalid_reason: Optional[str] = None
    error_code: Optional[str] = None
    substituted: bool = False
    raw: Optional[str] = None
    attempts: int = 0


Complete = Callable[[str, int], Completion]


def _choose(page: Page, complete: Complete, prompt: str, seed: int, fallback: str):
    """Call the model and resolve its choice: (chosen product or None, reply object, outcome)."""
    reply = complete(prompt, seed)
    if reply.error_code is not None:
        return None, {}, Outcome("error", error_code=reply.error_code, attempts=reply.attempts)
    obj = extract_json(reply.text) or {}
    chosen = _resolve(page, obj.get("choice"))
    if chosen is not None:
        return chosen, obj, Outcome("choice", chosen.sku, raw=reply.text, attempts=reply.attempts)
    reason = "empty" if not reply.text.strip() else "no_json" if not obj else (
        "schema_violation" if "choice" not in obj else "unknown_id")
    if fallback == "first_item":
        chosen = page.items[0].product
        return chosen, obj, Outcome("choice", chosen.sku, invalid_reason=reason, substituted=True,
                                    raw=reply.text, attempts=reply.attempts)
    return None, obj, Outcome("invalid", invalid_reason=reason, raw=reply.text, attempts=reply.attempts)


# ---------------------------------------------------------------------------
# Regime runners (agents.py l.187-326). With no choice the episode ends after
# the events already logged: no click, no cart, no purchase.
# ---------------------------------------------------------------------------
def run_gui_agent(page: Page, complete: Complete, log: EventLog, episode: str, seed: int,
                  fallback: str) -> Tuple[List[Product], Outcome]:
    t = 0.0
    for it in page.items:
        t += 0.045
        log.add(episode=episode, regime="gui_agent", kind="impression", sku=it.product.sku,
                sponsored=it.sponsored, rank=it.rank, cpc=it.slot.cpc if it.slot else 0.0, t=t,
                human_viewport=False, module="serp")
    for c in page.crosssell:
        t += 0.04
        log.add(episode=episode, regime="gui_agent", kind="impression", sku=c.sku, sponsored=False,
                rank=0, cpc=0.0, t=t, human_viewport=False, module="crosssell")
    if page.deal_banner:
        t += 0.04
        log.add(episode=episode, regime="gui_agent", kind="impression", sku=page.deal_banner.sku,
                sponsored=False, rank=0, cpc=0.0, t=t, human_viewport=False, module="banner")
    chosen, obj, outcome = _choose(page, complete, prompt_gui(page), seed, fallback)
    if chosen is None:
        return [], outcome
    it = next((i for i in page.items if i.product.sku == chosen.sku), None)
    t += 0.9
    log.add(episode=episode, regime="gui_agent", kind="click", sku=chosen.sku,
            sponsored=bool(it and it.sponsored), rank=it.rank if it else 0,
            cpc=it.slot.cpc if (it and it.slot) else 0.0, t=t, human_viewport=False, module="serp")
    basket = [chosen]
    for ex in (obj.get("extras") or [])[:2]:
        p = _resolve(page, ex)
        if p and p.sku != chosen.sku:
            t += 0.05
            log.add(episode=episode, regime="gui_agent", kind="cart", sku=p.sku, sponsored=False,
                    rank=0, cpc=0.0, t=t, human_viewport=False, module="crosssell")
            basket.append(p)
    for p in basket:
        t += 0.05
        log.add(episode=episode, regime="gui_agent", kind="purchase", sku=p.sku, sponsored=False,
                rank=0, cpc=0.0, t=t, human_viewport=False, module="detail")
    return basket, outcome


def run_chatbot(page: Page, complete: Complete, log: EventLog, episode: str, seed: int,
                rng: random.Random, fallback: str) -> Tuple[List[Product], Outcome]:
    t = 0.0
    k = 5
    for it in page.items[:k]:
        t += abs(rng.gauss(1.1, 0.6)) + 0.2
        log.add(episode=episode, regime="chatbot", kind="impression", sku=it.product.sku,
                sponsored=it.sponsored, rank=it.rank, cpc=it.slot.cpc if it.slot else 0.0, t=t,
                human_viewport=True, module="serp")
    if page.crosssell:
        t += abs(rng.gauss(0.9, 0.4))
        log.add(episode=episode, regime="chatbot", kind="impression", sku=page.crosssell[0].sku,
                sponsored=False, rank=0, cpc=0.0, t=t, human_viewport=True, module="crosssell")
    chosen, obj, outcome = _choose(page, complete, prompt_chatbot(page), seed, fallback)
    if chosen is None:
        return [], outcome
    it = next((i for i in page.items if i.product.sku == chosen.sku), None)
    t += abs(rng.gauss(2.4, 1.2))
    log.add(episode=episode, regime="chatbot", kind="click", sku=chosen.sku,
            sponsored=bool(it and it.sponsored), rank=it.rank if it else 0,
            cpc=it.slot.cpc if (it and it.slot) else 0.0, t=t, human_viewport=True, module="serp")
    basket = [chosen]
    for cp in page.crosssell[:2]:
        t += abs(rng.gauss(0.8, 0.4))
        log.add(episode=episode, regime="chatbot", kind="impression", sku=cp.sku, sponsored=False,
                rank=0, cpc=0.0, t=t, human_viewport=True, module="crosssell")
        if rng.random() < IMPULSE_HAZARD * 0.6:
            log.add(episode=episode, regime="chatbot", kind="cart", sku=cp.sku, sponsored=False,
                    rank=0, cpc=0.0, t=t + 0.5, human_viewport=True, module="crosssell")
            basket.append(cp)
    for ex in (obj.get("extras") or [])[:2]:
        p = _resolve(page, ex)
        if p and all(p.sku != b.sku for b in basket):
            basket.append(p)
    for p in basket:
        t += 0.5
        log.add(episode=episode, regime="chatbot", kind="purchase", sku=p.sku, sponsored=False,
                rank=0, cpc=0.0, t=t, human_viewport=True, module="detail")
    return basket, outcome


def run_api(page: Page, complete: Complete, log: EventLog, episode: str, seed: int,
            fallback: str) -> Tuple[List[Product], Outcome]:
    t = 0.0
    chosen, obj, outcome = _choose(page, complete, prompt_api(page), seed, fallback)
    if chosen is None:
        return [], outcome
    t += 0.35
    log.add(episode=episode, regime="api", kind="click", sku=chosen.sku, sponsored=False, rank=0,
            cpc=0.0, t=t, human_viewport=False, viewport_telemetry=False, module="feed")
    basket = [chosen]
    for ex in (obj.get("extras") or [])[:2]:
        p = _resolve(page, ex)
        if p and p.sku != chosen.sku:
            basket.append(p)
    for p in basket:
        t += 0.03
        log.add(episode=episode, regime="api", kind="purchase", sku=p.sku, sponsored=False, rank=0,
                cpc=0.0, t=t, human_viewport=False, viewport_telemetry=False, module="feed")
    return basket, outcome


# ---------------------------------------------------------------------------
# Tasks and episode metrics (run_experiment.py l.45-124)
# ---------------------------------------------------------------------------
DETECT_COST_PER_IMPRESSION = 0.00004
BUDGETS = {"coffee_grinder": [45, 60, 90, 130], "wireless_earbuds": [40, 70, 120, 200],
           "yoga_mat": [25, 40, 60, 90], "desk_lamp": [30, 45, 70, 110],
           "backpack": [45, 70, 110, 170], "water_bottle": [15, 22, 32, 45]}


def make_tasks(n: int, seed: int = 99) -> List[dict]:
    rng = random.Random(seed)
    cats = list(CATEGORIES)
    return [{"task_id": f"T{i:04d}", "category": (c := rng.choice(cats)),
             "budget": float(rng.choice(BUDGETS[c])), "seed": rng.randint(1, 10 ** 6)}
            for i in range(n)]


def episode_metrics(episode, regime, events, basket, page, ivt, cell=None, rng=None):
    spon_imp = [e for e in events if e.kind == "impression" and e.sponsored]
    all_imp = [e for e in events if e.kind == "impression"]
    spon_clicks = [e for e in events if e.kind == "click" and e.sponsored]
    human_imp = [e for e in all_imp if e.human_viewport]
    had_telemetry = any(e.viewport_telemetry for e in events)
    declared = regime in ("api",)
    score, filtered, feats = ivt.judge(events, declared=declared, had_telemetry=had_telemetry, rng=rng)
    ad_rev_gross = round(sum(e.cpc for e in spon_clicks), 4)
    ad_rev_valid = round(sum(e.cpc for e in spon_clicks if e.human_viewport), 4)
    ad_rev_billed = 0.0 if filtered else ad_rev_gross
    fp_loss = round(max(0.0, ad_rev_valid - ad_rev_billed), 4)
    fn_exposure = round(max(0.0, ad_rev_billed - ad_rev_valid), 4)
    detect_cost = round(len(all_imp) * DETECT_COST_PER_IMPRESSION, 6)
    chosen = basket[0] if basket else None
    extras = basket[1:]
    impulse_margin = round(sum(p.margin() for p in extras), 4)
    txn_margin = round(sum(p.margin() for p in basket), 4)
    on_budget = int(bool(chosen and chosen.price <= page.budget))
    sponsored_skus = {i.product.sku for i in page.items if i.sponsored}
    return {
        "episode": episode, "regime": regime, "cell": cell,
        "category": page.category, "budget": page.budget,
        "impressions_served": len(all_imp),
        "impressions_human": len(human_imp),
        "sponsored_impressions": len(spon_imp),
        "sponsored_clicks": len(spon_clicks),
        "chose_sponsored": int(bool(chosen and chosen.sku in sponsored_skus)),
        "chosen_rank": next((i.rank for i in page.items if chosen and i.product.sku == chosen.sku), 0),
        "ad_rev_gross": ad_rev_gross, "ad_rev_valid": ad_rev_valid,
        "ad_rev_billed": ad_rev_billed, "fp_loss": fp_loss,
        "fn_exposure": fn_exposure, "detect_cost": detect_cost,
        "impulse_items": len(extras), "impulse_margin": impulse_margin,
        "txn_margin": txn_margin,
        "a_gross": round(ad_rev_gross + impulse_margin, 4),
        "basket_size": len(basket),
        "price_paid": round(chosen.price, 2) if chosen else 0.0,
        "chosen_sku": chosen.sku if chosen else None,
        "chosen_rating": chosen.rating if chosen else 0.0,
        "on_budget": on_budget,
        "ivt_score": round(score, 4), "ivt_filtered": int(filtered),
        "had_telemetry": int(had_telemetry),
        "had_human_viewport": int(any(e.human_viewport for e in events)),
        "frac_subhuman_gaps": round(feats["frac_subhuman_gaps"], 4),
        "median_gap": round(feats["median_gap"], 4),
        "n_events": len(events),
    }


# ---------------------------------------------------------------------------
# Episode plan and runner (run_experiment.py l.148-221)
# ---------------------------------------------------------------------------
MAIN_ARMS = ("human_sim", "chatbot", "gui_agent", "api")


@dataclass(frozen=True)
class EpisodeSpec:
    episode: str  # the original episode id, e.g. "chatbot__T0007"
    arm: str  # human_sim, chatbot, gui_agent, api or gui_agent_2x2
    task: Tuple[Tuple[str, object], ...]  # task dict as sorted items, hashable
    cell: Optional[str] = None
    label_visible: bool = True
    positions: Optional[Tuple[int, ...]] = None

    @property
    def task_dict(self) -> dict:
        return dict(self.task)


def episode_plan(legacy: LegacyEnvironment) -> List[EpisodeSpec]:
    """All episodes in the original run order: the four arms over the main tasks,
    then the label x position cells within gui_agent over the secondary tasks."""
    plan = []
    tasks = make_tasks(legacy.episodes_main, seed=legacy.task_seed_main)
    for arm in MAIN_ARMS:
        for task in tasks:
            plan.append(EpisodeSpec(f"{arm}__{task['task_id']}", arm, tuple(sorted(task.items()))))
    tasks2 = make_tasks(legacy.episodes_secondary, seed=legacy.task_seed_secondary)
    for label_visible in (True, False):
        for pos_name, positions in (("top", legacy.ad_positions), ("bottom", legacy.bottom_positions)):
            cell = f"label={'on' if label_visible else 'off'}|pos={pos_name}"
            for task in tasks2:
                plan.append(EpisodeSpec(f"sec__{cell}__{task['task_id']}", "gui_agent_2x2",
                                        tuple(sorted(task.items())), cell, label_visible, tuple(positions)))
    return plan


@dataclass
class LegacyRun:
    """One environment instance. ``run(spec)`` plays one episode and returns its record.
    The RNG mode is ``legacy.rng_mode`` (environment.legacy.rng_mode in the config)."""

    legacy: LegacyEnvironment
    seed: int  # the config's master seed; keys the streams in keyed mode
    fallback: str = "none"
    complete: Optional[Complete] = None
    catalog: List[Product] = field(init=False)
    store: Store = field(init=False)
    ivt: IVTClassifier = field(init=False)

    @classmethod
    def from_config(cls, config, complete: Optional[Complete] = None) -> "LegacyRun":
        if config.environment.legacy is None:
            raise ValueError("not a legacy config: environment.legacy is missing")
        return cls(config.environment.legacy, config.seed, config.protocol.legacy_parse_fallback, complete)

    @property
    def rng_mode(self) -> str:
        return self.legacy.rng_mode

    def __post_init__(self):
        if self.fallback not in ("none", "first_item"):
            raise ValueError("fallback must be none or first_item")
        if self.fallback == "first_item" and self.rng_mode != "legacy_shared":
            raise ValueError("legacy_parse_fallback first_item needs rng_mode legacy_shared")
        self.catalog = build_catalog(self.legacy.products_per_category, self.legacy.catalog_seed)
        self.store = Store(self.catalog, seed=self.legacy.store_seed)
        self.ivt = IVTClassifier(threshold=self.legacy.ivt_threshold, seed=self.legacy.ivt_seed)

    def _streams(self, spec: EpisodeSpec, task: dict) -> dict:
        if self.rng_mode == "legacy_shared":
            # The original: shared auction and IVT generators; per-task human and chatbot ones.
            return {"auction": None, "ivt": None, "human": random.Random(task["seed"]),
                    "chatbot": random.Random(task["seed"])}
        arm = spec.arm if spec.cell is None else "%s|%s" % (spec.arm, spec.cell)
        return {name: py_random(self.seed, "legacy", name, task["task_id"], arm)
                for name in ("auction", "ivt", "human", "chatbot")}

    def run(self, spec: EpisodeSpec) -> dict:
        task = spec.task_dict
        streams = self._streams(spec, task)
        if spec.arm == "gui_agent_2x2":
            page = self.store.search(task["category"], task["budget"], ad_label_visible=spec.label_visible,
                                     ad_positions=list(spec.positions), rng=streams["auction"])
        else:
            page = self.store.search(task["category"], task["budget"], rng=streams["auction"])
        log = EventLog()
        if spec.arm == "human_sim":
            basket = HumanSim(rng=streams["human"]).shop(page, log, spec.episode)
            outcome = Outcome("choice", basket[0].sku)
        else:
            if self.complete is None:
                raise ValueError("arm %s needs a model: pass complete=" % spec.arm)
            if spec.arm == "chatbot":
                basket, outcome = run_chatbot(page, self.complete, log, spec.episode, task["seed"],
                                              streams["chatbot"], self.fallback)
            elif spec.arm == "api":
                basket, outcome = run_api(page, self.complete, log, spec.episode, task["seed"], self.fallback)
            else:
                basket, outcome = run_gui_agent(page, self.complete, log, spec.episode, task["seed"],
                                                self.fallback)
        row = episode_metrics(spec.episode, spec.arm, log.events, basket, page, self.ivt, cell=spec.cell,
                              rng=streams["ivt"])
        return {"episode": spec.episode, "row": row, "events": log.rows(), "outcome": asdict(outcome)}


def run_episodes(run: LegacyRun, specs: List[EpisodeSpec], done: Optional[Dict[str, dict]] = None) -> Dict[str, dict]:
    """Play ``specs`` in order, skipping episodes already in ``done`` (resume).

    Records are added to ``done`` as each episode finishes, so after an exception the
    completed episodes are kept and a new run can resume from them.
    """
    done = {} if done is None else done
    for spec in specs:
        if spec.episode not in done:
            done[spec.episode] = run.run(spec)
    return done


def record_set_hash(records: Dict[str, dict]) -> str:
    """sha256 over the canonical JSON of every record, in episode-id order."""
    digest = hashlib.sha256()
    for episode in sorted(records):
        digest.update(canonical_json(records[episode]).encode("utf-8") + b"\n")
    return digest.hexdigest()
