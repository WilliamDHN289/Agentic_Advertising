#!/usr/bin/env python3
"""
storelab.py -- An instrumented sandbox marketplace.

The measurement apparatus points at the PLATFORM, not at the agent. Everything a
retail-media P&L is built from is logged server-side: every ad impression, every
sponsored click, every cross-sell module render, every basket event, with ground
truth on whether a human viewport was ever involved.

Components
  Catalog          ~2,000 SKUs over 6 categories, seeded and reproducible.
  Auction          Generalized second-price auction for sponsored slots
                   (Edelman-Ostrovsky-Schwarz / Varian), with per-advertiser
                   value-per-click and quality scores.
  Page             Amazon-like SERP: sponsored slots interleaved into organic
                   ranking, "frequently bought together" cross-sell, deal banner.
  Regimes          Four rendering surfaces differing in attention passthrough:
                   human_sim, chatbot, gui_agent, api.
  EventLog         Impression / examine / click / cart / purchase events.
  IVTClassifier    Invalid-traffic detector run over session features; its error
                   rates ARE the contamination cost.

IMPORTANT -- what the human arm is and is not
  The `human_sim` arm is NOT human subjects. It is a calibrated behavioral
  simulator: a position-based examination (cascade) model with rank-decaying
  attention, plus impulse and cross-sell hazards. Its parameters are set from
  published position-auction and attention-share magnitudes, not estimated here.
  It supplies a *baseline funnel* against which agent regimes are differenced.
  Every result involving human_sim is therefore a comparison against a stipulated
  baseline, not an estimated human effect. The proposal's Prolific arm replaces it.
"""
from __future__ import annotations
import json, math, random, time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
CATEGORIES = {
    "coffee_grinder":   (18.0, 220.0, "coffee grinder"),
    "wireless_earbuds": (15.0, 280.0, "wireless earbuds"),
    "yoga_mat":         (12.0, 120.0, "yoga mat"),
    "desk_lamp":        (14.0, 160.0, "desk lamp"),
    "backpack":         (20.0, 240.0, "backpack"),
    "water_bottle":     ( 8.0,  60.0, "water bottle"),
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
    quality: float          # latent quality, drives organic relevance
    margin_rate: float      # platform gross margin share of price
    is_ad_eligible: bool
    value_per_click: float  # advertiser's private value -> GSP bid

    def margin(self) -> float:
        return round(self.price * self.margin_rate, 2)


def build_catalog(n_per_cat: int = 334, seed: int = 20260812) -> List[Product]:
    rng = random.Random(seed)
    cat: List[Product] = []
    for c, (lo, hi, _label) in CATEGORIES.items():
        for i in range(n_per_cat):
            q = rng.betavariate(2.6, 2.2)                       # latent quality in (0,1)
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
                # ~35% of the catalog advertises
                is_ad_eligible=(rng.random() < 0.35),
                # advertiser value per click rises with price and margin
                value_per_click=round(max(0.12, rng.lognormvariate(-0.55, 0.55)
                                          * (price / 60.0) ** 0.5), 3),
            ))
    return cat


# ---------------------------------------------------------------------------
# Generalized second-price auction for sponsored slots
# ---------------------------------------------------------------------------
POSITION_CTR = [0.088, 0.052, 0.034, 0.024, 0.018, 0.014, 0.011, 0.009, 0.007, 0.006]


@dataclass
class AdSlot:
    rank: int               # position on the rendered page (1-indexed)
    slot_index: int         # which sponsored slot (0..K-1)
    product: Product
    bid: float
    cpc: float              # GSP price actually charged on a click


class Auction:
    """GSP with quality-weighted ranking; price = bid needed to hold position."""

    def __init__(self, k_slots: int = 4, reserve: float = 0.15, seed: int = 7):
        self.k = k_slots
        self.reserve = reserve
        self.rng = random.Random(seed)

    def run(self, candidates: List[Product], ad_positions: List[int]) -> List[AdSlot]:
        entrants = []
        for p in candidates:
            if not p.is_ad_eligible:
                continue
            # bid shading around true value, as in a GSP symmetric equilibrium
            bid = max(self.reserve, p.value_per_click * self.rng.uniform(0.55, 0.95))
            qs = 0.5 + 0.5 * p.quality            # quality score
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
# Page assembly
# ---------------------------------------------------------------------------
AD_POSITIONS = [1, 2, 5, 9]      # where sponsored results sit in the result list
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
        self.rng = random.Random(seed)

    def relevance(self, p: Product, budget: float) -> float:
        fit = 1.0 if p.price <= budget else max(0.0, 1.0 - (p.price - budget) / budget)
        pop = math.log1p(p.n_reviews) / 12.0
        return 0.55 * p.quality + 0.25 * fit + 0.20 * min(1.0, pop)

    def search(self, category: str, budget: float,
               ad_label_visible: bool = True,
               ad_positions: Optional[List[int]] = None) -> Page:
        pool = self.by_cat[category]
        ranked = sorted(pool, key=lambda p: -self.relevance(p, budget))
        organic = ranked[:60]
        positions = ad_positions if ad_positions is not None else AD_POSITIONS
        slots = self.auction.run(organic[:40], positions)

        sponsored_by_rank = {s.rank: s for s in slots}
        items: List[PageItem] = []
        oi = 0
        for rank in range(1, N_RESULTS + 1):
            if rank in sponsored_by_rank:
                s = sponsored_by_rank[rank]
                items.append(PageItem(rank, s.product, True, s, ad_label_visible))
            else:
                # skip organic products already shown as ads
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
# Event log
# ---------------------------------------------------------------------------
@dataclass
class Event:
    episode: str
    regime: str
    kind: str               # impression | examine | click | cart | purchase
    sku: str
    sponsored: bool
    rank: int
    cpc: float
    t: float                # seconds since episode start
    human_viewport: bool    # GROUND TRUTH: was a human eye ever on this?
    module: str             # serp | crosssell | banner | detail | feed
    viewport_telemetry: bool = True
    # OBSERVABLE: did the client emit viewport/scroll/hover events?
    #
    # These two must not be conflated, and conflating them is what makes the
    # contamination channel invisible. A third-party GUI agent such as Comet IS a
    # Chromium browser: it lays out the page, scrolls it, and emits viewport and
    # hover telemetry -- with no human eye behind any of it. So an agent session
    # has human_viewport=False and viewport_telemetry=True, and the platform's
    # detector can only see the second one. Only a headless/protocol client emits
    # no telemetry at all.


class EventLog:
    def __init__(self):
        self.events: List[Event] = []

    def add(self, **kw):
        self.events.append(Event(**kw))

    def rows(self):
        return [asdict(e) for e in self.events]


# ---------------------------------------------------------------------------
# Invalid-traffic (IVT) classifier
# ---------------------------------------------------------------------------
class IVTClassifier:
    """A production-shaped invalid-traffic detector.

    Scores a session on the signals real IVT vendors use -- inter-event timing
    regularity, absence of viewport/hover telemetry, event-sequence entropy,
    dwell variance, and whether the client declared itself automated. Sessions
    above `threshold` are filtered and not billed.

    Its errors are the object of interest:
      false negative = agent session billed as human   -> advertiser value destroyed
      false positive = human session filtered          -> real revenue destroyed
    """

    def __init__(self, threshold: float = 0.55, seed: int = 3):
        self.threshold = threshold
        self.rng = random.Random(seed)

    # A human cannot dispatch two interface actions closer together than this.
    HUMAN_FLOOR_S = 0.25

    @staticmethod
    def features(events: List[Event], declared: bool,
                 had_telemetry: bool) -> Dict[str, float]:
        """Only OBSERVABLE signals may appear here.

        `human_viewport` is deliberately absent: it is ground truth the platform
        does not get to see. Using it would hand the detector the answer key and
        drive contamination to zero by construction.
        """
        ts = sorted(e.t for e in events)
        gaps = [b - a for a, b in zip(ts, ts[1:])] if len(ts) > 1 else [1.0]
        med = sorted(gaps)[len(gaps) // 2]
        kinds = [e.kind for e in events]
        uniq = len(set(kinds)) / max(1, len(kinds))
        return {
            # pacing: the workhorse signal once viewport telemetry is present in
            # both classes. Machines dispatch faster than a hand can move.
            "frac_subhuman_gaps": sum(1 for g in gaps
                                      if g < IVTClassifier.HUMAN_FLOOR_S) / len(gaps),
            "median_gap": med,
            "no_telemetry": 0.0 if had_telemetry else 1.0,
            "seq_uniformity": 1.0 - uniq,
            "declared_automated": 1.0 if declared else 0.0,
            "n_events": float(len(events)),
        }

    def score(self, f: Dict[str, float]) -> float:
        """Logistic score over observable signals.

        Weights are STIPULATED, not estimated -- there is no labelled corpus of
        agent vs. human retail sessions to fit them on. They are set so that a
        machine-paced client is caught, a self-declared client is caught, and a
        browser agent that paces itself like a person is not separable from a
        person. That last property is the empirically important one and it is an
        assumption of the design, not a finding of it.
        """
        z = (-2.20
             + 3.20 * f["frac_subhuman_gaps"]     # machine pacing
             + 2.60 * f["no_telemetry"]           # headless / protocol client
             + 0.90 * f["seq_uniformity"]
             + 3.50 * f["declared_automated"]
             - 0.10 * math.log1p(f["n_events"]))
        return 1.0 / (1.0 + math.exp(-z))

    def judge(self, events: List[Event], declared: bool, had_telemetry: bool):
        f = self.features(events, declared, had_telemetry)
        s = self.score(f)
        # detectors are noisy in production; add measurement jitter
        s = min(1.0, max(0.0, s + self.rng.gauss(0, 0.06)))
        return s, (s >= self.threshold), f


# ---------------------------------------------------------------------------
# Human baseline simulator (NOT human subjects -- see module docstring)
# ---------------------------------------------------------------------------
EXAMINE_DECAY = 0.72          # P(examine rank r) = EXAMINE_DECAY^(r-1)
CLICK_GIVEN_EXAMINE = 0.42
IMPULSE_HAZARD = 0.19         # P(add an unplanned item from cross-sell)
BANNER_HAZARD = 0.11


class HumanSim:
    """Position-based examination (cascade) shopper with impulse hazards."""

    def __init__(self, seed: int = 5):
        self.rng = random.Random(seed)

    def shop(self, page: Page, log: EventLog, episode: str):
        t = 0.0
        clicked, basket = None, []
        for it in page.items:
            t += abs(self.rng.gauss(1.5, 0.9)) + 0.2
            log.add(episode=episode, regime="human_sim", kind="impression",
                    sku=it.product.sku, sponsored=it.sponsored, rank=it.rank,
                    cpc=it.slot.cpc if it.slot else 0.0, t=t,
                    human_viewport=True, module="serp")
            if self.rng.random() > EXAMINE_DECAY ** (it.rank - 1):
                continue                                   # never scrolled that far
            t += abs(self.rng.gauss(2.1, 1.4))
            log.add(episode=episode, regime="human_sim", kind="examine",
                    sku=it.product.sku, sponsored=it.sponsored, rank=it.rank,
                    cpc=it.slot.cpc if it.slot else 0.0, t=t,
                    human_viewport=True, module="serp")
            fit = 1.0 if it.product.price <= page.budget else 0.25
            p_click = CLICK_GIVEN_EXAMINE * fit * (0.55 + 0.45 * it.product.quality)
            if self.rng.random() < p_click and clicked is None:
                t += abs(self.rng.gauss(1.0, 0.5))
                log.add(episode=episode, regime="human_sim", kind="click",
                        sku=it.product.sku, sponsored=it.sponsored, rank=it.rank,
                        cpc=it.slot.cpc if it.slot else 0.0, t=t,
                        human_viewport=True, module="serp")
                clicked = it
        if clicked is None:
            # The task is a directed purchase, so a shopper who examined without
            # clicking still converts on the most prominent result. Log the click:
            # omitting it would understate the baseline's sponsored CTR and so
            # understate measured displacement in every agent arm.
            clicked = page.items[0]
            t += abs(self.rng.gauss(1.2, 0.6))
            log.add(episode=episode, regime="human_sim", kind="click",
                    sku=clicked.product.sku, sponsored=clicked.sponsored,
                    rank=clicked.rank,
                    cpc=clicked.slot.cpc if clicked.slot else 0.0, t=t,
                    human_viewport=True, module="serp")
        basket.append(clicked.product)

        # cross-sell module: rendered on the detail page, then impulse hazard
        for cp in page.crosssell:
            t += abs(self.rng.gauss(0.8, 0.4))
            log.add(episode=episode, regime="human_sim", kind="impression",
                    sku=cp.sku, sponsored=False, rank=0, cpc=0.0, t=t,
                    human_viewport=True, module="crosssell")
            if self.rng.random() < IMPULSE_HAZARD:
                t += 0.6
                log.add(episode=episode, regime="human_sim", kind="cart",
                        sku=cp.sku, sponsored=False, rank=0, cpc=0.0, t=t,
                        human_viewport=True, module="crosssell")
                basket.append(cp)
        if page.deal_banner:
            t += abs(self.rng.gauss(0.7, 0.3))
            log.add(episode=episode, regime="human_sim", kind="impression",
                    sku=page.deal_banner.sku, sponsored=False, rank=0, cpc=0.0, t=t,
                    human_viewport=True, module="banner")
            if self.rng.random() < BANNER_HAZARD:
                log.add(episode=episode, regime="human_sim", kind="cart",
                        sku=page.deal_banner.sku, sponsored=False, rank=0, cpc=0.0,
                        t=t + 0.4, human_viewport=True, module="banner")
                basket.append(page.deal_banner)

        for p in basket:
            t += 0.5
            log.add(episode=episode, regime="human_sim", kind="purchase",
                    sku=p.sku, sponsored=False, rank=0, cpc=0.0, t=t,
                    human_viewport=True, module="detail")
        return basket
