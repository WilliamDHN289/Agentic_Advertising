#!/usr/bin/env python3
"""
agents.py -- The three LLM-mediated interface regimes.

Each regime differs ONLY in the surface the decision-maker perceives and in which
server-side events that surface fires. The catalog, ranking, auction, and prices
are identical across regimes, so any difference in platform revenue is caused by
the interface, not by the assortment.

  chatbot    Platform-curated conversational cards (top-k) with a sponsored
             prompt slot. The model picks; a human then clicks through to a
             detail page. Impressions fire and ARE human-viewed -> valid.
  gui_agent  Full rendered SERP text: every sponsored slot, cross-sell module and
             banner is served and logged. No human eye is behind any of it, but
             the client is a real browser, so it still emits viewport telemetry.
             These are the contaminating impressions.
  api        Structured product feed. No ad slots, no modules, nothing rendered,
             and no viewport telemetry at all. Zero impressions -- clean, total
             displacement.

Telemetry, per regime (human_viewport = truth / viewport_telemetry = observable):
  human_sim  True  / True      chatbot  True  / True
  gui_agent  False / True      api      False / False

The gui_agent row is the whole point. A browser-driving agent is observationally
a browser; only the pacing of its actions distinguishes it, and an agent that
paces itself like a person is not separable from a person at all.

Backend: a local model served by ollama. Everything is offline.
"""
from __future__ import annotations
import json, re, time, urllib.request
from typing import Dict, List, Optional, Tuple

from storelab import (Page, PageItem, EventLog, Product, IMPULSE_HAZARD,
                      EXAMINE_DECAY, CLICK_GIVEN_EXAMINE)

OLLAMA = "http://127.0.0.1:11434/api/generate"


class LocalLLM:
    """Thin ollama client with retry and JSON extraction."""

    def __init__(self, model: str = "qwen2:7b", temperature: float = 0.7,
                 num_predict: int = 120, timeout: int = 180):
        self.model, self.temperature = model, temperature
        self.num_predict, self.timeout = num_predict, timeout
        self.n_calls = 0
        self.n_parse_fail = 0
        self.total_latency = 0.0

    def generate(self, prompt: str, seed: Optional[int] = None) -> str:
        payload = {
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": self.temperature,
                        "num_predict": self.num_predict, "top_p": 0.9},
        }
        if seed is not None:
            payload["options"]["seed"] = seed
        data = json.dumps(payload).encode()
        for attempt in range(3):
            try:
                t0 = time.time()
                req = urllib.request.Request(
                    OLLAMA, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read().decode())
                self.n_calls += 1
                self.total_latency += time.time() - t0
                return out.get("response", "")
            except Exception:
                time.sleep(2 * (attempt + 1))
        return ""

    @staticmethod
    def extract_json(text: str) -> Optional[dict]:
        if not text:
            return None
        text = re.sub(r"```(?:json)?", "", text)
        # first balanced {...}
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


# ---------------------------------------------------------------------------
# Surface renderers
# ---------------------------------------------------------------------------
def render_gui(page: Page) -> str:
    """Full storefront as an autonomous browser agent would parse it."""
    L = [f'Search results for "{page.query}"  |  budget guide: ${page.budget:.0f}', ""]
    for it in page.items:
        tag = ""
        if it.sponsored and it.label_visible:
            tag = " [Sponsored]"
        elif it.sponsored:
            tag = ""                      # label hidden -- secondary manipulation
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
    """Platform-curated conversational cards, with one sponsored prompt."""
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
    """Structured feed: no ad slots, no modules, no presentation."""
    feed = [{"id": it.product.sku, "title": it.product.title,
             "price": it.product.price, "rating": it.product.rating,
             "reviews": it.product.n_reviews} for it in page.items]
    return json.dumps({"query": page.query, "budget": page.budget,
                       "products": feed}, indent=None)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Regime runners
# ---------------------------------------------------------------------------
def _resolve(page: Page, sku: Optional[str]) -> Optional[Product]:
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


def run_gui_agent(page: Page, llm: LocalLLM, log: EventLog, episode: str,
                  seed: int) -> Tuple[List[Product], dict]:
    """Every rendered element is served and logged -- with no human viewport."""
    t = 0.0
    # The agent fetches and parses the whole page: all impressions fire.
    for it in page.items:
        t += 0.045                     # machine-paced, near-constant inter-event gap
        log.add(episode=episode, regime="gui_agent", kind="impression",
                sku=it.product.sku, sponsored=it.sponsored, rank=it.rank,
                cpc=it.slot.cpc if it.slot else 0.0, t=t,
                human_viewport=False, module="serp")
    for c in page.crosssell:
        t += 0.04
        log.add(episode=episode, regime="gui_agent", kind="impression", sku=c.sku,
                sponsored=False, rank=0, cpc=0.0, t=t,
                human_viewport=False, module="crosssell")
    if page.deal_banner:
        t += 0.04
        log.add(episode=episode, regime="gui_agent", kind="impression",
                sku=page.deal_banner.sku, sponsored=False, rank=0, cpc=0.0, t=t,
                human_viewport=False, module="banner")

    raw = llm.generate(prompt_gui(page), seed=seed)
    obj = llm.extract_json(raw) or {}
    chosen = _resolve(page, obj.get("choice"))
    if chosen is None:
        llm.n_parse_fail += 1
        chosen = page.items[0].product

    it = next((i for i in page.items if i.product.sku == chosen.sku), None)
    t += 0.9
    log.add(episode=episode, regime="gui_agent", kind="click", sku=chosen.sku,
            sponsored=bool(it and it.sponsored), rank=it.rank if it else 0,
            cpc=it.slot.cpc if (it and it.slot) else 0.0, t=t,
            human_viewport=False, module="serp")

    basket = [chosen]
    for ex in (obj.get("extras") or [])[:2]:
        p = _resolve(page, ex)
        if p and p.sku != chosen.sku:
            t += 0.05
            log.add(episode=episode, regime="gui_agent", kind="cart", sku=p.sku,
                    sponsored=False, rank=0, cpc=0.0, t=t,
                    human_viewport=False, module="crosssell")
            basket.append(p)
    for p in basket:
        t += 0.05
        log.add(episode=episode, regime="gui_agent", kind="purchase", sku=p.sku,
                sponsored=False, rank=0, cpc=0.0, t=t,
                human_viewport=False, module="detail")
    return basket, {"raw": raw[:200], "parsed": bool(obj)}


def run_chatbot(page: Page, llm: LocalLLM, log: EventLog, episode: str,
                seed: int, rng) -> Tuple[List[Product], dict]:
    """Curated cards are shown to a human, who then clicks through.

    Only the k cards the assistant surfaces are impressed, but a human sees them,
    so these impressions are valid and billable.
    """
    t = 0.0
    k = 5
    for it in page.items[:k]:
        t += abs(rng.gauss(1.1, 0.6)) + 0.2
        log.add(episode=episode, regime="chatbot", kind="impression",
                sku=it.product.sku, sponsored=it.sponsored, rank=it.rank,
                cpc=it.slot.cpc if it.slot else 0.0, t=t,
                human_viewport=True, module="serp")
    if page.crosssell:
        t += abs(rng.gauss(0.9, 0.4))
        log.add(episode=episode, regime="chatbot", kind="impression",
                sku=page.crosssell[0].sku, sponsored=False, rank=0, cpc=0.0, t=t,
                human_viewport=True, module="crosssell")

    raw = llm.generate(prompt_chatbot(page), seed=seed)
    obj = llm.extract_json(raw) or {}
    chosen = _resolve(page, obj.get("choice"))
    if chosen is None:
        llm.n_parse_fail += 1
        chosen = page.items[0].product

    it = next((i for i in page.items if i.product.sku == chosen.sku), None)
    t += abs(rng.gauss(2.4, 1.2))
    log.add(episode=episode, regime="chatbot", kind="click", sku=chosen.sku,
            sponsored=bool(it and it.sponsored), rank=it.rank if it else 0,
            cpc=it.slot.cpc if (it and it.slot) else 0.0, t=t,
            human_viewport=True, module="serp")

    basket = [chosen]
    # the human is still present on the detail page, so the impulse hazard applies,
    # attenuated because the conversation already narrowed the consideration set
    for cp in page.crosssell[:2]:
        t += abs(rng.gauss(0.8, 0.4))
        log.add(episode=episode, regime="chatbot", kind="impression", sku=cp.sku,
                sponsored=False, rank=0, cpc=0.0, t=t,
                human_viewport=True, module="crosssell")
        if rng.random() < IMPULSE_HAZARD * 0.6:
            log.add(episode=episode, regime="chatbot", kind="cart", sku=cp.sku,
                    sponsored=False, rank=0, cpc=0.0, t=t + 0.5,
                    human_viewport=True, module="crosssell")
            basket.append(cp)
    for ex in (obj.get("extras") or [])[:2]:
        p = _resolve(page, ex)
        if p and all(p.sku != b.sku for b in basket):
            basket.append(p)
    for p in basket:
        t += 0.5
        log.add(episode=episode, regime="chatbot", kind="purchase", sku=p.sku,
                sponsored=False, rank=0, cpc=0.0, t=t,
                human_viewport=True, module="detail")
    return basket, {"raw": raw[:200], "parsed": bool(obj)}


def run_api(page: Page, llm: LocalLLM, log: EventLog, episode: str,
            seed: int) -> Tuple[List[Product], dict]:
    """Structured feed. Nothing is rendered, so nothing is impressed."""
    t = 0.0
    raw = llm.generate(prompt_api(page), seed=seed)
    obj = llm.extract_json(raw) or {}
    chosen = _resolve(page, obj.get("choice"))
    if chosen is None:
        llm.n_parse_fail += 1
        chosen = page.items[0].product

    t += 0.35
    # a feed read is a transaction, not a page view: no impression, no ad slot
    log.add(episode=episode, regime="api", kind="click", sku=chosen.sku,
            sponsored=False, rank=0, cpc=0.0, t=t,
            human_viewport=False, viewport_telemetry=False, module="feed")
    basket = [chosen]
    for ex in (obj.get("extras") or [])[:2]:
        p = _resolve(page, ex)
        if p and p.sku != chosen.sku:
            basket.append(p)
    for p in basket:
        t += 0.03
        log.add(episode=episode, regime="api", kind="purchase", sku=p.sku,
                sponsored=False, rank=0, cpc=0.0, t=t,
                human_viewport=False, viewport_telemetry=False, module="feed")
    return basket, {"raw": raw[:200], "parsed": bool(obj)}
