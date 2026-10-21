#!/usr/bin/env python3
"""
run_experiment.py -- The interface-regime experiment.

Design
  Main:      4 regimes x N episodes on an identical randomized task list.
             regimes = human_sim (calibrated baseline), chatbot, gui_agent, api.
  Secondary: within gui_agent, a 2x2 on sponsored-slot LABEL (visible/hidden)
             x POSITION (top / bottom), testing whether an autonomous agent is
             steerable by paid placement.

Task list is held fixed across regimes -- the same (category, budget, seed)
triples are pushed through every arm -- so regime differences are not driven by
assortment or query mix.

Economics per episode
  txn_margin      platform transaction margin on everything purchased
  ad_rev_gross    GSP charge on sponsored clicks the auction would bill
  ad_rev_billed   what survives the IVT filter
  ad_rev_valid    ground-truth value delivered (human viewport actually saw it)
  fp_loss         human revenue destroyed by over-filtering
  fn_exposure     agent clicks billed as human -> advertiser value destroyed
  detect_cost     verification run over every served impression

  a_gross = ad_rev_gross + impulse_margin      (the attention rent)
  contamination = fp_loss + gamma*fn_exposure + detect_cost
"""
from __future__ import annotations
import json, os, random, sys, time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storelab import (build_catalog, Store, EventLog, IVTClassifier, HumanSim,
                      CATEGORIES, AD_POSITIONS)
from agents import LocalLLM, run_gui_agent, run_chatbot, run_api

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "logs")
os.makedirs(RESULTS, exist_ok=True); os.makedirs(LOGS, exist_ok=True)

MODEL = os.environ.get("STORELAB_MODEL", "qwen2:7b")
N_MAIN = int(os.environ.get("STORELAB_N", "90"))       # episodes per regime
N_SECONDARY = int(os.environ.get("STORELAB_N2", "40")) # episodes per 2x2 cell
DETECT_COST_PER_IMPRESSION = 0.00004                   # verification opex
BUDGETS = {"coffee_grinder": [45, 60, 90, 130], "wireless_earbuds": [40, 70, 120, 200],
           "yoga_mat": [25, 40, 60, 90], "desk_lamp": [30, 45, 70, 110],
           "backpack": [45, 70, 110, 170], "water_bottle": [15, 22, 32, 45]}
BOTTOM_POSITIONS = [9, 10, 11, 12]


def make_tasks(n, seed=99):
    rng = random.Random(seed)
    cats = list(CATEGORIES)
    return [{"task_id": f"T{i:04d}", "category": (c := rng.choice(cats)),
             "budget": float(rng.choice(BUDGETS[c])), "seed": rng.randint(1, 10 ** 6)}
            for i in range(n)]


def episode_metrics(episode, regime, events, basket, page, ivt, cell=None):
    spon_imp = [e for e in events if e.kind == "impression" and e.sponsored]
    all_imp = [e for e in events if e.kind == "impression"]
    spon_clicks = [e for e in events if e.kind == "click" and e.sponsored]
    human_imp = [e for e in all_imp if e.human_viewport]

    # The detector sees telemetry, not truth. A browser-driving agent emits
    # viewport events with no human behind them, so had_telemetry is True for
    # gui_agent while human_viewport is False for every event in that session.
    had_telemetry = any(e.viewport_telemetry for e in events)
    declared = regime in ("api",)          # only protocol traffic self-declares
    score, filtered, feats = ivt.judge(events, declared=declared,
                                       had_telemetry=had_telemetry)

    ad_rev_gross = round(sum(e.cpc for e in spon_clicks), 4)
    ad_rev_valid = round(sum(e.cpc for e in spon_clicks if e.human_viewport), 4)
    ad_rev_billed = 0.0 if filtered else ad_rev_gross
    fp_loss = round(max(0.0, ad_rev_valid - ad_rev_billed), 4)         # human, filtered
    fn_exposure = round(max(0.0, ad_rev_billed - ad_rev_valid), 4)     # agent, billed
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
        # funnel
        "impressions_served": len(all_imp),
        "impressions_human": len(human_imp),
        "sponsored_impressions": len(spon_imp),
        "sponsored_clicks": len(spon_clicks),
        "chose_sponsored": int(bool(chosen and chosen.sku in sponsored_skus)),
        "chosen_rank": next((i.rank for i in page.items
                             if chosen and i.product.sku == chosen.sku), 0),
        # revenue
        "ad_rev_gross": ad_rev_gross, "ad_rev_valid": ad_rev_valid,
        "ad_rev_billed": ad_rev_billed, "fp_loss": fp_loss,
        "fn_exposure": fn_exposure, "detect_cost": detect_cost,
        "impulse_items": len(extras), "impulse_margin": impulse_margin,
        "txn_margin": txn_margin,
        # The attention rent has two components on very different scales: ad
        # revenue is cents per click, impulse margin is dollars per added item.
        # Keep them separate; a_gross is reported only as their sum for the
        # aggregate revenue-at-risk arithmetic, never as a single "effect size".
        "a_gross": round(ad_rev_gross + impulse_margin, 4),
        # basket / choice
        "basket_size": len(basket),
        "price_paid": round(chosen.price, 2) if chosen else 0.0,
        "chosen_sku": chosen.sku if chosen else None,
        "chosen_rating": chosen.rating if chosen else 0.0,
        "on_budget": on_budget,
        # IVT
        "ivt_score": round(score, 4), "ivt_filtered": int(filtered),
        "had_telemetry": int(had_telemetry),
        "had_human_viewport": int(any(e.human_viewport for e in events)),
        "frac_subhuman_gaps": round(feats["frac_subhuman_gaps"], 4),
        "median_gap": round(feats["median_gap"], 4),
        "n_events": len(events),
    }


def checkpoint_path(name):
    return os.path.join(RESULTS, f"_ckpt_{name}.json")


def save_checkpoint(name, rows, events):
    """Persist one completed arm so an interrupted run loses at most that arm."""
    with open(checkpoint_path(name), "w") as f:
        json.dump({"rows": rows, "events": events}, f)


def load_checkpoint(name):
    p = checkpoint_path(name)
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
        return d["rows"], d["events"]
    except Exception:
        return None


def main():
    t_start = time.time()
    print(f"StoreLab -- interface-regime experiment\nmodel={MODEL}  "
          f"N_main={N_MAIN}/regime  N_secondary={N_SECONDARY}/cell\n")

    catalog = build_catalog()
    store = Store(catalog)
    ivt = IVTClassifier()
    llm = LocalLLM(model=MODEL)
    print(f"catalog: {len(catalog)} SKUs across {len(CATEGORIES)} categories, "
          f"{sum(p.is_ad_eligible for p in catalog)} ad-eligible\n")

    tasks = make_tasks(N_MAIN)
    rows, all_events = [], []

    # ---------------------------------------------------------------- main design
    for regime in ["human_sim", "chatbot", "gui_agent", "api"]:
        ck = load_checkpoint(regime)
        if ck:
            rows.extend(ck[0]); all_events.extend(ck[1])
            print(f"  {regime:10s} RESUMED from checkpoint ({len(ck[0])} episodes)\n",
                  flush=True)
            continue
        arm_rows, arm_events = [], []
        t0 = time.time()
        for i, task in enumerate(tasks):
            ep = f"{regime}__{task['task_id']}"
            page = store.search(task["category"], task["budget"])
            log = EventLog()
            rng = random.Random(task["seed"])
            if regime == "human_sim":
                basket = HumanSim(seed=task["seed"]).shop(page, log, ep)
            elif regime == "chatbot":
                basket, _ = run_chatbot(page, llm, log, ep, task["seed"], rng)
            elif regime == "gui_agent":
                basket, _ = run_gui_agent(page, llm, log, ep, task["seed"])
            else:
                basket, _ = run_api(page, llm, log, ep, task["seed"])
            arm_rows.append(episode_metrics(ep, regime, log.events, basket, page, ivt))
            arm_events.extend(log.rows())
            if (i + 1) % 15 == 0:
                print(f"  {regime:10s} {i+1:3d}/{len(tasks)}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        save_checkpoint(regime, arm_rows, arm_events)
        rows.extend(arm_rows); all_events.extend(arm_events)
        print(f"  {regime:10s} DONE in {time.time()-t0:.0f}s  (checkpointed)\n", flush=True)

    # ------------------------------------------------- secondary 2x2 within GUI
    print("secondary 2x2 (label x position) within gui_agent")
    tasks2 = make_tasks(N_SECONDARY, seed=555)
    for label_visible in (True, False):
        for pos_name, positions in (("top", AD_POSITIONS), ("bottom", BOTTOM_POSITIONS)):
            cell = f"label={'on' if label_visible else 'off'}|pos={pos_name}"
            ck = load_checkpoint("sec_" + cell.replace("|", "_").replace("=", ""))
            if ck:
                rows.extend(ck[0]); all_events.extend(ck[1])
                print(f"  {cell:26s} RESUMED ({len(ck[0])} episodes)", flush=True)
                continue
            cell_rows, cell_events = [], []
            t0 = time.time()
            for task in tasks2:
                ep = f"sec__{cell}__{task['task_id']}"
                page = store.search(task["category"], task["budget"],
                                    ad_label_visible=label_visible,
                                    ad_positions=positions)
                log = EventLog()
                basket, _ = run_gui_agent(page, llm, log, ep, task["seed"])
                cell_rows.append(episode_metrics(ep, "gui_agent_2x2", log.events,
                                                 basket, page, ivt, cell=cell))
                cell_events.extend(log.rows())
            save_checkpoint("sec_" + cell.replace("|", "_").replace("=", ""),
                            cell_rows, cell_events)
            rows.extend(cell_rows); all_events.extend(cell_events)
            print(f"  {cell:26s} n={len(tasks2)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- persistence
    with open(os.path.join(RESULTS, "episodes.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(LOGS, "events.json"), "w") as f:
        json.dump(all_events, f)
    meta = {"model": MODEL, "n_main": N_MAIN, "n_secondary": N_SECONDARY,
            "catalog_size": len(catalog), "llm_calls": llm.n_calls,
            "llm_parse_failures": llm.n_parse_fail,
            "mean_llm_latency_s": round(llm.total_latency / max(1, llm.n_calls), 2),
            "detect_cost_per_impression": DETECT_COST_PER_IMPRESSION,
            "runtime_s": round(time.time() - t_start, 1),
            "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(RESULTS, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    print(f"\n{len(rows)} episodes, {len(all_events)} events")
    print(f"LLM: {llm.n_calls} calls, {llm.n_parse_fail} parse failures "
          f"({llm.n_parse_fail/max(1,llm.n_calls):.1%}), "
          f"{meta['mean_llm_latency_s']}s mean latency")
    print(f"total runtime {meta['runtime_s']}s")


if __name__ == "__main__":
    main()
