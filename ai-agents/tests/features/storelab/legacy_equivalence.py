"""SL-04 equivalence harness: run in a fresh process as ``original`` or ``port``.

    python -B tests/features/storelab/legacy_equivalence.py original|port

Prints JSON: a sha256 per item of tickets.md SL-04 step 2, computed the same way on
both sides, arms in the original order, so the two outputs must be identical:

* catalog: every field of the 2,004 SKUs;
* pages_first_90 / pages_all_520: the pages of make_tasks(90) as the first arm saw
  them, and every page of the run in run order (one shared auction generator);
* prompts: the rendered prompt of every LLM episode;
* human_sim: baskets, metrics rows and events of the 90 human_sim episodes;
* ivt_saved: IVT score, filter flag and features for the saved events of all 520
  episodes, judged in run order by one classifier;
* replay_rows / replay_events: the whole run replayed with the choices and extras
  each LLM episode made in the saved events.

Also prints the hashes of the saved episodes.json and events.json, and how many
saved IVT scores the run reproduces. Nothing is written to disk.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

AI_AGENTS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AI_AGENTS))

from features.storelab import legacy_baseline  # noqa: E402
from features.storelab.config_files import load_config  # noqa: E402
from features.storelab.core.config import canonical_json  # noqa: E402
import hashlib  # noqa: E402

SAVED = legacy_baseline.snapshot_dir()
ROWS = json.loads((SAVED / "results" / "episodes.json").read_text(encoding="utf-8"))
EVENTS = json.loads((SAVED / "logs" / "events.json").read_text(encoding="utf-8"))
LEGACY = load_config(AI_AGENTS / "features" / "storelab" / "configs" / "legacy_2026_08.yaml").environment.legacy


def sha(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def page_json(page) -> dict:
    return {"query": page.query, "category": page.category, "budget": page.budget,
            "items": [{"rank": it.rank, "sku": it.product.sku, "sponsored": it.sponsored,
                       "label_visible": it.label_visible,
                       "slot": None if it.slot is None else {
                           "rank": it.slot.rank, "slot_index": it.slot.slot_index, "sku": it.slot.product.sku,
                           "bid": it.slot.bid, "cpc": it.slot.cpc}} for it in page.items],
            "crosssell": [p.sku for p in page.crosssell],
            "deal": page.deal_banner.sku if page.deal_banner else None}


def saved_replies() -> dict:
    """The reply each saved LLM episode amounts to: its clicked product and its extras."""
    by_episode = defaultdict(list)
    for event in EVENTS:
        by_episode[event["episode"]].append(event)
    replies = {}
    for episode, events in by_episode.items():
        if episode.startswith("human_sim__"):
            continue
        click = next(e["sku"] for e in events if e["kind"] == "click")
        carts = [e["sku"] for e in events if e["kind"] == "cart"]
        purchases = [e["sku"] for e in events if e["kind"] == "purchase"]
        if episode.startswith(("gui_agent__", "sec__")):
            extras = carts
        elif episode.startswith("chatbot__"):
            extras = purchases[1 + len(carts):]  # chosen, impulse carts, then extras
        else:
            extras = purchases[1:]
        replies[episode] = json.dumps({"choice": click, "extras": extras, "reason": "replayed"})
    return replies


def plan():
    """(episode, arm, task, cell, label_visible, positions) in run order (run_experiment.py l.163-221)."""
    import random

    def make_tasks(n, seed):  # run_experiment.py l.52-57, inlined so both sides share the task list
        rng = random.Random(seed)
        budgets = {"coffee_grinder": [45, 60, 90, 130], "wireless_earbuds": [40, 70, 120, 200],
                   "yoga_mat": [25, 40, 60, 90], "desk_lamp": [30, 45, 70, 110],
                   "backpack": [45, 70, 110, 170], "water_bottle": [15, 22, 32, 45]}
        cats = list(budgets)
        return [{"task_id": f"T{i:04d}", "category": (c := rng.choice(cats)),
                 "budget": float(rng.choice(budgets[c])), "seed": rng.randint(1, 10 ** 6)} for i in range(n)]

    out = []
    for arm in ("human_sim", "chatbot", "gui_agent", "api"):
        for task in make_tasks(LEGACY.episodes_main, LEGACY.task_seed_main):
            out.append((f"{arm}__{task['task_id']}", arm, task, None, True, None))
    for label_visible in (True, False):
        for pos_name, positions in (("top", LEGACY.ad_positions), ("bottom", LEGACY.bottom_positions)):
            cell = f"label={'on' if label_visible else 'off'}|pos={pos_name}"
            for task in make_tasks(LEGACY.episodes_secondary, LEGACY.task_seed_secondary):
                out.append((f"sec__{cell}__{task['task_id']}", "gui_agent_2x2", task, cell, label_visible,
                            list(positions)))
    return out


def report(env, run_replay) -> dict:
    """``env`` exposes the original API; ``run_replay(replies)`` -> (rows, events) of the run."""
    catalog = env.build_catalog(LEGACY.products_per_category, LEGACY.catalog_seed)
    store = env.Store(catalog, seed=LEGACY.store_seed)
    pages, prompts = [], []
    prompt_of = {"chatbot": env.prompt_chatbot, "gui_agent": env.prompt_gui, "api": env.prompt_api,
                 "gui_agent_2x2": env.prompt_gui}
    for episode, arm, task, cell, label_visible, positions in plan():
        if positions is None:
            page = store.search(task["category"], task["budget"])
        else:
            page = store.search(task["category"], task["budget"], ad_label_visible=label_visible,
                                ad_positions=positions)
        pages.append(page_json(page))
        if arm != "human_sim":
            prompts.append(prompt_of[arm](page))

    store, ivt = env.Store(catalog, seed=LEGACY.store_seed), env.IVTClassifier(LEGACY.ivt_threshold, LEGACY.ivt_seed)
    human = []
    for episode, arm, task, *_ in plan()[:LEGACY.episodes_main]:
        page = store.search(task["category"], task["budget"])
        log = env.EventLog()
        basket = env.HumanSim(seed=task["seed"]).shop(page, log, episode)
        human.append({"basket": [p.sku for p in basket],
                      "row": env.episode_metrics(episode, arm, log.events, basket, page, ivt),
                      "events": log.rows()})

    ivt, by_episode, judged, matches = env.IVTClassifier(LEGACY.ivt_threshold, LEGACY.ivt_seed), defaultdict(list), [], 0
    for event in EVENTS:
        by_episode[event["episode"]].append(event)
    for row in ROWS:
        events = [env.Event(**e) for e in by_episode[row["episode"]]]
        score, filtered, feats = ivt.judge(events, declared=row["regime"] in ("api",),
                                           had_telemetry=any(e.viewport_telemetry for e in events))
        judged.append([row["episode"], score, filtered, feats])
        matches += round(score, 4) == row["ivt_score"] and int(filtered) == row["ivt_filtered"]

    rows, events = run_replay(saved_replies())
    return {"catalog": sha([asdict(p) for p in catalog]), "catalog_size": len(catalog),
            "pages_first_90": sha(pages[:90]), "pages_all_520": sha(pages), "pages": len(pages),
            "prompts": sha(prompts), "human_sim": sha(human), "ivt_saved": sha(judged),
            "ivt_saved_matches": matches, "replay_rows": sha(rows), "replay_events": sha(events),
            "saved_rows": sha(ROWS), "saved_events": sha(EVENTS)}


def original() -> dict:
    vendored = AI_AGENTS / "features" / "storelab" / "legacy_2026_08"
    sys.path.insert(0, str(vendored))
    real_makedirs = os.makedirs
    os.makedirs = lambda *args, **kwargs: None  # run_experiment.py l.40 would create results/ and logs/
    try:
        import agents
        import run_experiment
        import storelab
    finally:
        os.makedirs = real_makedirs

    class ReplayLLM:  # stands in for agents.LocalLLM
        extract_json = staticmethod(agents.LocalLLM.extract_json)
        n_parse_fail = 0
        reply = ""

        def generate(self, prompt, seed=None):
            return self.reply

    def run_replay(replies):  # run_experiment.main l.153-221 without checkpoints
        import random
        catalog = storelab.build_catalog()
        store, ivt, llm = storelab.Store(catalog), storelab.IVTClassifier(), ReplayLLM()
        rows, events = [], []
        for episode, arm, task, cell, label_visible, positions in plan():
            if positions is None:
                page = store.search(task["category"], task["budget"])
            else:
                page = store.search(task["category"], task["budget"], ad_label_visible=label_visible,
                                    ad_positions=positions)
            log = storelab.EventLog()
            llm.reply = replies.get(episode, "")
            if arm == "human_sim":
                basket = storelab.HumanSim(seed=task["seed"]).shop(page, log, episode)
            elif arm == "chatbot":
                basket, _ = agents.run_chatbot(page, llm, log, episode, task["seed"], random.Random(task["seed"]))
            elif arm == "api":
                basket, _ = agents.run_api(page, llm, log, episode, task["seed"])
            else:
                basket, _ = agents.run_gui_agent(page, llm, log, episode, task["seed"])
            rows.append(run_experiment.episode_metrics(episode, arm, log.events, basket, page, ivt, cell=cell))
            events.extend(log.rows())
        return rows, events

    class Env:
        build_catalog, Store, IVTClassifier = storelab.build_catalog, storelab.Store, storelab.IVTClassifier
        EventLog, HumanSim, Event = storelab.EventLog, storelab.HumanSim, storelab.Event
        prompt_chatbot, prompt_gui, prompt_api = agents.prompt_chatbot, agents.prompt_gui, agents.prompt_api
        episode_metrics = staticmethod(run_experiment.episode_metrics)

    return report(Env, run_replay)


def port() -> dict:
    from features.storelab.core import legacy_env
    from features.storelab.core.protocol import Completion

    def run_replay(replies):
        current = {}
        config = load_config(AI_AGENTS / "features" / "storelab" / "configs" / "legacy_2026_08.yaml")
        run = legacy_env.LegacyRun.from_config(config, lambda prompt, seed: Completion(text=current["reply"], attempts=1))
        rows, events = [], []
        for spec in legacy_env.episode_plan(LEGACY):
            current["reply"] = replies.get(spec.episode, "")
            record = run.run(spec)
            rows.append(record["row"])
            events.extend(record["events"])
        return rows, events

    return report(legacy_env, run_replay)


if __name__ == "__main__":
    side = sys.argv[1] if len(sys.argv) > 1 else ""
    if side not in ("original", "port"):
        raise SystemExit("usage: legacy_equivalence.py original|port")
    print(json.dumps(original() if side == "original" else port(), indent=1, sort_keys=True))
