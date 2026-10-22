"""Original StoreLab events (storelab.py l.219-240) as v2 ``EventRecord``s, losslessly.

Each of the 11 legacy fields maps to exactly one EventRecord field (FIELD_MAP), and
``event_to_legacy(event_from_legacy(e))`` returns ``e`` unchanged. The v2-only fields
are derived: ``trial_key`` from the episode id, ``seq`` from the event's order in its
episode, and ``simulated`` (every legacy purchase is simulated). The table is
repeated in docs/protocol.md.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

from .config import LEGACY_EXPERIMENT
from .records import EventRecord

FIELD_MAP = {
    "episode": "trial_label",
    "regime": "interface",
    "kind": "kind",
    "sku": "product_id",
    "sponsored": "sponsored",
    "rank": "position",
    "cpc": "cpc",
    "t": "t_s",
    "human_viewport": "human_viewport",
    "module": "module",
    "viewport_telemetry": "viewport_telemetry",
}
REGIME_TO_INTERFACE = {"human_sim": "legacy_human_sim", "chatbot": "legacy_chatbot",
                       "gui_agent": "legacy_gui_agent", "api": "legacy_api"}
INTERFACE_TO_REGIME = {interface: regime for regime, interface in REGIME_TO_INTERFACE.items()}


def legacy_trial_key(episode: str) -> str:
    return hashlib.sha256(("%s\0%s" % (LEGACY_EXPERIMENT, episode)).encode("utf-8")).hexdigest()


def event_from_legacy(event: Mapping[str, Any], seq: int) -> EventRecord:
    if set(event) != set(FIELD_MAP):
        raise ValueError("legacy event fields %s, expected %s" % (sorted(event), sorted(FIELD_MAP)))
    values = {FIELD_MAP[key]: value for key, value in event.items()}
    values["interface"] = REGIME_TO_INTERFACE[event["regime"]]
    return EventRecord(trial_key=legacy_trial_key(event["episode"]), seq=seq,
                       simulated=event["kind"] == "purchase", **values)


def event_to_legacy(record: EventRecord) -> dict[str, Any]:
    data = record.model_dump()
    legacy = {key: data[field] for key, field in FIELD_MAP.items()}
    legacy["regime"] = INTERFACE_TO_REGIME[record.interface]
    return legacy


def events_from_legacy(events: Iterable[Mapping[str, Any]]) -> list[EventRecord]:
    """Convert an events.json list; ``seq`` counts events within each episode."""
    counters: dict[str, int] = {}
    records = []
    for event in events:
        seq = counters.get(event["episode"], 0)
        counters[event["episode"]] = seq + 1
        records.append(event_from_legacy(event, seq))
    return records
