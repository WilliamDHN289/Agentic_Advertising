"""SL-03 protocol conformance checks for any choice source (used from SL-06 on).

A backend conforms when, for an observation and the trial's interface:

- every action it emits is a protocol ``Action`` (RequestPage, Select, Leave,
  Invalid) and round-trips through the discriminated union;
- the actions are page requests followed by exactly one final Select, Leave or
  Invalid; page requests occur only under ``browser_tool``;
- every product id it names is a candidate of the observation;
- its ``TrialOutcome`` validates, matches the final action (choice ↔ Select,
  outside ↔ Leave, invalid ↔ Invalid with the same reason), lists the page requests
  in order, names its backend and model, and fits in a ``TrialRecord``.
"""
from __future__ import annotations

import unittest

from features.storelab.core.protocol import (ACTION, Invalid, Leave, Observation, RequestPage, Select,
                                             TrialOutcome)
from features.storelab.core.records import TrialRecord

ACTION_TYPES = (RequestPage, Select, Leave, Invalid)
FINAL_TYPES = (Select, Leave, Invalid)


def check_actions(test: unittest.TestCase, observation: Observation, interface: str, actions: list) -> None:
    test.assertGreaterEqual(len(actions), 1)
    for action in actions:
        test.assertIsInstance(action, ACTION_TYPES)
        test.assertEqual(ACTION.validate_python(action.model_dump(mode="json")), action)
        if isinstance(action, (RequestPage, Select)):
            test.assertIn(action.product_id, observation.candidate_ids)
    test.assertIsInstance(actions[-1], FINAL_TYPES)
    test.assertTrue(all(isinstance(a, RequestPage) for a in actions[:-1]))
    if interface != "browser_tool":
        test.assertFalse(any(isinstance(a, RequestPage) for a in actions))


def check_outcome(test: unittest.TestCase, observation: Observation, interface: str, outcome: TrialOutcome,
                  actions: list | None = None) -> None:
    test.assertEqual(TrialOutcome.model_validate(outcome.model_dump(mode="json")), outcome)
    test.assertIsNotNone(outcome.requested_model)
    if outcome.chosen_id is not None:
        test.assertIn(outcome.chosen_id, observation.candidate_ids)
    test.assertTrue(set(outcome.page_requests) <= observation.candidate_ids)
    if actions is not None:
        check_actions(test, observation, interface, actions)
        final = actions[-1]
        expected = {Select: "choice", Leave: "outside", Invalid: "invalid"}[type(final)]
        test.assertEqual(outcome.status, expected)
        if isinstance(final, Select):
            test.assertEqual(outcome.chosen_id, final.product_id)
        if isinstance(final, Invalid):
            test.assertEqual(outcome.invalid_reason, final.reason)
        test.assertEqual(outcome.page_requests, [a.product_id for a in actions if isinstance(a, RequestPage)])
    TrialRecord(run_id="1b4e28ba-2fa1-4d2e-883f-0016d3cca427", trial_key="0" * 64, config_hash="1" * 64,
                task_id=observation.task.task_id, set_id=observation.set_id,
                presentation_id=observation.presentation_id, interface=interface, model_key="conformance",
                temperature=None, rep=0, candidate_ids=sorted(observation.candidate_ids), outcome=outcome)
