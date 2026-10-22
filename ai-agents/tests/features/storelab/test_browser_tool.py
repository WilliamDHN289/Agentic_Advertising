"""SL-10 browser_tool: one event per tool call in each provider's fixture transcript
(AC-1); the turn limit gives invalid/max_turns, never a choice (AC-2); identical
observations, pages and served bytes across modes (AC-3); the documented limitation
(AC-4); plus the storefront rules, the loop's error paths, the simulated agent acting
through the same tools, and events stored with their trial records.

The transcripts in fixtures/browser are served through the real pinned SDKs by the
in-process ``Server`` of test_providers (skipped where the SDKs are not installed);
everything else uses mock providers. No test uses the network.
"""
from __future__ import annotations

import itertools
import json
import threading
import unittest
from pathlib import Path

from features.storelab import api
from features.storelab.backends.browser_tool import first_messages, simulated_decision
from features.storelab.backends.llm import LLMBackend
from features.storelab.backends.provider import KillSwitchEngaged
from features.storelab.backends.simulated import SimulatedAgent, params_from_config
from features.storelab.core.cache import LLMRequest, ProviderResponse, call_key
from features.storelab.core.protocol import Usage
from features.storelab.core.render import build_observation, page_model
from features.storelab.core.runner import Decision, Trial
from features.storelab.core.store import LocalStore
from features.storelab.core.storefront import (SYSTEM_PROMPT, TOOL_SCHEMA_VERSION, BrowserSession, ToolCall,
                                               tool_definitions, tool_schema_sha256)
from tests.features.storelab.test_cache import Env
from tests.features.storelab.test_providers import HAVE_SDKS, Server, policy
from tests.features.storelab.test_render import PRESENTATION, PRODUCTS, TASK

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "browser"
DOCS = Path(api.__file__).resolve().parent / "docs"
PINNED_TOOLS = {TOOL_SCHEMA_VERSION: "e529ac9e93638e9288d59ec65795f739bd9fdb5803bb8f6350eabf90ba876299"}
BROWSER = {"interfaces": ["browser_tool"], "protocol.prompt_templates": {"browser_tool": "e1_browser_tool@1"}}
SPECS = {"openai": {"key": "gpt_4o_20240806", "provider": "openai", "model": "gpt-4o-2024-08-06",
                    "temperatures": [None]},
         "anthropic": {"key": "claude_sonnet_45_20250929", "provider": "anthropic",
                       "model": "claude-sonnet-4-5-20250929", "temperatures": [None]},
         "google": {"key": "gemini_20_flash_001", "provider": "google", "model": "gemini-2.0-flash-001",
                    "temperatures": [None]}}
IDS = ["B0AAA00003", "B0AAA00001", "B0AAA00002"]  # PRESENTATION's candidates in position order


def call(name: str, product_id: str | None = None, **arguments) -> ToolCall:
    if product_id is not None:
        arguments["product_id"] = product_id
    return ToolCall(id="c-%s-%s" % (name, product_id), name=name, arguments=arguments)


class EnvCase(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        self.addCleanup(self.env.cleanup)

    def config(self, provider="openai", **overrides):
        spec = dict(SPECS[provider], model=overrides.pop("model", SPECS[provider]["model"]))
        return self.env.config(**{"agents.models": [spec], **BROWSER, **overrides})

    def observation(self, config):
        return build_observation(config, TASK, PRESENTATION, PRODUCTS)

    @staticmethod
    def trial(config, key="a" * 64) -> Trial:
        return Trial(trial_key=key, task_id=TASK.task_id, k=3, set_id=PRESENTATION.set_id,
                     presentation_id=PRESENTATION.presentation_id, interface="browser_tool",
                     model_key=config.agents.models[0].key, temperature=None, rep=0)


class ScriptedProvider:
    """A provider whose n-th call returns ``script(n, request)``: a list of tool calls, a
    ProviderResponse, or an exception to raise."""

    name, adapter_version = "openai", "scripted@1"

    def __init__(self, script):
        self.script, self.requests, self.lock = script, [], threading.Lock()

    def complete(self, request: LLMRequest) -> ProviderResponse:
        with self.lock:
            self.requests.append(request)
            n = len(self.requests)
        result = self.script(n, request)
        if isinstance(result, BaseException):
            raise result
        if isinstance(result, ProviderResponse):
            return result
        return ProviderResponse(text="", tool_calls=result or None, attempts=1, latency_s=0.5,
                                returned_model=request.requested_model,
                                usage=Usage(input_tokens=100 * n, output_tokens=10))


# ------------------------------------------------------------------- AC-1
def calls_in(provider: str, body: dict) -> list[tuple[str, dict]]:
    """(tool name, arguments) of each tool call in a provider response body, read
    independently of the adapters."""
    if provider == "openai":
        return [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                for c in body["choices"][0]["message"].get("tool_calls") or []]
    if provider == "anthropic":
        return [(b["name"], b["input"]) for b in body["content"] if b["type"] == "tool_use"]
    return [(p["functionCall"]["name"], p["functionCall"]["args"]) for p in body["candidates"][0]["content"]["parts"]
            if "functionCall" in p]


@unittest.skipUnless(HAVE_SDKS, "the pinned provider SDKs are not installed")
class TranscriptFixtureTest(EnvCase):
    """AC-1: every tool call in a fixture transcript is exactly one event with its
    product id and turn index, through the real SDKs."""

    def fixtures(self):
        names = sorted(p.name for p in FIXTURES.glob("*.json"))
        self.assertEqual({n.split("_")[0] for n in names}, {"openai", "anthropic", "google"})
        return [json.loads((FIXTURES / n).read_text(encoding="utf-8")) | {"name": n} for n in names]

    def run_fixture(self, data, cache=None, policy_name="off"):
        config = self.config(data["provider"], model=data["model"])
        turns = data["turns"]
        server = Server(data["provider"], lambda n, m, b: (turns[n - 1]["status"], turns[n - 1]["headers"],
                                                           turns[n - 1]["json"]))
        backend = LLMBackend(config, config.agents.models[0].key, provider=server.adapter(policy=policy()),
                             cache=cache, policy=policy_name, clock=itertools.count(0.0, 0.5).__next__)
        return config, server, backend.decide(self.trial(config), self.observation(config))

    def test_every_tool_call_is_exactly_one_event_with_product_id_and_turn(self):
        for data in self.fixtures():
            with self.subTest(fixture=data["name"]):
                config, server, decision = self.run_fixture(data)
                self.assertIsInstance(decision, Decision)
                expected = []  # (turn, tool, product id named by the call), from the fixture alone
                for turn, response in enumerate(data["turns"], 1):
                    expected += [(turn, name, args.get("product_id")) for name, args in
                                 calls_in(data["provider"], response["json"])]
                events = decision.events
                self.assertEqual(len(events), len(expected))
                self.assertEqual(len(server.requests), len(data["turns"]))
                self.assertEqual([e.seq for e in events], list(range(len(events))))
                for event, (turn, tool, product_id) in zip(events, expected):
                    self.assertEqual((event.turn, event.tool, event.trial_key), (turn, tool, "a" * 64))
                    self.assertEqual(event.interface, "browser_tool")
                    if tool != "back":
                        self.assertEqual(event.product_id, product_id)
                self.assertEqual([{"turn": e.turn, "tool": e.tool, "kind": e.kind, "product_id": e.product_id}
                                  for e in events], data["expect"]["events"])
                outcome = decision.outcome
                self.assertEqual(outcome.model_dump(include=set(data["expect"]["outcome"])),
                                 data["expect"]["outcome"])
                self.assertEqual((outcome.requested_model, outcome.returned_model), (data["model"],) * 2)
                self.assertEqual(outcome.attempts, len(data["turns"]))
                self.assertGreater(outcome.usage.input_tokens, 0)
                # Page requests are recorded apart from the effective choice, per product.
                self.assertEqual([e.product_id for e in events if e.kind == "page_request"], outcome.page_requests)
                for event in events:
                    if event.product_id in IDS:
                        candidate = next(c for c in PRESENTATION.candidates if c.product_id == event.product_id)
                        self.assertEqual((event.position, event.sponsored),
                                         (candidate.position, candidate.sponsored_label))

    def test_each_sdk_sends_the_tools_and_the_pages_served(self):
        for data in self.fixtures():
            with self.subTest(fixture=data["name"]):
                config, server, _ = self.run_fixture(data)
                observation = self.observation(config)
                first, second = server.requests[0]["json"], server.requests[1]["json"]
                opened = calls_in(data["provider"], data["turns"][0]["json"])
                pages = [observation.pages[args["product_id"]] for name, args in opened]
                if data["provider"] == "openai":
                    self.assertEqual(first["messages"][:2], [{"role": "system", "content": SYSTEM_PROMPT},
                                                             {"role": "user",
                                                              "content": observation.views["browser_tool"]}])
                    self.assertEqual((first["tool_choice"], first["parallel_tool_calls"]), ("required", False))
                    self.assertEqual([t["function"]["name"] for t in first["tools"]],
                                     ["open_product", "back", "select", "leave"])
                    self.assertEqual([m["content"] for m in second["messages"] if m["role"] == "tool"], pages)
                elif data["provider"] == "anthropic":
                    self.assertEqual(first["system"], SYSTEM_PROMPT)
                    self.assertEqual(first["messages"], [{"role": "user",
                                                          "content": observation.views["browser_tool"]}])
                    self.assertEqual(first["tool_choice"], {"type": "any", "disable_parallel_tool_use": True})
                    self.assertEqual([t["name"] for t in first["tools"]], ["open_product", "back", "select", "leave"])
                    results = second["messages"][-1]["content"]
                    self.assertEqual([r["content"] for r in results if r["type"] == "tool_result"], pages)
                    self.assertEqual(second["messages"][-2]["content"][0],
                                     {"type": "text", "text": "I will check the glass mug first."})
                else:
                    self.assertEqual(first["systemInstruction"]["parts"][0]["text"], SYSTEM_PROMPT)
                    self.assertEqual(first["toolConfig"], {"functionCallingConfig": {"mode": "ANY"}})
                    self.assertEqual([d["name"] for d in first["tools"][0]["functionDeclarations"]],
                                     ["open_product", "back", "select", "leave"])
                    parts = second["contents"][-1]["parts"]
                    self.assertEqual([p["functionResponse"]["response"]["content"] for p in parts], pages)

    def test_the_transcript_is_cached_and_replays_with_the_same_events(self):
        cache = api.open_cache(self.env.cache_root)
        for data in self.fixtures():
            with self.subTest(fixture=data["name"]):
                config, server, live = self.run_fixture(data, cache, "read_write")
                key = live.outcome.raw_ref.split(":", 1)[1]
                entry = next(e for e in cache.entries() if e.call_key == key)
                # The last turn's entry holds the whole conversation.
                self.assertEqual(sum(m.role == "assistant" for m in entry.request.messages), len(data["turns"]) - 1)
                self.assertEqual(entry.request.tool_schema_version, TOOL_SCHEMA_VERSION)
                self.assertIsNone(entry.parsed_action)
                replay = LLMBackend(config, config.agents.models[0].key, provider=None, cache=cache,
                                    policy="read_only").decide(self.trial(config), self.observation(config))
                self.assertEqual([e.model_dump(exclude={"t_s"}) for e in replay.events],
                                 [e.model_dump(exclude={"t_s"}) for e in live.events])
                self.assertEqual(replay.outcome.model_dump(exclude={"latency_s"}),
                                 live.outcome.model_dump(exclude={"latency_s"}))
        self.assertEqual(cache.verify(), {})


# ------------------------------------------------------------------- AC-2
class TurnLimitTest(EnvCase):
    """AC-2: hitting max_turns gives invalid/max_turns, never a choice."""

    def test_the_session_never_turns_the_limit_into_a_choice(self):
        observation = self.observation(self.config())
        for max_turns in (1, 2, 8):
            with self.subTest(max_turns=max_turns):
                session = BrowserSession(observation, "a" * 64, max_turns)
                for turn in range(max_turns):
                    session.step([call("open_product", IDS[turn % 3])])
                self.assertEqual(session.result(), {"status": "invalid", "invalid_reason": "max_turns",
                                                    "page_requests": [IDS[t % 3] for t in range(max_turns)],
                                                    "turns": max_turns})
                with self.assertRaises(RuntimeError):
                    session.step([call("select", IDS[0])])  # no turn after the limit
        session = BrowserSession(observation, "a" * 64, 3)
        session.step([call("open_product", IDS[0])])
        session.step([call("back")])
        session.step([call("select", IDS[1])])  # on the last allowed turn
        self.assertEqual(session.result()["status"], "choice")

    def test_a_model_that_never_decides_makes_max_turns_calls_and_no_choice(self):
        for max_turns in (3, 8):
            with self.subTest(max_turns=max_turns):
                config = self.config(**{"protocol.max_turns": max_turns})
                provider = ScriptedProvider(lambda n, r: [call("open_product", IDS[n % 3])])
                backend = LLMBackend(config, config.agents.models[0].key, provider=provider, cache=None, policy="off")
                decision = backend.decide(self.trial(config), self.observation(config))
                outcome = decision.outcome
                self.assertEqual((outcome.status, outcome.invalid_reason, outcome.chosen_id),
                                 ("invalid", "max_turns", None))
                self.assertEqual((outcome.turns, outcome.attempts, len(provider.requests)), (max_turns,) * 3)
                self.assertEqual([e.kind for e in decision.events], ["page_request"] * max_turns)
                self.assertEqual([e.turn for e in decision.events], list(range(1, max_turns + 1)))

    def test_a_select_on_the_last_turn_counts_and_none_is_asked_after_it(self):
        config = self.config(**{"protocol.max_turns": 3})
        provider = ScriptedProvider(lambda n, r: [call("open_product", IDS[0])] if n < 3 else [call("select", IDS[2])])
        decision = LLMBackend(config, config.agents.models[0].key, provider=provider, cache=None,
                              policy="off").decide(self.trial(config), self.observation(config))
        self.assertEqual((decision.outcome.status, decision.outcome.chosen_id), ("choice", IDS[2]))
        self.assertEqual(len(provider.requests), 3)

    def test_a_run_records_max_turns_as_invalid(self):
        config = self.config(**{"protocol.max_turns": 2, "repetitions": 2})
        provider = ScriptedProvider(lambda n, r: [call("back")])
        run_dir, result = api.run(config, snapshot=self.env.snapshot, root=self.env.runs,
                                  environ={"OPENAI_API_KEY": "test"}, estimate_cost=lambda c, t, m: 0.0,
                                  providers={"openai": provider}, cache_root=self.env.cache_root)
        store = LocalStore(run_dir)
        records = store.records().values()
        self.assertTrue(records)
        for record in records:
            self.assertEqual((record.outcome.status, record.outcome.invalid_reason, record.outcome.chosen_id),
                             ("invalid", "max_turns", None))
        self.assertEqual(result.status, "COMPLETED_WITH_GAPS")
        self.assertEqual(len(provider.requests), 2 * len(records))
        events = store.events()
        self.assertEqual(sorted(events), sorted(r.trial_key for r in records))
        self.assertTrue(all([e.turn for e in ev] == [1, 2] for ev in events.values()))


# ------------------------------------------------------------------- AC-3
class ToolMock(ScriptedProvider):
    """Opens the first listed product, goes back, opens the last and selects it; records
    the pages it is served."""

    def __init__(self):
        super().__init__(self.answer)
        self.served = []

    def answer(self, n, request):
        turn = sum(m.role == "assistant" for m in request.messages) + 1
        ids = [line.rsplit("/product/", 1)[1] for line in request.messages[1].content.splitlines()
               if "/product/" in line]
        with self.lock:
            self.served += [(request.messages[1].content, m.content) for m in request.messages if m.role == "tool"]
        return {1: [call("open_product", ids[0])], 2: [call("back")], 3: [call("open_product", ids[-1])]}.get(
            turn, [call("select", ids[-1])])


class Recording:
    """Wraps a backend; records each observation it is given."""

    def __init__(self, inner, seen):
        self.inner, self.seen = inner, seen
        self.backend, self.model = inner.backend, inner.model

    def decide(self, trial, observation):
        self.seen[(trial.presentation_id, trial.rep)] = observation
        return self.inner.decide(trial, observation)


class AcrossModesTest(EnvCase):
    """AC-3: identical observations render identically across modes (hash test)."""

    def run_mode(self, config, mode, providers=None):
        run_dir, planned = api.plan_run(config, snapshot=self.env.snapshot, root=self.env.runs,
                                        environ={"OPENAI_API_KEY": "test"} if mode == "sandbox" else {},
                                        estimate_cost=lambda c, t, m: 0.0)
        stimuli = api.load_plan(LocalStore(run_dir))[1]
        cache = api.open_cache(self.env.cache_root)
        built = api.backends_for(config, stimuli, planned.manifest.mode.resolved, providers=providers, cache=cache,
                                 run_id=planned.manifest.run_id)
        seen = {}
        result = api.execute_run(run_dir, backends={k: Recording(b, seen) for k, b in built.items()})
        return run_dir, result, seen

    def test_every_mode_sees_the_same_observation_and_is_served_the_same_pages(self):
        base = {"repetitions": 2, "cache.policy": "read_write", "budget.max_concurrency": 2}
        offline = self.config(**base, mode="offline")
        sandbox = self.config(**base, mode="sandbox")
        mock = ToolMock()
        _, offline_result, offline_seen = self.run_mode(offline, "offline")
        source_dir, sandbox_result, sandbox_seen = self.run_mode(sandbox, "sandbox", {"openai": mock})
        replay_config = api.replay_config(source_dir)
        _, replay_result, replay_seen = self.run_mode(replay_config, "replay")
        self.assertEqual(sandbox_result.status, "COMPLETED")
        self.assertEqual(replay_result.status, "COMPLETED")
        self.assertEqual(offline_seen.keys(), sandbox_seen.keys())
        self.assertEqual(sandbox_seen.keys(), replay_seen.keys())
        for coordinates, observation in offline_seen.items():
            with self.subTest(presentation=coordinates):
                hashes = {o.observation_hash() for o in (observation, sandbox_seen[coordinates],
                                                         replay_seen[coordinates])}
                self.assertEqual(len(hashes), 1)
                self.assertEqual(sorted(observation.pages), sorted(c.product_id for c in observation.candidates))
        # What the model was served is the observation's bytes: results page and product pages.
        pages = {page for o in offline_seen.values() for page in o.pages.values()}
        views = {o.views["browser_tool"] for o in offline_seen.values()}
        self.assertTrue(mock.served)
        for view, page in mock.served:
            self.assertIn(view, views)
            self.assertIn(page, pages | views)
        self.assertEqual({r.messages[1].content for r in mock.requests}, views)
        self.assertEqual({r.messages[0].content for r in mock.requests}, {SYSTEM_PROMPT})

    def test_the_pages_are_the_page_model_text(self):
        config = self.config()
        observation = self.observation(config)
        model = page_model(TASK, PRESENTATION, {k: p.truncated(config.environment.description_chars)
                                                for k, p in PRODUCTS.items()})
        self.assertEqual(observation.views["browser_tool"], model.serp_text)
        self.assertEqual(observation.pages, {k: p.text for k, p in model.pages.items()})
        text_only = self.env.config()
        self.assertIsNone(build_observation(text_only, TASK, PRESENTATION, PRODUCTS).pages)
        self.assertNotIn("pages", build_observation(text_only, TASK, PRESENTATION, PRODUCTS).model_dump())

    def test_the_tools_are_pinned_to_their_version(self):
        self.assertEqual(tool_schema_sha256(), PINNED_TOOLS[TOOL_SCHEMA_VERSION])
        self.assertEqual([t["name"] for t in tool_definitions(TOOL_SCHEMA_VERSION)],
                         ["open_product", "back", "select", "leave"])
        config = self.config()
        trial, observation = self.trial(config), self.observation(config)
        backend = LLMBackend(config, config.agents.models[0].key, provider=ScriptedProvider(lambda n, r: []),
                             cache=None, policy="off")
        request = backend.request(trial, observation, first_messages(observation))
        self.assertEqual(request.tool_schema_version, TOOL_SCHEMA_VERSION)
        other = request.model_copy(update={"tool_schema_version": "storelab-browser-tools@2"})
        self.assertNotEqual(call_key(request), call_key(other))


# ------------------------------------------------------------------- AC-4
class DocsTest(unittest.TestCase):
    def test_the_docs_state_the_h1e_limitation(self):
        text = " ".join((DOCS / "browser_tool.md").read_text(encoding="utf-8").split())
        self.assertIn("H1e conclusions apply to `browser_tool` only unless SL-10b runs", text)


# -------------------------------------------------------- storefront rules
class StorefrontTest(EnvCase):
    def session(self, max_turns=8):
        return BrowserSession(self.observation(self.config()), "a" * 64, max_turns)

    def test_tools_serve_the_observation_and_record_one_event_each(self):
        session = self.session()
        observation = session.observation
        self.assertEqual(session.step([call("open_product", IDS[2])])[0].content, observation.pages[IDS[2]])
        self.assertEqual(session.step([call("back")])[0].content, observation.views["browser_tool"])
        self.assertEqual(session.step([call("back")])[0].content, observation.views["browser_tool"])
        self.assertEqual(session.step([call("open_product", " %s " % IDS[0])])[0].content, observation.pages[IDS[0]])
        self.assertEqual(session.step([call("select", IDS[0])]), [])
        self.assertEqual([(e.turn, e.kind, e.product_id, e.module, e.position, e.sponsored) for e in session.events],
                         [(1, "page_request", IDS[2], "serp", 3, True), (2, "back", IDS[2], "detail", 3, True),
                          (3, "back", None, "serp", 0, False), (4, "page_request", IDS[0], "serp", 1, True),
                          (5, "select", IDS[0], "detail", 1, True)])
        self.assertEqual(session.result(), {"status": "choice", "chosen_id": IDS[0], "turns": 5,
                                            "page_requests": [IDS[2], IDS[0]]})

    def test_invalid_calls_are_recorded_and_never_executed(self):
        cases = [
            ([call("open_product", "B0ZZZ99999")], "unknown_id", "B0ZZZ99999"),
            ([call("select", "B0ZZZ99999")], "unknown_id", "B0ZZZ99999"),
            ([call("select", "   ")], "empty", None),
            ([call("fetch_url", url="http://example.com")], "schema_violation", None),
            ([call("open_product", IDS[0], extra=1)], "schema_violation", IDS[0]),
            ([call("leave", reason="none")], "schema_violation", None),
            ([ToolCall(id="x", name="open_product", arguments='{"product_id": "B0AA')], "schema_violation", None),
            ([ToolCall(id="x", name="select", arguments={"product_id": 7})], "schema_violation", None),
        ]
        for calls, reason, product_id in cases:
            with self.subTest(calls=calls[0].name, arguments=calls[0].arguments):
                session = self.session()
                self.assertEqual(session.step(calls), [])  # nothing served
                self.assertEqual(session.result(), {"status": "invalid", "invalid_reason": reason,
                                                    "page_requests": [], "turns": 1})
                (event,) = session.events
                self.assertEqual((event.kind, event.invalid_reason, event.product_id, event.turn),
                                 ("invalid", reason, product_id, 1))

    def test_a_final_call_must_end_its_response(self):
        session = self.session()
        results = session.step([call("open_product", IDS[0]), call("select", IDS[0]), call("open_product", IDS[1])])
        self.assertEqual(len(results), 1)
        self.assertEqual(session.result()["status"], "invalid")
        self.assertEqual(session.result()["invalid_reason"], "multiple_choices")
        self.assertEqual([(e.kind, e.invalid_reason) for e in session.events],
                         [("page_request", None), ("invalid", "multiple_choices"), ("invalid", "multiple_choices")])
        session = self.session()
        session.step([call("open_product", "nope"), call("open_product", IDS[1])])
        self.assertEqual([(e.kind, e.invalid_reason) for e in session.events],
                         [("invalid", "unknown_id"), ("invalid", "unknown_id")])
        self.assertEqual(session.page_requests, [])

    def test_a_response_without_a_tool_call(self):
        for text, reason in ((None, "empty"), ("", "empty"), ("I pick the mug.", "schema_violation")):
            session = self.session()
            session.step([], text)
            self.assertEqual(session.result()["invalid_reason"], reason)
            self.assertEqual(session.events, [])


# ------------------------------------------------------- loop error paths
class LoopTest(EnvCase):
    def decide(self, script, clock=None, **overrides):
        config = self.config(**overrides)
        provider = ScriptedProvider(script)
        kwargs = {"clock": clock} if clock else {}
        backend = LLMBackend(config, config.agents.models[0].key, provider=provider, cache=None, policy="off",
                             **kwargs)
        return provider, backend.decide(self.trial(config), self.observation(config))

    def test_a_provider_error_keeps_the_events_so_far(self):
        def script(n, request):
            if n == 3:
                return ProviderResponse(error_code="rate_limited", attempts=4, latency_s=2.0)
            return [call("open_product", IDS[n - 1])]
        _, decision = self.decide(script)
        outcome = decision.outcome
        self.assertEqual((outcome.status, outcome.error_code, outcome.turns, outcome.attempts),
                         ("error", "rate_limited", 2, 6))
        self.assertEqual(outcome.page_requests, IDS[:2])
        self.assertEqual(len(decision.events), 2)
        self.assertEqual((outcome.usage.input_tokens, outcome.latency_s), (300, 3.0))

    def test_a_refusal_ends_the_trial(self):
        def script(n, request):
            if n == 2:
                return ProviderResponse(text="", refusal=True, attempts=1, returned_model="gpt-4o-2024-08-06")
            return [call("open_product", IDS[0])]
        _, decision = self.decide(script)
        self.assertEqual((decision.outcome.status, decision.outcome.turns), ("refusal", 1))
        self.assertEqual(len(decision.events), 1)

    def test_a_kill_switch_mid_trial_is_not_run_without_events(self):
        def script(n, request):
            if n == 2:
                return KillSwitchEngaged("engaged")
            return [call("open_product", IDS[0])]
        _, decision = self.decide(script)
        self.assertEqual((decision.outcome.status, decision.outcome.error_code, decision.outcome.attempts),
                         ("not_run", "kill_switch", 0))
        self.assertEqual(decision.events, [])

    def test_the_trial_timeout_stops_the_loop_before_the_next_call(self):
        provider, decision = self.decide(lambda n, r: [call("open_product", IDS[0])],
                                         clock=itertools.count(0.0, 50.0).__next__,
                                         **{"failure_policy.trial_timeout_s": 120.0})
        self.assertEqual((decision.outcome.status, decision.outcome.error_code), ("error", "provider_unavailable"))
        self.assertEqual(len(provider.requests), decision.outcome.turns)
        self.assertLess(len(provider.requests), 8)

    def test_the_local_model_has_no_tool_loop(self):
        config = self.env.config(**{"agents.models": [{"key": "qwen2_7b", "provider": "ollama", "model": "qwen2:7b"}],
                                    "agents.offline_backend": "local_llm", "mode": "offline", **BROWSER})
        with self.assertRaisesRegex(ValueError, "tool calling"):
            api.backends_for(config, None, "offline")


# --------------------------------------------- simulated agents and store
class SimulatedTest(EnvCase):
    def test_the_simulated_agent_acts_through_the_same_tools(self):
        config = self.config(mode="offline")
        stimuli = api.build_stimuli(config, self.env.snapshot)
        params = params_from_config(config, stimuli.tasks)
        observation = self.observation(config)
        scores = {p: 1.0 for p in IDS}
        model = params.model(config.agents.models[0].key).model_copy(update={
            "scores": {TASK.task_id: scores}, "page_request_prob": 0.6})
        agent = SimulatedAgent(model, seed=7, max_turns=config.protocol.max_turns)
        requested = 0
        for i in range(60):
            trial = self.trial(config, key="%064x" % i)
            decision = simulated_decision(agent, trial, observation, config.protocol.max_turns)
            self.assertEqual(decision.outcome, agent.decide(observation, "browser_tool", trial.trial_key))
            actions = agent.act(observation, "browser_tool", trial.trial_key)
            self.assertEqual([e.kind for e in decision.events],
                             [{"request_page": "page_request"}.get(a.type, a.type) for a in actions])
            self.assertTrue(all(e.simulated for e in decision.events))
            self.assertEqual([e.turn for e in decision.events], list(range(1, len(actions) + 1)))
            requested += len(decision.outcome.page_requests)
        self.assertGreater(requested, 0)

    def test_an_offline_run_stores_one_event_per_action(self):
        config = self.config(mode="offline", repetitions=3, **{"agents.simulated.page_request_prob": 0.5})
        run_dir, result = api.run(config, snapshot=self.env.snapshot, root=self.env.runs, environ={},
                                  estimate_cost=lambda c, t, m: 0.0)
        self.assertEqual(result.status, "COMPLETED")
        store = LocalStore(run_dir)
        records, events = store.records(), store.events()
        self.assertEqual(set(events), set(records))
        for key, record in records.items():
            kinds = [e.kind for e in events[key]]
            self.assertEqual(kinds.count("page_request"), len(record.outcome.page_requests))
            self.assertEqual(len(kinds), record.outcome.turns)
            self.assertEqual(kinds[-1], "select" if record.outcome.status == "choice" else "leave")
        self.assertGreater(sum(len(r.outcome.page_requests) for r in records.values()), 0)


class StoreTest(EnvCase):
    def test_events_are_kept_or_lost_with_their_record(self):
        config = self.config(mode="offline", repetitions=2, **{"agents.simulated.page_request_prob": 0.5})
        run_dir, _ = api.run(config, snapshot=self.env.snapshot, root=self.env.runs, environ={},
                             estimate_cost=lambda c, t, m: 0.0)
        records, full = LocalStore(run_dir).records(), LocalStore(run_dir).events()
        store = LocalStore(self.env.root / "copy")
        writer = store.open_writer(flush_every=1)  # one gzip member per trial
        for key, record in records.items():
            writer.append(record, full.get(key, []))
        (shard,) = store.shards()
        shard.write_bytes(shard.read_bytes()[:-7])  # a kill cuts the last member short
        cut_records, cut_events = store.records(), store.events()
        self.assertEqual(len(cut_records), len(records) - 1)
        self.assertEqual(set(cut_events), set(cut_records) & set(full))
        self.assertEqual(cut_events, {k: v for k, v in full.items() if k in cut_records})
        writer = store.open_writer(flush_every=1)  # resume: only the lost trial is written again
        self.assertEqual(sum(writer.append(r, full.get(k, [])) for k, r in records.items()), 1)
        writer.flush()
        self.assertEqual(store.records(), records)
        self.assertEqual(store.events(), full)


if __name__ == "__main__":
    unittest.main()
