"""SL-09 provider adapters: the configured model on every call and retry, no gateway
imports (AC-1); fixture contract tests (AC-2); the kill switch (AC-3); one call-log
line per live call under LocalGovernance and PlatformGovernance (AC-4); the live-smoke
report (AC-5); and failure injection per tickets.md section 3.5.

Two layers. ``FakeSDK`` stands in for each SDK's client object (no SDK needed). The
contract tests run the real pinned SDKs (openai 2.8.0, anthropic 1.17.0, google-genai
1.56.0) against an in-process HTTP transport (``Server``) serving the fixtures in
``fixtures/providers``; they are skipped where those SDKs are not installed. No test
uses the network.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import itertools
import json
import re
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from features.storelab import api, live_smoke
from features.storelab.backends import provider as provider_module
from features.storelab.backends.anthropic_adapter import AnthropicAdapter
from features.storelab.backends.google_adapter import GoogleAdapter
from features.storelab.backends.llm import LLMBackend
from features.storelab.backends.openai_adapter import OpenAIAdapter
from features.storelab.backends.provider import KillSwitchEngaged, NoGuard, code_for_status, retry_after_s
from features.storelab.core.cache import LLMRequest, Message
from features.storelab.core.config import FailurePolicyConfig
from features.storelab.core.runner import LocalGovernance, PlanError, Trial, kill_switch_engaged
from features.storelab.core.store import LocalStore
from features.storelab.governance import Platform, PlatformGovernance
from features.storelab.pricing import MODEL_PRICES, UnknownModelError, call_cost_usd
from features.storelab.settings import StorelabSettings, model_problems
from tests.features.storelab.test_cache import Env

STORELAB = Path(api.__file__).resolve().parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "providers"
KILL = "LOADTEST_LLM_KILL_SWITCH_FORCE"
HAVE_SDKS = all(importlib.util.find_spec(m) for m in ("openai", "anthropic", "httpx", "httpx2")) and \
    importlib.util.find_spec("google") is not None and importlib.util.find_spec("google.genai") is not None
MODELS = [
    {"key": "gpt_4o_20240806", "provider": "openai", "model": "gpt-4o-2024-08-06", "temperatures": [None, 0.0]},
    {"key": "gpt_41_20250414", "provider": "openai", "model": "gpt-4.1-2025-04-14", "temperatures": [None]},
    {"key": "claude_sonnet_45_20250929", "provider": "anthropic", "model": "claude-sonnet-4-5-20250929",
     "temperatures": [None, 0.0]},
    {"key": "gemini_20_flash_001", "provider": "google", "model": "gemini-2.0-flash-001", "temperatures": [None, 0.0]},
]
KEYS = {"OPENAI_API_KEY": "test-openai", "ANTHROPIC_API_KEY": "test-anthropic", "GOOGLE_API_KEY": "test-google"}
FAST = {"failure_policy.backoff_initial_s": 0.001, "failure_policy.backoff_max_s": 0.002}
ID = re.compile(r'ID: (\S+) \||"id": "([^"]+)"')


def request(provider="openai", model="gpt-4o-2024-08-06", **params) -> LLMRequest:
    return LLMRequest(provider=provider, requested_model=model,
                      params={"temperature": None, "top_p": None, "max_output_tokens": 256, **params},
                      prompt_template_sha="a" * 64, messages=[Message(role="user", content="prompt")], rep=0,
                      sample_namespace="ns")


def policy(**changes) -> FailurePolicyConfig:
    return FailurePolicyConfig(**{"max_attempts": 4, "backoff_initial_s": 1.0, "backoff_max_s": 60.0, **changes})


# ------------------------------------------------------------- SDK-free fakes
class APIError(Exception):
    """Shaped like the SDKs' status errors: ``status_code`` (OpenAI, Anthropic), ``code``
    on a class named APIError (google-genai), and ``response.headers``."""

    def __init__(self, status: int, headers: dict | None = None):
        super().__init__("HTTP %d" % status)
        self.status_code = self.code = status
        self.response = SimpleNamespace(headers=headers or {})


StatusError = APIError


class APITimeoutError(Exception):
    """Named like the SDKs' timeout error, which is how the adapters recognise it."""


class FakeSDK:
    """A client object with each SDK's call path; ``script(n)`` gives the n-th result."""

    def __init__(self, provider: str, script):
        self.provider, self.script, self.calls = provider, script, []
        create = self.create
        self.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=create)))
        self.messages = SimpleNamespace(with_raw_response=SimpleNamespace(create=create))
        self.models = SimpleNamespace(generate_content=create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.script(len(self.calls))
        if isinstance(result, BaseException):
            raise result
        return result

    def adapter(self, guard=None, **kwargs):
        cls = {"openai": OpenAIAdapter, "anthropic": AnthropicAdapter, "google": GoogleAdapter}[self.provider]
        return cls(self, kwargs.pop("policy", policy()), guard, **kwargs)


def openai_ok(model: str, text: str = '{"choice": null, "reason": "none fit"}'):
    completion = SimpleNamespace(
        model=model, system_fingerprint="fp_1", choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, refusal=None, tool_calls=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
        model_dump=lambda mode=None: {"model": model})
    return SimpleNamespace(parse=lambda: completion, headers={"x-request-id": "req_1"})


def sdk_ok(provider: str, model: str):
    if provider == "openai":
        return openai_ok(model)
    if provider == "anthropic":
        message = SimpleNamespace(model=model, content=[SimpleNamespace(type="text", text='{"choice": null}')],
                                  stop_reason="end_turn", usage=SimpleNamespace(input_tokens=100, output_tokens=10),
                                  model_dump=lambda mode=None: {"model": model})
        return SimpleNamespace(parse=lambda: message, headers={"request-id": "req_1"})
    part = SimpleNamespace(text='{"choice": null}', thought=None, function_call=None)
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason="STOP")],
                           prompt_feedback=None, usage_metadata=SimpleNamespace(prompt_token_count=100,
                                                                                candidates_token_count=10),
                           model_version=model, response_id="resp_1", model_dump=lambda **kw: {"model": model})


def requested_model(provider: str, kwargs: dict) -> str:
    return kwargs["model"]


class FakeSDKAdapterTest(unittest.TestCase):
    """AC-1 with the SDKs mocked: every call, including retries, names the configured model."""

    PROVIDERS = (("openai", "gpt-4o-2024-08-06"), ("anthropic", "claude-sonnet-4-5-20250929"),
                 ("google", "gemini-2.0-flash-001"))

    def test_every_call_and_retry_names_the_configured_model(self):
        for name, model in self.PROVIDERS:
            with self.subTest(provider=name):
                sleeps = []
                failures = {1: StatusError(429, {"retry-after": "3"}), 2: StatusError(503), 3: APITimeoutError()}
                sdk = FakeSDK(name, lambda n: failures.get(n) or sdk_ok(name, model))
                response = sdk.adapter(sleep=sleeps.append).complete(request(name, model))
                self.assertIsNone(response.error_code)
                self.assertEqual(response.attempts, 4)
                self.assertEqual([requested_model(name, c) for c in sdk.calls], [model] * 4)
                self.assertEqual(sleeps, [3.0, 2.0, 4.0])  # Retry-After 3 s beats 1 s; then 2 s, 4 s
                self.assertEqual(response.returned_model, model)

    def test_a_returned_alias_is_recorded_and_never_retried_with_another_model(self):
        sdk = FakeSDK("openai", lambda n: openai_ok("gpt-4o"))
        response = sdk.adapter().complete(request())
        self.assertEqual(response.returned_model, "gpt-4o")
        self.assertEqual([c["model"] for c in sdk.calls], ["gpt-4o-2024-08-06"])

    def test_sent_parameters(self):
        sdk = FakeSDK("openai", lambda n: openai_ok("gpt-4o-2024-08-06"))
        response = sdk.adapter().complete(request(seed=7))
        self.assertNotIn("temperature", sdk.calls[0])  # null temperature: omitted
        self.assertEqual((sdk.calls[0]["seed"], sdk.calls[0]["max_completion_tokens"]), (7, 256))
        self.assertEqual(response.sent_params, {"temperature": "provider_default", "max_output_tokens": 256,
                                                "seed": 7})
        self.assertTrue(response.seed_supported)
        self.assertEqual((response.request_id, response.system_fingerprint), ("req_1", "fp_1"))
        sdk.adapter().complete(request(temperature=0.0, top_p=0.9))
        self.assertEqual((sdk.calls[1]["temperature"], sdk.calls[1]["top_p"]), (0.0, 0.9))
        claude = FakeSDK("anthropic", lambda n: sdk_ok("anthropic", "claude-sonnet-4-5-20250929"))
        reply = claude.adapter().complete(request("anthropic", "claude-sonnet-4-5-20250929", temperature=0.0))
        self.assertEqual(claude.calls[0]["extra_body"], {"temperature": 0.0})
        self.assertFalse(reply.seed_supported)
        self.assertEqual(reply.api_version, "2023-06-01")
        claude.adapter().complete(request("anthropic", "claude-sonnet-4-5-20250929"))
        self.assertIsNone(claude.calls[1]["extra_body"])

    def test_an_adapter_refuses_another_providers_request(self):
        with self.assertRaisesRegex(ValueError, "adapter openai got a request for provider google"):
            FakeSDK("openai", lambda n: None).adapter().complete(request("google", "gemini-2.0-flash-001"))


class FailureInjectionTest(unittest.TestCase):
    """tickets.md section 3.5 at the adapter: codes, retries and attempt counts."""

    def test_error_codes_retries_and_attempts(self):
        cases = [  # (failure, code, HTTP calls)
            (StatusError(400), "bad_request", 1), (StatusError(422), "bad_request", 1),
            (StatusError(401), "auth", 1), (StatusError(403), "auth", 1), (StatusError(404), "model_unavailable", 1),
            (StatusError(429), "rate_limited", 4), (StatusError(500), "provider_unavailable", 4),
            (StatusError(529), "provider_unavailable", 4), (StatusError(408), "provider_unavailable", 4),
            (APITimeoutError(), "provider_unavailable", 4), (ConnectionResetError(), "provider_unavailable", 4)]
        for failure, code, calls in cases:
            with self.subTest(failure=repr(failure)):
                sdk = FakeSDK("openai", lambda n, f=failure: f)
                sleeps = []
                response = sdk.adapter(sleep=sleeps.append).complete(request())
                self.assertEqual((response.error_code, response.attempts, len(sdk.calls)), (code, calls, calls))
                self.assertEqual(len(sleeps), calls - 1)
                self.assertIsNone(response.text)

    def test_backoff_honours_retry_after_up_to_the_cap(self):
        sdk = FakeSDK("openai", lambda n: StatusError(429, {"retry-after-ms": "250"}) if n == 1 else
                      StatusError(429, {"retry-after": "120"}) if n == 2 else openai_ok("gpt-4o-2024-08-06"))
        sleeps = []
        sdk.adapter(sleep=sleeps.append, policy=policy(backoff_max_s=30.0)).complete(request())
        self.assertEqual(sleeps, [1.0, 30.0])
        self.assertEqual(retry_after_s({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"},
                                       now=lambda: 1792567670.0), 10.0)
        self.assertIsNone(retry_after_s({"retry-after": "soon"}))
        self.assertEqual([code_for_status(s) for s in (409, 418, 502)],
                         ["provider_unavailable", "bad_request", "provider_unavailable"])

    def test_a_kill_switch_between_retries_labels_the_call(self):
        environ = {}
        guard = LocalGovernance("run", 0.0, environ=environ)

        def script(n):
            environ[KILL] = "1"
            return StatusError(503)

        sdk = FakeSDK("openai", script)
        response = sdk.adapter(guard, sleep=lambda s: None).complete(request())
        self.assertEqual((response.error_code, response.attempts, response.stopped_by, len(sdk.calls)),
                         ("provider_unavailable", 1, "kill_switch", 1))
        clean = FakeSDK("openai", lambda n: StatusError(503)).adapter(sleep=lambda s: None).complete(request())
        self.assertIsNone(clean.stopped_by)
        self.assertNotIn("stopped_by", clean.model_dump(mode="json"))

    def test_an_unclassified_exception_is_a_defect_and_propagates(self):
        sdk = FakeSDK("anthropic", lambda n: KeyError("bug"))
        with self.assertRaises(KeyError):
            sdk.adapter().complete(request("anthropic", "claude-sonnet-4-5-20250929"))
        self.assertEqual(len(sdk.calls), 1)

    def test_refusals_are_responses(self):
        def refused(n):
            reply = openai_ok("gpt-4o-2024-08-06", text=None)
            completion = reply.parse()
            completion.choices[0].message.refusal = "I can't help with that."
            return reply

        response = FakeSDK("openai", refused).adapter().complete(request())
        self.assertTrue(response.refusal)
        self.assertIsNone(response.error_code)


# ------------------------------------------------------------- import boundary
class ImportTest(unittest.TestCase):
    """AC-1: no adapter reaches the gateway facade or its fallback settings."""

    FORBIDDEN = ("get_adgraph_client", "AdgraphClient", "AdgraphFallbackSettings", "model_gateway",
                 "_run_with_failover", "AdgraphModelTarget")
    FILES = ("backends/provider.py", "backends/openai_adapter.py", "backends/google_adapter.py",
             "backends/anthropic_adapter.py", "backends/llm.py", "governance.py", "settings.py", "pricing.py",
             "live_smoke.py", "api.py")

    def test_no_adapter_names_the_gateway_or_its_fallback(self):
        for name in self.FILES:
            with self.subTest(file=name):
                tree = ast.parse((STORELAB / name).read_text(encoding="utf-8"))
                names = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names.update(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        names.add(node.module or "")
                        names.update(alias.name for alias in node.names)
                    elif isinstance(node, ast.Name):
                        names.add(node.id)
                    elif isinstance(node, ast.Attribute):
                        names.add(node.attr)
                hits = sorted(n for n in names for f in self.FORBIDDEN if f in n)
                self.assertEqual(hits, [])

    def test_loading_the_adapters_loads_no_gateway_module(self):
        for module in list(sys.modules):
            self.assertNotIn("model_gateway", module)
        self.assertNotIn("features.adgraph", " ".join(sys.modules))


# ------------------------------------------------------------- HTTP contract
def reply_for(prompt: str) -> str:
    ids = [a or b for a, b in ID.findall(prompt)]
    return '{"choice": "%s", "reason": "fits the task"}' % ids[0]


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def response_json(provider: str, model: str, text: str) -> dict:
    """A success body in each API's format, from the authored fixtures."""
    body = json.loads(json.dumps(fixture("%s_ok.json" % provider)["response"]["json"]))
    if provider == "openai":
        body["model"], body["choices"][0]["message"]["content"] = model, text
    elif provider == "anthropic":
        body["model"], body["content"][0]["text"] = model, text
    else:
        body["modelVersion"], body["candidates"][0]["content"]["parts"][0]["text"] = model, text
    return body


ERROR_BODIES = {"openai": {"error": {"message": "injected", "type": "server_error", "code": None}},
                "anthropic": {"type": "error", "error": {"type": "api_error", "message": "injected"}},
                "google": {"error": {"code": 500, "message": "injected", "status": "INTERNAL"}}}


class Server:
    """One provider's API, in process: records each request and answers by ``script``."""

    def __init__(self, provider: str, script=None):
        import httpx
        import httpx2

        self.provider, self.module = provider, httpx2 if provider == "anthropic" else httpx
        self.script = script or (lambda n, model, body: (200, {}, response_json(provider, model, reply_for(
            self.prompt(body)))))
        self.requests, self.lock = [], threading.Lock()

    @staticmethod
    def prompt(body: dict) -> str:
        if "messages" in body:
            return body["messages"][-1]["content"]
        return body["contents"][-1]["parts"][0]["text"]

    def model(self, request, body) -> str:
        if self.provider == "google":
            return re.search(r"models/([^:/]+):generateContent", request.url.path).group(1)
        return body["model"]

    def handle(self, request):
        body = json.loads(request.content or b"null")
        with self.lock:
            self.requests.append({"path": request.url.path, "headers": dict(request.headers), "json": body,
                                  "model": self.model(request, body)})
            n = len(self.requests)
        result = self.script(n, self.model(request, body), body)
        if result == "timeout":
            raise self.module.ReadTimeout("injected timeout", request=request)
        status, headers, payload = result
        if status >= 400 and payload is None:
            payload = ERROR_BODIES[self.provider]
        return self.module.Response(status, headers={"content-type": "application/json", **headers}, json=payload,
                                    request=request)

    def client(self):
        return self.module.Client(transport=self.module.MockTransport(self.handle))

    def adapter(self, guard=None, **kwargs):
        cls = {"openai": OpenAIAdapter, "anthropic": AnthropicAdapter, "google": GoogleAdapter}[self.provider]
        return cls.from_key("test-key", kwargs.pop("policy", policy()), guard, http_client=self.client(), **kwargs)


@unittest.skipUnless(HAVE_SDKS, "the pinned provider SDKs are not installed")
class ContractTest(unittest.TestCase):
    """AC-2: requested id, returned id, usage and latency from each provider's fixtures,
    through the real SDKs; the request each SDK sends."""

    def fixtures(self):
        names = sorted(p.name for p in FIXTURES.glob("*.json"))
        self.assertGreaterEqual(len(names), 6)
        return [fixture(n) | {"name": n} for n in names]

    def expected(self, data: dict) -> dict:
        if "expect" in data:
            return data["expect"]
        body = data["response"]["json"]  # a recorded exchange
        if data["provider"] == "openai":
            return {"returned_model": body["model"], "input_tokens": body["usage"]["prompt_tokens"],
                    "output_tokens": body["usage"]["completion_tokens"],
                    "refusal": bool(body["choices"][0]["message"].get("refusal")),
                    "request_id": data["response"]["headers"].get("x-request-id")}
        if data["provider"] == "anthropic":
            return {"returned_model": body["model"], "input_tokens": body["usage"]["input_tokens"],
                    "output_tokens": body["usage"]["output_tokens"], "refusal": body["stop_reason"] == "refusal",
                    "request_id": data["response"]["headers"].get("request-id")}
        usage = body.get("usageMetadata", {})
        return {"returned_model": body["modelVersion"], "input_tokens": usage.get("promptTokenCount"),
                "output_tokens": usage.get("candidatesTokenCount"), "request_id": body.get("responseId"),
                "refusal": body.get("candidates", [{}])[0].get("finishReason") == "SAFETY"}

    def test_fixtures_map_to_requested_and_returned_id_usage_and_latency(self):
        for data in self.fixtures():
            with self.subTest(fixture=data["name"]):
                provider = data["provider"]
                model = data.get("model") or data["request"]["json"].get("model") or re.search(
                    r"models/([^:/]+):", data["request"]["path"]).group(1)
                server = Server(provider, lambda n, m, b, d=data: (d["response"]["status"], d["response"]["headers"],
                                                                   d["response"]["json"]))
                clock = itertools.count(10.0, 0.25).__next__
                response = server.adapter(clock=clock).complete(request(provider, model))
                want = self.expected(data)
                self.assertEqual(server.requests[0]["model"], model)
                self.assertEqual(response.returned_model, want["returned_model"])
                self.assertEqual((response.usage.input_tokens, response.usage.output_tokens),
                                 (want["input_tokens"], want["output_tokens"]))
                self.assertEqual(response.refusal, want["refusal"])
                self.assertEqual(response.request_id, want["request_id"])
                self.assertEqual(response.latency_s, 0.25)
                self.assertEqual(response.attempts, 1)
                if data["source"] == "authored":
                    self.assertTrue(data["note"].startswith("Authored"))

    def test_the_trial_outcome_carries_the_ids_usage_and_latency(self):
        env = Env()
        self.addCleanup(env.cleanup)
        for spec in MODELS[::2] + MODELS[3:]:
            with self.subTest(model=spec["model"]):
                config = env.config(**{"agents.models": [spec]})
                server = Server(spec["provider"])
                run_dir, result = api.run(config, snapshot=env.snapshot, root=env.runs, environ=KEYS,
                                          estimate_cost=lambda c, t, m: 0.0, cache_root=env.cache_root,
                                          http_clients={spec["provider"]: server.client()})
                self.assertEqual(result.status, "COMPLETED")
                records = LocalStore(run_dir).records().values()
                self.assertEqual(len(server.requests), len(records))
                for record in records:
                    outcome = record.outcome
                    self.assertEqual((outcome.status, outcome.backend), ("choice", spec["provider"]))
                    self.assertEqual((outcome.requested_model, outcome.returned_model), (spec["model"],) * 2)
                    self.assertGreater(outcome.usage.input_tokens, 0)
                    self.assertGreater(outcome.usage.output_tokens, 0)
                    self.assertGreaterEqual(outcome.latency_s, 0.0)
                manifest = LocalStore(run_dir).read_manifest()
                self.assertEqual(manifest.models[0].returned_ids, [spec["model"]])
                self.assertIsNotNone(manifest.models[0].last_access_utc)

    def test_each_sdk_sends_the_configured_request(self):
        cases = {"openai": ("gpt-4o-2024-08-06", {"seed": 11, "temperature": 0.0}),
                 "anthropic": ("claude-sonnet-4-5-20250929", {"temperature": 0.0}),
                 "google": ("gemini-2.0-flash-001", {"seed": 11, "temperature": 0.0})}
        for provider, (model, params) in cases.items():
            with self.subTest(provider=provider):
                server = Server(provider)
                server.script = lambda n, m, b, p=provider: (200, {}, response_json(p, m, '{"choice": null}'))
                adapter = server.adapter()
                adapter.complete(request(provider, model, **params))
                adapter.complete(request(provider, model))
                (first, second) = server.requests
                self.assertEqual((first["model"], second["model"]), (model, model))
                if provider == "openai":
                    self.assertEqual(adapter.client.max_retries, 0)
                    self.assertEqual({k: first["json"][k] for k in ("seed", "temperature", "max_completion_tokens")},
                                     {"seed": 11, "temperature": 0.0, "max_completion_tokens": 256})
                    self.assertNotIn("temperature", second["json"])
                elif provider == "anthropic":
                    self.assertEqual(adapter.client.max_retries, 0)
                    self.assertEqual(first["headers"]["anthropic-version"], "2023-06-01")
                    self.assertEqual((first["json"]["temperature"], first["json"]["max_tokens"]), (0.0, 256))
                    self.assertNotIn("seed", first["json"])
                    self.assertNotIn("temperature", second["json"])
                else:
                    config = first["json"]["generationConfig"]
                    self.assertEqual((config["seed"], config["temperature"], config["maxOutputTokens"]), (11, 0.0, 256))
                    self.assertNotIn("temperature", second["json"]["generationConfig"])
                    self.assertEqual(adapter.client._api_client._http_options.retry_options.attempts, 1)

    def test_injected_http_failures_through_each_sdk(self):
        for provider, model in (("openai", "gpt-4o-2024-08-06"), ("anthropic", "claude-sonnet-4-5-20250929"),
                                ("google", "gemini-2.0-flash-001")):
            for status, code, calls in ((400, "bad_request", 1), (401, "auth", 1), (403, "auth", 1),
                                        (404, "model_unavailable", 1), (429, "rate_limited", 3),
                                        (503, "provider_unavailable", 3), ("timeout", "provider_unavailable", 3)):
                with self.subTest(provider=provider, status=status):
                    server = Server(provider, lambda n, m, b, s=status: s if s == "timeout" else (s, {}, None))
                    sleeps = []
                    response = server.adapter(policy=policy(max_attempts=3), sleep=sleeps.append).complete(
                        request(provider, model))
                    self.assertEqual((response.error_code, response.attempts), (code, calls))
                    self.assertEqual(len(server.requests), calls)  # the SDK itself never retries
                    self.assertEqual({r["model"] for r in server.requests}, {model})
            with self.subTest(provider=provider, retry_after=True):
                server = Server(provider, lambda n, m, b, p=provider: (429, {"retry-after": "7"}, None) if n == 1 else
                                (200, {}, response_json(p, m, '{"choice": null}')))
                sleeps = []
                response = server.adapter(sleep=sleeps.append).complete(request(provider, model))
                self.assertEqual((response.error_code, response.attempts), (None, 2))
                self.assertEqual(sleeps, [7.0])


# ------------------------------------------------------------- runs (AC-1, 3, 4)
@unittest.skipUnless(HAVE_SDKS, "the pinned provider SDKs are not installed")
class SandboxRunTest(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        self.addCleanup(self.env.cleanup)

    def config(self, **overrides):
        return self.env.config(**{"agents.models": MODELS, "repetitions": 2, **FAST, **overrides})

    def run_with(self, config, servers, environ=None, **kwargs):
        return api.run(config, snapshot=self.env.snapshot, root=self.env.runs,
                       environ=KEYS if environ is None else environ, estimate_cost=lambda c, t, m: 0.0,
                       cache_root=self.env.cache_root, http_clients={p: s.client() for p, s in servers.items()},
                       **kwargs)

    def test_every_request_of_a_run_names_its_trials_model(self):
        """AC-1 over a run with injected 429, 5xx and timeouts on all three SDKs."""
        def flaky(provider):
            def script(n, model, body):
                if n % 3 == 1:
                    return [(429, {"retry-after": "0"}, None), (503, {}, None), "timeout"][(n // 3) % 3]
                return 200, {}, response_json(provider, model, reply_for(Server.prompt(body)))
            return script

        servers = {p: Server(p, flaky(p)) for p in ("openai", "anthropic", "google")}
        run_dir, result = self.run_with(self.config(), servers)
        self.assertEqual(result.status, "COMPLETED")
        store = LocalStore(run_dir)
        records = list(store.records().values())
        by_model = {}
        for record in records:
            spec = next(m for m in MODELS if m["key"] == record.model_key)
            self.assertEqual(record.outcome.requested_model, spec["model"])
            by_model[spec["model"]] = by_model.get(spec["model"], 0) + record.outcome.attempts
        requests = [r for s in servers.values() for r in s.requests]
        self.assertGreater(sum(r.outcome.attempts > 1 for r in records), 10)  # retries happened
        self.assertEqual({m: sum(r["model"] == m for r in requests) for m in by_model}, by_model)
        self.assertEqual(sorted({r["model"] for r in requests}), sorted(m["model"] for m in MODELS))

    def test_forced_kill_switch_cancels_the_run_with_no_later_calls(self):
        """AC-3: LOADTEST_LLM_KILL_SWITCH_FORCE=1 set mid-run."""
        environ = dict(KEYS)
        arrived_engaged = []

        def script(n, model, body):
            arrived_engaged.append(kill_switch_engaged(environ))
            if n == 5:
                environ[KILL] = "1"
            return 200, {}, response_json("openai", model, reply_for(Server.prompt(body)))

        servers = {"openai": Server("openai", script)}
        config = self.config(**{"agents.models": MODELS[:1], "budget.max_concurrency": 1})
        run_dir, result = self.run_with(config, servers, environ=environ)
        self.assertEqual((result.status, result.reason), ("CANCELLED", "kill_switch"))
        self.assertEqual(len(servers["openai"].requests), 5)
        self.assertEqual(arrived_engaged, [False] * 5)  # zero calls after the switch
        manifest = LocalStore(run_dir).read_manifest()
        self.assertEqual((manifest.status_history[-1].status, manifest.status_history[-1].reason),
                         ("CANCELLED", "kill_switch"))
        statuses = [r.outcome.status for r in LocalStore(run_dir).records().values()]
        self.assertEqual((statuses.count("choice"), statuses.count("not_run")), (5, 1))
        # resuming with the switch still engaged makes no call either
        with self.assertRaisesRegex(ValueError, "CANCELLED"):
            api.resume(run_dir, environ=environ)
        self.assertEqual(len(servers["openai"].requests), 5)

    def test_the_kill_switch_stops_retries_and_never_starts_a_call(self):
        environ = {}
        guard = LocalGovernance("run", 0.0, environ=environ)

        def script(n, model, body):
            environ[KILL] = "1"
            return 503, {}, None

        server = Server("openai", script)
        response = server.adapter(guard, sleep=lambda s: None).complete(request())
        self.assertEqual((response.error_code, response.attempts, len(server.requests)), ("provider_unavailable", 1, 1))
        self.assertEqual(response.stopped_by, "kill_switch")  # labelled as cut short by the switch
        with self.assertRaises(KillSwitchEngaged):
            server.adapter(guard).complete(request())
        self.assertEqual(len(server.requests), 1)
        for value, engaged in (("1", True), ("0", False), (" 0 ", False), ("", True), ("yes", True)):
            self.assertEqual(kill_switch_engaged({KILL: value}), engaged)
        self.assertFalse(kill_switch_engaged({}))

    def test_one_local_log_line_per_live_call(self):
        """AC-4 (LocalGovernance): invocations.jsonl has one line per live call with run_id
        and trial_key; cache hits and stopped calls write none."""
        def script(n, model, body):
            if n % 4 == 0:
                return 503, {}, None
            return 200, {}, response_json("openai", model, reply_for(Server.prompt(body)))

        servers = {"openai": Server("openai", script)}
        config = self.config(**{"agents.models": MODELS[:2]})
        run_dir, result = self.run_with(config, servers)
        self.assertEqual(result.status, "COMPLETED")
        store = LocalStore(run_dir)
        lines = [json.loads(x) for x in store.invocation_log.read_text(encoding="utf-8").splitlines()]
        records = store.records()
        run_id = store.read_manifest().run_id
        self.assertEqual(len(lines), len(records))
        self.assertEqual(sorted(x["trial_key"] for x in lines), sorted(records))
        self.assertEqual({x["run_id"] for x in lines}, {run_id})
        self.assertEqual(sum(x["attempts"] for x in lines), len(servers["openai"].requests))
        for x in lines:
            record = records[x["trial_key"]]
            self.assertEqual((x["attempts"], x["requested_model"]),
                             (record.outcome.attempts, record.outcome.requested_model))
            self.assertAlmostEqual(x["cost_usd"], call_cost_usd(x["requested_model"], record.outcome.usage))
        # a second run of the same config is served from the cache: no call, no line
        before = len(servers["openai"].requests)
        again_dir, again = self.run_with(config, servers)
        self.assertEqual(again.status, "COMPLETED")
        self.assertEqual(len(servers["openai"].requests), before)
        self.assertFalse(LocalStore(again_dir).invocation_log.exists())

    def test_platform_governance_writes_one_invocation_log_per_live_call(self):
        """AC-4 (PlatformGovernance) with the platform functions faked."""
        platform = FakePlatform()

        def script(n, model, body):
            if n % 5 == 0:
                return 500, {}, None
            return 200, {}, response_json("anthropic", model, reply_for(Server.prompt(body)))

        servers = {"anthropic": Server("anthropic", script)}
        config = self.config(**{"agents.models": MODELS[2:3]})
        run_dir, result = self.run_with(config, servers, platform=platform.build())
        self.assertEqual(result.status, "COMPLETED")
        store = LocalStore(run_dir)
        records, run_id = store.records(), store.read_manifest().run_id
        logs = platform.logs
        self.assertEqual(len(logs), len(records))
        self.assertEqual(sorted(x["task_ref"] for x in logs), sorted(records))
        self.assertEqual({(x["job_id"], x["invocation_kind"], x["task_type"]) for x in logs},
                         {(run_id, "storelab", "storelab_trial")})
        self.assertEqual({x["model_name"] for x in logs}, {"claude-sonnet-4-5-20250929"})
        requests = len(servers["anthropic"].requests)
        self.assertEqual(platform.counts["slot"], requests)
        self.assertEqual(platform.counts["release"], requests)
        self.assertEqual(platform.counts["success"] + platform.counts["failure"], requests)
        self.assertEqual(platform.counts["failure"], sum(1 for n in range(1, requests + 1) if n % 5 == 0))
        self.assertGreater(platform.counts["kill_checks"], requests)  # every trial and every attempt
        self.assertFalse(store.invocation_log.exists())

    def test_platform_kill_switch_breaker_and_slots(self):
        platform = FakePlatform(killed=True)
        governance = PlatformGovernance("run", 1.0, platform.build())
        trial = Trial(trial_key="a" * 64, task_id="t", set_id="s", presentation_id="p", interface="text",
                      model_key="m", temperature=None, rep=0, k=2)
        self.assertEqual(governance.authorize(trial, 0.0), "kill_switch")
        self.assertEqual(governance.before_attempt("openai", "gpt-4o-2024-08-06"), "kill_switch")
        open_breaker = FakePlatform(breaker_open=True)
        server = Server("openai")
        response = server.adapter(PlatformGovernance("run", 1.0, open_breaker.build())).complete(request())
        self.assertEqual((response.error_code, response.attempts, len(server.requests)), ("provider_unavailable", 0, 0))
        busy = FakePlatform(capacity_timeout=True)
        response = server.adapter(PlatformGovernance("run", 1.0, busy.build())).complete(request())
        self.assertEqual((response.error_code, response.attempts, len(server.requests)), ("rate_limited", 0, 0))
        self.assertEqual(busy.counts["slot"], busy.counts["release"])
        broken = FakePlatform(capacity_error=RuntimeError("limiter down"))
        with self.assertRaisesRegex(RuntimeError, "limiter down"):
            server.adapter(PlatformGovernance("run", 1.0, broken.build())).complete(request())
        self.assertEqual((broken.counts["slot"], broken.counts["release"], len(server.requests)), (1, 1, 0))

    def test_breaker_open_writes_no_log_and_pauses_the_run(self):
        platform = FakePlatform(breaker_open=True)
        servers = {"openai": Server("openai")}
        config = self.config(**{"agents.models": MODELS[:1], "budget.max_concurrency": 1,
                                "failure_policy.breaker.max_auto_resumes": 0})
        run_dir, result = self.run_with(config, servers, platform=platform.build())
        self.assertEqual(result.status, "FAILED")
        self.assertIn("breaker_open", result.reason)
        self.assertEqual((len(servers["openai"].requests), len(platform.logs)), (0, 0))

    def test_run_effects_of_provider_failures(self):
        """Section 3.5 run effects through the real adapters."""
        cases = {404: ("FAILED", "model_unavailable"), 401: ("FAILED", "auth"), 400: ("FAILED", "bad_request")}
        for status, (run_status, reason) in cases.items():
            with self.subTest(status=status):
                servers = {"openai": Server("openai", lambda n, m, b, s=status: (s, {}, None))}
                _, result = self.run_with(self.config(**{"agents.models": MODELS[:1]}), servers)
                self.assertEqual(result.status, run_status)
                self.assertIn(reason, result.reason)
        drift = {"google": Server("google", lambda n, m, b: (200, {}, response_json(
            "google", m if n < 4 else "gemini-2.0-flash-002", reply_for(Server.prompt(b)))))}
        _, result = self.run_with(self.config(**{"agents.models": MODELS[3:], "budget.max_concurrency": 1}), drift)
        self.assertEqual(result.status, "PAUSED")
        self.assertIn("model_drift", result.reason)


class FakePlatform:
    """The ai-agents guard functions, counted."""

    def __init__(self, killed=False, breaker_open=False, capacity_timeout=False, capacity_error=None):
        self.killed, self.breaker_open, self.capacity_timeout = killed, breaker_open, capacity_timeout
        self.capacity_error = capacity_error
        self.counts = {k: 0 for k in ("kill_checks", "slot", "release", "capacity", "success", "failure")}
        self.logs, self.lock = [], threading.Lock()

    def count(self, name):
        with self.lock:
            self.counts[name] += 1

    def build(self) -> Platform:
        fake = self

        class Aborted(RuntimeError):
            pass

        class CircuitOpen(RuntimeError):
            pass

        class Busy(RuntimeError):
            pass

        def enforce(call_site=""):
            fake.count("kill_checks")
            if fake.killed:
                raise Aborted("engaged")

        def capacity(model, provider):
            fake.count("capacity")
            if fake.capacity_timeout:
                raise TimeoutError
            if fake.capacity_error is not None:
                raise fake.capacity_error

        def allow(provider, model):
            if fake.breaker_open:
                raise CircuitOpen(provider)

        def write(**fields):
            with fake.lock:
                fake.logs.append(fields)

        breaker = SimpleNamespace(allow=allow, record_success=lambda p, m: fake.count("success"),
                                  record_failure=lambda p, m, e: fake.count("failure"))
        return Platform(enforce, Aborted, breaker, CircuitOpen, lambda p: fake.count("slot"),
                        lambda p: fake.count("release"), Busy, capacity, write)


# ------------------------------------------------------------- backend glue
class BackendTest(unittest.TestCase):
    def test_a_kill_switch_before_the_first_attempt_is_not_run(self):
        env = Env()
        self.addCleanup(env.cleanup)
        config = env.config(**{"agents.models": MODELS[:1]})
        guard = LocalGovernance("run", 0.0, environ={KILL: "1"})
        sdk = FakeSDK("openai", lambda n: openai_ok("gpt-4o-2024-08-06"))
        backend = LLMBackend(config, MODELS[0]["key"], provider=sdk.adapter(guard), cache=None, policy="off",
                             run_id=None, guard=guard)
        stimuli = api.build_stimuli(config, env.snapshot)
        from features.storelab.core.render import build_observation
        from features.storelab.core.runner import plan_trials
        trial = plan_trials(config, stimuli)[0]
        design = next(d for d in stimuli.designs if any(s.set_id == trial.set_id for s in d.sets))
        obs = build_observation(config, stimuli.tasks[0].task, design.presentation_for(trial.set_id, trial.rep),
                                stimuli.products)
        outcome = backend.decide(trial, obs)
        self.assertEqual((outcome.status, outcome.error_code, outcome.attempts), ("not_run", "kill_switch", 0))
        self.assertEqual(sdk.calls, [])

    def test_no_guard(self):
        guard = NoGuard()
        self.assertIsNone(guard.before_attempt("openai", "m"))
        self.assertIsNone(guard.log_call({}))


# ------------------------------------------------------------- settings, prices
class SettingsTest(unittest.TestCase):
    def test_settings_from_the_environment(self):
        settings = StorelabSettings.from_env({"OPENAI_API_KEY": "sk-1", "GEMINI_API_KEY": "g-1",
                                              "STORELAB_ALLOWED_MODELS": "gpt-4o-2024-08-06, gemini-2.0-flash-001",
                                              "STORELAB_BREAKER_FAILURE_THRESHOLD": "7"})
        self.assertEqual((settings.api_key("openai"), settings.api_key("google"), settings.api_key("anthropic")),
                         ("sk-1", "g-1", None))
        self.assertEqual(settings.allowed_models, ("gpt-4o-2024-08-06", "gemini-2.0-flash-001"))
        self.assertEqual(settings.breaker_failure_threshold, 7)
        self.assertNotIn("sk-1", repr(settings))
        self.assertEqual(StorelabSettings.from_env({}).allowed_models, tuple(sorted(MODEL_PRICES)))

    def test_a_model_must_be_pinned_allowed_and_priced(self):
        env = Env()
        self.addCleanup(env.cleanup)
        config = env.config(**{"agents.models": MODELS})
        self.assertEqual(model_problems(config, StorelabSettings()), [])
        unpriced = env.config(**{"agents.models": [{"key": "gpt_4o_20241120", "provider": "openai",
                                                    "model": "gpt-4o-2024-11-20"}]})
        self.assertEqual(model_problems(unpriced, StorelabSettings()), [
            "gpt_4o_20241120: gpt-4o-2024-11-20 is not in StorelabSettings.allowed_models",
            "gpt_4o_20241120: gpt-4o-2024-11-20 has no price row (pricing.py)"])
        with self.assertRaisesRegex(PlanError, "has no price row"):
            api.plan_run(unpriced, snapshot=env.snapshot, root=env.runs, environ=KEYS)
        with self.assertRaisesRegex(PlanError, "gemini-2.0-flash-001 is not in StorelabSettings.allowed_models"):
            api.plan_run(config, snapshot=env.snapshot, root=env.runs,
                         environ={**KEYS, "STORELAB_ALLOWED_MODELS": "gpt-4o-2024-08-06,gpt-4.1-2025-04-14,"
                                                                     "claude-sonnet-4-5-20250929"})
        self.assertFalse(env.runs.exists() and any(env.runs.iterdir()))

    def test_prices(self):
        from features.storelab.core.protocol import Usage
        self.assertAlmostEqual(call_cost_usd("claude-sonnet-4-20250514", Usage(input_tokens=1_000_000,
                                                                               output_tokens=100_000)), 4.5)
        self.assertEqual(call_cost_usd("gpt-4o-2024-08-06", Usage()), 0.0)
        with self.assertRaises(UnknownModelError):
            call_cost_usd("gpt-4o", Usage(input_tokens=1))

    def test_adapters_need_keys(self):
        env = Env()
        self.addCleanup(env.cleanup)
        config = env.config(**{"agents.models": MODELS})
        with self.assertRaisesRegex(ValueError, "no API key for anthropic"):
            provider_module.adapters_for(config, StorelabSettings.from_env({"OPENAI_API_KEY": "k"}))


# ------------------------------------------------------------- live smoke (AC-5)
@unittest.skipUnless(HAVE_SDKS, "the pinned provider SDKs are not installed")
class LiveSmokeTest(unittest.TestCase):
    """AC-5, with the providers served in process: the report lists returned id, access
    date and cost for every configured model. The live run itself needs keys."""

    def test_report_lists_returned_id_access_date_and_cost_per_model(self):
        env = Env()
        self.addCleanup(env.cleanup)
        config = live_smoke.smoke_config(live_smoke.Path(api.__file__).parent / "configs" / "e1_pilot.yaml",
                                         {"environment.dataset": {"name": "amazon_reviews_2023",
                                                                  "sha256": env.snapshot.name}})
        self.assertEqual((config.repetitions, config.interfaces, config.cache.policy, config.mode),
                         (1, ["text"], "off", "sandbox"))
        bounds = live_smoke.worst_case_usd(config, env.snapshot)
        self.assertEqual(set(bounds), {m.key for m in config.agents.models})
        self.assertTrue(all(0 < b["usd"] <= 1.0 for b in bounds.values()))
        servers = {p: Server(p) for p in ("openai", "anthropic", "google")}
        run_dir, result = api.run(config, snapshot=env.snapshot, root=env.runs, environ=KEYS,
                                  estimate_cost=lambda c, t, m: 1.0, cache_root=env.cache_root,
                                  http_clients={p: s.client() for p, s in servers.items()})
        self.assertEqual(result.status, "COMPLETED")
        data = live_smoke.report(run_dir, bounds)
        self.assertEqual(len(data["models"]), len(config.agents.models))
        for m in data["models"]:
            self.assertEqual(m["returned_ids"], [m["requested_model"]])
            self.assertRegex(m["access_date_utc"], r"^\d{4}-\d{2}-\d{2}$")
            self.assertEqual(m["calls"], bounds[m["key"]]["trials"])
            self.assertGreater(m["cost_usd"], 0.0)
            self.assertLessEqual(m["cost_usd"], m["bound_usd"])
        text = live_smoke.markdown(data)
        for m in data["models"]:
            self.assertIn("| %s | %s | %s | %s |" % (m["key"], m["requested_model"], m["requested_model"],
                                                     m["access_date_utc"]), text)
        self.assertEqual(data["anthropic_api_version"], "2023-06-01")

    def test_the_script_calls_nothing_without_live_or_keys(self):
        env = Env()
        self.addCleanup(env.cleanup)
        args = ["features/storelab/configs/e1_pilot.yaml", "--snapshot", str(env.snapshot), "--set",
                "environment.dataset={name: amazon_reviews_2023, sha256: %s}" % env.snapshot.name]
        args[0] = str(Path(api.__file__).parent / "configs" / "e1_pilot.yaml")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(live_smoke.main(args), 0)
        self.assertIn("dry run", err.getvalue())
        self.assertIn("bound_usd", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(live_smoke.main(args + ["--max-usd-per-model", "0.000001"]), 2)
        self.assertIn("cost bound above", err.getvalue())


if __name__ == "__main__":
    unittest.main()
