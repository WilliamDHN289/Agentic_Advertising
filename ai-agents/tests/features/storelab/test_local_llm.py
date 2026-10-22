"""SL-04 local LLM backend (backends/local_llm.py) and its manifest hooks.

The closed-port tests need no server. The other tests start a stdlib fake ollama on
127.0.0.1 and skip, with that reason, where binding a socket is not allowed.
"""
from __future__ import annotations

import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from features.storelab.backends.local_llm import LocalLLM
from features.storelab.config_files import load_config
from features.storelab.core.legacy_env import LegacyRun, episode_plan, run_episodes
from features.storelab.core.manifest import new_manifest, record_model_digest

CONFIGS = Path(__file__).resolve().parents[3] / "features" / "storelab" / "configs"
LEGACY_CONFIG = load_config(CONFIGS / "legacy_2026_08.yaml")


def closed_port() -> int:
    """A localhost port with nothing listening (a stopped ollama)."""
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]
    except OSError:
        return 1  # binding is not allowed here; port 1 has no listener either


def client(port: int, **kwargs) -> LocalLLM:
    kwargs.setdefault("max_attempts", 3)
    return LocalLLM("qwen2:7b", base_url="http://127.0.0.1:%d" % port, backoff_s=0.0, timeout_s=5.0, **kwargs)


class StoppedOllamaTest(unittest.TestCase):
    """AC-4: with ollama stopped, trials are error/provider_unavailable and none is a choice."""

    def test_calls_fail_with_provider_unavailable_after_every_attempt(self):
        waits = []
        llm = client(closed_port(), sleep=waits.append)
        reply = llm.complete("Pick one.", seed=7)
        self.assertEqual((reply.text, reply.error_code, reply.attempts), (None, "provider_unavailable", 3))
        self.assertEqual(waits, [0.0, 0.0])
        outcome = llm.decide("Pick one.", ["A1", "B2"], seed=7)
        self.assertEqual((outcome.status, outcome.error_code, outcome.chosen_id), ("error", "provider_unavailable", None))
        self.assertEqual((outcome.backend, outcome.requested_model, outcome.attempts), ("local_llm", "qwen2:7b", 3))

    def test_the_legacy_run_records_errors_and_no_choice(self):
        llm = client(closed_port(), max_attempts=2)
        for fallback in ("first_item", "none"):
            with self.subTest(fallback=fallback):
                config = LEGACY_CONFIG.model_copy(update={"protocol": LEGACY_CONFIG.protocol.model_copy(
                    update={"legacy_parse_fallback": fallback})})
                specs = [s for s in episode_plan(config.environment.legacy) if s.arm != "human_sim"]
                records = run_episodes(LegacyRun.from_config(config, llm.complete), specs)
                outcomes = [r["outcome"] for r in records.values()]
                self.assertEqual(len(outcomes), 430)
                self.assertEqual({(o["status"], o["error_code"]) for o in outcomes}, {("error", "provider_unavailable")})
                self.assertFalse(any(o["substituted"] or o["chosen_sku"] for o in outcomes))
                kinds = {e["kind"] for r in records.values() for e in r["events"]}
                self.assertFalse(kinds & {"click", "cart", "purchase"})
                self.assertTrue(all(r["row"]["basket_size"] == 0 for r in records.values()))


class FakeOllama:
    """A minimal ollama: scripted responses per path; records each request body."""

    def __init__(self, responses: dict):
        self.responses, self.requests = responses, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self):
                length = int(self.headers.get("Content-Length") or 0)
                outer.requests.append((self.path, json.loads(self.rfile.read(length) or b"null")))
                queue = outer.responses[self.path]
                status, body = queue.pop(0) if len(queue) > 1 else queue[0]
                data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _reply

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class FakeServerTest(unittest.TestCase):
    def serve(self, responses: dict) -> FakeOllama:
        try:
            fake = FakeOllama(responses)
        except OSError as exc:
            self.skipTest("cannot bind a local socket: %s" % exc)
        self.addCleanup(fake.close)
        return fake

    def test_a_reply_is_parsed_strictly_and_the_options_are_sent(self):
        reply = {"model": "qwen2:7b", "response": '{"choice": "B2", "reason": "cheapest"}'}
        fake = self.serve({"/api/generate": [(200, reply)]})
        llm = client(fake.port, temperature=0.7, top_p=0.9, num_predict=120)
        outcome = llm.decide("Pick one.", ["A1", "B2"], seed=42)
        self.assertEqual((outcome.status, outcome.chosen_id, outcome.returned_model), ("choice", "B2", "qwen2:7b"))
        self.assertEqual(fake.requests[0][1], {"model": "qwen2:7b", "prompt": "Pick one.", "stream": False,
                                               "options": {"num_predict": 120, "temperature": 0.7, "top_p": 0.9,
                                                           "seed": 42}})
        llm_default = client(fake.port)
        llm_default.complete("x")
        self.assertEqual(fake.requests[1][1]["options"], {"num_predict": 256})  # null temperature: omitted

    def test_invalid_and_outside_replies(self):
        fake = self.serve({"/api/generate": [(200, {"response": "I pick B2."}), (200, {"response": '{"choice": null, "reason": "none fit"}'})]})
        llm = client(fake.port)
        self.assertEqual((o := llm.decide("x", ["A1"])).status, "invalid")
        self.assertEqual(o.invalid_reason, "no_json")
        self.assertEqual(llm.decide("x", ["A1"]).status, "outside")

    def test_http_errors_map_to_section_3_5_codes(self):
        cases = [(404, "model_unavailable", 1), (400, "bad_request", 1), (403, "auth", 1),
                 (500, "provider_unavailable", 3), (429, "rate_limited", 3)]
        for status, code, attempts in cases:
            with self.subTest(status=status):
                fake = self.serve({"/api/generate": [(status, {"error": "x"})]})
                reply = client(fake.port).complete("x")
                self.assertEqual((reply.error_code, reply.attempts, len(fake.requests)), (code, attempts, attempts))

    def test_malformed_responses_are_provider_unavailable(self):
        for body in ("not json", {"no_response": 1}, {"response": 3}):
            with self.subTest(body=body):
                fake = self.serve({"/api/generate": [(200, body)]})
                self.assertEqual(client(fake.port, max_attempts=2).complete("x").error_code, "provider_unavailable")

    def test_a_transient_failure_is_retried(self):
        fake = self.serve({"/api/generate": [(503, {}), (200, {"response": '{"choice": "A1", "reason": "x"}'})]})
        outcome = client(fake.port).decide("x", ["A1"])
        self.assertEqual((outcome.status, outcome.attempts), ("choice", 2))

    def test_model_digest_from_show_or_from_the_tags_listing(self):
        digest = "sha256:" + "a" * 64
        fake = self.serve({"/api/show": [(200, {"digest": digest})]})
        self.assertEqual(client(fake.port).model_digest(), digest)
        fake = self.serve({"/api/show": [(200, {"details": {}})],
                           "/api/tags": [(200, {"models": [{"name": "qwen2:7b", "digest": digest}]})]})
        self.assertEqual(client(fake.port).model_digest(), digest)
        fake = self.serve({"/api/show": [(404, {"error": "model 'qwen2:7b' not found"})]})
        self.assertIsNone(client(fake.port).model_digest())
        self.assertIsNone(client(closed_port()).model_digest())


class ManifestHookTest(unittest.TestCase):
    def manifest(self, name: str):
        config = load_config(CONFIGS / (name + ".yaml"))
        templates = {t: "text of " + t for t in config.protocol.prompt_templates.values()}
        return new_manifest(config, resolved_mode="offline", mode_reason="test", governance="local",
                            protocol_version="storelab-protocol@1", prompt_templates=templates)

    def test_the_digest_goes_into_the_manifest(self):
        manifest = self.manifest("legacy_2026_08")
        record_model_digest(manifest, "qwen2_7b", "sha256:" + "b" * 64)
        self.assertEqual(manifest.models[0].digest, "sha256:" + "b" * 64)
        with self.assertRaises(ValueError):
            record_model_digest(manifest, "qwen2_7b", "sha256:" + "c" * 64)
        with self.assertRaises(KeyError):
            record_model_digest(manifest, "nope", "x")

    def test_the_legacy_fallback_flags_the_manifest(self):
        self.assertTrue(self.manifest("legacy_2026_08").contains_substituted_outcomes)
        self.assertFalse(self.manifest("smoke_offline").contains_substituted_outcomes)
        data = self.manifest("legacy_2026_08").model_dump()
        data["contains_substituted_outcomes"] = False
        with self.assertRaisesRegex(ValueError, "contains_substituted_outcomes"):
            type(self.manifest("smoke_offline")).model_validate(data)


if __name__ == "__main__":
    unittest.main()
