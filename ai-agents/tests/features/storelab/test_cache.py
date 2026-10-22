"""SL-08 response cache and replay: replay fidelity with zero calls (AC-1), cold-cache
call counts (AC-2), read_only misses (AC-3), verify / purge / export with audit
(AC-4), the S3 store, and a short local_llm run through the cache.

Uses a counting mock provider; writes only to temporary directories.
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import io
import json
import re
import shutil
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path

from features.storelab import api, cli
from features.storelab.backends.llm import LLMBackend, OllamaProvider, build_request, call_seed
from features.storelab.cache_s3 import S3CacheStore
from features.storelab.config_files import read_config_data
from features.storelab.core.cache import (Filters, LLMRequest, LocalAuditLog, LocalCacheStore, Message,
                                          ProviderResponse, ResponseCache, TamperedEntry, call_key, new_entry)
from features.storelab.core.config import apply_overrides, resolve_config
from features.storelab.core.protocol import Completion, Select, Usage
from features.storelab.core.runner import (RUN_IDENTITY_FIELDS, VOLATILE_FIELDS, PlanError, comparable,
                                           record_set_hash)
from features.storelab.core.store import LocalStore
from tests.features.storelab.test_catalog_amazon import fixture_snapshot

CONFIG = Path(cli.__file__).resolve().parent / "configs" / "e1_pilot.yaml"
GPT = {"key": "gpt_4o_20240806", "provider": "openai", "model": "gpt-4o-2024-08-06", "temperatures": [None]}
ID = re.compile(r'ID: (\S+) \||"id": "([^"]+)"')


class CountingMockProvider:
    """Answers from the prompt's candidate ids, deterministically per call_key; counts calls."""

    name, adapter_version = "openai", "mock@1"

    def __init__(self):
        self.calls, self.keys, self.lock = 0, [], threading.Lock()

    def complete(self, request: LLMRequest) -> ProviderResponse:
        key = call_key(request)
        with self.lock:
            self.calls += 1
            self.keys.append(key)
        ids = [a or b for a, b in ID.findall(request.messages[-1].content)]
        h = int(key[:8], 16)
        if h % 10 == 0:
            text = '{"choice": null, "reason": "nothing fits"}'
        elif h % 10 == 1:
            text = "I would pick the second one."
        else:
            text = '{"choice": "%s", "reason": "good value"}' % ids[h % len(ids)]
        return ProviderResponse(text=text, attempts=1 + h % 2, latency_s=0.001 * (h % 50),
                                returned_model=request.requested_model, raw={"id": key[:12]},
                                usage=Usage(input_tokens=len(request.messages[-1].content) // 4, output_tokens=12))


class Env:
    """Fixture snapshot, run root and cache root for one test."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp())
        self.snapshot = fixture_snapshot(self.root)
        self.cache_root = self.root / "cache"
        self.runs = self.root / "runs"

    def config(self, **overrides):
        base = {"environment.dataset": {"name": "amazon_reviews_2023", "sha256": self.snapshot.name},
                "environment.categories": ["Home_and_Kitchen"], "environment.tasks_per_category": 1,
                "design.eligible_size": 4, "design.set_sizes": [2], "repetitions": 10, "interfaces": ["text"],
                "protocol.prompt_templates": {"text": "e1_text@1"}, "agents.models": [GPT], "mode": "sandbox",
                "budget.max_concurrency": 4}
        return resolve_config(apply_overrides(read_config_data(CONFIG), {**base, **overrides}))

    def run(self, config, provider, cache_root=None):
        run_dir, result = api.run(config, snapshot=self.snapshot, root=self.runs, environ={"OPENAI_API_KEY": "test"},
                                  estimate_cost=lambda c, t, m: 0.0, providers={"openai": provider},
                                  cache_root=cache_root or self.cache_root)
        return run_dir, result

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


def entries(cache_root: Path) -> int:
    return sum(1 for _ in LocalCacheStore(cache_root).paths())


class CacheTestCase(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        self.addCleanup(self.env.cleanup)


# --------------------------------------------------------------------- call key
class CallKeyTest(CacheTestCase):
    def request(self, **changes):
        base = dict(provider="openai", requested_model="gpt-4o-2024-08-06",
                    params={"temperature": None, "max_output_tokens": 256}, prompt_template_sha="a" * 64,
                    messages=[Message(role="user", content="prompt")], rep=0, sample_namespace="ns")
        return LLMRequest(**{**base, **changes})

    def test_every_input_is_in_the_key(self):
        base = call_key(self.request())
        self.assertEqual(call_key(self.request()), base)
        for change in ({"rep": 1}, {"sample_namespace": "other"}, {"params": {"temperature": 0.0,
                                                                              "max_output_tokens": 256}},
                       {"messages": [Message(role="user", content="prompt!")]}, {"provider": "google"},
                       {"requested_model": "gpt-4o-2024-11-20"}, {"prompt_template_sha": "b" * 64},
                       {"tool_schema_version": "e1_tools@1"}):
            with self.subTest(change=change):
                self.assertNotEqual(call_key(self.request(**change)), base)

    def test_requests_and_seeds_do_not_depend_on_the_config_hash(self):
        config = self.env.config()
        other = resolve_config(apply_overrides(config.model_dump(mode="json"), {
            "mode": "replay", "source_run_id": "1b4e28ba-2fa1-4d2e-883f-0016d3cca427", "cache.policy": "read_only"}))
        self.assertNotEqual(config.config_hash, other.config_hash)
        stimuli = api.build_stimuli(config, self.env.snapshot)
        from features.storelab.core.runner import plan_trials
        from features.storelab.core.render import build_observation
        a, b = plan_trials(config, stimuli)[5], plan_trials(other, stimuli)[5]
        self.assertNotEqual(a.trial_key, b.trial_key)
        self.assertEqual(call_seed(config, a), call_seed(other, b))
        spec = {t.task.task_id: t.task for t in stimuli.tasks}[a.task_id]
        design = next(d for d in stimuli.designs if any(s.set_id == a.set_id for s in d.sets))
        obs = build_observation(config, spec, design.presentation_for(a.set_id, a.rep), stimuli.products)
        self.assertEqual(call_key(build_request(config, GPT["key"], a, obs)),
                         call_key(build_request(other, GPT["key"], b, obs)))
        self.assertIn("seed", build_request(config, GPT["key"], a, obs).params)


# -------------------------------------------------------------------- replay
class ReplayTest(CacheTestCase):
    """AC-1: replaying a completed mock-provider run makes 0 provider calls and
    reproduces the record-set hash; volatile fields are enumerated and excluded."""

    def test_replay_makes_no_calls_and_reproduces_the_hash(self):
        mock = CountingMockProvider()
        source_dir, result = self.env.run(self.env.config(), mock)
        self.assertIn(result.status, ("COMPLETED", "COMPLETED_WITH_GAPS"))  # the mock answers some invalid
        calls = mock.calls
        self.assertEqual(calls, 60)
        replay_dir, replayed = api.replay(source_dir, cache_root=self.env.cache_root, providers={"openai": mock},
                                          environ={})
        self.assertEqual(mock.calls, calls)  # zero provider calls
        source, replay = LocalStore(source_dir), LocalStore(replay_dir)
        self.assertEqual(record_set_hash(replay.records()), record_set_hash(source.records()))
        self.assertEqual(api.status(replay_dir)["record_set_hash"], api.status(source_dir)["record_set_hash"])
        manifest = replay.read_manifest()
        self.assertEqual((manifest.mode.requested, manifest.mode.resolved), ("replay", "replay"))
        self.assertEqual(manifest.config.source_run_id, source.read_manifest().run_id)  # the parent link
        self.assertIn(source.read_manifest().run_id, manifest.mode.reason)
        self.assertEqual(manifest.config.cache.policy, "read_only")
        self.assertEqual(replayed.status, result.status)
        mine, theirs = replay.records(), source.records()
        self.assertTrue(set(mine).isdisjoint(theirs))  # other config_hash, other trial keys
        by_coordinates = {(r.set_id, r.rep): r for r in theirs.values()}
        for record in mine.values():
            original = by_coordinates[record.set_id, record.rep]
            self.assertEqual(record.outcome.raw_ref, original.outcome.raw_ref)
            self.assertEqual(comparable(record), comparable(original))

    def test_volatile_and_identity_fields_are_enumerated_and_only_they_are_excluded(self):
        self.assertEqual(VOLATILE_FIELDS, ("started_utc", "finished_utc", "outcome.latency_s"))
        self.assertEqual(RUN_IDENTITY_FIELDS, ("run_id", "config_hash", "trial_key"))
        run_dir, _ = self.env.run(self.env.config(repetitions=1), CountingMockProvider())
        records = LocalStore(run_dir).records()
        key, record = next(iter(records.items()))
        base = record_set_hash(records)
        later = datetime.datetime(2030, 1, 1, tzinfo=datetime.timezone.utc)
        same = record.model_copy(update={"run_id": "6f1a2c3d-4b5e-4f60-9a1b-2c3d4e5f6a7b", "config_hash": "f" * 64,
                                         "trial_key": "e" * 64, "started_utc": later, "finished_utc": later,
                                         "outcome": record.outcome.model_copy(update={"latency_s": 99.0})})
        self.assertEqual(record_set_hash({**records, key: same}), base)
        for change in ({"attempts": record.outcome.attempts + 1}, {"returned_model": "gpt-4o-2024-11-20"},
                       {"usage": Usage(input_tokens=1)}, {"raw_ref": "cache:" + "0" * 64}):
            with self.subTest(change=change):
                altered = record.model_copy(update={"outcome": record.outcome.model_copy(update=change)})
                self.assertNotEqual(record_set_hash({**records, key: altered}), base)

    def test_a_replay_config_must_match_its_source(self):
        mock = CountingMockProvider()
        source_dir, _ = self.env.run(self.env.config(repetitions=1), mock)
        data = api.replay_config(source_dir).model_dump(mode="json")
        data["repetitions"] = 2
        with self.assertRaisesRegex(PlanError, "differs from source run .* in \\['repetitions'\\]"):
            api.plan_run(resolve_config(data), root=self.env.runs)
        data["repetitions"], data["source_run_id"] = 1, "1b4e28ba-2fa1-4d2e-883f-0016d3cca427"
        with self.assertRaisesRegex(PlanError, "not found"):
            api.plan_run(resolve_config(data), root=self.env.runs)

    def test_cli_replay(self):
        mock = CountingMockProvider()
        source_dir, _ = self.env.run(self.env.config(repetitions=2), mock)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["replay", str(source_dir), "--cache-root", str(self.env.cache_root)])
        summary = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(summary["record_set_hash"], api.status(source_dir)["record_set_hash"])
        self.assertEqual(summary["mode"]["resolved"], "replay")


# ------------------------------------------------------------- cold cache
class ColdCacheTest(CacheTestCase):
    """AC-2: on a cold cache a run with r = 10 makes exactly 10 × presentations calls."""

    def test_ten_repetitions_are_ten_calls_per_presentation(self):
        mock = CountingMockProvider()
        run_dir, _ = self.env.run(self.env.config(), mock)
        trials, _ = api.load_plan(LocalStore(run_dir))
        presentations = {t.presentation_id for t in trials}
        self.assertEqual(len(presentations), 6)  # per_set: one presentation per set, shown r times
        self.assertEqual(mock.calls, 10 * len(presentations))
        self.assertEqual(len(set(mock.keys)), mock.calls)  # repetitions are distinct samples
        self.assertEqual(entries(self.env.cache_root), mock.calls)

    def test_calls_scale_with_interfaces_and_temperatures(self):
        mock = CountingMockProvider()
        config = self.env.config(**{"interfaces": ["text", "structured"],
                                    "protocol.prompt_templates": {"text": "e1_text@1",
                                                                  "structured": "e1_structured@1"},
                                    "agents.models": [{**GPT, "temperatures": [None, 0.0]}]})
        self.env.run(config, mock)
        self.assertEqual(mock.calls, 10 * 6 * 2 * 2)

    def test_per_repetition_presentations(self):
        mock = CountingMockProvider()
        run_dir, _ = self.env.run(self.env.config(**{"design.randomisation": "per_repetition"}), mock)
        trials, _ = api.load_plan(LocalStore(run_dir))
        self.assertEqual(len({t.presentation_id for t in trials}), 60)
        self.assertEqual(mock.calls, 60)

    def test_policies(self):
        config = self.env.config()
        first = CountingMockProvider()
        first_dir, _ = self.env.run(config, first)
        warm = CountingMockProvider()
        warm_dir, _ = self.env.run(config, warm)  # read_write, warm
        self.assertEqual(warm.calls, 0)
        self.assertEqual(api.status(warm_dir)["record_set_hash"], api.status(first_dir)["record_set_hash"])
        before = {e.call_key: e.created_utc for e in api.open_cache(self.env.cache_root).entries()}
        fresh = CountingMockProvider()
        self.env.run(self.env.config(**{"cache.policy": "write_only"}), fresh)
        self.assertEqual(fresh.calls, 60)
        after = {e.call_key: e.created_utc for e in api.open_cache(self.env.cache_root).entries()}
        self.assertEqual(set(after), set(before))
        self.assertTrue(all(after[k] > before[k] for k in before))  # replaced by the fresh samples
        off = CountingMockProvider()
        empty = self.env.root / "cache-off"
        off_dir, _ = self.env.run(self.env.config(**{"cache.policy": "off"}), off, cache_root=empty)
        self.assertEqual(off.calls, 60)
        self.assertEqual(entries(empty), 0)
        self.assertTrue(all(r.outcome.raw_ref is None for r in LocalStore(off_dir).records().values()))


# ------------------------------------------------------------- read_only miss
class ReadOnlyMissTest(CacheTestCase):
    """AC-3: a read_only miss yields error/cache_miss and zero live calls."""

    def test_replay_against_an_empty_cache(self):
        mock = CountingMockProvider()
        source_dir, _ = self.env.run(self.env.config(), mock)
        calls = mock.calls
        empty = self.env.root / "empty-cache"
        replay_dir, result = api.replay(source_dir, cache_root=empty, providers={"openai": mock}, environ={})
        self.assertEqual(mock.calls, calls)
        records = LocalStore(replay_dir).records()
        self.assertEqual({(r.outcome.status, r.outcome.error_code, r.outcome.attempts) for r in records.values()},
                         {("error", "cache_miss", 0)})
        self.assertEqual(len(records), 60)
        self.assertEqual(result.status, "COMPLETED_WITH_GAPS")
        self.assertIn("60 replay cache misses", result.reason)

    def test_partial_misses(self):
        mock = CountingMockProvider()
        source_dir, _ = self.env.run(self.env.config(), mock)
        store = LocalCacheStore(self.env.cache_root)
        removed = list(store.paths())[:7]
        for path in removed:
            store.delete(path)
        calls = mock.calls
        replay_dir, _ = api.replay(source_dir, cache_root=self.env.cache_root, providers={"openai": mock}, environ={})
        self.assertEqual(mock.calls, calls)
        statuses = [r.outcome.error_code for r in LocalStore(replay_dir).records().values()]
        self.assertEqual(statuses.count("cache_miss"), 7)

    def test_the_backend_never_calls_under_read_only(self):
        config = self.env.config()
        backend = LLMBackend(config, GPT["key"], provider=None, cache=api.open_cache(self.env.root / "x"),
                             policy="read_only")
        stimuli = api.build_stimuli(config, self.env.snapshot)
        from features.storelab.core.render import build_observation
        from features.storelab.core.runner import plan_trials
        trial = plan_trials(config, stimuli)[0]
        design = stimuli.designs[0]
        obs = build_observation(config, stimuli.tasks[0].task, design.presentation_for(trial.set_id, 0),
                                stimuli.products)
        outcome = backend.decide(trial, obs)
        self.assertEqual((outcome.status, outcome.error_code, outcome.attempts), ("error", "cache_miss", 0))
        with self.assertRaisesRegex(ValueError, "needs a provider"):
            LLMBackend(config, GPT["key"], provider=None, cache=api.open_cache(self.env.root / "x"),
                       policy="read_write")


# ---------------------------------------------------------------- management
class ManagementTest(CacheTestCase):
    """AC-4: verify detects a tampered entry; purge and export write an audit entry."""

    def populate(self):
        mock = CountingMockProvider()
        run_dir, _ = self.env.run(self.env.config(repetitions=2), mock)
        return run_dir, api.open_cache(self.env.cache_root)

    def edit(self, path: Path, change) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        change(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_verify_detects_tampering(self):
        _, cache = self.populate()
        self.assertEqual(cache.verify(), {})
        paths = sorted(cache.store.paths())
        root = self.env.cache_root
        self.edit(root / paths[0], lambda d: d["response"].update(text='{"choice": null, "reason": "edited"}'))
        self.edit(root / paths[1], lambda d: d["request"]["messages"][0].update(content="edited prompt"))
        (root / paths[2]).write_text("{not json", encoding="utf-8")
        moved = root / "openai" / "gpt-4o-2024-08-06" / ("f" * 64 + ".json")
        shutil.copyfile(root / paths[3], moved)
        bad = cache.verify()
        self.assertEqual(set(bad), {paths[0], paths[1], paths[2], moved.relative_to(root).as_posix()})
        self.assertIn("content_sha256 does not match the stored content", bad[paths[0]])
        self.assertIn("call_key does not match the stored request", bad[paths[1]])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["cache", "verify", "--cache-root", str(root)]), 1)
        self.assertEqual(len(json.loads(out.getvalue())["tampered"]), 4)

    def test_a_tampered_entry_fails_the_run_instead_of_being_served(self):
        _, cache = self.populate()
        path = sorted(cache.store.paths())[0]
        self.edit(self.env.cache_root / path, lambda d: d["response"].update(text='{"choice": null, "reason": "x"}'))
        key = Path(path).stem
        with self.assertRaises(TamperedEntry):
            cache.get("openai", "gpt-4o-2024-08-06", key)
        mock = CountingMockProvider()
        _, result = self.env.run(self.env.config(repetitions=2), mock)
        self.assertEqual(result.status, "FAILED")
        self.assertIn("TamperedEntry", result.reason)

    def test_purge_needs_a_reason_and_a_filter_and_is_audited(self):
        _, cache = self.populate()
        total = sum(1 for _ in cache.store.paths())
        with self.assertRaisesRegex(ValueError, "reason"):
            cache.purge(Filters(namespace="e1_pilot"), "")
        with self.assertRaisesRegex(ValueError, "filter"):
            cache.purge(Filters(), "cleanup")
        self.assertEqual(cache.purge(Filters(namespace="other"), "nothing matches")["entries"], 0)
        event = cache.purge(Filters(namespace="e1_pilot", model="gpt-4o-2024-08-06"), "re-run with a new prompt")
        self.assertEqual(event["entries"], total)
        self.assertEqual(sum(1 for _ in cache.store.paths()), 0)
        events = LocalAuditLog(self.env.cache_root / "audit.jsonl").events()
        self.assertEqual([e["event"] for e in events], ["cache_purge", "cache_purge"])
        self.assertEqual(events[-1]["reason"], "re-run with a new prompt")
        self.assertEqual(events[-1]["filters"]["namespace"], "e1_pilot")
        self.assertTrue(events[-1]["user"])

    def test_export_writes_a_tarball_and_is_audited(self):
        _, cache = self.populate()
        target = self.env.root / "release" / "cache.tar.gz"
        with self.assertRaisesRegex(ValueError, "reason"):
            cache.export(Filters(), " ", target)
        event = cache.export(Filters(provider="openai"), "release v2", target)
        with tarfile.open(target) as tar:
            names = tar.getnames()
        self.assertEqual(len(names), event["entries"])
        self.assertTrue(all(n.startswith("cache/openai/gpt-4o-2024-08-06/") for n in names))
        self.assertEqual(event["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertEqual(LocalAuditLog(self.env.cache_root / "audit.jsonl").events()[-1]["event"], "cache_export")

    def test_cli_list_inspect_purge_export(self):
        _, cache = self.populate()
        root = str(self.env.cache_root)

        def run(*argv):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cli.main(list(argv))
            return code, out.getvalue(), err.getvalue()

        code, out, _ = run("cache", "list", "--cache-root", root, "--namespace", "e1_pilot")
        listed = json.loads(out)
        self.assertEqual(len(listed), 12)
        code, out, _ = run("cache", "inspect", listed[0]["call_key"], "--cache-root", root)
        self.assertEqual(json.loads(out)["call_key"], listed[0]["call_key"])
        code, _, err = run("cache", "purge", "--cache-root", root, "--namespace", "e1_pilot")
        self.assertEqual(code, 2)
        self.assertIn("reason", err)
        code, out, _ = run("cache", "export", "--cache-root", root, "--reason", "share", "--out",
                           str(self.env.root / "x.tar.gz"))
        self.assertEqual((code, json.loads(out)["entries"]), (0, 12))
        tomorrow = (datetime.date.today() + datetime.timedelta(days=2)).isoformat()
        code, out, _ = run("cache", "purge", "--cache-root", root, "--before", tomorrow, "--reason", "expire")
        self.assertEqual((code, json.loads(out)["entries"]), (0, 12))
        events = [e["event"] for e in LocalAuditLog(self.env.cache_root / "audit.jsonl").events()]
        self.assertEqual(events, ["cache_export", "cache_purge"])


# ---------------------------------------------------------------- S3 store
class FakeS3:
    class NoSuchKey(Exception):
        response = {"Error": {"Code": "NoSuchKey"}}

    def __init__(self, page_size=3):
        self.objects, self.page_size = {}, page_size

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + self.page_size]
        more = start + self.page_size < len(keys)
        return {"Contents": [{"Key": k} for k in page], "IsTruncated": more,
                **({"NextContinuationToken": str(start + self.page_size)} if more else {})}


class S3StoreTest(unittest.TestCase):
    def test_entries_round_trip_under_the_prefix(self):
        client = FakeS3()
        cache = ResponseCache(S3CacheStore("bucket", client=client))
        made = []
        for rep in range(7):
            request = LLMRequest(provider="openai", requested_model="gpt-4o-2024-08-06", params={},
                                 prompt_template_sha="a" * 64, messages=[Message(role="user", content="p")], rep=rep,
                                 sample_namespace="ns")
            entry = new_entry(request, ProviderResponse(text='{"choice": "B1", "reason": "x"}'),
                              Select(product_id="B1"), "mock@1")
            cache.put(entry)
            made.append(entry)
        self.assertTrue(all(k.startswith("storelab/cache/openai/gpt-4o-2024-08-06/") for k in client.objects))
        self.assertEqual(len(list(cache.entries())), 7)  # across three pages
        self.assertEqual(cache.get("openai", "gpt-4o-2024-08-06", made[3].call_key), made[3])
        self.assertIsNone(cache.get("openai", "gpt-4o-2024-08-06", "0" * 64))
        self.assertEqual(cache.verify(), {})
        key = "storelab/cache/" + made[0].path
        client.objects[key] = client.objects[key].replace(b'"B1"', b'"B2"', 1)
        self.assertEqual(list(cache.verify()), [made[0].path])
        self.assertEqual(cache.purge(Filters(provider="openai"), "test")["entries"], 7)
        self.assertEqual(client.objects, {})


# ---------------------------------------------------------- local_llm run
class FakeLocalLLM:
    calls = 0

    def __init__(self, model, **kwargs):
        self.model, self.kwargs = model, kwargs

    def complete(self, prompt, seed=None):
        FakeLocalLLM.calls += 1
        ids = [a or b for a, b in ID.findall(prompt)]
        return Completion(text='{"choice": "%s", "reason": "seed %d"}' % (ids[(seed or 0) % len(ids)], seed or 0),
                          attempts=1, latency_s=0.01, returned_model=self.model)


class LocalLLMRunTest(CacheTestCase):
    """A short local_llm run through the cache (ollama replaced by a fake client)."""

    def test_local_llm_run_is_cached_and_replayable(self):
        config = self.env.config(**{"mode": "offline", "agents.offline_backend": "local_llm", "repetitions": 2,
                                    "agents.models": [{"key": "qwen2_7b", "provider": "ollama", "model": "qwen2:7b",
                                                       "temperatures": [0.7]}]})
        provider = OllamaProvider(config, client_factory=FakeLocalLLM)
        FakeLocalLLM.calls = 0
        run = lambda: api.run(config, snapshot=self.env.snapshot, root=self.env.runs,  # noqa: E731
                              providers={"ollama": provider}, cache_root=self.env.cache_root, environ={})
        first_dir, first = run()
        self.assertEqual((first.status, FakeLocalLLM.calls), ("COMPLETED", 12))
        records = LocalStore(first_dir).records().values()
        self.assertEqual({r.outcome.backend for r in records}, {"local_llm"})
        entry = next(api.open_cache(self.env.cache_root).entries())
        self.assertEqual((entry.request.provider, entry.adapter_version), ("ollama", "ollama-generate@1"))
        self.assertIn("seed", entry.request.params)
        second_dir, _ = run()
        self.assertEqual(FakeLocalLLM.calls, 12)  # served from the cache
        replay_dir, _ = api.replay(first_dir, cache_root=self.env.cache_root, environ={})
        self.assertEqual(FakeLocalLLM.calls, 12)
        hashes = {api.status(d)["record_set_hash"] for d in (first_dir, second_dir, replay_dir)}
        self.assertEqual(len(hashes), 1)


if __name__ == "__main__":
    unittest.main()
