"""SL-02 manifest and version-capture tests.

Read-only except the tests that write a manifest file (skipped if no temporary
directory is writable):

    cd ai-agents && python -B -m unittest discover -s tests/features/storelab -t . -v
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jsonschema
from pydantic import ValidationError

from features.storelab import cli
from features.storelab.config_files import load_config
from features.storelab.core import versioning
from features.storelab.core.manifest import (Manifest, append_status, manifest_json, new_manifest, read_manifest,
                                             record_model_access, write_manifest)

CONFIGS = Path(cli.__file__).resolve().parent / "configs"
REPO = Path(cli.__file__).resolve().parents[3]
SCHEMA = json.loads((cli.CONTRACTS / "manifest.schema.json").read_text(encoding="utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)
NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
TEMPLATES = {
    "e1_pilot": {"e1_text@1": "TEXT {items}", "e1_structured@1": "JSON {items}", "e1_browser_tool@1": "SERP {items}"},
    "e2_core": {},
    "e2_stipulated": {},
    "legacy_2026_08": {"legacy_chatbot@1": "chat", "legacy_gui_agent@1": "gui", "legacy_api@1": "api"},
    "smoke_offline": {"e1_text@1": "TEXT {items}", "e1_structured@1": "JSON {items}"},
}
RESOLVED = {"auto": "offline", "offline": "offline"}


def build(name: str) -> Manifest:
    config = load_config(CONFIGS / (name + ".yaml"))
    return new_manifest(config, resolved_mode=RESOLVED[config.mode], mode_reason="no credentials for any provider",
                        governance="local", protocol_version="storelab-protocol@1", prompt_templates=TEMPLATES[name],
                        overrides={"seed": config.seed}, now=NOW, repo_dir=REPO)


def writable_tmp(test: unittest.TestCase) -> tempfile.TemporaryDirectory:
    try:
        return tempfile.TemporaryDirectory()
    except OSError as exc:
        test.skipTest("no writable temporary directory: %s" % exc)


class SchemaContractTest(unittest.TestCase):
    def test_committed_schemas_are_current_and_valid(self):
        for name in cli.SCHEMAS:
            with self.subTest(schema=name):
                committed = (cli.CONTRACTS / name).read_text(encoding="utf-8")
                self.assertEqual(committed, cli.schema_text(name), "regenerate with: python -m features.storelab.cli schemas")
                jsonschema.Draft202012Validator.check_schema(json.loads(committed))


class WrittenManifestTest(unittest.TestCase):
    """AC-4: a written manifest validates against manifest.schema.json."""

    def test_manifest_text_validates_for_every_example_config(self):
        for name in TEMPLATES:
            with self.subTest(config=name):
                manifest = build(name)
                if manifest.models:  # e2_stipulated has stipulated agents only, no provider models
                    record_model_access(manifest, manifest.models[0].key, "returned-model-id", NOW)
                append_status(manifest, "QUEUED", NOW + timedelta(seconds=1))
                VALIDATOR.validate(json.loads(manifest_json(manifest)))

    def test_written_file_validates_and_reads_back(self):
        with writable_tmp(self) as directory:
            manifest = build("e1_pilot")
            record_model_access(manifest, "gpt_4o_20240806", "gpt-4o-2024-08-06", NOW)
            append_status(manifest, "RUNNING", NOW + timedelta(minutes=1), "started")
            path = write_manifest(manifest, Path(directory) / "manifest.json")
            VALIDATOR.validate(json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(read_manifest(path), manifest)
            write_manifest(manifest, path)  # replacing an existing file
            self.assertEqual(os.listdir(directory), ["manifest.json"])
            umask = os.umask(0)
            os.umask(umask)
            self.assertEqual(path.stat().st_mode & 0o777, 0o666 & ~umask, "permissions follow the umask")

    def test_schema_rejects_missing_and_unknown_fields(self):
        data = json.loads(manifest_json(build("e1_pilot")))
        for broken in ({k: v for k, v in data.items() if k != "config_hash"}, {**data, "colour": "blue"},
                       {**data, "config": {**data["config"], "colour": "blue"}}):
            with self.subTest(keys=sorted(set(broken) ^ set(data)) or "config.colour"):
                self.assertFalse(VALIDATOR.is_valid(broken))


class ManifestContentTest(unittest.TestCase):
    """Each item tickets.md SL-02 says the manifest records."""

    def test_records_every_required_item(self):
        manifest = build("e1_pilot")
        data = json.loads(manifest_json(manifest))
        self.assertRegex(data["run_id"], r"^[0-9a-f]{8}-[0-9a-f]{4}-4")
        self.assertEqual(data["config_hash"], manifest.config.config_hash)
        self.assertEqual(data["config"]["seed"], 2026102101)
        self.assertEqual(data["overrides"], {"seed": 2026102101})
        self.assertEqual(data["mode"], {"requested": "auto", "resolved": "offline",
                                        "reason": "no credentials for any provider"})
        self.assertRegex(data["versions"]["git"]["sha"], r"^[0-9a-f]{40}$")
        self.assertIsInstance(data["versions"]["git"]["dirty"], bool)
        self.assertEqual(data["versions"]["python"]["version"].split(".")[:2], ["3", "11"])
        for package in ("numpy", "scipy", "cvxpy", "pydantic", "openai", "anthropic", "google-genai"):
            self.assertIn(package, data["versions"]["packages"])
        self.assertEqual(data["versions"]["packages"]["pydantic"], __import__("pydantic").VERSION)
        self.assertRegex(data["versions"]["env_version"], r"^storelab-core\+[0-9a-f]{16}$")
        self.assertEqual(data["versions"]["protocol_version"], "storelab-protocol@1")
        self.assertEqual(data["prompt_templates"][0], {
            "id": "e1_browser_tool@1", "sha256": hashlib.sha256(b"SERP {items}").hexdigest()})
        self.assertEqual([m["requested_id"] for m in data["models"]],
                         [m.model for m in manifest.config.agents.models])
        self.assertEqual(data["models"][0]["temperatures"], ["provider_default", 0.0])
        self.assertEqual([m["seed_supported"] for m in data["models"]], [True, True, False, False, True, True])
        self.assertEqual(data["models"][0]["returned_ids"], [])
        self.assertIsNone(data["models"][0]["first_access_utc"])
        self.assertEqual(data["cache"], {"policy": "read_write", "namespace": "e1_pilot"})
        self.assertEqual(data["governance"], "local")
        self.assertEqual(data["status_history"], [{"status": "PLANNED", "at_utc": "2026-10-21T12:00:00Z",
                                                   "reason": "no credentials for any provider"}])

    def test_dataset_snapshot_hash_comes_from_the_config(self):
        legacy = build("legacy_2026_08")
        self.assertEqual(legacy.dataset_snapshot.sha256,
                         "9491773fa9aa2601d95f6396b39a08c6887f77a7bd053dff9e756b91872bbe07")
        self.assertEqual(build("e1_pilot").dataset_snapshot.model_dump(), {  # the SL-05 snapshot
            "name": "amazon_reviews_2023", "sha256": "d7750e73c2e13f9706db2499fe0cff0267a31ced8d04727b96dd933359d9870b"})

    def test_model_access_tracks_returned_ids_and_the_access_window(self):
        manifest = build("e1_pilot")
        later, earlier = NOW + timedelta(hours=2), NOW - timedelta(hours=1)
        for at, returned in ((NOW, "gpt-4o-2024-08-06"), (later, "gpt-4o-2024-08-06"), (earlier, "gpt-4o-2024-11-20")):
            record_model_access(manifest, "gpt_4o_20240806", returned, at)
        record = manifest.models[0]
        self.assertEqual(record.returned_ids, ["gpt-4o-2024-08-06", "gpt-4o-2024-11-20"])
        self.assertEqual((record.first_access_utc, record.last_access_utc), (earlier, later))
        with self.assertRaises(KeyError):
            record_model_access(manifest, "no_such_model", "x", NOW)
        with self.assertRaises(ValueError):
            record_model_access(manifest, "gpt_4o_20240806", "x", NOW.astimezone(timezone(timedelta(hours=2))))

    def test_a_manifest_that_disagrees_with_its_config_is_rejected(self):
        data = json.loads(manifest_json(build("e1_pilot")))
        cases = {
            "config_hash": {**data, "config_hash": "0" * 64},
            "mode.resolved": {**data, "mode": {**data["mode"], "resolved": "replay"}},
            "status_history": {**data, "status_history": [{"status": "RUNNING", "at_utc": "2026-10-21T12:00:00Z"}]},
            "prompt_templates": {**data, "prompt_templates": data["prompt_templates"][:1]},
            "cache": {**data, "cache": {"policy": "off", "namespace": "e1_pilot"}},
            "created_utc": {**data, "created_utc": "2026-10-21T14:00:00+02:00"},
        }
        for field, broken in cases.items():
            with self.subTest(field=field), self.assertRaises(ValidationError) as caught:
                Manifest.model_validate(broken)
            self.assertIn(field.split(".")[0], str(caught.exception))


class VersioningTest(unittest.TestCase):
    def test_fingerprint_normalises_line_endings_and_covers_names_and_content(self):
        base = versioning.fingerprint([("a.py", b"x = 1\n"), ("b.py", b"y = 2\n")])
        self.assertEqual(versioning.fingerprint([("b.py", b"y = 2\r\n"), ("a.py", b"x = 1\n")]), base)
        self.assertNotEqual(versioning.fingerprint([("a.py", b"x = 1\n"), ("c.py", b"y = 2\n")]), base)
        self.assertNotEqual(versioning.fingerprint([("a.py", b"x = 2\n"), ("b.py", b"y = 2\n")]), base)

    def test_env_version_fingerprints_the_core_source(self):
        files = sorted(p for p in versioning.CORE_DIR.rglob("*.py") if "__pycache__" not in p.parts)
        expected = versioning.fingerprint((p.relative_to(versioning.CORE_DIR).as_posix(), p.read_bytes()) for p in files)
        self.assertEqual(versioning.env_version(), "storelab-core+" + expected[:16])
        self.assertIn("config.py", [p.name for p in files])

    def test_git_state_outside_a_repository_is_null(self):
        self.assertEqual(versioning.git_state(Path("/")), {"sha": None, "dirty": None})

    def test_git_state_inside_the_repository(self):
        state = versioning.git_state(REPO)
        self.assertRegex(state["sha"], r"^[0-9a-f]{40}$")
        self.assertIsInstance(state["dirty"], bool)

    def test_missing_packages_are_recorded_as_null(self):
        versions = versioning.package_versions(["pydantic", "no-such-package-sl02"])
        self.assertTrue(re.match(r"^\d+\.\d+", versions["pydantic"]))
        self.assertIsNone(versions["no-such-package-sl02"])


if __name__ == "__main__":
    unittest.main()
