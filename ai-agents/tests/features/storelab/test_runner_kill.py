"""SL-07 AC-1: an offline run killed at 5 random points finishes with the same
record-set hash as an uninterrupted run.

Real processes: ``python -m features.storelab.cli resume RUN`` executes a planned run
and is killed with SIGKILL once a random number of records is on disk; the next
``resume`` continues it. Writes only to a temporary directory.
"""
from __future__ import annotations

import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from features.storelab import api, cli
from features.storelab.core.runner import record_set_hash
from features.storelab.core.store import LocalStore, gzip_members
from features.storelab.config_files import read_config_data
from features.storelab.core.config import apply_overrides, resolve_config
from tests.features.storelab.test_catalog_amazon import fixture_snapshot

AI_AGENTS = Path(__file__).resolve().parents[3]
CONFIG = Path(cli.__file__).resolve().parent / "configs" / "e1_pilot.yaml"
OVERRIDES = {"repetitions": 3, "agents.models": [
    {"key": "gpt_4o_20240806", "provider": "openai", "model": "gpt-4o-2024-08-06", "temperatures": [None]},
    {"key": "gemini_20_flash_001", "provider": "google", "model": "gemini-2.0-flash-001", "temperatures": [0.0]}]}
KILLS = 5


def records_on_disk(store: LocalStore) -> int:
    return sum(member.count(b"\n") for shard in store.shards() for member in gzip_members(shard.read_bytes()))


@unittest.skipUnless(hasattr(signal, "SIGKILL"), "needs SIGKILL")
class KillAndResumeTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.env = {k: v for k, v in os.environ.items()
                    if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")}
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"

    def resume(self, run_dir: Path) -> subprocess.Popen:
        self.log = self.root / "stderr.log"
        with open(self.log, "wb") as err:
            return subprocess.Popen([sys.executable, "-B", "-m", "features.storelab.cli", "resume", str(run_dir)],
                                    cwd=AI_AGENTS, env=self.env, stdout=subprocess.DEVNULL, stderr=err)

    def test_killed_five_times_then_resumed_gives_the_uninterrupted_hash(self):
        snapshot = fixture_snapshot(self.root)
        cfg = resolve_config(apply_overrides(read_config_data(CONFIG), {
            **OVERRIDES, "environment.dataset": {"name": "amazon_reviews_2023", "sha256": snapshot.name}}))
        reference_dir, planned = api.plan_run(cfg, snapshot=snapshot, root=self.root / "reference", environ={})
        killed_dir, _ = api.plan_run(cfg, snapshot=snapshot, root=self.root / "killed", environ={})
        total = len(planned.trials)
        self.assertEqual(total, 4 * 63 * 3 * 3 * 2)

        process = self.resume(reference_dir)
        self.assertEqual(process.wait(timeout=600), 0, self.log.read_text())
        reference = LocalStore(reference_dir)
        self.assertEqual(reference.read_manifest().status_history[-1].status, "COMPLETED")
        expected = record_set_hash(reference.records())

        store = LocalStore(killed_dir)
        rng = random.Random(707)
        targets = sorted(rng.sample(range(50, total - 800), KILLS))
        progress = []
        for target in targets:
            process = self.resume(killed_dir)
            deadline = time.monotonic() + 600
            while records_on_disk(store) < target:
                self.assertIsNone(process.poll(), "the run ended before the kill point %d" % target)
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.005)
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=60)
            self.assertEqual(process.returncode, -signal.SIGKILL)
            on_disk = records_on_disk(store)
            progress.append(on_disk)
            self.assertLess(on_disk, total)
            self.assertEqual(store.read_manifest().status_history[-1].status, "RUNNING")  # no final status
        self.assertEqual(progress, sorted(progress))

        process = self.resume(killed_dir)
        self.assertEqual(process.wait(timeout=600), 0, self.log.read_text())
        manifest = store.read_manifest()
        statuses = [e.status for e in manifest.status_history]
        self.assertEqual(statuses[-1], "COMPLETED")
        self.assertEqual(statuses.count("STALLED"), KILLS)
        records = store.records()
        self.assertEqual(len(records), total)
        self.assertEqual(sum(1 for _ in store.iter_records()), total)  # no trial recorded twice
        self.assertEqual(record_set_hash(records), expected)
        self.assertNotEqual(manifest.run_id, reference.read_manifest().run_id)


if __name__ == "__main__":
    unittest.main()
