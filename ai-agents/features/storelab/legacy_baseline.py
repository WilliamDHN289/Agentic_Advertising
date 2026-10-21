"""Hash-identified baseline of the original StoreLab run (SL-00).

The original experiment (``~/Desktop/agentic_amazon/5_storelab``, run 2026-08-14)
is not under version control. It is frozen here in two parts:

* the code, vendored byte-for-byte in ``legacy_2026_08/`` (excluded from ruff and
  coverage);
* the saved run data, as a content-addressed dataset snapshot in the local
  artifact store: ``<artifact root>/datasets/legacy_storelab_2026_08/<sha256>/``.

Standard library only, so it runs under the ai-agents interpreter (3.11) and the
original one (3.9.13). Every check is read-only; ``reproduce --mode copy`` writes
only to a temporary directory, and ``snapshot`` is the one-off command that
creates the store entry.

    python -m features.storelab.legacy_baseline verify [--source DIR]
    python -m features.storelab.legacy_baseline reproduce [--python EXE] [--mode in-memory|copy]
    python -m features.storelab.legacy_baseline catalog [--python EXE]
    python -m features.storelab.legacy_baseline snapshot --source DIR --access-date YYYY-MM-DD
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LEGACY_DIR = Path(__file__).resolve().parent / "legacy_2026_08"
VENDORED_FILES = ("storelab.py", "agents.py", "run_experiment.py", "analyze.py", "make_figures.py")

DATASET = "legacy_storelab_2026_08"
# sha256 over the sorted (path, sha256, size) entries of STORED_FILES; see snapshot_digest().
SNAPSHOT_SHA256 = "65d6137645b7585a04de285892ab5da52910d90cf77efed3028f9b11f68f1228"
STORED_FILES = (
    "logs/events.json",
    "logs/run.log",
    "results/_ckpt_api.json",
    "results/_ckpt_chatbot.json",
    "results/_ckpt_gui_agent.json",
    "results/_ckpt_human_sim.json",
    "results/_ckpt_sec_labeloff_posbottom.json",
    "results/_ckpt_sec_labeloff_postop.json",
    "results/_ckpt_sec_labelon_posbottom.json",
    "results/_ckpt_sec_labelon_postop.json",
    "results/episodes.json",
    "results/findings.md",
    "results/run_meta.json",
)
# What analyze.py reads (analyze.py l.113-115) and the file it writes (l.437-439).
ANALYZE_INPUTS = ("results/episodes.json", "logs/events.json", "results/run_meta.json")
FINDINGS = "results/findings.md"
RUN_LOG_CATALOG = re.compile(r"catalog: (\d+) SKUs across (\d+) categories, (\d+) ad-eligible")

# Interpreter flags for every child process: ignore PYTHON* variables (-E) and user
# site-packages (-s), and write no bytecode (-B), so runs are read-only and hermetic.
_FLAGS = ("-E", "-s", "-B")

# Runs analyze.main() from the unmodified vendored file with its I/O redirected:
# inputs are read from the snapshot and the findings.md write is captured in memory.
_IN_MEMORY = r"""
import io, os, sys
code_dir, data_dir = sys.argv[1], sys.argv[2]
sys.path[:] = [code_dir] + [p for p in sys.path if p not in ("", os.getcwd())]
import analyze
target = os.path.abspath(os.path.join(data_dir, "results", "findings.md"))
captured = []

class _Capture(io.StringIO):
    def close(self):
        captured.append(self.getvalue())
        super().close()

def _open(path, mode="r", *args, **kwargs):
    if set(mode) & set("wax+"):
        if os.path.abspath(path) != target:
            raise PermissionError("unexpected write by analyze.py: %s" % path)
        return _Capture()
    return open(path, mode, *args, **kwargs)

analyze.RES = os.path.join(data_dir, "results")
analyze.LOGS = os.path.join(data_dir, "logs")
analyze.open = _open
analyze.print = lambda *args, **kwargs: None
analyze.main()
if len(captured) != 1:
    raise SystemExit("analyze.main() wrote findings.md %d times" % len(captured))
sys.stdout.buffer.write(captured[0].encode("utf-8"))
"""

_CATALOG = r"""
import hashlib, json, os, sys
from dataclasses import asdict
sys.path[:] = [sys.argv[1]] + [p for p in sys.path if p not in ("", os.getcwd())]
from storelab import build_catalog
catalog = build_catalog()
blob = json.dumps([asdict(p) for p in catalog], sort_keys=True).encode()
print(json.dumps({
    "skus": len(catalog),
    "categories": len({p.category for p in catalog}),
    "ad_eligible": sum(p.is_ad_eligible for p in catalog),
    "catalog_sha256": hashlib.sha256(blob).hexdigest(),
}))
"""


def default_source() -> Path:
    return Path(os.environ.get("STORELAB_LEGACY_SOURCE", "~/Desktop/agentic_amazon/5_storelab")).expanduser()


def default_artifact_root() -> Path:
    # ai-agents/features/storelab/legacy_baseline.py -> <repository root>/artifacts
    return Path(os.environ.get("STORELAB_ARTIFACT_ROOT", Path(__file__).resolve().parents[3] / "artifacts"))


def snapshot_dir(root: Path | None = None) -> Path:
    return Path(root or default_artifact_root()) / "datasets" / DATASET / SNAPSHOT_SHA256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_entries(base: Path, names) -> list[dict]:
    return [{"path": name, "sha256": sha256_file(base / name), "size": (base / name).stat().st_size}
            for name in names]


def snapshot_digest(entries) -> str:
    """sha256 of the compact JSON of the sorted [path, sha256, size] rows."""
    rows = sorted([e["path"], e["sha256"], e["size"]] for e in entries)
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def check_vendored(source: Path | None = None) -> list[str]:
    """Problems with the vendored code; compared with ``source`` when given."""
    problems = []
    present = sorted(p.name for p in LEGACY_DIR.glob("*.py"))
    if present != sorted(VENDORED_FILES):
        problems.append("vendored .py files %s != expected %s" % (present, sorted(VENDORED_FILES)))
    if source is not None:
        for name in VENDORED_FILES:
            if not (LEGACY_DIR / name).is_file():
                continue
            ours, theirs = sha256_file(LEGACY_DIR / name), sha256_file(source / name)
            if ours != theirs:
                problems.append("%s: vendored %s != source %s" % (name, ours, theirs))
    return problems


def check_snapshot(root: Path | None = None, source: Path | None = None) -> list[str]:
    """Problems with the stored snapshot; compared with ``source`` when given."""
    target = snapshot_dir(root)
    manifest_path = target / "MANIFEST.json"
    if not manifest_path.is_file():
        return ["missing snapshot manifest %s" % manifest_path]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    if sorted(e["path"] for e in manifest["files"]) != sorted(STORED_FILES):
        problems.append("manifest file list differs from STORED_FILES")
    if snapshot_digest(manifest["files"]) != SNAPSHOT_SHA256 or manifest["snapshot_sha256"] != SNAPSHOT_SHA256:
        problems.append("manifest digest does not match the snapshot address %s" % SNAPSHOT_SHA256)
    stored = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} - {"MANIFEST.json"}
    if stored != set(STORED_FILES):
        problems.append("snapshot holds unexpected or missing files: %s" % sorted(stored ^ set(STORED_FILES)))
    for entry in manifest["files"]:
        path = target / entry["path"]
        if path.is_file() and (sha256_file(path), path.stat().st_size) != (entry["sha256"], entry["size"]):
            problems.append("%s: stored bytes do not match the manifest" % entry["path"])
        if source is not None and sha256_file(source / entry["path"]) != entry["sha256"]:
            problems.append("%s: source bytes do not match the manifest" % entry["path"])
    return problems


def write_snapshot(source: Path, root: Path, access_date: str) -> Path:
    """Copy STORED_FILES from ``source`` into a new, immutable snapshot directory."""
    entries = file_entries(source, STORED_FILES)
    digest = snapshot_digest(entries)
    if digest != SNAPSHOT_SHA256:
        raise ValueError("source digest %s != pinned SNAPSHOT_SHA256 %s" % (digest, SNAPSHOT_SHA256))
    target = Path(root) / "datasets" / DATASET / digest
    if target.exists():
        raise FileExistsError("snapshot already exists: %s" % target)
    for name in STORED_FILES:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target / name)
    run_meta = json.loads((source / "results/run_meta.json").read_text(encoding="utf-8"))
    manifest = {
        "dataset": DATASET,
        "snapshot_sha256": digest,
        "digest": "sha256 of compact JSON of sorted [path, sha256, size] rows",
        "source": str(source).replace(str(Path.home()), "~", 1),
        "access_date": access_date,
        "run_utc": run_meta["run_utc"],
        "model": run_meta["model"],
        "files": entries,
    }
    (target / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    problems = check_snapshot(root, source)
    if problems:
        raise RuntimeError("snapshot verification failed: %s" % problems)
    return target


def _child(python: str, args, cwd: Path | None = None) -> bytes:
    result = subprocess.run([python, *_FLAGS, *args], cwd=cwd, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("%s exited %d: %s" % (python, result.returncode, result.stderr.decode(errors="replace")))
    return result.stdout


def regenerate_findings(python: str = sys.executable, mode: str = "in-memory",
                        data_dir: Path | None = None) -> bytes:
    """Run the vendored analyze.py on the saved data and return the findings.md bytes.

    ``in-memory`` writes nothing anywhere. ``copy`` reproduces the original layout
    (code next to ``results/`` and ``logs/``) in a temporary directory and runs
    ``python analyze.py`` there unchanged.
    """
    data_dir = Path(data_dir or snapshot_dir())
    if mode == "in-memory":
        return _child(python, ["-c", _IN_MEMORY, str(LEGACY_DIR), str(data_dir)])
    if mode != "copy":
        raise ValueError("mode must be 'in-memory' or 'copy'")
    with tempfile.TemporaryDirectory(prefix="storelab-sl00-") as tmp:
        work = Path(tmp)
        for name in VENDORED_FILES:
            shutil.copyfile(LEGACY_DIR / name, work / name)
        for name in ANALYZE_INPUTS:
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(data_dir / name, work / name)
        _child(python, ["analyze.py"], cwd=work)
        return (work / FINDINGS).read_bytes()


def findings_diff(expected: bytes, actual: bytes) -> list[str]:
    """Unified diff of two findings.md versions; empty means byte-identical."""
    if expected == actual:
        return []
    lines = difflib.unified_diff(expected.decode("utf-8", "replace").splitlines(True),
                                 actual.decode("utf-8", "replace").splitlines(True),
                                 "stored/findings.md", "regenerated/findings.md")
    return list(lines) or ["files differ only in bytes not visible as text (encoding or newlines)\n"]


def catalog_summary(python: str = sys.executable) -> dict:
    """Run the vendored build_catalog() and summarise it."""
    return json.loads(_child(python, ["-c", _CATALOG, str(LEGACY_DIR)]))


def run_log_catalog(data_dir: Path | None = None) -> dict:
    """The catalog line printed by the original run (run_experiment.py l.157-158)."""
    text = (Path(data_dir or snapshot_dir()) / "logs/run.log").read_text(encoding="utf-8")
    match = RUN_LOG_CATALOG.search(text)
    if not match:
        raise ValueError("run.log has no catalog line")
    skus, categories, ad_eligible = map(int, match.groups())
    return {"skus": skus, "categories": categories, "ad_eligible": ad_eligible}


def interpreter(python: str) -> str:
    return _child(python, ["-c", "import sys; print(sys.version.split()[0], sys.executable)"]).decode().strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify the SL-00 StoreLab baseline.")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="check vendored code and the stored snapshot")
    verify.add_argument("--source", type=Path, help="also compare with the original directory")
    reproduce = sub.add_parser("reproduce", help="regenerate findings.md and diff it against the stored one")
    reproduce.add_argument("--python", default=sys.executable)
    reproduce.add_argument("--mode", choices=("in-memory", "copy"), default="in-memory")
    catalog = sub.add_parser("catalog", help="run build_catalog() and compare with run.log")
    catalog.add_argument("--python", default=sys.executable)
    snapshot = sub.add_parser("snapshot", help="create the artifact-store snapshot (once)")
    snapshot.add_argument("--source", type=Path, required=True)
    snapshot.add_argument("--access-date", required=True)
    args = parser.parse_args(argv)

    if args.command == "verify":
        problems = check_vendored(args.source) + check_snapshot(source=args.source)
        print(json.dumps({"vendored": list(VENDORED_FILES), "snapshot": str(snapshot_dir()),
                          "compared_with_source": str(args.source) if args.source else None,
                          "problems": problems}, indent=1))
        return 1 if problems else 0
    if args.command == "reproduce":
        expected = (snapshot_dir() / FINDINGS).read_bytes()
        actual = regenerate_findings(args.python, args.mode)
        diff = findings_diff(expected, actual)
        sys.stdout.writelines(diff)
        print(json.dumps({"interpreter": interpreter(args.python), "mode": args.mode,
                          "stored_sha256": hashlib.sha256(expected).hexdigest(),
                          "regenerated_sha256": hashlib.sha256(actual).hexdigest(),
                          "diff_lines": len(diff)}, indent=1))
        return 1 if diff else 0
    if args.command == "catalog":
        summary = catalog_summary(args.python)
        logged = run_log_catalog()
        ok = summary["skus"] == logged["skus"] and summary["ad_eligible"] == logged["ad_eligible"]
        print(json.dumps({"interpreter": interpreter(args.python), "build_catalog": summary,
                          "run_log": logged, "match": ok}, indent=1))
        return 0 if ok else 1
    print(write_snapshot(args.source.expanduser(), default_artifact_root(), args.access_date))
    return 0


if __name__ == "__main__":
    sys.exit(main())
