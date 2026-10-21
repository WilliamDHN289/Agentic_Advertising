"""Version capture for the run manifest (SL-02).

``env_version`` follows ``source_fingerprint`` in webserver-v2
``services/engine_version.py``: a digest of the environment's own source, so it
changes whenever any rule, weight or threshold in core/ does. Unlike that helper it
also hashes each file's relative path, so moving code between files changes it too.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Iterable

CORE_DIR = Path(__file__).resolve().parent
ENV_NAME = "storelab-core"
# Packages whose versions can change trial outcomes or their analysis.
TRACKED_PACKAGES = ("numpy", "scipy", "pandas", "pyarrow", "pydantic", "cvxpy",
                    "openai", "anthropic", "google-genai")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(files: Iterable[tuple[str, bytes]]) -> str:
    """sha256 over (relative path, content) pairs sorted by path; CRLF read as LF."""
    digest = hashlib.sha256()
    for name, content in sorted(files):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(content.replace(b"\r\n", b"\n")).digest())
    return digest.hexdigest()


def source_fingerprint(root: Path, pattern: str = "*.py") -> str:
    root = Path(root)
    return fingerprint((p.relative_to(root).as_posix(), p.read_bytes())
                       for p in root.rglob(pattern) if p.is_file() and "__pycache__" not in p.parts)


def env_version(root: Path = CORE_DIR) -> str:
    """``storelab-core+<16 hex digits of the core/ source fingerprint>``."""
    return "%s+%s" % (ENV_NAME, source_fingerprint(root)[:16])


def git_state(path: Path = CORE_DIR) -> dict[str, object]:
    """HEAD commit and whether the work tree differs from it; nulls outside a repository."""
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", "--no-optional-locks", "-C", str(path), *args],
                                    capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout if result.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain") if sha else None
    return {"sha": sha.strip() if sha else None, "dirty": bool(status.strip()) if status is not None else None}


def python_info() -> dict[str, str]:
    return {"version": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform()}


def package_versions(names: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed distribution versions; null for a package that is not installed."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions
