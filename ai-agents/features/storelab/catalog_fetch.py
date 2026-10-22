"""Stream Amazon Reviews 2023 item metadata from Hugging Face into a snapshot (SL-05).

    python -m features.storelab.catalog_fetch [--categories Electronics Home_and_Kitchen]
        [--root ARTIFACT_ROOT] [--access-date YYYY-MM-DD]

Rows are read with the generic ``json`` builder over an ``hf://`` path pinned to
core/catalog_amazon.py SOURCE_REVISION, in streaming mode, so the multi-gigabyte
source files are never written to disk; only the normalised products that pass the
rating floor are stored (core/catalog_amazon.py describes the snapshot). The
repository's own loading script is not used: recent ``datasets`` releases no longer
run it. Before streaming, each file's LFS sha256 and size at the revision are
checked against SOURCE_FILES.

Outside core/ because ``datasets`` and ``huggingface_hub`` are not on its import
allow-list. They need pyarrow >= 24 and so numpy >= 2, which the ai-agents pins
exclude, so this tool runs in its own environment (requirements-storelab-fetch.txt).
"""
from __future__ import annotations

import argparse
import datetime
import platform
import shutil
import sys
import time
import uuid
from importlib import metadata
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .core.catalog_amazon import (ATTRIBUTE_KEYS, DATASET, MIN_RATINGS, NORMALISER_VERSION, SOURCE_FILES,
                                  SOURCE_REPO, SOURCE_REVISION, STORED_DESCRIPTION_CHARS, CategoryCounts,
                                  ProductWriter, SnapshotManifest, SourceFile, StoredFile, check_snapshot,
                                  items_path, normalise, snapshot_digest, snapshot_dir, write_manifest)
from .legacy_baseline import default_artifact_root

PROGRESS_EVERY = 250_000
# Every column of the metadata files. Types are declared because inferring them from
# the first chunk fails later in the file (``author`` is null at first, then a
# struct). Json keeps each value as decoded JSON; normalise() checks the types.
SOURCE_COLUMNS = ("main_category", "title", "average_rating", "rating_number", "features", "description", "price",
                  "images", "videos", "store", "categories", "details", "parent_asin", "bought_together",
                  "subtitle", "author")


def data_file_url(category: str, revision: str = SOURCE_REVISION) -> str:
    return "hf://datasets/%s@%s/%s" % (SOURCE_REPO, revision, SOURCE_FILES[category][0])


def stream_rows(category: str) -> Iterable[Mapping]:
    """The source rows of ``category``, streamed."""
    from datasets import Features, Json, Value, load_dataset

    features = Features({c: Value("string") if c == "parent_asin" else Json() for c in SOURCE_COLUMNS})
    return load_dataset("json", data_files=data_file_url(category), split="train", streaming=True,
                        features=features)


def source_files(categories: Iterable[str]) -> list[SourceFile]:
    """The pinned source files, after checking the hub reports the same sha256 and size."""
    from huggingface_hub import HfApi

    categories = list(categories)
    paths = [SOURCE_FILES[c][0] for c in categories]
    infos = {i.path: i for i in HfApi().get_paths_info(SOURCE_REPO, paths, repo_type="dataset",
                                                       revision=SOURCE_REVISION)}
    files = []
    for category in categories:
        path, sha256, size = SOURCE_FILES[category]
        info = infos.get(path)
        lfs = getattr(info, "lfs", None)
        if lfs is None or (lfs.sha256, info.size) != (sha256, size):
            raise RuntimeError("%s at %s: hub reports %s, pinned sha256 %s size %d"
                               % (path, SOURCE_REVISION, lfs or info, sha256, size))
        files.append(SourceFile(category=category, path=path, sha256=sha256, size=size))
    return files


def tool_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("datasets", "huggingface_hub", "pyarrow", "pydantic"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def store_category(rows: Iterable[Mapping], category: str, target: Path,
                   log: Callable[[str], None] = lambda _: None) -> tuple[StoredFile, CategoryCounts]:
    """Normalise ``rows`` and write the products with >= MIN_RATINGS ratings to ``target``."""
    counts = dict(rows=0, stored=0, no_parent_asin=0, duplicate_parent_asin=0, missing_rating_number=0,
                  below_min_ratings=0)
    stored_ids: set[str] = set()
    started = time.monotonic()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        writer = ProductWriter(handle)
        for raw in rows:
            counts["rows"] += 1
            if counts["rows"] % PROGRESS_EVERY == 0:
                log("%s: %d rows, %d stored, %.0f rows/s" % (category, counts["rows"], counts["stored"],
                                                             counts["rows"] / (time.monotonic() - started)))
            product = normalise(raw, category, STORED_DESCRIPTION_CHARS)
            if product is None:
                counts["no_parent_asin"] += 1
            elif product.rating_number is None:
                counts["missing_rating_number"] += 1
            elif product.rating_number < MIN_RATINGS:
                counts["below_min_ratings"] += 1
            elif product.product_id in stored_ids:
                counts["duplicate_parent_asin"] += 1
            else:
                stored_ids.add(product.product_id)
                writer.write(product)
                counts["stored"] += 1
        sha256, size = writer.close()
    log("%s: done, %d rows, %d stored" % (category, counts["rows"], counts["stored"]))
    return StoredFile(path=items_path(category), sha256=sha256, size=size), CategoryCounts(**counts)


def write_snapshot(root: Path, sources: list[SourceFile], access_date: str,
                   rows_for: Callable[[str], Iterable[Mapping]] = stream_rows,
                   tools: Mapping[str, str | None] | None = None,
                   log: Callable[[str], None] = lambda _: None) -> Path:
    """Build a snapshot of ``sources`` under ``root``; returns its directory.

    Files are written to a staging directory and moved to their address once the
    manifest is complete, so an interrupted fetch leaves no partial snapshot. If a
    snapshot with the same address exists, it is kept and the staging copy removed.
    """
    base = Path(root) / "datasets" / DATASET
    staging = base / (".staging-%s" % uuid.uuid4().hex)
    try:
        files, counts = [], {}
        for source in sources:
            stored, counts[source.category] = store_category(rows_for(source.category), source.category,
                                                             staging / items_path(source.category), log)
            files.append(stored)
        digest = snapshot_digest(files)
        manifest = SnapshotManifest(
            snapshot_sha256=digest, source_repo=SOURCE_REPO, source_revision=SOURCE_REVISION,
            source_files=sources, access_date=access_date, normaliser=NORMALISER_VERSION,
            min_ratings=MIN_RATINGS, description_chars=STORED_DESCRIPTION_CHARS,
            attribute_keys=list(ATTRIBUTE_KEYS), counts=counts, files=files,
            tools=dict(tools if tools is not None else tool_versions()))
        write_manifest(staging, manifest)
        target = snapshot_dir(root, digest)
        if target.exists():
            log("snapshot %s already exists; keeping it" % digest)
        else:
            staging.rename(target)
        problems = check_snapshot(target)
        if problems:
            raise RuntimeError("snapshot verification failed: %s" % problems)
        return target
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m features.storelab.catalog_fetch")
    parser.add_argument("--categories", nargs="+", default=list(SOURCE_FILES), choices=sorted(SOURCE_FILES))
    parser.add_argument("--root", type=Path, default=None, help="artifact root (default: <repository>/artifacts)")
    parser.add_argument("--access-date", default=datetime.datetime.now(datetime.timezone.utc).date().isoformat())
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    target = write_snapshot(args.root or default_artifact_root(), source_files(args.categories),
                            args.access_date, log=log)
    print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
