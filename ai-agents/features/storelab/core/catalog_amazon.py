"""Amazon Reviews 2023 item metadata for E1 (SL-05): normaliser, loader and snapshot.

Source: Hou et al. (2024), the McAuley-Lab/Amazon-Reviews-2023 dataset on Hugging
Face, files ``raw/meta_categories/meta_<Category>.jsonl`` at a pinned revision
(SOURCE). ``catalog_fetch.py`` streams them through ``normalise`` and stores a
snapshot under ``datasets/amazon_reviews_2023/<snapshot_sha256>/``:

    MANIFEST.json              source revision and file sha256, access date, normaliser
                               settings, per-category counts, stored file digests
    items/<Category>.jsonl.gz  one Product per line (canonical JSON), stream order

A snapshot stores the products with at least MIN_RATINGS ratings, the floor of the
E1 screening (core/tasks.py); fields missing in the source stay null and are counted
when tasks exclude them. Stored files are addressed by the sha256 and size of their
decompressed bytes, so the address does not depend on the zlib build. Terms (A7):
research use; only ids and derived fields may be redistributed until confirmed, so
the items files stay out of version control.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import uuid
from pathlib import Path
from typing import IO, Annotated, Any, Iterable, Iterator, Literal, Mapping

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt, StringConstraints

from .config import Ident, NonEmptyStr, Section, Sha256, canonical_json
from .protocol import ProductId

GitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
IsoDate = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]

DATASET = "amazon_reviews_2023"
NORMALISER_VERSION = "amazon_meta@1"
SOURCE_REPO = "McAuley-Lab/Amazon-Reviews-2023"
SOURCE_REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
# Category -> (path in the repository, LFS sha256, size in bytes) at SOURCE_REVISION.
SOURCE_FILES = {
    "Electronics": ("raw/meta_categories/meta_Electronics.jsonl",
                    "5eb4dabd539586c49654297319ef178b62a2fe8e43108376feda125f04455d93", 5246144134),
    "Home_and_Kitchen": ("raw/meta_categories/meta_Home_and_Kitchen.jsonl",
                         "691353084748d985180deb13dc8de59afe4969933ad3692ea06b80460510aa56", 11788767944),
}
MIN_RATINGS = 50
# Descriptions are stored up to this length; loaders cut them to the run's
# environment.description_chars, which must not exceed it.
STORED_DESCRIPTION_CHARS = 1000
ATTRIBUTE_MAX_CHARS = 80
# The ``details`` entries kept as attributes: those E1 requirements may name
# (core/tasks.py REQUIREMENT_KEYS) and Brand, shown on product pages.
ATTRIBUTE_KEYS = ("Brand", "Material", "Color", "Style", "Shape", "Pattern", "Finish Type",
                  "Connectivity Technology", "Form Factor", "Power Source")
REQUIRED_FIELDS = ("title", "price", "average_rating", "rating_number", "category_path")
DIGEST_RULE = "sha256 of compact JSON of sorted [path, sha256, size] rows; sha256 and size of the decompressed file"


class Product(Section):
    """One item of the metadata, normalised. ``None`` marks a field the source lacks."""

    product_id: ProductId  # parent_asin
    category: NonEmptyStr  # the source file's category, e.g. Electronics
    category_path: list[NonEmptyStr] = []  # top level first; the last entry is the leaf
    title: NonEmptyStr | None = None
    price: PositiveFloat | None = None  # USD
    average_rating: float | None = Field(default=None, ge=1.0, le=5.0)
    rating_number: NonNegativeInt | None = None
    description: str = ""
    attributes: dict[NonEmptyStr, NonEmptyStr] = {}

    def missing(self) -> list[str]:
        """The REQUIRED_FIELDS this product lacks."""
        return [name for name in REQUIRED_FIELDS if getattr(self, name) in (None, [])]

    def truncated(self, description_chars: int) -> Product:
        return self.model_copy(update={"description": truncate(self.description, description_chars)})


def truncate(text: str, chars: int) -> str:
    """The first ``chars`` characters, without trailing whitespace."""
    if chars < 1:
        raise ValueError("chars must be positive")
    return text[:chars].rstrip()


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return " ".join(value.split()) or None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip().removeprefix("$").replace(",", ""))
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def normalise(raw: Mapping[str, Any], category: str,
              description_chars: int = STORED_DESCRIPTION_CHARS) -> Product | None:
    """A ``Product`` from one metadata row; ``None`` if the row has no usable parent_asin.

    Text is whitespace-collapsed. The description joins the row's paragraphs and keeps
    ``description_chars`` characters. A price must be positive, an average rating in
    [1, 5] and a rating count a non-negative integer; anything else becomes null.
    Attributes keep the ATTRIBUTE_KEYS entries of ``details`` with a text or number
    value of at most ATTRIBUTE_MAX_CHARS characters.
    """
    product_id = _text(raw.get("parent_asin"))
    if product_id is None or " " in product_id or len(product_id) > 64:
        return None
    price = _number(raw.get("price"))
    rating = _number(raw.get("average_rating"))
    count = _number(raw.get("rating_number"))
    paragraphs = raw.get("description")
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    description = " ".join(t for t in (_text(p) for p in paragraphs or []) if t) if isinstance(paragraphs, list) else ""
    details = raw.get("details")
    attributes = {}
    if isinstance(details, Mapping):
        for key in ATTRIBUTE_KEYS:
            value = details.get(key)
            value = _text(str(value)) if isinstance(value, (int, float)) and not isinstance(value, bool) else _text(value)
            if value is not None and len(value) <= ATTRIBUTE_MAX_CHARS:
                attributes[key] = value
    path = raw.get("categories")
    return Product(
        product_id=product_id,
        category=category,
        category_path=[t for t in (_text(c) for c in path) if t] if isinstance(path, list) else [],
        title=_text(raw.get("title")),
        price=price if price is not None and price > 0 else None,
        average_rating=rating if rating is not None and 1.0 <= rating <= 5.0 else None,
        rating_number=int(count) if count is not None and count >= 0 and count == int(count) else None,
        description=truncate(description, description_chars) if description else "",
        attributes=attributes,
    )


# ------------------------------------------------------------------- snapshot
class SourceFile(Section):
    category: NonEmptyStr
    path: NonEmptyStr
    sha256: Sha256  # LFS sha256 reported by the hub at the pinned revision
    size: PositiveInt


class StoredFile(Section):
    path: NonEmptyStr
    sha256: Sha256  # of the decompressed bytes
    size: NonNegativeInt  # of the decompressed bytes


class CategoryCounts(Section):
    rows: NonNegativeInt  # rows read from the source file
    stored: NonNegativeInt
    no_parent_asin: NonNegativeInt
    duplicate_parent_asin: NonNegativeInt
    missing_rating_number: NonNegativeInt
    below_min_ratings: NonNegativeInt


class SnapshotManifest(Section):
    dataset: Literal["amazon_reviews_2023"] = DATASET
    snapshot_sha256: Sha256
    digest: Literal[DIGEST_RULE] = DIGEST_RULE
    source_repo: NonEmptyStr
    source_revision: GitSha
    source_files: list[SourceFile] = Field(min_length=1)
    access_date: IsoDate
    normaliser: NonEmptyStr
    min_ratings: NonNegativeInt
    description_chars: PositiveInt
    attribute_keys: list[NonEmptyStr]
    counts: dict[NonEmptyStr, CategoryCounts]
    files: list[StoredFile] = Field(min_length=1)
    tools: dict[Ident, str | None] = {}  # versions of the fetch tooling
    terms: NonEmptyStr = "A7: research use; redistribute ids and derived fields only until confirmed"

    @property
    def categories(self) -> list[str]:
        return [f.category for f in self.source_files]


def items_path(category: str) -> str:
    return "items/%s.jsonl.gz" % category


def snapshot_digest(files: Iterable[StoredFile]) -> str:
    rows = sorted([f.path, f.sha256, f.size] for f in files)
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def snapshot_dir(root: Path, snapshot_sha256: str) -> Path:
    """``<root>/datasets/amazon_reviews_2023/<snapshot_sha256>``; root is the artifact root."""
    return Path(root) / "datasets" / DATASET / snapshot_sha256


class ProductWriter:
    """Writes products as gzip JSONL (mtime 0, no file name) and digests the plain bytes."""

    def __init__(self, target: IO[bytes]):
        self._gzip = gzip.GzipFile(fileobj=target, mode="wb", mtime=0, filename="")
        self._digest = hashlib.sha256()
        self.size = 0
        self.count = 0

    def write(self, product: Product) -> None:
        line = (canonical_json(product.model_dump(mode="json")) + "\n").encode("utf-8")
        self._gzip.write(line)
        self._digest.update(line)
        self.size += len(line)
        self.count += 1

    def close(self) -> tuple[str, int]:
        """Finish the gzip stream; returns (sha256, size) of the decompressed bytes."""
        self._gzip.close()
        return self._digest.hexdigest(), self.size


def _lines(path: Path) -> Iterator[bytes]:
    with gzip.open(path, "rb") as handle:
        yield from handle


def iter_products(directory: Path, category: str) -> Iterator[Product]:
    """The stored products of ``category``, in stream order."""
    for line in _lines(Path(directory) / items_path(category)):
        yield Product.model_validate_json(line)


def read_manifest(directory: Path) -> SnapshotManifest:
    return SnapshotManifest.model_validate_json((Path(directory) / "MANIFEST.json").read_bytes())


def write_manifest(directory: Path, manifest: SnapshotManifest) -> None:
    text = json.dumps(manifest.model_dump(mode="json"), indent=1, ensure_ascii=False) + "\n"
    _atomic_write(Path(directory) / "MANIFEST.json", text.encode("utf-8"))


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(".%s.%s.tmp" % (path.name, uuid.uuid4().hex))
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def check_snapshot(directory: Path, verify_files: bool = True) -> list[str]:
    """Problems with a stored snapshot: manifest, address and (optionally) file digests."""
    directory = Path(directory)
    try:
        manifest = read_manifest(directory)
    except (OSError, ValueError) as error:
        return ["unreadable manifest in %s: %s" % (directory, error)]
    problems = []
    if snapshot_digest(manifest.files) != manifest.snapshot_sha256:
        problems.append("manifest files do not match its snapshot_sha256")
    if directory.name != manifest.snapshot_sha256:
        problems.append("directory name %s is not the snapshot_sha256 %s" % (directory.name, manifest.snapshot_sha256))
    expected = {items_path(c) for c in manifest.categories}
    if {f.path for f in manifest.files} != expected:
        problems.append("manifest files %s != items of its categories %s"
                        % (sorted(f.path for f in manifest.files), sorted(expected)))
    for f in manifest.files if verify_files else ():
        path = directory / f.path
        if not path.is_file():
            problems.append("%s: missing" % f.path)
            continue
        digest, size = hashlib.sha256(), 0
        for line in _lines(path):
            digest.update(line)
            size += len(line)
        if (digest.hexdigest(), size) != (f.sha256, f.size):
            problems.append("%s: stored bytes do not match the manifest" % f.path)
    return problems


def load_products(directory: Path, categories: Iterable[str], description_chars: int,
                  ids: Iterable[str] | None = None) -> dict[str, Product]:
    """Products of ``categories`` (only ``ids`` if given), descriptions cut to
    ``description_chars``. Raises if the snapshot stored shorter descriptions."""
    manifest = read_manifest(directory)
    if description_chars > manifest.description_chars:
        raise ValueError("description_chars %d exceeds the %d characters the snapshot stored"
                         % (description_chars, manifest.description_chars))
    wanted = set(ids) if ids is not None else None
    found = {}
    for category in categories:
        if category not in manifest.categories:
            raise ValueError("category %s is not in snapshot %s" % (category, manifest.snapshot_sha256))
        for product in iter_products(directory, category):
            if wanted is None or product.product_id in wanted:
                found[product.product_id] = product.truncated(description_chars)
    if wanted is not None and wanted - set(found):
        raise KeyError("products not in the snapshot: %s" % sorted(wanted - set(found))[:10])
    return found


def write_product_file(path: Path | IO[bytes], products: Iterable[Product]) -> tuple[str, int]:
    """Write ``products`` as one items file; returns (sha256, size) of the plain bytes."""
    if isinstance(path, (str, os.PathLike)):
        buffer = io.BytesIO()
        writer = ProductWriter(buffer)
        for product in products:
            writer.write(product)
        result = writer.close()
        _atomic_write(Path(path), buffer.getvalue())
        return result
    writer = ProductWriter(path)
    for product in products:
        writer.write(product)
    return writer.close()
