"""SL-05 catalog: normaliser, snapshot files and the streaming fetch (no network).

Writes only to temporary directories.
"""
from __future__ import annotations

import gzip
import io
import tempfile
import unittest
from pathlib import Path

from features.storelab import catalog_fetch
from features.storelab.core.catalog_amazon import (MIN_RATINGS, STORED_DESCRIPTION_CHARS, Product, SourceFile,
                                                   check_snapshot, items_path, iter_products, load_products,
                                                   normalise, read_manifest, write_product_file)
from tests.features.storelab import e1_fixture

SOURCES = [SourceFile(category=c, path="raw/meta_categories/meta_%s.jsonl" % c, sha256="0" * 64, size=1)
           for c in ("Electronics", "Home_and_Kitchen")]
TOOLS = {"python": "3.11"}


def fixture_snapshot(root: Path) -> Path:
    return catalog_fetch.write_snapshot(root, SOURCES, "2026-10-21", rows_for=e1_fixture.raw_rows, tools=TOOLS)


class NormaliseTest(unittest.TestCase):
    def test_a_complete_row(self):
        p = normalise({"parent_asin": " B01 ", "title": "  Glass   mug ", "price": 24.95, "average_rating": 4.6,
                       "rating_number": 18, "description": ["First  part.", "", "Second\npart."],
                       "categories": ["Home & Kitchen", " Mugs "],
                       "details": {"Material": "Glass", "Brand": 7, "Color": ["x"], "Date First Available": "2014",
                                   "Style": "s" * 81}}, "Home_and_Kitchen")
        self.assertEqual(p, Product(product_id="B01", category="Home_and_Kitchen",
                                    category_path=["Home & Kitchen", "Mugs"], title="Glass mug", price=24.95,
                                    average_rating=4.6, rating_number=18, description="First part. Second part.",
                                    attributes={"Brand": "7", "Material": "Glass"}))
        self.assertEqual(p.missing(), [])

    def test_unusable_values_become_null_and_are_reported_missing(self):
        p = normalise({"parent_asin": "B02", "title": " ", "price": "None", "average_rating": 0.0,
                       "rating_number": 3.5, "categories": [], "description": None, "details": None}, "Electronics")
        self.assertEqual((p.title, p.price, p.average_rating, p.rating_number, p.category_path, p.description,
                          p.attributes), (None, None, None, None, [], "", {}))
        self.assertEqual(p.missing(), ["title", "price", "average_rating", "rating_number", "category_path"])
        for price, expected in (("$1,299.50", 1299.5), (0, None), (-2, None), ("1e999", None), (True, None),
                                ("12.99 - 15.99", None), (12, 12.0)):
            with self.subTest(price=price):
                self.assertEqual(normalise({"parent_asin": "B", "price": price}, "Electronics").price, expected)

    def test_rows_without_a_usable_parent_asin_are_dropped(self):
        for asin in (None, "", "  ", "B 1", 123, "B" * 65):
            with self.subTest(asin=asin):
                self.assertIsNone(normalise({"parent_asin": asin, "title": "x"}, "Electronics"))

    def test_descriptions_are_cut(self):
        p = normalise({"parent_asin": "B", "description": "word " * 500}, "Electronics", 12)
        self.assertEqual(p.description, "word word wo")
        self.assertEqual(p.truncated(7).description, "word wo")
        self.assertEqual(p.truncated(5).description, "word")


class SnapshotFileTest(unittest.TestCase):
    def test_items_files_are_byte_deterministic_and_round_trip(self):
        products = e1_fixture.products("Electronics")
        first, second = io.BytesIO(), io.BytesIO()
        digest = write_product_file(first, products)
        self.assertEqual(write_product_file(second, products), digest)
        self.assertEqual(first.getvalue(), second.getvalue())
        plain = gzip.decompress(first.getvalue())
        self.assertEqual(len(plain), digest[1])
        self.assertEqual([Product.model_validate_json(line) for line in plain.splitlines()], products)


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))

    def test_snapshot_counts_files_and_address(self):
        target = fixture_snapshot(self.root)
        manifest = read_manifest(target)
        self.assertEqual(target.name, manifest.snapshot_sha256)
        self.assertEqual(check_snapshot(target), [])
        self.assertEqual(manifest.source_revision, catalog_fetch.SOURCE_REVISION)
        self.assertEqual((manifest.min_ratings, manifest.description_chars), (MIN_RATINGS, STORED_DESCRIPTION_CHARS))
        for category in ("Electronics", "Home_and_Kitchen"):
            counts = manifest.counts[category]
            rows = e1_fixture.raw_rows(category)
            stored = list(iter_products(target, category))
            with self.subTest(category=category):
                self.assertEqual(counts.rows, len(rows))
                self.assertEqual((counts.no_parent_asin, counts.missing_rating_number, counts.duplicate_parent_asin,
                                  counts.below_min_ratings), (1, 1, 1, 2))
                self.assertEqual(counts.stored, len(stored))
                self.assertEqual(stored, e1_fixture.products(category))
                self.assertEqual(counts.rows, counts.stored + 1 + 1 + 1 + 2)

    def test_the_same_rows_give_the_same_snapshot_and_no_staging_is_left(self):
        first = fixture_snapshot(self.root)
        second = fixture_snapshot(self.root)
        self.assertEqual(first, second)
        self.assertEqual(sorted(p.name for p in first.parent.iterdir()), [first.name])

    def test_an_interrupted_fetch_leaves_nothing(self):
        def failing(category):
            yield from e1_fixture.raw_rows(category)[:10]
            raise ConnectionError("stream dropped")

        with self.assertRaises(ConnectionError):
            catalog_fetch.write_snapshot(self.root, SOURCES, "2026-10-21", rows_for=failing, tools=TOOLS)
        base = self.root / "datasets" / "amazon_reviews_2023"
        self.assertEqual(list(base.iterdir()) if base.exists() else [], [])

    def test_tampering_is_detected(self):
        target = fixture_snapshot(self.root)
        path = target / items_path("Electronics")
        products = list(iter_products(target, "Electronics"))
        write_product_file(path, products[:-1])
        self.assertTrue(any("do not match" in p for p in check_snapshot(target)))
        self.assertEqual(check_snapshot(target, verify_files=False), [])
        renamed = target.with_name("f" * 64)
        target.rename(renamed)
        self.assertTrue(any("directory name" in p for p in check_snapshot(renamed, verify_files=False)))

    def test_load_products_cuts_descriptions_and_checks_ids(self):
        target = fixture_snapshot(self.root)
        loaded = load_products(target, ["Electronics"], 5)
        self.assertEqual(len(loaded), len(e1_fixture.products("Electronics")))
        self.assertTrue(all(len(p.description) <= 5 for p in loaded.values()))
        some = sorted(loaded)[:3]
        self.assertEqual(sorted(load_products(target, ["Electronics"], 300, ids=some)), some)
        with self.assertRaises(KeyError):
            load_products(target, ["Electronics"], 300, ids=["NOPE"])
        with self.assertRaises(ValueError):
            load_products(target, ["Electronics"], STORED_DESCRIPTION_CHARS + 1)
        with self.assertRaises(ValueError):
            load_products(target, ["Books"], 300)

    def test_the_stream_is_the_pinned_revision_with_declared_types(self):
        url = catalog_fetch.data_file_url("Electronics")
        self.assertEqual(url, "hf://datasets/McAuley-Lab/Amazon-Reviews-2023@%s/raw/meta_categories/"
                              "meta_Electronics.jsonl" % catalog_fetch.SOURCE_REVISION)
        self.assertIn("author", catalog_fetch.SOURCE_COLUMNS)
        self.assertEqual(len(set(catalog_fetch.SOURCE_COLUMNS)), 16)


if __name__ == "__main__":
    unittest.main()
