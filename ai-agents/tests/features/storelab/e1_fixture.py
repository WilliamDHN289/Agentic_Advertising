"""A synthetic catalog in the Amazon Reviews 2023 metadata schema, for the SL-05 tests.

Per category: two leaves that pass screening (KEPT), one whose only supported task is
the price band (BAND_ONLY) and one below the density floor (SPARSE); plus rows the
fetch must skip or count (no parent_asin, too few ratings, no rating count, a
duplicate). In each kept leaf some rows lack a required field, one product's
attribute is spelled in another case, and a few prices lie outside the band, so the
exclusion counts are known exactly (EXPECTED).
"""
from __future__ import annotations

from features.storelab.core.catalog_amazon import Product, normalise

TOP = {"Electronics": "Electronics", "Home_and_Kitchen": "Home & Kitchen"}
# category -> leaf name -> (attribute key, value, products with it, band price)
KEPT = {
    "Electronics": {"Headphones": ("Connectivity Technology", "Bluetooth", 44, 40.0),
                    "Chargers": ("Power Source", "USB", 38, 15.0)},
    "Home_and_Kitchen": {"Mugs": ("Material", "Glass", 46, 20.0),
                         "Kettles": ("Material", "Stainless Steel", 36, 30.0)},
}
BAND_ONLY = {"Electronics": "Cables", "Home_and_Kitchen": "Spoons"}
SPARSE = {"Electronics": "Adapters", "Home_and_Kitchen": "Forks"}
# Per kept leaf: rows missing price, title, average_rating; in-band rows missing the
# attribute; complete rows priced outside the band; rows with the attribute in
# another case (eligible) and with another value (not eligible).
MISSING_PRICE, MISSING_TITLE, MISSING_RATING, MISSING_ATTRIBUTE = 2, 1, 1, 3
OUTSIDE_BAND, OTHER_CASE, OTHER_VALUE = 4, 2, 5


def _row(asin, path, title, price, rating, count, attributes, description="A product."):
    return {"main_category": "x", "title": title, "average_rating": rating, "rating_number": count,
            "features": [], "description": [description, "Second paragraph."], "price": price, "images": [],
            "videos": [], "store": "Shop", "categories": path, "details": dict(attributes),
            "parent_asin": asin, "bought_together": None}


def raw_rows(category: str) -> list[dict]:
    """The source rows of ``category``, in a fixed order."""
    top, prefix, rows = TOP[category], category[0], []
    n = 0

    def asin():
        nonlocal n
        n += 1
        return "%s%07d" % (prefix, n)

    for leaf, (key, value, count, price) in KEPT[category].items():
        path = [top, leaf]
        for i in range(count):  # in the band, with the attribute
            rows.append(_row(asin(), path, "%s %d" % (leaf, i), price * (0.9 + 0.01 * (i % 20)), 3.0 + (i % 5) / 2,
                             60 + 7 * i, {key: value, "Brand": "B%d" % (i % 4)}))
        for i in range(OTHER_CASE):
            rows.append(_row(asin(), path, "%s case %d" % (leaf, i), price, 4.0, 80, {key: value.upper()}))
        for i in range(OTHER_VALUE):
            rows.append(_row(asin(), path, "%s other %d" % (leaf, i), price, 4.0, 80, {key: "Other"}))
        for i in range(OUTSIDE_BAND):
            rows.append(_row(asin(), path, "%s dear %d" % (leaf, i), price * 3, 4.0, 90, {key: value}))
        for i in range(MISSING_PRICE):
            rows.append(_row(asin(), path, "%s noprice %d" % (leaf, i), None, 4.0, 70, {key: value}))
        for i in range(MISSING_TITLE):
            rows.append(_row(asin(), path, None, price, 4.0, 70, {key: value}))
        for i in range(MISSING_RATING):
            rows.append(_row(asin(), path, "%s norating %d" % (leaf, i), price, None, 70, {key: value}))
        for i in range(MISSING_ATTRIBUTE):
            rows.append(_row(asin(), path, "%s noattr %d" % (leaf, i), price, 4.0, 70, {"Brand": "Z"}))
        rows.append(_row(asin(), path, "%s few ratings" % leaf, price, 5.0, 49, {key: value}))
    for i in range(35):
        rows.append(_row(asin(), [top, BAND_ONLY[category]], "band only %d" % i, 9.0, 4.0, 100, {}))
    for i in range(12):
        rows.append(_row(asin(), [top, SPARSE[category]], "sparse %d" % i, 5.0, 4.0, 100, {"Material": "Metal"}))
    rows.append(_row(None, [top, "Headless"], "no asin", 5.0, 4.0, 100, {}))
    rows.append(_row(asin(), [top, "Uncounted"], "no count", 5.0, 4.0, None, {}))
    rows.append(dict(rows[0]))  # a duplicate parent_asin
    return rows


# Kept leaf -> expected (leaf products stored, excluded, missing counts, outside requirements, pool).
def expected(category: str, leaf: str) -> dict:
    key, _, count, _ = KEPT[category][leaf]
    stored = count + OTHER_CASE + OTHER_VALUE + OUTSIDE_BAND + MISSING_PRICE + MISSING_TITLE + MISSING_RATING \
        + MISSING_ATTRIBUTE
    return {"leaf_products": stored,
            "excluded_products": MISSING_PRICE + MISSING_TITLE + MISSING_RATING + MISSING_ATTRIBUTE,
            "excluded_missing": dict(sorted({"price": MISSING_PRICE, "title": MISSING_TITLE,
                                             "average_rating": MISSING_RATING,
                                             "attribute:%s" % key: MISSING_ATTRIBUTE}.items())),
            "failed_requirements": OTHER_VALUE + OUTSIDE_BAND,
            "pool_size": count + OTHER_CASE}


def products(category: str) -> list[Product]:
    """The stored products of ``category`` (as the fetch would keep them)."""
    seen, out = set(), []
    for row in raw_rows(category):
        p = normalise(row, category)
        if p is None or p.rating_number is None or p.rating_number < 50 or p.product_id in seen:
            continue
        seen.add(p.product_id)
        out.append(p)
    return out
