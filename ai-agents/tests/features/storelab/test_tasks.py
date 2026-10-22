"""SL-05 tasks: screening, requirements, eligible sets and the utility u (AC-5)."""
from __future__ import annotations

import math
import unittest

from pydantic import ValidationError

from features.storelab.core.catalog_amazon import Product
from features.storelab.core.tasks import (MIN_ELIGIBLE, PRIOR_RATINGS, Requirements, TaskStimuli, instruction,
                                          screen, screen_leaf, shrunk_utility, task_stimuli)
from tests.features.storelab import e1_fixture

CATEGORIES = ["Electronics", "Home_and_Kitchen"]
CATALOG = [p for c in CATEGORIES for p in e1_fixture.products(c)]
SCREENING = screen(CATALOG, CATEGORIES, 2)


class ScreeningTest(unittest.TestCase):
    def test_the_densest_kept_leaves_give_one_attribute_task_each(self):
        chosen = [(t.task_id, t.leaf_path[-1], list(t.requirements.attributes.items())) for t in SCREENING.tasks]
        self.assertEqual(chosen, [
            ("electronics-1", "Headphones", [("Connectivity Technology", "Bluetooth")]),
            ("electronics-2", "Chargers", [("Power Source", "USB")]),
            ("home-and-kitchen-1", "Mugs", [("Material", "Glass")]),
            ("home-and-kitchen-2", "Kettles", [("Material", "Stainless Steel")]),
        ])
        for task in SCREENING.tasks:
            self.assertEqual(task.instruction, instruction(task.leaf_path[-1], task.requirements))

    def test_leaves_below_the_floors_are_not_kept(self):
        for category in SCREENING.categories:
            leaves = {c.leaf_path[-1]: c for c in category.candidates}
            band_only = e1_fixture.BAND_ONLY[category.category]
            with self.subTest(category=category.category):
                self.assertEqual((leaves[band_only].density, leaves[band_only].supported, leaves[band_only].kept),
                                 (35, 1, False))
                self.assertNotIn(e1_fixture.SPARSE[category.category], leaves)  # density 12 < MIN_ELIGIBLE
                self.assertTrue(all(c.density >= MIN_ELIGIBLE for c in category.candidates))
                self.assertEqual(category.leaves_kept, 2)
                self.assertEqual([c.density for c in category.candidates],
                                 sorted((c.density for c in category.candidates), reverse=True))

    def test_the_band_is_thirty_percent_around_the_median_complete_price(self):
        leaf = next(c for c in SCREENING.categories[0].candidates if c.leaf_path[-1] == "Headphones")
        products = [p for p in CATALOG if p.category_path == ["Electronics", "Headphones"]]
        complete = sorted(p.price for p in products if not p.missing())
        median = (complete[len(complete) // 2 - 1] + complete[len(complete) // 2]) / 2 if len(complete) % 2 == 0 \
            else complete[len(complete) // 2]
        self.assertAlmostEqual(leaf.reference_price, median)
        self.assertEqual((leaf.price_min, leaf.price_max), (round(0.7 * median, 2), round(1.3 * median, 2)))
        self.assertEqual(leaf.density, sum(leaf.price_min <= p <= leaf.price_max for p in complete))
        self.assertEqual(screen_leaf("Electronics", ["X"], [p.model_copy(update={"price": None}) for p in products]),
                         None)

    def test_too_few_kept_leaves_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "2 leaves pass screening"):
            screen(CATALOG, CATEGORIES, 3)
        with self.assertRaisesRegex(ValueError, "no products"):
            screen(CATALOG, ["Books"], 1)


class RequirementsTest(unittest.TestCase):
    def test_matching_and_missing_fields(self):
        req = Requirements(price_min=10, price_max=20, attributes={"Material": "Glass"})
        base = dict(product_id="B", category="Home_and_Kitchen", category_path=["H", "M"], title="t", price=15.0,
                    average_rating=4.0, rating_number=60, attributes={"Material": " glass"})
        self.assertTrue(req.satisfied_by(Product(**base)))
        for change, missing in (({"price": None}, ["price"]), ({"attributes": {}}, ["attribute:Material"]),
                                ({"title": None, "average_rating": None}, ["title", "average_rating"])):
            with self.subTest(change=change):
                self.assertEqual(req.missing(Product(**{**base, **change})), missing)
                self.assertFalse(req.satisfied_by(Product(**{**base, **change})))
        for change in ({"price": 20.01}, {"price": 9.99}, {"attributes": {"Material": "Glas"}}):
            self.assertFalse(req.satisfied_by(Product(**{**base, **change})))
        self.assertTrue(req.satisfied_by(Product(**{**base, "price": 20.0})))
        with self.assertRaises(ValidationError):
            Requirements(price_min=5, price_max=4)
        self.assertEqual(instruction("Mugs", req), 'Find a product in the category "Mugs" priced between $10.00 and '
                                                   '$20.00 with Material: Glass. Choose at most one product; choose '
                                                   'none if no product fits.')


class StimuliTest(unittest.TestCase):
    def test_exclusions_are_counted_and_u_is_defined_for_every_eligible_product(self):
        """AC-5."""
        for task in SCREENING.tasks:
            stimuli = task_stimuli(task, CATALOG, 10, seed=11)
            expected = e1_fixture.expected(task.category, task.leaf_path[-1])
            with self.subTest(task=task.task_id):
                self.assertEqual({k: getattr(stimuli, k) for k in expected}, expected)
                self.assertEqual(len(stimuli.eligible_ids), 10)
                self.assertEqual(sorted(stimuli.utility), stimuli.eligible_ids)
                self.assertTrue(all(math.isfinite(u) for u in stimuli.utility.values()))
                by_id = {p.product_id: p for p in CATALOG}
                chosen = [by_id[i] for i in stimuli.eligible_ids]
                self.assertTrue(all(task.eligible(p) and not p.missing() for p in chosen))
                mean = sum(p.average_rating for p in chosen) / len(chosen)
                self.assertAlmostEqual(stimuli.mean_rating, mean)
                for p in chosen:
                    self.assertAlmostEqual(stimuli.utility[p.product_id], (p.rating_number * p.average_rating
                                                                           + PRIOR_RATINGS * mean)
                                           / (p.rating_number + PRIOR_RATINGS))

    def test_the_eligible_draw_depends_on_the_seed_only(self):
        task = SCREENING.tasks[0]
        a = task_stimuli(task, CATALOG, 10, seed=11)
        self.assertEqual(task_stimuli(task, list(reversed(CATALOG)), 10, seed=11), a)
        self.assertNotEqual(task_stimuli(task, CATALOG, 10, seed=12).eligible_ids, a.eligible_ids)
        with self.assertRaisesRegex(ValueError, "eligible_size 47"):
            task_stimuli(task, CATALOG, 47, seed=11)

    def test_utility_must_cover_exactly_the_eligible_set(self):
        stimuli = task_stimuli(SCREENING.tasks[0], CATALOG, 10, seed=11)
        data = stimuli.model_dump()
        data["utility"].popitem()
        with self.assertRaises(ValidationError):
            TaskStimuli.model_validate(data)

    def test_shrinkage(self):
        products = [Product(product_id=i, category="c", title="t", price=1.0, average_rating=r, rating_number=n)
                    for i, r, n in (("a", 5.0, 0), ("b", 3.0, 10_000))]
        mean, u = shrunk_utility(products)
        self.assertEqual(mean, 4.0)
        self.assertEqual(u["a"], 4.0)  # no ratings: the prior mean
        self.assertAlmostEqual(u["b"], (30_000 + 4.0 * PRIOR_RATINGS) / (10_000 + PRIOR_RATINGS))


if __name__ == "__main__":
    unittest.main()
