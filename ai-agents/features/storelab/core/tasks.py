"""E1 tasks (SL-05): category screening, tasks with explicit requirements, eligible
sets and the ground-truth utility u(x, j). Rules: docs/e1_environment.md.

Screening (SCREENING_VERSION). A leaf is a full category path of the snapshot, whose
products all have at least MIN_RATINGS ratings (core/catalog_amazon.py). A product
is complete when it has every REQUIRED_FIELDS value. The leaf's reference price p is
the median price of its complete products; its price band is [0.7 p, 1.3 p] rounded
to cents; its substitute density is the number of complete products in the band.
Each leaf offers the task templates ``band`` (price in the band) and, for every key
of REQUIREMENT_KEYS, ``band_attribute`` (in the band and that attribute equal to its
most frequent value among band products). A template supports a task when at least
MIN_ELIGIBLE products are eligible for it, and a leaf is kept when it supports at
least MIN_TASKS_PER_LEAF tasks. Per category, kept leaves are ranked by density
(then path) and the first ``tasks_per_category`` each give one task: their
``band_attribute`` template with the most eligible products, so every task states a
price range and one required attribute.

Eligibility. A product is eligible for a task when it is in the task's leaf, is
complete, has the required attribute and satisfies every requirement. Products of the
leaf lacking a required field (an attribute named by the task included) are excluded
and counted by field. The eligible set E(x) is ``eligible_size`` products drawn
without replacement from the eligible pool with ``stream(seed, "eligible", task_id)``.

Utility (UTILITY_DEFINITION). u(x, j) = (n_j r_j + m r̄_x) / (n_j + m): the average
rating r_j shrunk towards the mean rating r̄_x of E(x) with prior weight m =
PRIOR_RATINGS ratings, n_j the rating count. It is defined for every j in E(x), since
eligible products are complete.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Annotated, Iterable, Literal

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt, StringConstraints, model_validator

from .catalog_amazon import MIN_RATINGS, REQUIRED_FIELDS, Product
from .config import NonEmptyStr, Section
from .protocol import ProductId
from .rng import stream

SCREENING_VERSION = "e1_screening@1"
INSTRUCTION_TEMPLATE = "e1_task_instruction@1"
UTILITY_DEFINITION = "shrunk_rating@1"
PRIOR_RATINGS = 20.0  # m
PRICE_BAND = 0.30
MIN_ELIGIBLE = 30
MIN_TASKS_PER_LEAF = 2
# Attributes a task may require: product properties a shopper states, not the brand.
REQUIREMENT_KEYS = ("Material", "Color", "Style", "Shape", "Pattern", "Finish Type", "Connectivity Technology",
                    "Form Factor", "Power Source")

TaskId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=64)]


def criteria() -> dict:
    """The registered screening and utility constants, recorded with every build."""
    return {"screening": SCREENING_VERSION, "instruction_template": INSTRUCTION_TEMPLATE,
            "utility": UTILITY_DEFINITION, "prior_ratings": PRIOR_RATINGS, "price_band": PRICE_BAND,
            "min_ratings": MIN_RATINGS, "min_eligible": MIN_ELIGIBLE, "min_tasks_per_leaf": MIN_TASKS_PER_LEAF,
            "required_fields": list(REQUIRED_FIELDS), "requirement_keys": list(REQUIREMENT_KEYS)}


def _fold(text: str) -> str:
    return " ".join(text.split()).casefold()


class Requirements(Section):
    price_min: PositiveFloat  # USD, inclusive
    price_max: PositiveFloat  # USD, inclusive: the price ceiling
    attributes: dict[NonEmptyStr, NonEmptyStr] = {}  # key -> value, compared case-insensitively

    @model_validator(mode="after")
    def _range(self) -> Requirements:
        if self.price_min > self.price_max:
            raise ValueError("price_min exceeds price_max")
        return self

    def missing(self, product: Product) -> list[str]:
        """Fields the requirements need that ``product`` lacks."""
        return product.missing() + ["attribute:%s" % k for k in self.attributes if k not in product.attributes]

    def satisfied_by(self, product: Product) -> bool:
        """True if ``product`` has every needed field and meets every requirement."""
        if self.missing(product):
            return False
        return (self.price_min <= product.price <= self.price_max
                and all(_fold(product.attributes[k]) == _fold(v) for k, v in self.attributes.items()))


def instruction(leaf: str, requirements: Requirements) -> str:
    """The task instruction (INSTRUCTION_TEMPLATE)."""
    text = 'Find a product in the category "%s" priced between $%.2f and $%.2f' % (
        leaf, requirements.price_min, requirements.price_max)
    if requirements.attributes:
        text += " with " + " and ".join("%s: %s" % kv for kv in requirements.attributes.items())
    return text + ". Choose at most one product; choose none if no product fits."


class TaskSpec(Section):
    task_id: TaskId
    category: NonEmptyStr
    leaf_path: list[NonEmptyStr] = Field(min_length=1)
    template: Literal["band", "band_attribute"]
    requirements: Requirements
    instruction: NonEmptyStr

    def in_leaf(self, product: Product) -> bool:
        return product.category == self.category and product.category_path == self.leaf_path

    def eligible(self, product: Product) -> bool:
        return self.in_leaf(product) and self.requirements.satisfied_by(product)


class LeafScreen(Section):
    category: NonEmptyStr
    leaf_path: list[NonEmptyStr]
    products: NonNegativeInt
    complete: NonNegativeInt
    reference_price: PositiveFloat
    price_min: PositiveFloat
    price_max: PositiveFloat
    density: NonNegativeInt  # complete products in the price band
    templates: dict[NonEmptyStr, NonNegativeInt]  # "band" or "band_attribute:<key>" -> eligible products
    attribute_values: dict[NonEmptyStr, NonEmptyStr] = {}  # key -> the most frequent value in the band
    supported: NonNegativeInt  # templates with >= MIN_ELIGIBLE eligible products
    kept: bool


class CategoryScreen(Section):
    category: NonEmptyStr
    products: NonNegativeInt
    leaves: NonNegativeInt
    leaves_kept: NonNegativeInt
    # Leaves whose density reaches MIN_ELIGIBLE (only these can be kept), ranked.
    candidates: list[LeafScreen]


class Screening(Section):
    criteria: dict
    categories: list[CategoryScreen]
    tasks: list[TaskSpec]


class TaskStimuli(Section):
    """A task with its eligible set E(x) and u(x, j) for every j in it."""

    task: TaskSpec
    leaf_products: NonNegativeInt
    excluded_missing: dict[NonEmptyStr, NonNegativeInt]  # field -> products of the leaf lacking it
    excluded_products: NonNegativeInt  # products of the leaf lacking at least one needed field
    failed_requirements: NonNegativeInt  # complete products of the leaf outside the requirements
    pool_size: PositiveInt  # eligible products
    eligible_ids: list[ProductId] = Field(min_length=1)  # E(x), sorted
    utility_definition: Literal["shrunk_rating@1"] = UTILITY_DEFINITION
    prior_ratings: PositiveFloat = PRIOR_RATINGS
    mean_rating: float  # r̄_x over E(x)
    utility: dict[ProductId, float]

    @model_validator(mode="after")
    def _utility_for_every_eligible_product(self) -> TaskStimuli:
        if sorted(self.utility) != self.eligible_ids or self.eligible_ids != sorted(set(self.eligible_ids)):
            raise ValueError("utility must be defined for exactly the sorted, distinct eligible_ids")
        return self


# ------------------------------------------------------------------- screening
def _band(prices: list[float]) -> tuple[float, float, float]:
    reference = statistics.median(prices)
    return reference, round(reference * (1 - PRICE_BAND), 2), round(reference * (1 + PRICE_BAND), 2)


def screen_leaf(category: str, path: list[str], products: list[Product]) -> LeafScreen | None:
    """Screen one leaf; ``None`` if it has no complete product."""
    complete = [p for p in products if not p.missing()]
    if not complete:
        return None
    reference, low, high = _band([p.price for p in complete])
    band = [p for p in complete if low <= p.price <= high]
    templates, values = {"band": len(band)}, {}
    for key in REQUIREMENT_KEYS:
        spellings = Counter(p.attributes[key] for p in band if key in p.attributes)
        if not spellings:
            continue
        folded = Counter()
        for spelling, n in spellings.items():
            folded[_fold(spelling)] += n
        top = min(folded, key=lambda v: (-folded[v], v))
        values[key] = min((s for s in spellings if _fold(s) == top), key=lambda s: (-spellings[s], s))
        templates["band_attribute:%s" % key] = folded[top]
    supported = sum(n >= MIN_ELIGIBLE for n in templates.values())
    return LeafScreen(category=category, leaf_path=path, products=len(products), complete=len(complete),
                      reference_price=reference, price_min=low, price_max=high, density=len(band),
                      templates=templates, attribute_values=values, supported=supported,
                      kept=supported >= MIN_TASKS_PER_LEAF)


def _task_for(leaf: LeafScreen, task_id: str) -> TaskSpec:
    options = [(n, REQUIREMENT_KEYS.index(name.split(":", 1)[1]), name.split(":", 1)[1])
               for name, n in leaf.templates.items() if name.startswith("band_attribute:") and n >= MIN_ELIGIBLE]
    _, _, key = max(options, key=lambda o: (o[0], -o[1]))
    requirements = Requirements(price_min=leaf.price_min, price_max=leaf.price_max,
                                attributes={key: leaf.attribute_values[key]})
    return TaskSpec(task_id=task_id, category=leaf.category, leaf_path=leaf.leaf_path, template="band_attribute",
                    requirements=requirements, instruction=instruction(leaf.leaf_path[-1], requirements))


def task_id_for(category: str, index: int) -> str:
    return "%s-%d" % (category.lower().replace("_", "-"), index)


def screen(products: Iterable[Product], categories: list[str], tasks_per_category: int) -> Screening:
    """Screen the leaves of ``categories`` and choose ``tasks_per_category`` tasks in each."""
    leaves: dict[tuple[str, tuple[str, ...]], list[Product]] = defaultdict(list)
    for product in products:
        if product.category in categories:
            leaves[product.category, tuple(product.category_path)].append(product)
    screens, tasks = [], []
    for category in categories:
        own = {path: items for (c, path), items in leaves.items() if c == category}
        if not own:
            raise ValueError("category %s has no products in the snapshot" % category)
        results = [screen_leaf(category, list(path), items) for path, items in own.items() if path]
        results = [r for r in results if r is not None]
        candidates = sorted((r for r in results if r.density >= MIN_ELIGIBLE),
                            key=lambda r: (-r.density, r.leaf_path))
        kept = [r for r in candidates if r.kept]
        if len(kept) < tasks_per_category:
            raise ValueError("category %s: %d leaves pass screening %s, %d tasks requested"
                             % (category, len(kept), SCREENING_VERSION, tasks_per_category))
        tasks += [_task_for(leaf, task_id_for(category, i + 1)) for i, leaf in enumerate(kept[:tasks_per_category])]
        screens.append(CategoryScreen(category=category, products=sum(map(len, own.values())), leaves=len(own),
                                      leaves_kept=len(kept), candidates=candidates))
    return Screening(criteria=criteria(), categories=screens, tasks=tasks)


# ------------------------------------------------------------ eligible sets, u
def shrunk_utility(products: list[Product], prior_ratings: float = PRIOR_RATINGS) -> tuple[float, dict[str, float]]:
    """(r̄, u) with u_j = (n_j r_j + m r̄) / (n_j + m) and r̄ the mean rating of ``products``."""
    if not products:
        raise ValueError("no products")
    mean = statistics.fmean(p.average_rating for p in products)
    return mean, {p.product_id: (p.rating_number * p.average_rating + prior_ratings * mean)
                  / (p.rating_number + prior_ratings) for p in products}


def task_stimuli(task: TaskSpec, products: Iterable[Product], eligible_size: int, seed: int) -> TaskStimuli:
    """E(x) and u for ``task``; ``products`` must include every product of its leaf."""
    leaf = [p for p in products if task.in_leaf(p)]
    missing: Counter = Counter()
    excluded = failed = 0
    pool = []
    for product in leaf:
        lacks = task.requirements.missing(product)
        if lacks:
            excluded += 1
            missing.update(lacks)
        elif task.requirements.satisfied_by(product):
            pool.append(product)
        else:
            failed += 1
    if len(pool) < eligible_size:
        raise ValueError("task %s: %d eligible products, eligible_size %d" % (task.task_id, len(pool), eligible_size))
    pool.sort(key=lambda p: p.product_id)
    picks = stream(seed, "eligible", task.task_id).choice(len(pool), size=eligible_size, replace=False)
    chosen = sorted((pool[int(i)] for i in picks), key=lambda p: p.product_id)
    mean, utility = shrunk_utility(chosen)
    return TaskStimuli(task=task, leaf_products=len(leaf), excluded_missing=dict(sorted(missing.items())),
                       excluded_products=excluded, failed_requirements=failed, pool_size=len(pool),
                       eligible_ids=[p.product_id for p in chosen], mean_rating=mean, utility=utility)
