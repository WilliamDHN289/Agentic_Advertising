"""SL-05 renderers: golden snapshots, template digests and mode/backend invariance (AC-4).

Golden files are in fixtures/render/. After an intended template change, bump the
template id, then rewrite them with

    cd ai-agents && STORELAB_WRITE_GOLDEN=1 python -B -m unittest tests.features.storelab.test_render
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from features.storelab.config_files import read_config_data
from features.storelab.core.catalog_amazon import Product
from features.storelab.core.config import apply_overrides, resolve_config
from features.storelab.core.design import Presentation, build_design
from features.storelab.core.protocol import Candidate
from features.storelab.core.render import (TEMPLATES, build_observation, page_model, render_structured, render_text,
                                           render_views, template_sha256)
from features.storelab.core.tasks import Requirements, TaskSpec, instruction

CONFIGS = Path(__file__).resolve().parents[3] / "features" / "storelab" / "configs"
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "render"
PINNED_TEMPLATES = {
    "e1_text@1": "b268d7bb2cb174810b3c39eae2d7105986d49661dff95d0c5a1936576bdd6437",
    "e1_structured@1": "f34c9b579c5825300bed3d28a574790a8223ebe4ea56834176ff605416af1926",
    "e1_browser_tool@1": "541ca8e9e4f322c120c6aead3210593807df55a5606a444603cb998ed7735541",
}

REQUIREMENTS = Requirements(price_min=14.0, price_max=26.0, attributes={"Material": "Glass"})
TASK = TaskSpec(task_id="home-and-kitchen-1", category="Home_and_Kitchen", leaf_path=["Home & Kitchen", "Mugs"],
                template="band_attribute", requirements=REQUIREMENTS, instruction=instruction("Mugs", REQUIREMENTS))
PRODUCTS = {p.product_id: p for p in (
    Product(product_id="B0AAA00001", category="Home_and_Kitchen", category_path=["Home & Kitchen", "Mugs"],
            title='Irish Coffee Mug <Set of 4> & "Footed"', price=24.95, average_rating=4.6, rating_number=1234,
            description="Clear glass, 10.5 oz. Thick walls for durability; dishwasher safe. Made for café drinks.",
            attributes={"Brand": "LavoHome", "Material": "Glass", "Color": "Clear"}),
    Product(product_id="B0AAA00002", category="Home_and_Kitchen", category_path=["Home & Kitchen", "Mugs"],
            title="Double-wall mug", price=14.0, average_rating=4.0, rating_number=50, description="",
            attributes={"Material": "glass"}),
    Product(product_id="B0AAA00003", category="Home_and_Kitchen", category_path=["Home & Kitchen", "Mugs"],
            title="Espresso cups, 2 pack", price=1234.5, average_rating=3.25, rating_number=98765,
            description="Small cups.", attributes={"Material": "Glass", "Style": "Modern"}),
)}
PRESENTATION = Presentation(presentation_id="home-and-kitchen-1/k3/s000/p0", set_id="home-and-kitchen-1/k3/s000",
                            draw=0, candidates=[
                                Candidate(product_id="B0AAA00003", position=1, sponsored_label=True),
                                Candidate(product_id="B0AAA00001", position=2, sponsored_label=False),
                                Candidate(product_id="B0AAA00002", position=3, sponsored_label=True)])


def rendered() -> dict[str, str]:
    pages = page_model(TASK, PRESENTATION, PRODUCTS)
    return {"text.txt": render_text(TASK, PRESENTATION, PRODUCTS),
            "structured.json": render_structured(TASK, PRESENTATION, PRODUCTS),
            "serp.html": pages.serp_html, "serp.txt": pages.serp_text,
            "page_B0AAA00001.html": pages.pages["B0AAA00001"].html,
            "page_B0AAA00001.txt": pages.pages["B0AAA00001"].text,
            "page_B0AAA00002.txt": pages.pages["B0AAA00002"].text}


class GoldenTest(unittest.TestCase):
    def test_renderers_match_the_golden_files(self):
        outputs = rendered()
        if os.environ.get("STORELAB_WRITE_GOLDEN") == "1":
            GOLDEN.mkdir(parents=True, exist_ok=True)
            for name, text in outputs.items():
                (GOLDEN / name).write_bytes(text.encode("utf-8"))
        self.assertEqual(sorted(p.name for p in GOLDEN.iterdir()), sorted(outputs))
        for name, text in outputs.items():
            with self.subTest(name=name):
                self.assertEqual(text.encode("utf-8"), (GOLDEN / name).read_bytes())

    def test_template_text_is_pinned_to_its_id(self):
        self.assertEqual(set(TEMPLATES), set(PINNED_TEMPLATES))
        for template_id, digest in PINNED_TEMPLATES.items():
            with self.subTest(template=template_id):
                self.assertEqual(template_sha256(template_id), digest)

    def test_text_is_escaped_in_html_only(self):
        outputs = rendered()
        self.assertIn('Irish Coffee Mug <Set of 4> & "Footed"', outputs["text.txt"])
        self.assertIn("Irish Coffee Mug &lt;Set of 4&gt; &amp; &quot;Footed&quot;", outputs["serp.html"])
        self.assertNotIn("<Set", outputs["serp.html"])
        self.assertIn("[Sponsored]", outputs["serp.txt"])

    def test_templates_must_exist_and_match_the_interface(self):
        with self.assertRaisesRegex(ValueError, "unknown prompt template"):
            render_views(TASK, PRESENTATION, PRODUCTS, {"text": "e1_text@2"})
        with self.assertRaisesRegex(ValueError, "renders interface structured"):
            render_views(TASK, PRESENTATION, PRODUCTS, {"text": "e1_structured@1"})
        broken = {**PRODUCTS, "B0AAA00002": PRODUCTS["B0AAA00002"].model_copy(update={"price": None})}
        with self.assertRaisesRegex(ValueError, "lacks"):
            render_text(TASK, PRESENTATION, broken)


class ModeInvarianceTest(unittest.TestCase):
    """AC-4: the same (set, presentation) renders byte-identically in every mode and backend."""

    def configs(self):
        base = read_config_data(CONFIGS / "e1_pilot.yaml")
        ollama = {"agents.offline_backend": "local_llm", "mode": "offline",
                  "agents.models": [{"key": "qwen2_7b", "provider": "ollama", "model": "qwen2:7b"}]}
        variants = {"auto/simulated": {}, "offline/simulated": {"mode": "offline"},
                    "sandbox/providers": {"mode": "sandbox"},
                    "replay": {"mode": "replay", "source_run_id": "1b4e28ba-2fa1-4d2e-883f-0016d3cca427",
                               "cache.policy": "read_only"},
                    "offline/local_llm": ollama,
                    "cache off": {"cache.policy": "off"}, "other budget": {"budget.max_usd": 1}}
        return {name: resolve_config(apply_overrides(base, change)) for name, change in variants.items()}

    def test_observations_are_identical_across_modes_and_backends(self):
        configs = self.configs()
        self.assertEqual(len({c.config_hash for c in configs.values()}), len(configs) - 1)  # budget is not hashed
        designs = [build_design(TASK.task_id, list(PRODUCTS), k, 10, 0.3, "per_set", 3) for k in (2, 3)]
        presentations = [PRESENTATION] + [p for d in designs for p in d.presentations]
        for presentation in presentations:
            observations = {name: build_observation(c, TASK, presentation, PRODUCTS) for name, c in configs.items()}
            hashes = {o.observation_hash() for o in observations.values()}
            with self.subTest(presentation=presentation.presentation_id):
                self.assertEqual(len(hashes), 1)
                views = [o.views for o in observations.values()]
                self.assertTrue(all(v == views[0] for v in views))
                self.assertEqual(sorted(views[0]), ["browser_tool", "structured", "text"])

    def test_the_description_length_is_the_only_config_input_besides_templates(self):
        base = read_config_data(CONFIGS / "e1_pilot.yaml")
        short = resolve_config(apply_overrides(base, {"environment.description_chars": 20}))
        full = resolve_config(base)
        self.assertNotEqual(build_observation(short, TASK, PRESENTATION, PRODUCTS).observation_hash(),
                            build_observation(full, TASK, PRESENTATION, PRODUCTS).observation_hash())
        self.assertIn("Clear glass, 10.5 oz", build_observation(short, TASK, PRESENTATION, PRODUCTS).views["text"])
        self.assertNotIn("Thick walls", build_observation(short, TASK, PRESENTATION, PRODUCTS).views["text"])


if __name__ == "__main__":
    unittest.main()
