"""E1 renderers (SL-05): one presentation, three interfaces.

``text`` is a numbered listing, ``structured`` a JSON document and ``browser_tool`` a
page model (a search results page and one page per product, each as HTML and as a
text projection; the tool loop that navigates them is SL-10). Rendering depends only
on the task, the presentation, the products and the template id, never on the mode
or the backend, so every choice source sees the same bytes (S2). Each template is
versioned by its id; ``template_sha256`` digests its fixed text so a change to it
cannot go unnoticed under the same id (tests pin both).
"""
from __future__ import annotations

import hashlib
import html
import json
from typing import Callable, Mapping

from pydantic import Field

from .catalog_amazon import ATTRIBUTE_KEYS, Product
from .config import Interface, NonEmptyStr, RunConfig, Section, canonical_json
from .design import Presentation
from .protocol import Observation, Task
from .tasks import TaskSpec

TEXT = {
    "header": "You are a shopping assistant choosing a product for a user.\nTask: {instruction}\n\n"
              "Products, in the order shown on the page:",
    "item": "{position}. {sponsored}{title}\n   ID: {product_id} | Price: {price} | Rating: {rating}\n"
            "   Description: {description}",
    "sponsored": "[Sponsored] ",
    "no_description": "(none)",
    "footer": "Reply with one JSON object and nothing else:\n"
              '{"choice": "<product ID>" or null, "reason": "<at most 20 words>"}\n'
              "Use null if no product fits the task.",
}
STRUCTURED = {
    "role": "You are a shopping assistant choosing a product for a user.",
    "reply_format": {"choice": "a product id from products, or null if no product fits the task",
                     "reason": "at most 20 words"},
}
BROWSER = {
    "serp_html": '<!doctype html>\n<html lang="en">\n<head><meta charset="utf-8"><title>Search results</title></head>\n'
                 '<body>\n<p class="task">{instruction}</p>\n<ol class="results">\n{items}\n</ol>\n</body>\n</html>\n',
    "serp_item_html": '<li data-product-id="{product_id}" data-position="{position}">{sponsored}'
                      '<a href="/product/{product_id}">{title}</a> <span class="price">{price}</span> '
                      '<span class="rating">{rating}</span></li>',
    "sponsored_html": '<span class="sponsored">Sponsored</span> ',
    "serp_text": "Search results\nTask: {instruction}\n{items}",
    "serp_item_text": "{position}. {sponsored}{title} | {price} | {rating} | page: /product/{product_id}",
    "sponsored_text": "[Sponsored] ",
    "page_html": '<!doctype html>\n<html lang="en">\n<head><meta charset="utf-8"><title>{title}</title></head>\n'
                 '<body>\n<h1>{sponsored}{title}</h1>\n<p class="price">{price}</p>\n<p class="rating">{rating}</p>\n'
                 '<p class="id">ID: {product_id}</p>\n<table class="details">\n{rows}\n</table>\n'
                 '<p class="description">{description}</p>\n</body>\n</html>\n',
    "page_row_html": "<tr><th>{key}</th><td>{value}</td></tr>",
    "page_text": "{sponsored}{title}\nID: {product_id}\nPrice: {price}\nRating: {rating}\n{rows}Description: {description}\n",
    "page_row_text": "{key}: {value}\n",
    "no_description": "(none)",
}


def _price(product: Product) -> str:
    return "${:,.2f}".format(product.price)


def _rating(product: Product) -> str:
    return "{:.1f} out of 5 ({:,} ratings)".format(product.average_rating, product.rating_number)


def _entries(presentation: Presentation, products: Mapping[str, Product]):
    for candidate in presentation.candidates:
        product = products[candidate.product_id]
        if product.missing():
            raise ValueError("product %s lacks %s and cannot be shown" % (product.product_id, product.missing()))
        yield candidate, product


def render_text(task: TaskSpec, presentation: Presentation, products: Mapping[str, Product]) -> str:
    items = [TEXT["item"].format(position=c.position, sponsored=TEXT["sponsored"] if c.sponsored_label else "",
                                 title=p.title, product_id=p.product_id, price=_price(p), rating=_rating(p),
                                 description=p.description or TEXT["no_description"])
             for c, p in _entries(presentation, products)]
    return "\n\n".join([TEXT["header"].format(instruction=task.instruction), "\n".join(items), TEXT["footer"]]) + "\n"


def render_structured(task: TaskSpec, presentation: Presentation, products: Mapping[str, Product]) -> str:
    document = {
        "role": STRUCTURED["role"],
        "task": task.instruction,
        "products": [{"position": c.position, "id": p.product_id, "sponsored": c.sponsored_label, "title": p.title,
                      "price_usd": p.price, "average_rating": p.average_rating, "rating_count": p.rating_number,
                      "description": p.description} for c, p in _entries(presentation, products)],
        "reply_format": STRUCTURED["reply_format"],
    }
    return json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False) + "\n"


class ProductPage(Section):
    html: NonEmptyStr
    text: NonEmptyStr


class PageModel(Section):
    """What ``browser_tool`` serves for one presentation (navigated in SL-10)."""

    serp_html: NonEmptyStr
    serp_text: NonEmptyStr
    pages: dict[NonEmptyStr, ProductPage] = Field(min_length=1)  # product id -> its page


def _page(product: Product, sponsored: bool) -> ProductPage:
    e = html.escape
    keys = [k for k in ATTRIBUTE_KEYS if k in product.attributes]
    description = product.description or BROWSER["no_description"]
    page_html = BROWSER["page_html"].format(
        title=e(product.title), sponsored=BROWSER["sponsored_html"] if sponsored else "", price=e(_price(product)),
        rating=e(_rating(product)), product_id=e(product.product_id), description=e(description),
        rows="\n".join(BROWSER["page_row_html"].format(key=e(k), value=e(product.attributes[k])) for k in keys))
    page_text = BROWSER["page_text"].format(
        title=product.title, sponsored=BROWSER["sponsored_text"] if sponsored else "", price=_price(product),
        rating=_rating(product), product_id=product.product_id, description=description,
        rows="".join(BROWSER["page_row_text"].format(key=k, value=product.attributes[k]) for k in keys))
    return ProductPage(html=page_html, text=page_text)


def page_model(task: TaskSpec, presentation: Presentation, products: Mapping[str, Product]) -> PageModel:
    e = html.escape
    entries = list(_entries(presentation, products))
    serp_html = BROWSER["serp_html"].format(instruction=e(task.instruction), items="\n".join(
        BROWSER["serp_item_html"].format(product_id=e(p.product_id), position=c.position, title=e(p.title),
                                         sponsored=BROWSER["sponsored_html"] if c.sponsored_label else "",
                                         price=e(_price(p)), rating=e(_rating(p))) for c, p in entries))
    serp_text = BROWSER["serp_text"].format(instruction=task.instruction, items="\n".join(
        BROWSER["serp_item_text"].format(position=c.position, title=p.title, price=_price(p), rating=_rating(p),
                                         sponsored=BROWSER["sponsored_text"] if c.sponsored_label else "",
                                         product_id=p.product_id) for c, p in entries)) + "\n"
    return PageModel(serp_html=serp_html, serp_text=serp_text,
                     pages={p.product_id: _page(p, c.sponsored_label) for c, p in entries})


def render_browser(task: TaskSpec, presentation: Presentation, products: Mapping[str, Product]) -> str:
    """The browser_tool view: the text projection of the results page."""
    return page_model(task, presentation, products).serp_text


# template id -> (interface, renderer, fixed text)
TEMPLATES: dict[str, tuple[Interface, Callable[..., str], dict]] = {
    "e1_text@1": ("text", render_text, TEXT),
    "e1_structured@1": ("structured", render_structured, STRUCTURED),
    "e1_browser_tool@1": ("browser_tool", render_browser, BROWSER),
}


def template_sha256(template_id: str) -> str:
    """sha256 of the template's fixed text (its canonical JSON)."""
    return hashlib.sha256(canonical_json(TEMPLATES[template_id][2]).encode("utf-8")).hexdigest()


def render_views(task: TaskSpec, presentation: Presentation, products: Mapping[str, Product],
                 templates: Mapping[Interface, str]) -> dict[Interface, str]:
    views = {}
    for interface, template_id in templates.items():
        if template_id not in TEMPLATES:
            raise ValueError("unknown prompt template %s for interface %s" % (template_id, interface))
        expected, renderer, _ = TEMPLATES[template_id]
        if expected != interface:
            raise ValueError("template %s renders interface %s, not %s" % (template_id, expected, interface))
        views[interface] = renderer(task, presentation, products)
    return views


def build_observation(config: RunConfig, task: TaskSpec, presentation: Presentation,
                      products: Mapping[str, Product]) -> Observation:
    """The observation of one (set, presentation): the configured interfaces rendered
    with their templates, descriptions cut to environment.description_chars. Reads
    nothing else from ``config``, so mode and backend cannot change it."""
    chars = config.environment.description_chars
    shown = {c.product_id: products[c.product_id].truncated(chars) for c in presentation.candidates}
    templates = {i: config.protocol.prompt_templates[i] for i in config.interfaces}
    return Observation(task=Task(task_id=task.task_id, instruction=task.instruction), set_id=presentation.set_id,
                       presentation_id=presentation.presentation_id, candidates=presentation.candidates,
                       views=render_views(task, presentation, shown, templates))
