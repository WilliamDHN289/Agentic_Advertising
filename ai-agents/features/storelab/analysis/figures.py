"""Figure output shared by the pre-E1 simulation reports (SL-19S, SL-20S, SL-21S).

Every figure carries ``LABEL`` and is written to ``artifacts/figures/<ticket>/`` as PDF
and PNG, next to a CSV of the plotted data (one row per plotted point). The output is
deterministic: no creation date in the PDF, fixed DPI and font, and the CSV uses
``repr`` floats. Legends go in a band between the title and the axes (``save(...,
legend=ncol)``), so no legend covers plotted data (SL-24).
"""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABEL = "Stipulated parameters (A17), pre-E1, exploratory"
REPO = Path(__file__).resolve().parents[4]
FIGURES = REPO / "artifacts" / "figures"
DPI = 150
STYLE = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
         "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.hashsalt": "storelab",
         "pdf.compression": 6}


def figure(rows: int = 1, cols: int = 1, width: float = 7.0, height: float = 3.2):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
    return fig, axes


TITLE_IN, LEGEND_ROW_IN, LABEL_IN = 0.32, 0.15, 0.2  # heights in inches


def save(fig, ticket: str, name: str, header: Sequence[str], rows: Iterable[Sequence], title: str,
         label: str = LABEL, legend: int | None = None, note: str | None = None) -> list[Path]:
    """Stamp the label, write <name>.pdf, .png and .csv; return the paths.

    ``legend`` (a column count) gathers the labelled artists of every axes, first label
    first, into one legend between the title and the axes. ``note`` is a line under the
    label, e.g. which markets a figure shows (A17 §5.1)."""
    out = FIGURES / ticket
    out.mkdir(parents=True, exist_ok=True)
    height = fig.get_size_inches()[1]
    fig.suptitle(title, fontsize=10, y=0.995)
    top = 1 - TITLE_IN / height
    if legend is not None:
        entries: dict[str, object] = {}
        for ax in fig.axes:
            for handle, text in zip(*ax.get_legend_handles_labels()):
                entries.setdefault(text, handle)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        fig.legend(list(entries.values()), list(entries), loc="upper center", bbox_to_anchor=(0.5, top),
                   ncol=legend, frameon=False)
        top -= (math.ceil(len(entries) / legend) * LEGEND_ROW_IN + 0.05) / height
    bottom = LABEL_IN / height
    if note is not None:
        fig.text(0.5, bottom, note, ha="center", va="bottom", fontsize=7, color="0.3", linespacing=1.3)
        bottom += (0.15 * (note.count("\n") + 1) + 0.04) / height
    fig.text(0.5, 0.005, label, ha="center", va="bottom", fontsize=7, style="italic", color="0.3")
    fig.tight_layout(rect=(0, bottom, 1, top))
    pdf, png, data = out / (name + ".pdf"), out / (name + ".png"), out / (name + ".csv")
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": label})
    fig.savefig(png, dpi=DPI, metadata={"Software": None, "Comment": label})
    plt.close(fig)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    buffer.write("# " + label + "\n")
    writer.writerow(list(header))
    for row in rows:
        writer.writerow([repr(v) if isinstance(v, float) else v for v in row])
    data.write_text(buffer.getvalue(), encoding="utf-8")
    return [pdf, png, data]
