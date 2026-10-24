"""Figure output shared by the pre-E1 simulation reports (SL-19S, SL-20S, SL-21S).

Every figure carries ``LABEL`` and is written to ``artifacts/figures/<ticket>/`` as PDF
and PNG, next to a CSV of the plotted data (one row per plotted point). The output is
deterministic: no creation date in the PDF, fixed DPI and font, and the CSV uses
``repr`` floats.
"""
from __future__ import annotations

import csv
import io
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


def save(fig, ticket: str, name: str, header: Sequence[str], rows: Iterable[Sequence], title: str,
         label: str = LABEL) -> list[Path]:
    """Stamp the label, write <name>.pdf, .png and .csv; return the paths. Figures that do not
    depend on the stipulated parameters (an implementation check) pass their own label."""
    out = FIGURES / ticket
    out.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title, fontsize=10, y=0.995)
    fig.text(0.5, 0.005, label, ha="center", va="bottom", fontsize=7, style="italic", color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
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
