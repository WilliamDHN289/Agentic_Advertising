#!/usr/bin/env python3
"""
make_figures.py -- Paper figures for the StoreLab interface-regime experiment.

Every panel that reports a mean or a rate carries a 95% bootstrap CI resampled
over episodes (the unit of randomisation), 4000 draws, seed fixed. The detection
frontier is re-derived here rather than read off findings.md, so its CI band comes
from the same episode-level resampling as the point estimates.

Outputs figures/fig{1..5}_*.png (+ .pdf).
"""
import json, os, random, sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, HERE)
from storelab import IVTClassifier, Event
from analyze import build_evasive, judge_dicts

BOOT = 4000
RNG = random.Random(20260827)
NP = np.random.default_rng(20260827)

# ---- design tokens (dataviz reference palette, light surface) --------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2df"
S1 = "#2a78d6"   # categorical slot 1
S2 = "#eb6834"   # categorical slot 2
MUTED = "#9a9993"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
})

REGIMES = ["human_sim", "chatbot", "gui_agent", "api"]
LABEL = {"human_sim": "Human baseline\n(simulated)", "chatbot": "Chatbot",
         "gui_agent": "GUI agent", "api": "API checkout"}


def boot_ci(vals, stat, n=BOOT, alpha=0.05):
    """Percentile CI for `stat` applied to a resample of the episode list."""
    k = len(vals)
    draws = sorted(stat([vals[RNG.randrange(k)] for _ in range(k)]) for _ in range(n))
    return draws[int(alpha / 2 * n)], draws[int((1 - alpha / 2) * n)]


def titled(ax, title, subtitle=None):
    """Title above the axes, optional subtitle under it -- no collisions."""
    ax.text(0, 1.16 if subtitle else 1.06, title, transform=ax.transAxes,
            fontsize=10.5, color=INK, va="bottom")
    if subtitle:
        ax.text(0, 1.05, subtitle, transform=ax.transAxes, fontsize=8,
                color=INK2, va="bottom")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


eps = json.load(open(os.path.join(HERE, "results", "episodes.json")))
events = json.load(open(os.path.join(HERE, "logs", "events.json")))
by_ep = defaultdict(list)
for e in events:
    by_ep[e["episode"]].append(e)
by_reg = defaultdict(list)
for e in eps:
    by_reg[e["regime"]].append(e)

print("figures ->", FIG)

# =========================================================== Fig 1: funnel
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

ax = axes[0]
x = np.arange(len(REGIMES)); w = 0.38
for i, (key, colour, lab) in enumerate([
        ("impressions_served", MUTED, "Ad impressions served"),
        ("impressions_human", S1, "Seen by a human eye")]):
    m = [np.mean([e[key] for e in by_reg[r]]) for r in REGIMES]
    lo, hi = zip(*[boot_ci([e[key] for e in by_reg[r]], np.mean) for r in REGIMES])
    err = np.array([np.array(m) - np.array(lo), np.array(hi) - np.array(m)])
    ax.bar(x + (i - 0.5) * (w + 0.02), m, w, color=colour, label=lab,
           edgecolor=SURFACE, linewidth=1.2)
    ax.errorbar(x + (i - 0.5) * (w + 0.02), m, yerr=err, fmt="none",
                ecolor=INK2, elinewidth=1, capsize=2.5)
ax.set_xticks(x); ax.set_xticklabels([LABEL[r] for r in REGIMES], fontsize=8)
ax.set_ylabel("Impressions per session")
ax.set_ylim(0, 18)
ax.legend(fontsize=8, loc="upper right")
ax.text(2 + (w + 0.02) / 2, 0.45, "0.0", ha="center", fontsize=8.5,
        color=S1, fontweight="bold")

ax = axes[1]
ctr_m, ctr_lo, ctr_hi = [], [], []
for r in REGIMES:
    v = by_reg[r]
    f = lambda s: (sum(e["sponsored_clicks"] for e in s)
                   / max(1e-9, sum(e["sponsored_impressions"] for e in s))
                   if sum(e["sponsored_impressions"] for e in s) else 0.0)
    ctr_m.append(f(v)); a, b = boot_ci(v, f); ctr_lo.append(a); ctr_hi.append(b)
ctr_m = np.array(ctr_m)
err = np.array([ctr_m - np.array(ctr_lo), np.array(ctr_hi) - ctr_m])
ax.bar(x, ctr_m, 0.55, color=S1, edgecolor=SURFACE, linewidth=1.2)
ax.errorbar(x, ctr_m, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
for xi, v, h in zip(x, ctr_m, ctr_hi):
    ax.text(xi, max(v, h) + 0.009, f"{v*100:.1f}%", ha="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels([LABEL[r] for r in REGIMES], fontsize=8)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylabel("Sponsored CTR")
ax.set_ylim(0, 0.235)
save(fig, "fig1_attention_funnel")

# ================================================= Fig 2: detection frontier
human_eps = [e["episode"] for e in by_reg["human_sim"]]
gui_eps = [e["episode"] for e in by_reg["gui_agent"]]
human_gaps = []
for ep in human_eps:
    ts = sorted(ev["t"] for ev in by_ep[ep])
    human_gaps += [b - a for a, b in zip(ts, ts[1:])]
evasive = build_evasive(by_ep, gui_eps, human_gaps or [1.0])

THR = [0.03, 0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25,
       0.30, 0.40, 0.50, 0.60, 0.75, 0.90]
flags = {}   # threshold -> (human flags, naive flags, evasive flags)
for thr in THR:
    c = IVTClassifier(threshold=thr, seed=99)
    hf = [judge_dicts(c, by_ep[ep], False, True)[1] for ep in human_eps]
    nf = [judge_dicts(c, by_ep[ep], False, True)[1] for ep in gui_eps]
    ef = [judge_dicts(c, evasive[ep], False, True)[1] for ep in gui_eps]
    flags[thr] = (hf, nf, ef)

idx_h = list(range(len(human_eps))); idx_g = list(range(len(gui_eps)))
boot_curves = []
for _ in range(1500):
    rh = [RNG.randrange(len(idx_h)) for _ in idx_h]
    rg = [RNG.randrange(len(idx_g)) for _ in idx_g]
    boot_curves.append([(np.mean([flags[t][0][i] for i in rh]),
                         np.mean([flags[t][2][i] for i in rg])) for t in THR])
boot_curves = np.array(boot_curves)          # (B, T, 2)
fp = np.array([np.mean(flags[t][0]) for t in THR])
naive = np.array([np.mean(flags[t][1]) for t in THR])
evas = np.array([np.mean(flags[t][2]) for t in THR])
lo = np.percentile(boot_curves[:, :, 1], 2.5, axis=0)
hi = np.percentile(boot_curves[:, :, 1], 97.5, axis=0)

fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.fill_between(fp, lo, hi, color=S2, alpha=0.16, lw=0, zorder=2)
ax.plot(fp, evas, color=S2, lw=2, marker="o", ms=4.5, zorder=4,
        markeredgecolor=SURFACE, markeredgewidth=1, label="Evasive GUI agent (human-paced)")
ax.plot(fp, naive, color=S1, lw=2, marker="o", ms=4.5, zorder=3,
        markeredgecolor=SURFACE, markeredgewidth=1, label="Naive GUI agent (machine-paced)")
for t, a, b in [(0.20, fp[THR.index(0.20)], evas[THR.index(0.20)]),
                (0.16, fp[THR.index(0.16)], evas[THR.index(0.16)])]:
    ax.annotate(f"threshold {t:.2f}", (a, b), textcoords="offset points",
                xytext=(10, -12), fontsize=8, color=INK2)
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xlabel("Human sessions wrongly filtered (false-positive rate)")
ax.set_ylabel("Agent sessions caught (detection rate)")
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.02, 1.05)
save(fig, "fig2_detection_frontier")

# ============================================ Fig 3: displacement vs contamination
disp = {"chatbot": 0.5491, "gui_agent": 0.8431, "api": 0.8431, "gui_evasive": 0.8431}
cont = {"chatbot": -0.00029, "gui_agent": 0.0, "api": -0.00053, "gui_evasive": 0.32542}
order = ["chatbot", "gui_agent", "api", "gui_evasive"]
lab3 = {"chatbot": "Chatbot", "gui_agent": "GUI agent\n(machine-paced)",
        "api": "API checkout", "gui_evasive": "GUI agent\n(evasive)"}

fig, ax = plt.subplots(figsize=(6.4, 3.9))
x = np.arange(len(order))
d = np.array([disp[k] for k in order]); c = np.array([max(0, cont[k]) for k in order])
ax.bar(x, d, 0.5, color=S1, edgecolor=SURFACE, linewidth=1.5, label="Displacement (rent destroyed)")
ax.bar(x, c, 0.5, bottom=d, color=S2, edgecolor=SURFACE, linewidth=1.5,
       label="Contamination (mis-billed)")
for xi, (a, b) in enumerate(zip(d, c)):
    ax.text(xi, a + b + 0.022, f"${a+b:.3f}", ha="center", fontsize=8.5, color=INK)
ax.axhline(disp["api"], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
ax.text(-0.32, disp["api"] + 0.02, "API checkout level", fontsize=7.5, color=MUTED, ha="left")
ax.set_xticks(x); ax.set_xticklabels([lab3[k] for k in order], fontsize=8)
ax.set_ylabel("Ad-channel cost per session ($)")
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(0, 1.34)
save(fig, "fig3_displacement_contamination")

# ==================================================== Fig 4: 2x2 steerability
cells = {c: [e for e in by_reg["gui_agent_2x2"] if e["cell"] == c]
         for c in ["label=off|pos=bottom", "label=on|pos=bottom",
                   "label=off|pos=top", "label=on|pos=top"]}
fig, ax = plt.subplots(figsize=(5.6, 3.9))
pos_x = np.arange(2); w = 0.34
for i, (lab, colour) in enumerate([("off", S1), ("on", S2)]):
    keys = [f"label={lab}|pos=bottom", f"label={lab}|pos=top"]
    m = [np.mean([e["chose_sponsored"] for e in cells[k]]) for k in keys]
    ci = [boot_ci([e["chose_sponsored"] for e in cells[k]], np.mean) for k in keys]
    err = np.array([[a - c[0] for a, c in zip(m, ci)], [c[1] - a for a, c in zip(m, ci)]])
    ax.bar(pos_x + (i - 0.5) * (w + 0.02), m, w, color=colour, edgecolor=SURFACE,
           linewidth=1.5, label=f'"Sponsored" label {lab}')
    ax.errorbar(pos_x + (i - 0.5) * (w + 0.02), m, yerr=err, fmt="none",
                ecolor=INK2, elinewidth=1, capsize=3)
ax.set_xticks(pos_x); ax.set_xticklabels(["Ads at bottom", "Ads at top"])
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylabel("P(agent chose a sponsored item)")
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(0, 0.78)
save(fig, "fig4_position_vs_label")

# ================================================ Fig 5: contrasts vs baseline
outcomes = [("impressions_human", "Impressions seen by a human"),
            ("sponsored_clicks", "Sponsored clicks"),
            ("price_paid", "Price paid ($)")]
arms = ["chatbot", "gui_agent", "api"]
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
for ax, (key, title) in zip(axes, outcomes):
    base = [e[key] for e in by_reg["human_sim"]]
    ys = np.arange(len(arms))[::-1]
    for y, arm in zip(ys, arms):
        a = [e[key] for e in by_reg[arm]]
        diff = np.mean(a) - np.mean(base)
        draws = sorted(np.mean([a[RNG.randrange(len(a))] for _ in a])
                       - np.mean([base[RNG.randrange(len(base))] for _ in base])
                       for _ in range(BOOT))
        lo_, hi_ = draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]
        ax.plot([lo_, hi_], [y, y], color=S1, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([diff], [y], "o", ms=6, color=S1, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=3)
    ax.axvline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[a].replace("\n", " ") for a in arms]
                       if ax is axes[0] else [], fontsize=8.5)
    ax.set_xlabel(title, fontsize=9, color=INK2, labelpad=8)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(arms) - 0.4)
fig.text(0.008, 1.02, "Difference vs. the human baseline, 95% bootstrap CI.",
         fontsize=8, ha="left", color=INK2)
save(fig, "fig5_contrasts")
print("done.")
