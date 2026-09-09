#!/usr/bin/env python
"""Clean PM headline: which scoring METHOD (a) identifies binders and (b) tracks KD.
Two ranked bars, one highlighted winner, minimal text. Numbers from metric_analysis.csv
(Anthropic de novo set, 1,320 designs / 354 binders; consensus over 10 co-folding models).
AUROC = binder vs non-binder. KD = within-target Spearman(metric, pKd) among binders."""
import os, sys
import pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps  # figures4papers house style

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTE, GRID = ps.INK, ps.MUTED, ps.GRID
TEAL, AMBER = ps.NEUTRAL, ps.BLUE_MAIN   # non-winner (neutral) vs winner (emphasis)
WINNER = "ipSAE_min"

d = pd.read_csv(os.path.join(HERE, "metric_analysis.csv"))

ps.apply()
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.3))
fig.subplots_adjust(left=0.14, right=0.965, top=0.80, bottom=0.14, wspace=0.42)


def panel(ax, valcol, title, xlabel, xlim, refline=None, refkw=None, fmt="{:.2f}"):
    dd = d.sort_values(valcol).reset_index(drop=True)   # ascending -> best on top in barh
    y = range(len(dd))
    colors = [AMBER if lb == WINNER else TEAL for lb in dd.label]
    ax.barh(list(y), dd[valcol], color=colors, height=0.66, zorder=3)
    for i, (v, lb) in enumerate(zip(dd[valcol], dd.label)):
        ax.text(v + xlim[1] * 0.014, i, fmt.format(v),
                va="center", fontsize=9, color=INK,
                fontweight="bold" if lb == WINNER else "normal")
    if refline is not None:
        ax.axvline(refline, color=MUTE, lw=1.1, ls=(0, (3, 3)))
        if refkw: ax.text(refline, len(dd) - 0.35, refkw, color=MUTE, fontsize=8, ha="center", va="bottom")
    ax.set_yticks(list(y))
    ax.set_yticklabels(list(dd.label), fontsize=9.5)
    for t, lb in zip(ax.get_yticklabels(), dd.label):
        if lb == WINNER: t.set_color(AMBER); t.set_fontweight("bold")
    ax.set_xlim(*xlim); ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=8)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.tick_params(length=0); ax.grid(axis="x", color=GRID, lw=1.0); ax.set_axisbelow(True)


panel(a, "pooled_auroc", "Identify binders", "AUROC  (0.5 = coin flip)",
      (0.35, 0.85), refline=0.5, refkw="chance")
panel(b, "kd_rho", "Track affinity (KD)", "correlation with KD  (Spearman ρ)",
      (0.0, 0.40), refline=None, fmt="{:+.2f}")

fig.suptitle("Which scoring method identifies binders — and which tracks affinity?",
             x=0.5, y=0.945, fontsize=15, fontweight="bold", color=INK, ha="center")
fig.text(0.5, 0.035,
         "Anthropic de novo binder set · 1,320 designs (354 binders / 966 non-binders) · consensus over 10 co-folding models · "
         "AUROC = binder vs non-binder; KD = within-target ranking among binders.",
         fontsize=8, color=MUTE, ha="center")
out = os.path.join(HERE, "plots_anthropic", "metric_headline.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
print(out)
