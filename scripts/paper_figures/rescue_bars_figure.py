"""Rescue bar chart (apixaban): a real frozen backbone rescues the sequence layer; a reward-hacked one
does not. RFd3 (generated backbone) is the panel-wide plateau figure, not here."""
import os
from pathlib import Path as _P
DATADIR = str(_P(__file__).resolve().parents[1] / "data")
FIGDIR = os.environ.get("FIGDIR", str(_P(__file__).resolve().parents[2] / "figures"))
os.makedirs(FIGDIR, exist_ok=True)

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(FIGDIR + "/rescue.png")
labels = ["de novo\n(no backbone)", "reward-hacked\nbackbone", "real backbone\n+ MPNN"]
vals = [0.44, 0.39, 0.83]
cols = ["#cbc3b8", "#cbc3b8", "#35617a"]
REAL = 0.97

fig, ax = plt.subplots(figsize=(5.4, 3.7))
x = range(len(labels))
ax.bar(x, vals, color=cols, width=0.6, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333")
ax.axhline(REAL, ls="--", lw=1.3, color="#8f8880", zorder=2)
ax.text(len(labels) - 0.5, REAL + 0.005, f"real binder {REAL:.2f}", ha="right", va="bottom",
        fontsize=9, color="#8f8880")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("held-out interface pTM", fontsize=10)
ax.set_ylim(0, 1.05); ax.tick_params(axis="y", labelsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
