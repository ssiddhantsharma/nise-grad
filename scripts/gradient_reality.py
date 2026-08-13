"""Naive gradient ascent reward-hacks; a straight-through discrete step fixes it.

Measured (30aa binder vs benzoic acid, recycling=3, A6000). "discrete P(bind)" is the P(bind)
of the argmax sequence, refolded -- the number that matters. Naive optimization drives the soft
sequence to a degenerate string whose discrete P(bind) is low (poly-Leu at nss=2, poly-Phe at
nss=25). Straight-through (`optimize_pbind(..., straight_through=True)`) optimizes the discrete
sequence directly: higher discrete P(bind) and realistic composition.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.loads((Path(__file__).resolve().parent / "data" / "gradient_reality.json").read_text())
LABELS = ["naive nss=2", "naive nss=25", "STE nss=25"]
COLORS = ["#D55E00", "#D55E00", "#0072B2"]   # naive / naive / STE


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4))

    a.bar(LABELS, [D["discrete_pbind"][k] for k in LABELS], color=COLORS, width=0.6)
    a.axhline(0.5, ls="--", lw=1, color="0.6")
    a.set_ylabel("discrete P(bind)")
    a.set_title("discrete P(bind) of the design")
    a.set_ylim(0, 1)

    b.bar(LABELS, [D["hydrophobic"][k] for k in LABELS], color=COLORS, width=0.6)
    b.axhspan(0.4, 0.5, color="0.7", alpha=0.25)
    b.text(2, 0.45, "realistic", fontsize=8, color="0.4", ha="center", va="center")
    b.set_ylabel("hydrophobic fraction")
    b.set_title("composition (1.0 = degenerate collapse)")
    b.set_ylim(0, 1)

    for ax in (a, b):
        ax.set_xticklabels(LABELS, rotation=15, ha="right", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "gradient_reality.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
