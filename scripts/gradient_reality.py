"""Naive gradient ascent reward-hacks; a straight-through step (STE) fixes it.

Benchmark: 30aa binder, recycling=3, num_sampling_steps=25, 3 seeds each. "discrete P(bind)"
is the P(bind) of the argmax sequence, refolded, the number that matters. Naive optimizes the
soft sequence: its soft P(bind) is ~0.9 but the discrete design scores 0.33-0.41 and is
degenerate. STE optimizes the discrete sequence directly: 0.44-0.73 with realistic composition.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = json.loads((Path(__file__).resolve().parent / "data" / "benchmark.json").read_text())
GRAY, RED, BLUE = "0.6", "#D55E00", "#0072B2"
rng = np.random.RandomState(0)


def strip(ax, x, ys, color, marker="o"):
    ax.scatter(x + rng.uniform(-0.08, 0.08, len(ys)), ys, s=45, color=color,
               marker=marker, alpha=0.85, linewidths=0, zorder=3)


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4.2))

    # discrete P(bind): naive soft (illusion) -> naive discrete -> STE discrete
    a.axhline(0.5, ls="--", lw=1, color="0.7")
    strip(a, 0, D["naive_soft"], GRAY)
    strip(a, 1, D["naive_discrete"], RED)
    strip(a, 2, D["ste_benzoic"], BLUE, "o")
    strip(a, 2, D["ste_sulfonamide"], BLUE, "^")
    a.set_xticks([0, 1, 2])
    a.set_xticklabels(["naive\n(soft)", "naive\n(discrete)", "STE\n(discrete)"])
    a.set_ylabel("P(bind)")
    a.set_title("discrete P(bind) of the design")
    a.set_ylim(0, 1)
    a.scatter([], [], color=BLUE, marker="o", label="benzoic acid")
    a.scatter([], [], color=BLUE, marker="^", label="sulfonamide")
    a.legend(fontsize=8, loc="center right")

    # composition
    b.axhspan(0.4, 0.5, color="0.7", alpha=0.2)
    b.text(1, 0.45, "realistic", fontsize=8, color="0.4", ha="center", va="center")
    strip(b, 0, D["naive_hyd"], RED)
    strip(b, 1, D["ste_benzoic_hyd"] + D["ste_sulfonamide_hyd"], BLUE)
    b.set_xticks([0, 1])
    b.set_xticklabels(["naive", "STE"])
    b.set_ylabel("hydrophobic fraction")
    b.set_title("composition (1.0 = collapse)")
    b.set_ylim(0, 1)

    for ax in (a, b):
        ax.set_xlim(-0.5, 2.5 if ax is a else 1.5)
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
