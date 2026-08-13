"""The recipe, in one figure: recycling in the loop fixes gradient design, at a compute cost.

Reads the measured runs in scripts/data/ (30aa binder vs benzoic acid, A6000) and plots
figures/recycling.png. Regenerate the data with the recycling_steps override in optimize_pbind
(recycling=0 is the fast default; recycling=3 folds physical structures).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
D0 = json.loads((HERE / "data" / "scaling_rec0.json").read_text())
D3 = json.loads((HERE / "data" / "scaling_rec3.json").read_text())
C0, C3 = "#D55E00", "#0072B2"   # recycling 0 / 3


def sz(d, k):
    return [r["L"] for r in d["size"] if k in r], [r[k] for r in d["size"] if k in r]


def main():
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    (a, b), (c, d) = ax

    # outcome: P(bind) over optimization
    a.plot(D0["outcome"]["sig"], color=C0, lw=2, label="recycling 0 (noise structures)")
    a.plot(D3["outcome"]["sig"], color=C3, lw=2, label="recycling 3 (physical)")
    a.set(xlabel="gradient step", ylabel="P(bind)", ylim=(-0.03, 1))
    a.legend(fontsize=8, loc="center left")

    # outcome: hydrophobic fraction of the final design
    b.bar(["recycling 0", "recycling 3"], [D0["outcome"]["hyd"], D3["outcome"]["hyd"]],
          color=[C0, C3], width=0.6)
    b.axhline(0.45, ls="--", lw=1, color="0.5")
    b.text(1.35, 0.46, "realistic ~0.45", fontsize=8, color="0.4", ha="right")
    b.set(ylabel="hydrophobic fraction of design", ylim=(0, 1))
    b.text(0, 0.93, "collapse", ha="center", fontsize=8, color="0.4")

    # cost: time per step
    c.plot(*sz(D0, "sec"), "-o", color=C0, lw=2, ms=5, label="recycling 0")
    c.plot(*sz(D3, "sec"), "-o", color=C3, lw=2, ms=5, label="recycling 3")
    c.set(xlabel="binder length (aa)", ylabel="seconds / gradient step", yscale="log")
    c.legend(fontsize=8)

    # cost: peak GPU memory
    d.plot(*sz(D0, "gb"), "-o", color=C0, lw=2, ms=5, label="recycling 0")
    d.plot(*sz(D3, "gb"), "-o", color=C3, lw=2, ms=5, label="recycling 3")
    d.axhline(48, ls="--", lw=1, color="0.5")
    d.text(40, 45, "48 GB card", fontsize=8, color="0.4")
    d.set(xlabel="binder length (aa)", ylabel="GPU memory (GB)", ylim=(0, 52))

    for x in (a, b, c, d):
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
        x.grid(alpha=0.2)
    fig.tight_layout()
    out = HERE.parent / "figures" / "recycling.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
