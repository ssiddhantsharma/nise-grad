"""The honest result: gradient ascent on P(bind) reward-hacks, and its soft optimum does not
transfer to a discrete sequence.

Measured (30aa binder vs benzoic acid, recycling=3, A6000):
- soft P(bind) (the optimizer's objective, on the soft sequence) reaches ~0.8, but the
  P(bind) of the discrete argmax sequence, refolded, is 0.1-0.3 -- the soft-vs-hard gap.
- physical geometry needs num_sampling_steps>=25 (nss=2 is noise), which is ~25x slower.
Both settings produce degenerate designs (poly-Leu at nss=2, poly-Phe at nss=25).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.loads((Path(__file__).resolve().parent / "data" / "gradient_reality.json").read_text())
CS, CH = "#0072B2", "#D55E00"   # soft / hard


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4))

    # soft vs discrete P(bind)
    sh = D["soft_hard"]
    x = [0, 1]
    a.bar([i - 0.2 for i in x], [sh["nss2"]["soft"], sh["nss25"]["soft"]], width=0.4,
          color=CS, label="soft (optimizer objective)")
    a.bar([i + 0.2 for i in x], [sh["nss2"]["hard"], sh["nss25"]["hard"]], width=0.4,
          color=CH, label="discrete (refolded design)")
    a.axhline(0.5, ls="--", lw=1, color="0.6")
    a.set_xticks(x)
    a.set_xticklabels(["nss=2\n(noise)", "nss=25\n(physical)"])
    a.set_ylabel("P(bind)")
    a.set_title("the soft optimum does not transfer")
    a.legend(fontsize=8, loc="upper right")
    a.set_ylim(0, 1)

    # physicality vs sampling steps + cost
    p = D["physicality"]
    c = D["cost"]
    bars = b.bar(["nss=2", "nss=25"], [p["nss2_bonds"], p["nss25_bonds"]], color="#009E73", width=0.6)
    b.axhline(119, ls="--", lw=1, color="0.6")
    b.text(1.4, 119, "physical", fontsize=8, color="0.4", ha="right", va="bottom")
    b.set_ylabel("backbone bonds (of 119)")
    b.set_title("physical geometry needs nss>=25")
    for i, (bar, s) in enumerate(zip(bars, [c["nss2_sec"], c["nss25_sec"]])):
        b.text(bar.get_x() + bar.get_width() / 2, p[f"nss{[2, 25][i]}_bonds"] + 3,
               f"{s:.2f}s/step", ha="center", fontsize=8, color="0.4")
    b.set_ylim(0, 130)

    for x_ in (a, b):
        for s in ("top", "right"):
            x_.spines[s].set_visible(False)
        x_.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "gradient_reality.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
