"""The designs overfit the oracle they were optimized against.

STE reaches high Boltz P(bind) (0.82), but an independent oracle (Protenix-v2 ipTM) scores the
same designs 0.30-0.42 -- no better than a scramble. A real, experimentally-validated binder
(NISE's apixaban binder), folded under identical Protenix settings, scores 0.98, so the low
scores are real failures, not a compressed scale. Gradient design games the single oracle.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = json.loads((Path(__file__).resolve().parent / "data" / "xcheck.json").read_text())
BOLTZ, PROT, POS, GRAY, RED = "#0072B2", "#E69F00", "#009E73", "0.6", "#D55E00"


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # A: same designs, optimized oracle (high) vs independent oracle (low)
    names = list(D["designs"])
    x = np.arange(len(names))
    a.bar(x - 0.2, [D["designs"][n]["boltz"] for n in names], 0.4, color=BOLTZ,
          label="Boltz P(bind)  (optimized)")
    a.bar(x + 0.2, [D["designs"][n]["protenix"] for n in names], 0.4, color=PROT,
          label="Protenix ipTM  (independent)")
    a.set_xticks(x)
    a.set_xticklabels(names, fontsize=8)
    a.set_ylabel("score")
    a.set_title("high on the optimized oracle, low on an independent one")
    a.set_ylim(0, 1)
    a.legend(fontsize=8, loc="upper right")

    # B: Protenix is trustworthy (real binder 0.98)
    cal = D["calibration"]
    labels = list(cal)
    colors = [POS, GRAY, PROT, GRAY, RED]
    b.bar(labels, [cal[k] for k in labels], color=colors, width=0.65)
    b.axhline(cal["real binder"], ls="--", lw=1, color=POS)
    b.text(4.4, cal["real binder"], "real binder", fontsize=8, color=POS, ha="right", va="bottom")
    b.set_ylabel("Protenix ipTM")
    b.set_title("Protenix discriminates (real binder = 0.98)")
    b.set_ylim(0, 1)
    b.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "oracle_overfit.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
