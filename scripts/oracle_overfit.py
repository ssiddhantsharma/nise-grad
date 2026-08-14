"""The designs overfit the oracle they were optimized against; two independent oracles agree.

STE reaches high Boltz P(bind) (0.82), but two independent oracles (Protenix-v2 and jopendde
ipTM) both score the same designs low. A real, experimentally-validated binder (NISE's apixaban
binder) scores 0.98 on both, so the low scores are real failures. The independent oracles
disagree with Boltz on the hacked designs but agree on the real binder -- so they are
independent enough to optimize jointly (Route B), with Protenix held out as the transfer judge.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = json.loads((Path(__file__).resolve().parent / "data" / "xcheck.json").read_text())
BOLTZ, PROT, JOP, POS = "#0072B2", "#E69F00", "#CC79A7", "#009E73"


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    names = list(D["designs"])
    x = np.arange(len(names))
    a.bar(x - 0.25, [D["designs"][n]["boltz"] for n in names], 0.25, color=BOLTZ,
          label="Boltz P(bind)  (optimized)")
    a.bar(x, [D["designs"][n]["protenix"] for n in names], 0.25, color=PROT,
          label="Protenix ipTM  (independent)")
    a.bar(x + 0.25, [D["designs"][n]["jopendde"] for n in names], 0.25, color=JOP,
          label="jopendde ipTM  (independent)")
    a.set_xticks(x)
    a.set_xticklabels(names, fontsize=8)
    a.set_ylabel("score")
    a.set_title("hacked designs: high on Boltz, not on independent oracles")
    a.set_ylim(0, 1)
    a.legend(fontsize=7.5, loc="upper right")

    rb = D["real_binder"]
    b.bar(["Protenix", "jopendde"], [rb["protenix"], rb["jopendde"]], color=POS, width=0.6)
    b.axhline(0.5, ls="--", lw=1, color="0.7")
    b.set_ylabel("ipTM on a real binder")
    b.set_title("real binder: both agree (0.98)")
    b.set_ylim(0, 1)

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
