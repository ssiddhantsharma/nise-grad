"""Optimizing the Boltz proxy harder improves structure, not held-out binding.

Plain STE on apixaban at budgets 25/100/250 (n=20 each), scored on Boltz (the optimized proxy) and
Protenix-2 (held out). As the budget rises the Boltz P(bind) proxy climbs, but the held-out ipTM
does not follow it up (0.47 -> 0.43 -> 0.36); meanwhile gpde falls (1.60 -> 1.19), so the designs
get more confidently folded without binding better. Structure is optimisable, the binding interface
is not, the same picture DBMol (Qin et al. 2026) reports on the molecule side. Data: budget_scored.json.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROXY, HELD, GPDE = "#0072B2", "#E69F00", "#009E73"


def main():
    d = json.loads((Path(__file__).resolve().parent / "data" / "budget_scored.json").read_text())
    by = defaultdict(list)
    for r in d:
        if r["method"] == "ste" and r.get("protenix_iptm") is not None:
            by[r["budget"]].append(r)
    x = sorted(by)
    real = next(r["protenix_iptm"] for r in d if r["method"] == "anchor_real")

    def stat(fn):
        return (np.array([np.mean([fn(r) for r in by[b]]) for b in x]),
                np.array([np.std([fn(r) for r in by[b]]) for b in x]))

    boltz, bs = stat(lambda r: r["boltz_pbind"])
    prot, ps = stat(lambda r: r["protenix_iptm"])
    gpde, gs = stat(lambda r: r["protenix_gpde"])

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    h1 = ax.errorbar(x, boltz, bs, fmt="o-", color=PROXY, capsize=2, label="Boltz P(bind)  (optimized proxy)")
    h2 = ax.errorbar(x, prot, ps, fmt="s-", color=HELD, capsize=2, label="Protenix ipTM  (held-out)")
    ax.axhline(real, ls=":", lw=1, color="0.5")
    ax.text(x[-1], real + 0.01, "real binder apx1049", ha="right", va="bottom", fontsize=8, color="0.4")
    ax.set_xlabel("optimization budget (gradient steps)")
    ax.set_ylabel("score / ipTM")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)

    ax2 = ax.twinx()
    h3 = ax2.errorbar(x, gpde, gs, fmt="^--", color=GPDE, capsize=2, alpha=0.85, label="Protenix gpde  (lower = better)")
    ax2.set_ylabel("gpde", color=GPDE)
    ax2.tick_params(axis="y", colors=GPDE)
    ax2.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.legend([h1, h2, h3], [h.get_label() for h in (h1, h2, h3)],
              fontsize=7.5, loc="center right", frameon=False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "plateau.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
