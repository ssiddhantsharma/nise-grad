"""Optimizing the Boltz proxy harder does not improve held-out transfer (it hurts foldedness).

Independently mirrors DBMol (Qin et al. 2026), Fig 3 / Table 2: as optimization proceeds the
proxy climbs but the held-out metric stalls and validity degrades. Data: apix_prior_scored.json
(plain STE on apixaban at budgets 25/100/250, n=5), scored on Boltz (optimized) and Protenix-v2
(held-out); degeneracy = fraction of the single most common residue.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROXY, HELD, DEG = "#0072B2", "#E69F00", "#CC79A7"


def degeneracy(s):
    return max(s.count(c) for c in set(s)) / len(s)


def main():
    d = json.loads((Path(__file__).resolve().parent / "data" / "apix_prior_scored.json").read_text())
    by = defaultdict(list)
    for r in d:
        if r["method"] == "ste":
            by[r["budget"]].append(r)
    x = sorted(by)

    def stat(fn):
        m = [np.mean([fn(r) for r in by[b]]) for b in x]
        s = [np.std([fn(r) for r in by[b]]) for b in x]
        return np.array(m), np.array(s)

    boltz, bs = stat(lambda r: r["boltz_pbind"])
    prot, ps = stat(lambda r: r["protenix_iptm"])
    deg, ds = stat(lambda r: degeneracy(r["seq"]))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    h1 = ax.errorbar(x, boltz, bs, fmt="o-", color=PROXY, capsize=2, label="Boltz P(bind)  (optimized)")
    h2 = ax.errorbar(x, prot, ps, fmt="s-", color=HELD, capsize=2, label="Protenix ipTM  (held-out)")
    ax.axhline(0.98, ls=":", lw=1, color="0.5")
    ax.text(x[-1], 0.985, "real binder", ha="right", va="bottom", fontsize=8, color="0.4")
    ax.set_xlabel("optimization budget (folds)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)

    ax2 = ax.twinx()
    h3 = ax2.errorbar(x, deg, ds, fmt="^--", color=DEG, capsize=2, alpha=0.8, label="degeneracy")
    ax2.set_ylabel("degeneracy (max residue fraction)", color=DEG)
    ax2.tick_params(axis="y", colors=DEG)
    ax2.set_ylim(0, 0.6)

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.legend([h1, h2, h3], [h.get_label() for h in (h1, h2, h3)],
              fontsize=8, loc="lower center", frameon=False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "plateau.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
