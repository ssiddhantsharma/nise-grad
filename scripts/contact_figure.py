"""The distogram contact loss lifts held-out transfer but does not break the ceiling.

Mosaic's BinderTargetContact (DBMol Eq 3-6, binder residue -> ligand atom <8A) added to STE at
weight w. Held-out Protenix-2 ipTM/gpde vs w: the mean peaks at w=0.1 (0.45 -> 0.65), over-forcing
(w=0.5) hurts, and the best design still tops out near the plateau, below the real binder.
Data: apix_contact_scored.json (budget-25 designs, n=25 baseline / n=3 per weight).
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IPTM, GPDE = "#0072B2", "#CC79A7"


def weight_of(method):
    if method == "ste":
        return 0.0
    if method.startswith("ste_ct"):
        return float(method[len("ste_ct"):])
    return None


def main():
    d = json.loads((Path(__file__).resolve().parent / "data" / "apix_contact_scored.json").read_text())
    by = defaultdict(list)
    real = scram = None
    for r in d:
        if r["method"] == "anchor_real":
            real = r["protenix_iptm"]
        elif r["method"] == "anchor_scramble":
            scram = r["protenix_iptm"]
        w = weight_of(r["method"])
        if w is not None and r.get("protenix_iptm") is not None:
            by[w].append(r)
    x = sorted(by)

    def stat(fn):
        return (np.array([np.mean([fn(r) for r in by[w]]) for w in x]),
                np.array([np.std([fn(r) for r in by[w]]) for w in x]))

    iptm, ei = stat(lambda r: r["protenix_iptm"])
    gpde, eg = stat(lambda r: r["protenix_gpde"])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    h1 = ax.errorbar(x, iptm, ei, fmt="o-", color=IPTM, capsize=2, label="Protenix ipTM  (held-out)")
    ax.axhline(real, ls=":", lw=1, color="0.5")
    ax.text(x[-1], real + 0.005, "real binder", ha="right", va="bottom", fontsize=8, color="0.4")
    ax.axhline(scram, ls=":", lw=1, color="0.7")
    ax.text(x[-1], scram + 0.005, "scramble floor", ha="right", va="bottom", fontsize=8, color="0.6")
    ax.set_xlabel("contact-loss weight  $w_c$")
    ax.set_ylabel("Protenix ipTM", color=IPTM)
    ax.tick_params(axis="y", colors=IPTM)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)

    ax2 = ax.twinx()
    h2 = ax2.errorbar(x, gpde, eg, fmt="^--", color=GPDE, capsize=2, alpha=0.8, label="gpde")
    ax2.set_ylabel("gpde (lower = better)", color=GPDE)
    ax2.tick_params(axis="y", colors=GPDE)
    ax2.set_ylim(0, 3)

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.legend([h1, h2], [h.get_label() for h in (h1, h2)],
              fontsize=8, loc="upper center", frameon=False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "contact.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
