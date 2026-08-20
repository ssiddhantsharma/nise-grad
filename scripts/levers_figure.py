"""No objective lever moves the best design to the real-binder bar.

Held-out Protenix-2 ipTM per objective on apixaban: plain STE, then eight added terms (confidence,
contact, scaffold-init, pTMEnergy, decoupled-STE, KL-to-natural composition, anti-homopolymer
repetition). Bars are the per-lever mean +/- s.d.; the diamond is that lever's best single design.
Every one sits below the real crystal binder apx1049 (0.97, PDB 8VEZ). Small n (3-20) per lever, so
read the between-lever differences as within noise; the robust fact is the gap to the real binder.
Data: the *_scored.json experiment logs.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
BAR, BEST = "#0072B2", "#E69F00"

# (label, file, method, budget-filter or None)
LEVERS = [
    ("plain STE", "budget_scored.json", "ste", 25),
    ("+ confidence", "apix_conf_scored.json", "ste_c0.1", None),
    ("+ contact", "apix_contact_scored.json", "ste_ct0.1", None),
    ("+ scaffold", "apix_scaffold_scored.json", "ste_scaf", None),
    ("+ pTMEnergy", "apix_ptm_scored.json", "ste_ptm", None),
    ("+ decoupled-STE", "lever_dste_scored.json", "ste_dste0.5", None),
    ("+ composition", "lever_comp_scored.json", "ste_div0.1_0", None),
    ("+ repetition", "lever_rep_scored.json", "ste_div0_0.1", None),
]


def iptms(file, method, budget):
    d = json.loads((DATA / file).read_text())
    return [r["protenix_iptm"] for r in d if r["method"] == method
            and (budget is None or r.get("budget") == budget)
            and r.get("protenix_iptm") is not None]


def main():
    anch = json.loads((DATA / "apix_anchors_scored.json").read_text())
    real = next(r["protenix_iptm"] for r in anch if r["method"] == "anchor_real")
    scram = next(r["protenix_iptm"] for r in anch if r["method"] == "anchor_scramble")

    labels, means, sds, bests = [], [], [], []
    for lab, f, m, b in LEVERS:
        v = iptms(f, m, b)
        labels.append(lab)
        means.append(np.mean(v))
        sds.append(np.std(v))
        bests.append(np.max(v))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(x, means, yerr=sds, width=0.62, color=BAR, alpha=0.85, capsize=3, label="mean +/- s.d.")
    ax.scatter(x, bests, color=BEST, zorder=3, s=40, marker="D", label="best design")
    ax.axhline(real, ls=":", lw=1, color="0.5")
    ax.text(len(labels) - 0.5, real + 0.01, "real binder apx1049", ha="right", va="bottom",
            fontsize=8, color="0.4")
    ax.axhline(scram, ls=":", lw=1, color="0.7")
    ax.text(len(labels) - 0.5, scram + 0.01, "scramble floor", ha="right", va="bottom",
            fontsize=8, color="0.6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Protenix ipTM  (held-out)")
    ax.set_ylim(0, 1.05)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "levers.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
