"""Every gradient lever improves yield but none break the ceiling below the real binder.

Held-out Protenix-2 ipTM per lever on apixaban (budget-25 designs): plain STE, + confidence
(interface-PAE), + contact (mosaic BinderTargetContact), + scaffold-init from a NISE pocket fold.
Bars are the per-lever mean +/- s.d.; the diamond is that lever's best single design. All sit
well below the real binder (0.98), the plateau is robust to the objective, consistent with the
field's success requiring backbone/pocket design rather than sequence-only gradient.
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

# (label, file, method) for each lever
LEVERS = [
    ("plain STE", "apix_contact_scored.json", "ste"),
    ("+ confidence", "apix_conf_scored.json", "ste_c0.1"),
    ("+ contact", "apix_contact_scored.json", "ste_ct0.1"),
    ("+ scaffold", "apix_scaffold_scored.json", "ste_scaf"),
]


def iptms(file, method):
    d = json.loads((DATA / file).read_text())
    return [r["protenix_iptm"] for r in d if r["method"] == method and r.get("protenix_iptm") is not None]


def main():
    anch = json.loads((DATA / "apix_contact_scored.json").read_text())
    real = next(r["protenix_iptm"] for r in anch if r["method"] == "anchor_real")
    scram = next(r["protenix_iptm"] for r in anch if r["method"] == "anchor_scramble")

    labels, means, sds, bests = [], [], [], []
    for lab, f, m in LEVERS:
        v = iptms(f, m)
        labels.append(lab)
        means.append(np.mean(v))
        sds.append(np.std(v))
        bests.append(np.max(v))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(x, means, yerr=sds, width=0.6, color=BAR, alpha=0.85, capsize=3, label="mean +/- s.d.")
    ax.scatter(x, bests, color=BEST, zorder=3, s=40, marker="D", label="best design")
    ax.axhline(real, ls=":", lw=1, color="0.5")
    ax.text(len(labels) - 0.5, real + 0.01, "real binder", ha="right", va="bottom", fontsize=8, color="0.4")
    ax.axhline(scram, ls=":", lw=1, color="0.7")
    ax.text(len(labels) - 0.5, scram + 0.01, "scramble floor", ha="right", va="bottom", fontsize=8, color="0.6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Protenix ipTM  (held-out)")
    ax.set_ylim(0, 1.05)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, ncol=2)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "levers.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
