"""pTMEnergy fixes structural confidence (gpde) but not the binding interface (iptm).

Held-out Protenix-2 designs in (iptm, gpde) space. The affinity objective reward-hacks to high
gpde (poor structure); the pTMEnergy objective (Nori et al. 2025) pulls gpde down toward the real
binder, but iptm does not move right, so the binding-interface ceiling is separate and persists.
Data: apix_contact_scored.json (affinity baseline) + apix_ptm_scored.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent / "data"
AFF, PTM, REAL = "#999999", "#0072B2", "#009E73"


def points(file, method):
    d = json.loads((DATA / file).read_text())
    return [(r["protenix_iptm"], r["protenix_gpde"]) for r in d
            if r["method"] == method and r.get("protenix_iptm") is not None]


def main():
    aff = points("apix_contact_scored.json", "ste")
    ptm = points("apix_ptm_scored.json", "ste_ptm")
    real = points("apix_ptm_scored.json", "anchor_real")[0]

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.scatter(*zip(*aff, strict=True), c=AFF, s=28, label="affinity objective", alpha=0.7)
    ax.scatter(*zip(*ptm, strict=True), c=PTM, s=40, label="pTMEnergy objective", zorder=3)
    ax.scatter([real[0]], [real[1]], c=REAL, s=90, marker="*", zorder=4, label="real binder")
    ax.annotate("better structure", xy=(0.15, 0.5), xytext=(0.15, 1.4), fontsize=8, color="0.4",
                ha="center", arrowprops={"arrowstyle": "->", "color": "0.6"})
    ax.set_xlabel("Protenix ipTM  (binding interface)")
    ax.set_ylabel("Protenix gpde  (structure, lower better)")
    ax.set_xlim(0, 1.02)
    ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "ptm_energy.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
