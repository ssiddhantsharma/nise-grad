"""The decoupled straight-through estimator does not move the ceiling at any temperature.

Held-out Protenix ipTM of STE designs on apixaban with the decoupled-STE backward temperature swept
0.25/0.5/0.75/2.0 (arXiv 2410.13331, n=5 each). Flat at 0.36-0.42 throughout, far below the real
binder (0.97): the plateau is not a gradient-estimator artifact. Data: lever_dste_scored.json,
dste_{0.25,0.75,2.0}_scored.json, apix_anchors_scored.json.
"""

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
BAR = "#0072B2"

TEMPS = [("0.25", "dste_0.25_scored.json", "ste_dste0.25"),
         ("0.5", "lever_dste_scored.json", "ste_dste0.5"),
         ("0.75", "dste_0.75_scored.json", "ste_dste0.75"),
         ("2.0", "dste_2.0_scored.json", "ste_dste2")]


def vals(file, method):
    d = json.loads((DATA / file).read_text())
    return [r["protenix_iptm"] for r in d if r["method"] == method and r.get("protenix_iptm") is not None]


def main():
    anch = json.loads((DATA / "apix_anchors_scored.json").read_text())
    real = next(r["protenix_iptm"] for r in anch if r["method"] == "anchor_real")
    base = st.mean(vals("budget_scored.json", "ste"))   # plain STE, includes budget 25

    labels = [t[0] for t in TEMPS]
    means = [st.mean(vals(f, m)) for _, f, m in TEMPS]
    sds = [st.pstdev(vals(f, m)) for _, f, m in TEMPS]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.bar(x, means, yerr=sds, width=0.6, color=BAR, alpha=0.85, capsize=3)
    ax.axhline(real, ls=":", lw=1, color="#009E73")
    ax.text(len(labels) - 0.5, real + 0.01, "real binder", ha="right", va="bottom", fontsize=8, color="#009E73")
    ax.axhline(base, ls="--", lw=1, color="0.5")
    ax.text(0, base + 0.01, "plain STE", ha="left", va="bottom", fontsize=8, color="0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("decoupled-STE backward temperature")
    ax.set_ylabel("Protenix ipTM  (held-out)")
    ax.set_ylim(0, 1.05)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "dste_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
