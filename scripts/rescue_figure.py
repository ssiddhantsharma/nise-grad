"""The bottleneck is the backbone: supply a real pocket and the sequence layer recovers binders.

De-novo gradient design (no backbone) plateaus at held-out ipTM ~0.45. Freezing the real apx1049
crystal backbone and designing a fresh, random-init sequence to fit it (LigandMPNN NLL, then an
independent Protenix refold) jumps to 0.83, near the real binder (0.96) and far above scramble
(0.27). Same sequence machinery; the only added ingredient is a real pocket backbone. Data:
budget_scored.json (de-novo ste@25, n=20), rescue_scored.json (n=8).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
DENOVO, RESCUE, REAL, SCRAM = "#999999", "#0072B2", "#009E73", "#CCCCCC"


def vals(file, method, budget=None):
    d = json.loads((DATA / file).read_text())
    return [r["protenix_iptm"] for r in d if r["method"] == method
            and (budget is None or r.get("budget") == budget) and r.get("protenix_iptm") is not None]


def main():
    denovo = vals("budget_scored.json", "ste", 25)
    rescue = vals("rescue_scored.json", "rescue_realbb")
    resc = json.loads((DATA / "rescue_scored.json").read_text())
    real = next(r["protenix_iptm"] for r in resc if r["method"] == "anchor_real")
    scram = next(r["protenix_iptm"] for r in resc if r["method"] == "anchor_scramble")

    cats = ["de-novo STE\n(no backbone)", "fresh seq on\nreal backbone"]
    data = [denovo, rescue]
    colors = [DENOVO, RESCUE]
    x = np.arange(len(cats))

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.bar(x, [np.mean(v) for v in data], yerr=[np.std(v) for v in data], width=0.5,
           color=colors, alpha=0.85, capsize=3)
    for i, v in enumerate(data):
        ax.scatter([x[i]] * len(v), v, color="0.25", s=14, alpha=0.5, zorder=3)
    ax.axhline(real, ls=":", lw=1, color=REAL)
    ax.text(1.4, real + 0.005, "real binder apx1049", ha="right", va="bottom", fontsize=8, color=REAL)
    ax.axhline(scram, ls=":", lw=1, color="0.6")
    ax.text(1.4, scram + 0.005, "scramble", ha="right", va="bottom", fontsize=8, color="0.5")
    ax.annotate("", xy=(1, np.mean(rescue)), xytext=(1, np.mean(denovo)),
                arrowprops={"arrowstyle": "->", "color": RESCUE, "lw": 1.5})
    ax.text(1.08, (np.mean(rescue) + np.mean(denovo)) / 2, "supply\nthe pocket", fontsize=8,
            color=RESCUE, va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Protenix ipTM  (held-out)")
    ax.set_ylim(0, 1.05)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "rescue.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
