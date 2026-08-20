"""Structure quality is the bottleneck: what you fit the sequence to decides everything.

De-novo gradient design (no backbone) plateaus at held-out ipTM ~0.45. Fitting a sequence to a
design's own reward-hacked structure (the projector) makes it worse (0.39). Fitting a fresh
random-init sequence to the real apx1049 crystal backbone jumps to 0.83, near the real binder
(0.96). Same sequence machinery throughout; only the target structure changes. Data:
budget_scored.json (de-novo ste@25, n=20), apix_projector_scored.json (projector, n=5),
rescue_scored.json (n=8).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
DENOVO, PROJ, RESCUE, REAL = "#999999", "#D55E00", "#0072B2", "#009E73"


def vals(file, method, budget=None):
    d = json.loads((DATA / file).read_text())
    return [r["protenix_iptm"] for r in d if r["method"] == method
            and (budget is None or r.get("budget") == budget) and r.get("protenix_iptm") is not None]


def main():
    denovo = vals("budget_scored.json", "ste", 25)
    proj = vals("apix_projector_scored.json", "ste_proj")
    rescue = vals("rescue_scored.json", "rescue_realbb")
    resc = json.loads((DATA / "rescue_scored.json").read_text())
    real = next(r["protenix_iptm"] for r in resc if r["method"] == "anchor_real")
    scram = next(r["protenix_iptm"] for r in resc if r["method"] == "anchor_scramble")

    cats = ["de-novo STE\n(no backbone)", "onto own\nhacked structure", "fresh seq on\nreal backbone"]
    data = [denovo, proj, rescue]
    colors = [DENOVO, PROJ, RESCUE]
    x = np.arange(len(cats))

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(x, [np.mean(v) for v in data], yerr=[np.std(v) for v in data], width=0.55,
           color=colors, alpha=0.85, capsize=3)
    for i, v in enumerate(data):
        ax.scatter([x[i]] * len(v), v, color="0.25", s=14, alpha=0.5, zorder=3)
    ax.axhline(real, ls=":", lw=1, color=REAL)
    ax.text(2.4, real + 0.005, "real binder apx1049", ha="right", va="bottom", fontsize=8, color=REAL)
    ax.axhline(scram, ls=":", lw=1, color="0.6")
    ax.text(2.4, scram + 0.005, "scramble", ha="right", va="bottom", fontsize=8, color="0.5")
    ax.annotate("", xy=(2, np.mean(rescue)), xytext=(2, np.mean(denovo)),
                arrowprops={"arrowstyle": "->", "color": RESCUE, "lw": 1.5})
    ax.text(2.1, (np.mean(rescue) + np.mean(denovo)) / 2, "real\npocket", fontsize=8,
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
