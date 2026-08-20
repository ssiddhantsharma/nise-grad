"""The plateau generalizes to a second, chemically distinct target.

Held-out Protenix-2 ipTM of STE designs (budget 25) against two targets, each with a real
crystal-structure binder anchor: apixaban (apx1049, PDB 8VEZ) and cortisol (hcy129_mpnn5, PDB
8UQF). On both, the designs sit far below the real binder and near the scramble floor, so the
ceiling is not apixaban-specific. Data: budget_scored.json (apixaban ste@25, n=20),
cortisol_scored.json (cortisol ste, n=20), {apix,cort}_anchors_scored.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
DESIGN, REAL, SCRAM = "#0072B2", "#009E73", "#999999"


def vals(file, method, budget=None):
    d = json.loads((DATA / file).read_text())
    return [r["protenix_iptm"] for r in d if r["method"] == method
            and (budget is None or r.get("budget") == budget) and r.get("protenix_iptm") is not None]


def one(anchors_file):
    a = json.loads((DATA / anchors_file).read_text())
    return (next(r["protenix_iptm"] for r in a if r["method"] == "anchor_real"),
            next(r["protenix_iptm"] for r in a if r["method"] == "anchor_scramble"))


def main():
    apix_design = vals("budget_scored.json", "ste", 25)
    cort_design = vals("cortisol_scored.json", "ste")
    apix_real, apix_scr = one("apix_anchors_scored.json")
    cort_real, cort_scr = one("cort_anchors_scored.json")

    targets = ["apixaban", "cortisol"]
    designs = [apix_design, cort_design]
    reals = [apix_real, cort_real]
    scrams = [apix_scr, cort_scr]
    x = np.arange(len(targets))

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(x, [np.mean(v) for v in designs], yerr=[np.std(v) for v in designs], width=0.5,
           color=DESIGN, alpha=0.85, capsize=3, label="STE designs (mean +/- s.d.)")
    ax.scatter(x, reals, color=REAL, s=110, marker="*", zorder=4, label="real crystal binder")
    ax.scatter(x, scrams, color=SCRAM, s=45, marker="_", linewidth=2, zorder=3, label="scramble")
    for i, v in enumerate(designs):
        ax.scatter([x[i]] * len(v), v, color=DESIGN, s=12, alpha=0.4, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Protenix ipTM  (held-out)")
    ax.set_ylim(0, 1.05)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="upper center", frameon=False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "two_target.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
