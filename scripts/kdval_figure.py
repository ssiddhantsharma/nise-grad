"""The held-out judge tracks real measured affinity across independent de-novo binders.

Protenix score vs experimental pKd for 37 published de-novo protein->small-molecule binders (14
ligands, KD from ITC/SPR/MST/FP). gpde falls and ipTM rises with affinity (Spearman rho annotated);
the PARP set (one scaffold, four inhibitors) is ranked by gpde. Moderate but real, and it uses
others' wet-lab numbers, so the judge is grounded in experiment, not just discriminating scramble.
Data: kdval_scored.json (assembled from public de-novo binder papers).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent / "data"
OTHER, PARP = "#999999", "#D55E00"
PARP_LIGS = {"rucaparib", "mefuparib", "niraparib", "veliparib"}


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for i, o in enumerate(order):
            r[o] = i
        return r
    rx, ry = rank(xs), rank(ys); n = len(xs)
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry, strict=True)) / (n * (n * n - 1))


def main():
    d = [r for r in json.loads((DATA / "kdval_scored.json").read_text())
         if r.get("protenix_iptm") is not None and r.get("pKd") is not None]
    pk = [r["pKd"] for r in d]
    colors = [PARP if r["ligand_name"] in PARP_LIGS else OTHER for r in d]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, key, lab, rho in [
            (a1, "protenix_gpde", "Protenix gpde  (lower = better)", spearman([r["protenix_gpde"] for r in d], pk)),
            (a2, "protenix_iptm", "Protenix ipTM", spearman([r["protenix_iptm"] for r in d], pk))]:
        ax.scatter(pk, [r[key] for r in d], c=colors, s=34, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("measured pKd  (higher = tighter)")
        ax.set_ylabel(lab)
        ax.set_title(f"Spearman rho = {rho:+.2f}  (n={len(d)})", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    a1.invert_yaxis()
    a1.scatter([], [], c=PARP, s=34, label="PARP set (1 scaffold)")
    a1.scatter([], [], c=OTHER, s=34, label="other ligands")
    a1.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "kd_correlation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
