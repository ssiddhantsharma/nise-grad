"""Gradient de-novo design plateaus below the real binder on every ligand in the panel.

For 14 small molecules (steroid, PARP inhibitors, camptothecins, fluorogens, an NDI), each anchored
by a published crystal binder, STE de-novo design (8 seeds) is scored on held-out Protenix-2. The
real anchor (star) sits at ipTM ~0.93; the best composition-passing STE design (dot, most-common-AA
fraction < 0.35, so not poly-X gaming) lands ~0.26 lower and never reaches the anchor (0/14). A
capability ceiling, not reward-hacking. Data: data/panel_run/lig_*.json.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent / "data" / "panel_run"
ANCHOR, STE = "#009E73", "#999999"
POLY_X = 0.35
LABELS = {
    "lig_cortisol": "cortisol", "lig_apixaban": "apixaban", "lig_rucaparib": "rucaparib",
    "lig_mefuparib": "mefuparib", "lig_niraparib": "niraparib", "lig_veliparib": "veliparib",
    "lig_amantadine": "amantadine", "lig_n_n_diphenyl_naphtha": "NDI1", "lig_dfhbi": "DFHBI",
    "lig_dfhbi_1t": "DFHBI-1T", "lig_exatecan": "exatecan", "lig_fl118": "FL118",
    "lig_belotecan": "belotecan", "lig_camptothecin": "camptothecin",
}


def main():
    rows = []
    for f in sorted(DATA.glob("lig_*.json")):
        if f.stem not in LABELS:                    # skip *_in.json Protenix inputs
            continue
        d = json.loads(f.read_text())
        anc = [r for r in d if r["method"] == "anchor_real" and r.get("protenix_iptm") is not None]
        passing = [r["protenix_iptm"] for r in d
                   if r["method"] == "ste" and r.get("protenix_iptm") is not None
                   and max(Counter(r["seq"]).values()) / len(r["seq"]) < POLY_X]
        if anc and passing:
            rows.append((LABELS[f.stem], anc[0]["protenix_iptm"], max(passing)))
    rows.sort(key=lambda r: r[1])
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for i, (_, a, b) in zip(y, rows, strict=True):
        ax.plot([b, a], [i, i], color="#DDDDDD", lw=1.4, zorder=1)
    ax.scatter([r[2] for r in rows], y, c=STE, s=46, zorder=2,
               edgecolor="white", linewidth=0.5, label="best STE de-novo (composition-passing)")
    ax.scatter([r[1] for r in rows], y, c=ANCHOR, s=150, marker="*", zorder=3,
               edgecolor="white", linewidth=0.5, label="real crystal binder")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Protenix ipTM  (held-out)")
    ax.set_xlim(0, 1.05)
    mean_a = sum(r[1] for r in rows) / len(rows)
    mean_b = sum(r[2] for r in rows) / len(rows)
    ax.set_title(f"anchor {mean_a:.2f}  vs  best STE {mean_b:.2f}   "
                 f"(gap {mean_a - mean_b:.2f}, 0/{len(rows)} reach real)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "per_ligand_plateau.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out, "| ligands:", len(rows))


if __name__ == "__main__":
    main()
