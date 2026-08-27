"""RF2AA independence bar figure: a different-lineage predictor (RoseTTAFold All-Atom) reproduces the
ordering real ~ RFd3 << STE-plateau ~ scramble. Metric: RF2AA inter-chain (protein-ligand) PAE, lower =
more confident interface. Mean over 14 ligands, 95% bootstrap CI."""
import os
from pathlib import Path as _P
DATADIR = str(_P(__file__).resolve().parents[1] / "data")
FIGDIR = os.environ.get("FIGDIR", str(_P(__file__).resolve().parents[2] / "figures"))
os.makedirs(FIGDIR, exist_ok=True)

import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open(DATADIR + "/rf2aa_scored.json"))
OUT = Path(FIGDIR + "/rf2aa_independence.png")
rng = np.random.default_rng(0)

def ci(vals):
    v = np.array(vals, float); m = v.mean()
    b = [rng.choice(v, len(v), replace=True).mean() for _ in range(10000)]
    lo, hi = np.percentile(b, [2.5, 97.5]); return m, m - lo, hi - m

order = [("real\nbinder", "anchor_real", "#35617a"),
         ("RFd3\nbackbone", "rfd3", "#35617a"),
         ("STE\nplateau", "ste_plateau", "#b8b0a4"),
         ("scrambled", "anchor_scramble", "#cfc8bd")]
by = {}
for r in D:
    if r.get("rf2aa_iface_pae") is not None:
        by.setdefault(r["method"], []).append(r["rf2aa_iface_pae"])

fig, ax = plt.subplots(figsize=(5.0, 3.9))
xs = range(len(order))
for i, (lab, key, col) in enumerate(order):
    m, lo, hi = ci(by[key])
    ax.bar(i, m, color=col, width=0.62, zorder=3)
    ax.errorbar(i, m, yerr=[[lo], [hi]], color="#333", elinewidth=1.2, capsize=3, zorder=4)
    ax.text(i, m + hi + 0.6, f"{m:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333")

# bracket: real+RFd3 (confident) vs plateau+scramble (not)
ax.set_xticks(list(xs)); ax.set_xticklabels([o[0] for o in order], fontsize=9.5)
ax.set_ylabel("RF2AA inter-chain PAE\n(lower = more confident interface)", fontsize=9.5)
ax.set_ylim(0, 30)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", labelsize=9)
ax.set_title("A non-AF3 predictor reproduces the ordering (14/14 ligands)", fontsize=10, pad=8)
fig.tight_layout(); fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
