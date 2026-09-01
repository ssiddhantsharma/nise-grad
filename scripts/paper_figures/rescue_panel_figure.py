"""Panel-wide real-backbone rescue. Per-ligand best composition-passing held-out Protenix interface pTM
when a fresh sequence is designed onto each anchor's frozen real backbone (8 seeds). Every ligand lands
near the real binder and above the generated RFdiffusion3 backbone, far above the sequence gradient.
Data in data/rescue_panel/. Reference levels are panel means: real 0.93, RFd3 per-backbone-best 0.89,
STE best-of-40 0.73."""
import glob
import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATADIR = os.environ.get("DATADIR", str(Path(__file__).resolve().parents[1] / "data"))
FIGDIR = os.environ.get("FIGDIR", str(Path(__file__).resolve().parents[2] / "figures"))
POLY = 0.35

rows = []
for f in sorted(glob.glob(f"{DATADIR}/rescue_panel/*.json")):
    if f.endswith(("_designs.json", "_in.json")):
        continue
    with open(f) as fh:
        d = json.load(fh)
    vals = [r["protenix_iptm"] for r in d if r.get("protenix_iptm") is not None
            and (r.get("maxaa") is None or r["maxaa"] < POLY)]
    if vals:
        rows.append((Path(f).stem, max(vals)))
rows.sort(key=lambda x: x[1], reverse=True)
labels = [r[0].replace("_", " ") for r in rows]
best = [r[1] for r in rows]

fig, ax = plt.subplots(figsize=(6.2, 3.4))
xs = range(len(rows))
ax.bar(xs, best, color="#35617a", width=0.66, zorder=3, label="real-backbone rescue (best of 8)")
for lvl, txt, c in [(0.93, "real binder", "#333"), (0.89, "RFd3 backbone", "#7a9bb0"),
                    (0.73, "STE best-of-40", "#b8860b")]:
    ax.axhline(lvl, ls="--", lw=1, color=c, zorder=2)
    ax.text(len(rows) - 0.4, lvl + 0.004, txt, ha="right", va="bottom", fontsize=7.5, color=c)
ax.set_xticks(list(xs))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel("held-out interface pTM")
ax.set_ylim(0.6, 1.0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("A real backbone lifts every ligand near the real binder", fontsize=10, pad=8)
fig.tight_layout()
out = f"{FIGDIR}/rescue_panel.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
print("  panel-mean best:", round(float(np.mean(best)), 3), "| range:", round(min(best), 3), "-", round(max(best), 3))
