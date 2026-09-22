#!/usr/bin/env python
"""Rhodamine plateau: the gradient ceiling holds on a new chemotype.

Per de novo rhodamine (Rhobin), the best composition-passing held-out Protenix
interface pTM of the STE gradient designs vs the real rhobin9 binder anchor.
Reads the panel_run outputs (lig_<slug>.json). Design-side mirror of the 47->61
benchmark Rhobin addition.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(os.environ.get("RHO_DIR", "scripts/data/rhodamine_panel"))
PRETTY = {"jf646": "JF646", "jf660": "JF660", "jf669": "JF669",
          "rhodamine101": "Rhodamine 101", "sirh": "SiRh"}
BLUE, GREY, INK = "#0F4D92", "#CFCECE", "#272727"

rows = []
for slug, name in PRETTY.items():
    d = json.load(open(DATA / f"lig_{slug}.json"))
    anc = max((r["protenix_iptm"] for r in d
               if r.get("method") == "anchor_real" and r.get("protenix_iptm") is not None), default=None)
    ste = [r for r in d if str(r.get("method", "")).startswith("ste")]
    passing = [r["protenix_iptm"] for r in ste if r.get("protenix_iptm") is not None and r.get("maxaa", 1) < 0.35]
    disq = sum(1 for r in ste if r.get("protenix_iptm") is not None and r.get("maxaa", 1) >= 0.35)
    rows.append({"name": name, "anchor": anc, "grad": max(passing) if passing else 0.0,
                 "n": len(ste), "disq": disq})

rows.sort(key=lambda r: r["grad"])
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
                     "font.size": 12, "axes.linewidth": 1.6, "axes.edgecolor": INK})
fig, ax = plt.subplots(figsize=(7.2, 3.6))
y = range(len(rows))
ax.barh(list(y), [r["grad"] for r in rows], color=BLUE, height=0.5, zorder=3, label="gradient (STE) best")
ax.scatter([r["anchor"] for r in rows], list(y), marker="*", s=170, color=INK, zorder=4,
           label="real rhobin9 binder")
for yi, r in zip(y, rows):
    if r["disq"]:
        ax.text(0.01, yi, f"{r['disq']}/{r['n']} poly-X", va="center", ha="left", fontsize=8, color="#8a3a3a")
gm = sum(r["grad"] for r in rows) / len(rows)
am = sum(r["anchor"] for r in rows) / len(rows)
ax.axvline(gm, color=BLUE, ls=":", lw=1.3)
ax.axvline(am, color=INK, ls=":", lw=1.3)
ax.text(gm, len(rows) - 0.4, f" mean {gm:.2f}", color=BLUE, fontsize=8.5, va="top")
ax.text(am, len(rows) - 0.4, f" mean {am:.2f}", color=INK, fontsize=8.5, va="top")
ax.set_yticks(list(y)); ax.set_yticklabels([r["name"] for r in rows])
ax.set_xlabel("held-out Protenix interface pTM"); ax.set_xlim(0, 1.0)
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.set_title("The gradient plateau holds on rhodamines", fontsize=12, fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = Path(os.environ.get("OUT_PNG", "rhodamine_plateau.png"))
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out, "| gradient mean", round(gm, 3), "anchor mean", round(am, 3))
