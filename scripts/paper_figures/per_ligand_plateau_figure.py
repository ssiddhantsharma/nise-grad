"""Per-ligand plateau figure with the RFd3 backbone arm added: STE de novo (plateau) vs RFd3 backbone
vs real crystal binder, 14 ligands. Held-out Protenix interface pTM."""
import os
from pathlib import Path as _P
DATADIR = str(_P(__file__).resolve().parents[1] / "data")
FIGDIR = os.environ.get("FIGDIR", str(_P(__file__).resolve().parents[2] / "figures"))
os.makedirs(FIGDIR, exist_ok=True)

import glob, json, os, statistics as st
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PR = DATADIR + "/panel_run"
RF = DATADIR + "/rfd3_panel"
PANEL = DATADIR + "/ligand_panel.json"
OUT = Path(FIGDIR + "/per_ligand_plateau.png")

def maxaa(s): return max(Counter(s).values()) / len(s) if s else 1.0
ip = lambda r: r.get("protenix_iptm")

name_by_smi = {p["ligand"]: p["ligand_name"] for p in json.load(open(PANEL))}

# STE best-comp-passing + real anchor, by SMILES
ste, real = {}, {}
for f in glob.glob(f"{PR}/lig_*.json"):
    d = json.load(open(f))
    for r in d:
        if r["method"] == "anchor_real" and ip(r) is not None:
            real[r["ligand"]] = ip(r)
    v = [ip(r) for r in d if r["method"] == "ste" and ip(r) is not None and maxaa(r["seq"]) < 0.35]
    smi = next((r["ligand"] for r in d if r["method"] == "ste"), None)
    if v and smi:
        ste[smi] = max(v)

# RFd3 per-backbone-best mean, by SMILES
rfd3 = {}
for f in glob.glob(f"{RF}/*.json"):
    if f.endswith("_designs.json"): continue
    d = json.load(open(f))
    if not d: continue
    smi = d[0]["ligand"]
    byb = {}
    for r in d:
        if ip(r) is not None and maxaa(r["seq"]) < 0.35:
            byb.setdefault(r["note"], []).append(ip(r))
    pb = [max(v) for v in byb.values()]
    if pb: rfd3[smi] = st.mean(pb)

rows = []
for smi in ste:
    if smi in rfd3 and smi in real:
        nm = name_by_smi.get(smi, "?")
        nm = {"N,N'-diphenyl naphthalenediimide (NDI1)": "NDI"}.get(nm, nm)
        rows.append((nm, ste[smi], rfd3[smi], real[smi]))
rows.sort(key=lambda t: t[1])  # by STE plateau ascending

fig, ax = plt.subplots(figsize=(7.2, 5.0))
y = range(len(rows))
for i, (nm, s, r, a) in enumerate(rows):
    ax.plot([s, a], [i, i], color="#d8d2c8", lw=1.4, zorder=1)          # gap connector
ax.scatter([r[1] for r in rows], y, s=46, color="#cbc3b8", zorder=3, label="STE de novo (sequence gradient)")
ax.scatter([r[2] for r in rows], y, s=52, color="#35617a", zorder=4, label="RFd3 backbone + MPNN")
ax.scatter([r[3] for r in rows], y, s=90, marker="*", color="#111111", zorder=5, label="real crystal binder")
ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_xlabel("held-out interface pTM", fontsize=10)
ax.set_xlim(0.3, 1.02); ax.tick_params(axis="x", labelsize=9)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(fontsize=8.5, loc="upper left", frameon=False)
mste = st.mean([r[1] for r in rows]); mrf = st.mean([r[2] for r in rows]); mre = st.mean([r[3] for r in rows])
ax.set_title(f"best per ligand:   STE {mste:.2f}    RFd3 {mrf:.2f}    real {mre:.2f}   (n={len(rows)})",
             fontsize=10.5, pad=8)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT, "| means STE", round(mste, 2), "RFd3", round(mrf, 2), "real", round(mre, 2))
