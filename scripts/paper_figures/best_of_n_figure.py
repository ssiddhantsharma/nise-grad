"""Best-of-N budget sweep. More independent seeds help the sequence gradient (STE) but never close the
gap to the RFdiffusion3 backbone arm: at every matched budget N the backbone arm leads. Panel-mean best
composition-passing held-out Protenix interface pTM vs sample budget. Supports the matched-budget point in
the Reproducibility appendix. STE best-of-40 data in data/ste_best40/, RFd3 in data/rfd3_panel/."""
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
rng = np.random.default_rng(0)


def passing(rows, method):
    return [r["protenix_iptm"] for r in rows if r.get("method") == method
            and r.get("protenix_iptm") is not None and (r.get("maxaa") is None or r["maxaa"] < POLY)]


def ebest(vals, n, boot=3000):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return np.nan
    if len(v) <= n:
        return float(v.max())
    return float(np.mean([rng.choice(v, n, replace=False).max() for _ in range(boot)]))


def load(subdir, method):
    out = {}
    for f in glob.glob(f"{DATADIR}/{subdir}/*.json"):
        if f.endswith("_designs.json"):
            continue
        key = Path(f).stem.replace("lig_", "").replace("naphtha", "nap")
        with open(f) as fh:
            out[key] = passing(json.load(fh), method)
    return out


ste = load("ste_best40", "ste")
rfd3 = load("rfd3_panel", "rfd3_mpnn")
ligs = sorted(set(ste) & set(rfd3))
ns = [1, 2, 4, 8, 16, 40]
panel = lambda d: [float(np.mean([ebest(d[lg], n) for lg in ligs])) for n in ns]
ste_y, rfd3_y = panel(ste), panel(rfd3)

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(ns, rfd3_y, "-o", color="#35617a", label="RFdiffusion3 backbone")
ax.plot(ns, ste_y, "-s", color="#b8860b", label="sequence gradient (STE)")
ax.axhline(0.93, ls="--", color="#888", lw=1)
ax.text(40, 0.937, "real binder", ha="right", fontsize=8, color="#888")
ax.set_xscale("log", base=2)
ax.set_xticks(ns)
ax.set_xticklabels(ns)
ax.set_xlabel("sample budget N  (best of N, composition-passing)")
ax.set_ylabel("held-out interface pTM")
ax.set_ylim(0.4, 1.0)
ax.legend(frameon=False, fontsize=9, loc="center right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("More seeds help the gradient but never close the gap", fontsize=10, pad=8)
fig.tight_layout()
out = f"{FIGDIR}/best_of_n.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
print("  N       :", ns)
print("  STE     :", [round(y, 3) for y in ste_y])
print("  RFd3    :", [round(y, 3) for y in rfd3_y])
