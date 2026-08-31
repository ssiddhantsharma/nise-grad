"""Fig 7 (apixaban budget sweep, 3 trajectories): STE gradient, O3 latent search, Best-K-of-N, each
across budget 25/100/250, with the real-binder ceiling and scramble floor. The two optimisers (STE, O3)
decline as they reward-hack; pure sampling (Best-K-of-N) creeps up; all plateau far below the real binder.
Held-out Protenix interface pTM, mean over seeds, 95% bootstrap CI. Single target -> no statistic mixing."""
import os
from pathlib import Path as _P

DATADIR = str(_P(__file__).resolve().parents[1] / "data")
FIGDIR = os.environ.get("FIGDIR", str(_P(__file__).resolve().parents[2] / "figures"))
os.makedirs(FIGDIR, exist_ok=True)

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = DATADIR + ""
OUT = Path(FIGDIR + "/budget_ceiling.png")
rng = np.random.default_rng(0)
ip = lambda r: r.get("protenix_iptm")
BUD = [25, 100, 250]

def ci(vals):
    v = np.array(vals, float); m = v.mean()
    b = [rng.choice(v, len(v), replace=True).mean() for _ in range(10000)]
    lo, hi = np.percentile(b, [2.5, 97.5]); return m, m - lo, hi - m

def traj(rows, method):
    ms, los, his = [], [], []
    for bd in BUD:
        v = [ip(r) for r in rows if r["method"] == method and r["budget"] == bd and ip(r) is not None]
        m, lo, hi = ci(v); ms.append(m); los.append(lo); his.append(hi)
    return ms, los, his

sweep = json.loads(Path(f"{DATA}/budget_scored.json").read_text())          # STE trajectory + anchors
base = json.loads(Path(f"{DATA}/expB_apix_baselines_budget.json").read_text())  # bestn + o3 trajectories
ste = traj(sweep, "ste")
bn = traj(base, "bestn")
o3 = traj(base, "o3")
real = [r["protenix_iptm"] for r in sweep if r["method"] == "anchor_real"][0]
scr = [r["protenix_iptm"] for r in sweep if r["method"] == "anchor_scramble"][0]

fig, ax = plt.subplots(figsize=(5.8, 4.3))
INK = "#111111"
ax.axhline(real, ls="--", lw=1.4, color=INK, zorder=2)
ax.text(255, real + 0.008, f"real binder {real:.2f}", ha="right", va="bottom", fontsize=9, color=INK)
ax.axhline(scr, ls=":", lw=1.1, color="#9a9186", zorder=2)
ax.text(20.5, scr + 0.008, f"scrambled sequence {scr:.2f}", ha="left", va="bottom", fontsize=8, color="#9a9186")

series = [("STE (gradient)", ste, "#35617a", "o"),
          ("O3 (latent search)", o3, "#9c6d8e", "^"),
          ("Best-K-of-N (sampling)", bn, "#c2922f", "s")]
for label, (m, lo, hi), col, mk in series:
    ax.errorbar(BUD, m, yerr=[lo, hi], marker=mk, ms=7, lw=1.8, color=col,
                elinewidth=1.3, capsize=3, zorder=5, label=label)

ax.annotate("optimisers reward-hack\ndownward with budget", xy=(250, ste[0][2] + 0.01),
            xytext=(88, 0.70), fontsize=8.5, color="#35617a", ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color="#35617a", lw=1.0, connectionstyle="arc3,rad=-0.15"))

ax.set_xscale("log"); ax.set_xticks(BUD); ax.set_xticklabels([str(b) for b in BUD])
ax.set_xlim(19, 300); ax.set_ylim(0.0, 1.03)
ax.set_xlabel("optimisation budget (oracle folds)", fontsize=10)
ax.set_ylabel("held-out interface pTM", fontsize=10)
ax.tick_params(labelsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=8.2, loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.9))
ax.set_title("Budget does not close the gap (apixaban)", fontsize=10.5, pad=8)
fig.tight_layout(); fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
print(f"STE {[round(x,3) for x in ste[0]]} | O3 {[round(x,3) for x in o3[0]]} | bestn {[round(x,3) for x in bn[0]]} | real {real:.3f}")
