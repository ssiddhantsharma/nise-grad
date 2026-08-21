"""Held-out ipTM alone is fooled by poly-A; composition separates real designs from gaming.

Guided diffusion from poly-A conditioning produces poly-A sequences at every guidance scale: they
score a moderate held-out ipTM (0.7-0.8, above a scramble's 0.27) yet are 80-90% one residue, so the
judge is gamed. The rescue (fresh sequence on a real backbone) reaches the same ipTM band with
realistic composition. Structure source, not ipTM, decides whether the sequence is real. Data:
budget_scored.json, guided_s*_scored.json, rescue_scored.json, apix_anchors_scored.json.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
DENOVO, GUIDED, RESCUE, REAL = "#999999", "#D55E00", "#0072B2", "#009E73"


def pts(file, method, budget=None):
    d = json.loads((DATA / file).read_text())
    out = []
    for r in d:
        if r["method"] != method or r.get("protenix_iptm") is None:
            continue
        if budget is not None and r.get("budget") != budget:
            continue
        maxaa = max(Counter(r["seq"]).values()) / len(r["seq"])
        out.append((maxaa, r["protenix_iptm"]))
    return out


def main():
    denovo = pts("budget_scored.json", "ste", 25)
    guided = sum((pts(f"guided_s{s}_scored.json", "guided_design") for s in ["0.05", "0.1", "0.2", "0.3"]), [])
    rescue = pts("rescue_scored.json", "rescue_realbb")
    real = pts("apix_anchors_scored.json", "anchor_real")

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for data, c, lab in [(denovo, DENOVO, "de-novo STE"), (guided, GUIDED, "guided diffusion"),
                         (rescue, RESCUE, "rescue (real backbone)")]:
        if data:
            ax.scatter(*zip(*data), c=c, s=32, alpha=0.7, label=lab, edgecolor="white", linewidth=0.4)
    if real:
        ax.scatter(*zip(*real), c=REAL, s=120, marker="*", zorder=5, label="real binder apx1049")

    ax.axvspan(0, 0.25, color="#009E73", alpha=0.06)
    ax.text(0.125, 0.06, "designable\ncomposition", ha="center", fontsize=8, color="#009E73")
    ax.set_xlabel("most-common AA fraction  (right = poly-X, not designable)")
    ax.set_ylabel("Protenix ipTM  (held-out)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "guided_fail.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
