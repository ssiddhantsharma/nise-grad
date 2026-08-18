"""Optimized designs collapse to low-complexity sequences; the real binder does not.

Per-design fraction of the most common amino acid (maxAA), from the scored experiment logs. Every
gradient-optimized objective (affinity, and pTMEnergy despite its better structure) sits well above
the real apixaban binder; the scramble and random anchors do not, so the collapse is a property of
the optimization, not of length. Motivates the anti-collapse penalties (usage entropy + adjacent
repetition; arXiv 2602.00782). Data: scripts/data/apix_*_scored.json.
"""

import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent / "data"
GEN, PTM, REF = "#999999", "#0072B2", "#009E73"

# (label, method, color) top to bottom
GROUPS = [
    ("real binder", "anchor_real", REF),
    ("scramble", "anchor_scramble", REF),
    ("random", "anchor_random", REF),
    ("affinity STE", "ste", GEN),
    ("+ pTMEnergy", "ste_ptm", PTM),
]


def max_aa_fraction(seq):
    return max(Counter(seq).values()) / len(seq)


def seqs_by_method():
    out = {}
    for f in glob.glob(str(DATA / "apix_*_scored.json")):
        for r in json.loads(Path(f).read_text()):
            out.setdefault(r["method"], {})[r["seq"]] = 1   # dedup by sequence
    return {m: list(d) for m, d in out.items()}


def main():
    by_method = seqs_by_method()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for y, (label, method, color) in enumerate(GROUPS):
        vals = [max_aa_fraction(s) for s in by_method.get(method, [])]
        if not vals:
            continue
        ax.scatter(vals, [y] * len(vals), c=color, s=42, alpha=0.75,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter([sum(vals) / len(vals)], [y], marker="|", s=520, c="0.2", zorder=4)

    ax.axvline(1 / 20, ls=":", lw=1, color="0.6")
    ax.text(1 / 20, 4.6, "uniform (5%)", fontsize=7.5, color="0.5", ha="center")
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels([g[0] for g in GROUPS])
    ax.set_ylim(-0.6, len(GROUPS) - 0.2)
    ax.invert_yaxis()
    ax.set_xlabel("most-common AA fraction  (higher = more collapsed)")
    ax.set_xlim(0.03, 0.30)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "collapse.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
