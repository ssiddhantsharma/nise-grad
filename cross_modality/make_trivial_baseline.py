#!/usr/bin/env python
"""A 'cLogP for proteins': do trivial sequence properties rank K_d as well as the
dedicated co-folding metrics?

The small-molecule benchmark shows the affinity heads do not beat lipophilicity
(cLogP) at ranking measured pK_d. The protein analog: among the 354 measured
binders, compute within-target Spearman(feature, pK_d) for trivial sequence
descriptors (length, MW, hydrophobicity, charge/aromatic fraction) and compare
to the best co-folding metric (ipSAE, from metric_analysis.csv). If a trivial
descriptor matches the oracle, the oracle carries little affinity signal beyond
triviality. Writes trivial_baseline.csv + plots_anthropic/trivial_baseline.png.

Data: /tmp/design_summary.parquet (HF Anthropic/claude-protein-binder-design).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps

ps.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = "/tmp/design_summary.parquet"
RNG = np.random.default_rng(0)
MIN_PER_TARGET = 8

KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
MW = {"A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14, "Q": 128.13,
      "E": 129.12, "G": 57.05, "H": 137.14, "I": 113.16, "L": 113.16, "K": 128.17,
      "M": 131.19, "F": 147.18, "P": 97.12, "S": 87.08, "T": 101.10, "W": 186.21,
      "Y": 163.18, "V": 99.13}
CHARGED, AROMATIC = set("DEKR"), set("FWY")


def feats(seq):
    n = len(seq)
    if n == 0:
        return None
    return {
        "length": float(n),
        "mol_weight": sum(MW.get(a, 110) for a in seq),
        "hydrophobicity": np.mean([KD.get(a, 0) for a in seq]),
        "frac_charged": sum(a in CHARGED for a in seq) / n,
        "frac_aromatic": sum(a in AROMATIC for a in seq) / n,
        "frac_hydrophobic": np.mean([KD.get(a, 0) > 0 for a in seq]),
    }


def within_target_kd(d, col, boot=2000):
    b = d.dropna(subset=[col, "pKd"])
    per = {}
    for tgt, gg in b.groupby("target"):
        if len(gg) >= MIN_PER_TARGET and gg[col].nunique() > 2:
            rho = spearmanr(gg[col], gg.pKd).correlation
            if np.isfinite(rho):
                per[tgt] = rho
    vals = np.array(list(per.values()))
    if len(vals) < 3:
        return np.nan, np.nan, np.nan, len(vals)
    boots = [np.median(RNG.choice(vals, len(vals), replace=True)) for _ in range(boot)]
    return float(np.median(vals)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(vals)


def main():
    ds = pd.read_parquet(SUMMARY)
    b = ds[ds.binder_final == True].dropna(subset=["sequence", "kd_nM_final"]).copy()  # noqa: E712
    b["pKd"] = -np.log10(b.kd_nM_final * 1e-9)
    fr = b.sequence.apply(feats).apply(pd.Series)
    b = pd.concat([b[["target", "pKd"]].reset_index(drop=True), fr.reset_index(drop=True)], axis=1)
    print(f"binders with sequence + K_d = {len(b)}")

    cols = ["length", "mol_weight", "hydrophobicity", "frac_charged", "frac_aromatic", "frac_hydrophobic"]
    rows = []
    for c in cols:
        rho, lo, hi, nt = within_target_kd(b, c)
        rows.append({"feature": c, "kd_rho": rho, "lo": lo, "hi": hi, "n_targets": nt})
        print(f"{c:16s} within-target K_d rho = {rho:+.3f} [{lo:+.3f},{hi:+.3f}]  ({nt} targets)")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "trivial_baseline.csv"), index=False)

    # oracle reference from the existing metric analysis
    ma = pd.read_csv(os.path.join(HERE, "metric_analysis.csv"))
    best = ma.loc[ma.kd_rho.idxmax()]
    print(f"best co-folding metric: {best.label} K_d rho = {best.kd_rho:+.3f}")

    import matplotlib.pyplot as plt
    plot = out.copy()
    plot["abs"] = plot.kd_rho.abs()
    plot = plot.sort_values("abs")
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    y = range(len(plot))
    ax.barh(list(y), plot.kd_rho, color=ps.NEUTRAL, height=0.62, zorder=3)
    for yi, (_, r) in zip(y, plot.iterrows()):
        ax.plot([r.lo, r.hi], [yi, yi], color=ps.GREY, lw=0.9, zorder=2)
    ax.axvline(best.kd_rho, color=ps.BLUE_MAIN, ls=":", lw=1.6)
    ax.text(best.kd_rho, len(plot) - 0.4, f" best co-fold metric\n ({best.label}, {best.kd_rho:+.2f})",
            color=ps.BLUE_MAIN, fontsize=8.5, va="top")
    ax.set_yticks(list(y))
    ax.set_yticklabels(plot.feature)
    ax.set_xlabel("within-target Spearman rho vs measured pK_d")
    ax.axvline(0, color=ps.INK, lw=0.8)
    ax.set_title("Trivial sequence descriptors vs the co-folding oracle", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    dst = os.path.join(HERE, "plots_anthropic", "trivial_baseline.png")
    fig.savefig(dst, dpi=200, bbox_inches="tight")
    print("saved", dst)


if __name__ == "__main__":
    main()
