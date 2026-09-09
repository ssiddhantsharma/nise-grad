#!/usr/bin/env python
"""Per co-folding MODEL comparison on the Anthropic release, scored by ipSAE_min
(Anthropic's own ranking metric). Two questions per model:
  Q1 classification: macro-AP (within-target avg precision) binder vs non-binder.
  Q2 KD ranking:     within-target Spearman(ipSAE_min, pKd) among binders.
Plus two ensembles: mean ipSAE_min, and Anthropic's per-target z-score ensemble.

IMPORTANT caveat (stated on the figure): this is the RELEASED set (1,320), which
was already filtered on ESMFold2/ESMFold2-Fast/Protenix-v2 ipSAE -> post-selection,
so absolute values are compressed and the MODEL ranking here is NOT directly
comparable to Anthropic's Figure M1 (macro-AP on the unfiltered Overath 3,532 set:
ESMFold2-Fast 0.62, ESMFold2 0.61, AlphaFold3 0.55, Protenix-v2 0.52, ensemble 0.66).
This view answers "does the folder choice matter on the delivered designs" and is
a presentation option, not a re-derivation of their benchmark.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps  # figures4papers house style
ps.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = "/tmp/design_summary.parquet"
RNG = np.random.default_rng(0)
MINT = 8
# figures4papers roles: campaign=emphasis blue, re-score=teal, ensemble=violet, refs=red_strong
INK, SUB, GRID = ps.INK, ps.MUTED, ps.GRID
TEAL, AMBER, PURPLE = ps.TEAL, ps.BLUE_MAIN, ps.VIOLET
REF = ps.RED_STRONG

# column suffix -> display name; Anthropic's 3 campaign predictors + AF3 highlighted
MODELS = [("ef2fast", "ESMFold2-Fast"), ("ef2full", "ESMFold2"), ("ptxv2", "Protenix-v2"),
          ("af3of3", "AlphaFold3"), ("afm3", "AF-Multimer"), ("boltz2", "Boltz-2"),
          ("chai1", "Chai-1"), ("of3", "OpenFold3"), ("rf3", "RoseTTAFold3"), ("odde", "OpenDDE")]
CAMPAIGN = {"ESMFold2-Fast", "ESMFold2", "Protenix-v2"}
# Anthropic Fig M1 macro-AP on Overath (for reference annotation)
OVERATH = {"ESMFold2-Fast": 0.62, "ESMFold2": 0.61, "AlphaFold3": 0.55, "Protenix-v2": 0.52}


def macro_ap(d, col, y="y", boot=2000):
    per = {}
    for t, g in d.groupby("target"):
        g = g.dropna(subset=[col])
        nb = int(g[y].sum()); nn = int((1 - g[y]).sum())
        if nb >= 3 and nn >= 3:
            per[t] = average_precision_score(g[y].values, g[col].values)
    v = np.array(list(per.values()))
    if len(v) < 3: return np.nan, np.nan, np.nan
    bs = [np.mean(RNG.choice(v, len(v), replace=True)) for _ in range(boot)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def kd_rho(d, col, boot=2000):
    b = d[d.y == 1].dropna(subset=[col, "pKd"]); per = {}
    for t, g in b.groupby("target"):
        if len(g) >= MINT and g[col].nunique() > 2:
            r = spearmanr(g[col], g.pKd).correlation
            if np.isfinite(r): per[t] = r
    v = np.array(list(per.values()))
    if len(v) < 3: return np.nan, np.nan, np.nan
    bs = [np.median(RNG.choice(v, len(v), replace=True)) for _ in range(boot)]
    return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    d = pd.read_parquet(SUMMARY)
    d = d[d.binder_final.isin([True, False])].copy()
    d["y"] = d.binder_final.astype(int)
    d["pKd"] = -np.log10(d.kd_nM_final * 1e-9)
    cols = {}
    for suf, name in MODELS:
        c = f"ipsae_min_{suf}"
        if c in d: cols[name] = c
    # ensembles
    zc = []
    for name, c in cols.items():
        z = d.groupby("target")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
        d[f"z_{c}"] = z; zc.append(f"z_{c}")
    d["ens_mean"] = d[list(cols.values())].mean(axis=1)
    d["ens_zscore"] = d[zc].mean(axis=1)

    rows = []
    for name, c in cols.items():
        ap, alo, ahi = macro_ap(d, c); kr, klo, khi = kd_rho(d, c)
        rows.append(dict(model=name, kind="model", ap=ap, ap_lo=alo, ap_hi=ahi, kd=kr, kd_lo=klo, kd_hi=khi))
    for name, c in [("ensemble (mean)", "ens_mean"), ("ensemble (z-score)", "ens_zscore")]:
        ap, alo, ahi = macro_ap(d, c); kr, klo, khi = kd_rho(d, c)
        rows.append(dict(model=name, kind="ensemble", ap=ap, ap_lo=alo, ap_hi=ahi, kd=kr, kd_lo=klo, kd_hi=khi))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "per_model.csv"), index=False)
    print(df.to_string(index=False))

    plt.rcParams.update({"xtick.color": SUB, "ytick.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 1.4, "text.color": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.subplots_adjust(left=0.135, right=0.965, top=0.83, bottom=0.255, wspace=0.40)

    def barpanel(ax, valc, loc, hic, title, xlabel, xmax, ref=None):
        dd = df.sort_values(valc).reset_index(drop=True)
        y = range(len(dd))
        def col(m, k):
            if k == "ensemble": return PURPLE
            return AMBER if m in CAMPAIGN else TEAL
        ax.barh(list(y), dd[valc], color=[col(m, k) for m, k in zip(dd.model, dd.kind)], height=0.66, zorder=3)
        ax.errorbar(dd[valc], list(y), xerr=np.vstack([dd[valc]-dd[loc], dd[hic]-dd[valc]]),
                    fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=4)
        for i, v in zip(y, dd[valc]): ax.text(v + xmax*0.012, i, f"{v:.2f}", va="center", fontsize=8, color=INK)
        if ref:
            for i, m in zip(y, dd.model):
                if m in ref: ax.plot(ref[m], i, marker="D", ms=5, color=REF, zorder=6)
        ax.set_yticks(list(y)); ax.set_yticklabels(dd.model, fontsize=9)
        ax.set_xlim(0 if valc == "ap" else min(-0.05, dd[loc].min()-0.02), xmax)
        ax.set_title(title, loc="left"); ax.set_xlabel(xlabel)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ax.tick_params(length=0); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)

    barpanel(a, "ap", "ap_lo", "ap_hi", "Q1  Identify binders  (macro-AP)",
             "macro-AP (released set — post-selection)", 0.75, ref=OVERATH)
    barpanel(b, "kd", "kd_lo", "kd_hi", "Q2  Rank affinity / KD",
             "within-target Spearman ρ", 0.42)
    b.axvline(0.25, color=REF, lw=1.3, ls=(0, (4, 3)))
    b.text(0.256, 0.3, "Anthropic ρ=0.25", color=REF, fontsize=7.6, fontweight="bold", va="center")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[Patch(color=AMBER, label="Anthropic campaign model (ESMFold2 / -Fast / Protenix-v2)"),
                        Patch(color=TEAL, label="other re-score model"),
                        Patch(color=PURPLE, label="ensemble"),
                        Line2D([0],[0], marker="D", color="w", markerfacecolor=REF, ms=6,
                               label="Anthropic Overath macro-AP (Fig M1 ref)")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.135), frameon=False, fontsize=8.2)

    fig.suptitle("Which co-folding model ranks best? — Anthropic release, scored by ipSAE_min",
                 x=0.5, y=0.945, fontsize=14, fontweight="bold", color=INK, ha="center")
    fig.text(0.135, 0.055,
        "Per-model ipSAE_min on the 1,320 released designs. Q1 (left): on the delivered designs the models cluster tightly (macro-AP ~0.4-0.55) and the ENSEMBLE is best — but note the released set is POST-SELECTION\n"
        "(already filtered on ESMFold2/ESMFold2-Fast/Protenix-v2), so absolute values are compressed and the model ORDER differs from Anthropic's unfiltered benchmark (red diamonds = their Overath Fig M1 macro-AP). Use\n"
        "their Fig M1 for the authoritative model ranking; use this view for 'on the delivered set, does folder choice matter'. Q2 (right): folder choice barely affects KD ranking — every model, and the ensemble, sits at ρ≈0.25.",
        fontsize=8.2, color=SUB, ha="left")
    out = os.path.join(HERE, "figures", "per_model.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
