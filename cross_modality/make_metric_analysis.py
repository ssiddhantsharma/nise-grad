#!/usr/bin/env python
"""Bulletproof metric analysis for the Anthropic de novo binder release.

NOTE (deck): the PNG this writes (figures/metric_analysis.png) is NOT a deck figure.
Its Q1 uses macro-AP (avg precision, chance ~= binder prevalence 0.27-0.31), which
reads confusingly next to metric_headline.png's Q1 (AUROC, chance 0.50) for the same
"identify binders" question. The deck keeps metric_headline (one AUROC number).
This script is retained because it PRODUCES metric_analysis.csv, which make_headline.py
reads. Run it to refresh that CSV; the PNG is a diagnostic only.

Answers, for a PM:
  1. Was ipSAE the best metric?
  2. Detailed comparison across ALL co-folding metrics.
  3. How does it line up with what Anthropic themselves reported?

Two DISTINCT questions, two answers (this is the crux):
  * CLASSIFICATION (binder vs non-binder): which metric best identifies binders.
    Measured with average precision (AP) within each target, averaged over
    targets = "macro-AP" -- the SAME metric Anthropic use (their Fig M1 / Fig 4B).
  * KD RANKING (how *tightly*, among binders): within-target Spearman(metric, pKd).

Data: HF Anthropic/claude-protein-binder-design
  data/tables/insilico/cofold_predictions.parquet  (per design x model x seed)
  data/tables/design_summary.parquet               (binder_final, kd_nM_final)
Physics dG_bind (tmol/Rosetta) joined from tmol_ddg_kd.csv (this repo).

Aggregation (documented, matches Anthropic's spirit): per design x model, take the
best-over-seeds value in the favourable direction (max if higher-is-better, min
for pae); consensus = mean over the 10 co-folding models. Bootstrap 95% CIs:
over targets for macro-AP, over targets for the KD-rho median.

Anthropic's own reference numbers (paper: "Autonomous de novo protein binder
design with Claude", 2026-08-18), quoted for calibration:
  - Model macro-AP on the Overath benchmark (3,532 designs): ESMFold2-Fast 0.62,
    ESMFold2 0.61, AlphaFold3 0.55, Protenix v2 0.52; 3-model ensemble 0.66.
  - Ranking AP on the RELEASED set (post-selection, compressed): Claude's rank
    0.52, 7-predictor ensemble 0.57 (chance ~0.31).
  - KD ranking among 354 binders: Spearman rho = 0.16 pooled / 0.25 within-target.
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
COFOLD = "/tmp/cofold.parquet"
SUMMARY = "/tmp/design_summary.parquet"
TMOL = os.path.join(HERE, "tmol_ddg_kd.csv")
RNG = np.random.default_rng(0)
MIN_BINDERS, MIN_NONBIN = 3, 3     # per-target requirement for macro-AP
MIN_PER_TARGET_KD = 8              # per-target requirement for KD-rho
ANTHROPIC_ENSEMBLE_AP = 0.66      # their Fig M1 3-model ensemble (Overath)
ANTHROPIC_KD_RHO = 0.25           # their within-target KD Spearman

# (column, higher_is_better)
METRICS = [
    ("ipsae_min", True), ("ipsae_max", True), ("iptm_pae", True),
    ("ptm_pae", True), ("lis", True), ("sc_dockq", True), ("dockq_fnat", True),
    ("plddt_binder", True), ("plddt_target", True),
    ("pae_interface_min", False), ("pae_interface_mean", False),
    ("n_interface_contacts", True),
]
LABEL = {"ipsae_min": "ipSAE_min", "ipsae_max": "ipSAE_max", "iptm_pae": "iPTM",
         "ptm_pae": "pTM", "lis": "LIS", "sc_dockq": "sc_DockQ",
         "dockq_fnat": "DockQ_fnat", "plddt_binder": "pLDDT_binder",
         "plddt_target": "pLDDT_target", "pae_interface_min": "PAE_iface_min",
         "pae_interface_mean": "PAE_iface_mean",
         "n_interface_contacts": "n_iface_contacts"}

# figures4papers roles: ipSAE_min=emphasis blue, other=teal, physics=green, refs=red_strong
INK, SUB, GRID = ps.INK, ps.MUTED, ps.GRID
TEAL, AMBER, GREEN = ps.TEAL, ps.BLUE_MAIN, ps.GREEN_3
RED = PURPLE = ps.RED_STRONG   # Anthropic reference lines


def consensus_table():
    cf = pd.read_parquet(COFOLD)
    cols = [m for m, _ in METRICS]
    # seed-aggregate per (design, model) in the favourable direction
    aggs = {m: ("max" if hib else "min") for m, hib in METRICS}
    g = cf.groupby(["full_name", "cofolding_model"]).agg(aggs).reset_index()
    # consensus = mean over models
    cons = g.groupby("full_name")[cols].mean()
    ds = pd.read_parquet(SUMMARY)[
        ["full_name", "target", "binder_final", "kd_nM_final"]].set_index("full_name")
    d = cons.join(ds, how="inner")
    d = d[d.binder_final.isin([True, False])].copy()
    d["y"] = d.binder_final.astype(int)
    d["pKd"] = -np.log10(d.kd_nM_final * 1e-9)
    return d


def oriented(d, m, hib):
    v = d[m].values.astype(float)
    return v if hib else -v


def macro_ap(d, m, hib, boot=2000):
    """Within-target AP averaged over targets; bootstrap-over-targets 95% CI."""
    per = {}
    for tgt, gg in d.groupby("target"):
        gg = gg.dropna(subset=[m])
        nb = int(gg.y.sum()); nn = int((1 - gg.y).sum())
        if nb >= MIN_BINDERS and nn >= MIN_NONBIN:
            per[tgt] = average_precision_score(gg.y.values, oriented(gg, m, hib))
    tgts = list(per); vals = np.array([per[t] for t in tgts])
    if len(vals) < 3:
        return np.nan, np.nan, np.nan, len(vals)
    boots = [np.mean(RNG.choice(vals, len(vals), replace=True)) for _ in range(boot)]
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(vals)


def pooled_auroc(d, m, hib):
    gg = d.dropna(subset=[m])
    return roc_auc_score(gg.y.values, oriented(gg, m, hib))


def within_target_kd(d, m, hib, boot=2000):
    """Median within-target Spearman(metric, pKd) over binders; direction-corrected
    (so + = ranks tighter binding as expected). Bootstrap-over-targets CI."""
    b = d[d.y == 1].dropna(subset=[m, "pKd"])
    per = {}
    for tgt, gg in b.groupby("target"):
        if len(gg) >= MIN_PER_TARGET_KD and gg[m].nunique() > 2:
            rho = spearmanr(gg[m], gg.pKd).correlation
            if np.isfinite(rho):
                per[tgt] = rho if hib else -rho   # flip pae so + = expected
    vals = np.array(list(per.values()))
    if len(vals) < 3:
        return np.nan, np.nan, np.nan, len(vals)
    boots = [np.median(RNG.choice(vals, len(vals), replace=True)) for _ in range(boot)]
    return float(np.median(vals)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(vals)


def physics_kd(boot=2000):
    """Physics dG_bind within-target KD-rho from the tmol run (this repo)."""
    if not os.path.exists(TMOL):
        return None
    t = pd.read_csv(TMOL)
    per = {}
    for tgt, gg in t.groupby("target"):
        gg = gg.dropna(subset=["ddg_norep", "pKd"])
        if len(gg) >= MIN_PER_TARGET_KD and gg.ddg_norep.nunique() > 2:
            rho = spearmanr(gg.ddg_norep, gg.pKd).correlation
            if np.isfinite(rho):
                per[tgt] = -rho    # lower dG = tighter -> flip so + = expected
    vals = np.array(list(per.values()))
    if len(vals) < 3:
        return None
    boots = [np.median(RNG.choice(vals, len(vals), replace=True)) for _ in range(boot)]
    return float(np.median(vals)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(vals)


def main():
    d = consensus_table()
    n_all = len(d); n_bind = int(d.y.sum())
    rows = []
    for m, hib in METRICS:
        ap, lo, hi, nt = macro_ap(d, m, hib)
        auc = pooled_auroc(d, m, hib)
        kr, klo, khi, nk = within_target_kd(d, m, hib)
        rows.append(dict(metric=m, label=LABEL[m], higher_better=hib,
                         macro_ap=ap, ap_lo=lo, ap_hi=hi, n_targets_ap=nt,
                         pooled_auroc=auc, kd_rho=kr, kd_lo=klo, kd_hi=khi, n_targets_kd=nk))
    df = pd.DataFrame(rows)
    phys = physics_kd()

    out_csv = os.path.join(HERE, "metric_analysis.csv")
    df.to_csv(out_csv, index=False)
    print(f"designs={n_all} binders={n_bind}")
    print(df[["label", "macro_ap", "ap_lo", "ap_hi", "pooled_auroc",
              "kd_rho", "kd_lo", "kd_hi", "n_targets_kd"]].to_string(index=False))
    if phys:
        print(f"physics dG_bind KD-rho={phys[0]:+.3f} [{phys[1]:+.3f},{phys[2]:+.3f}] ({phys[3]} targets)")

    # ---------- figure ----------
    plt.rcParams.update({"xtick.color": SUB, "ytick.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 1.4, "text.color": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 7.2))
    fig.subplots_adjust(left=0.135, right=0.965, top=0.885, bottom=0.20, wspace=0.42)

    # Panel A: classification macro-AP (Anthropic's metric), sorted
    da = df.sort_values("macro_ap")
    y = range(len(da))
    xerr = np.vstack([da.macro_ap - da.ap_lo, da.ap_hi - da.macro_ap])
    bars_c = [AMBER if lb == "ipSAE_min" else TEAL for lb in da.label]
    a.barh(list(y), da.macro_ap, color=bars_c, height=0.66, zorder=3)
    a.errorbar(da.macro_ap, list(y), xerr=xerr, fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=4)
    a.axvline(ANTHROPIC_ENSEMBLE_AP, color=PURPLE, lw=1.4, ls=(0, (4, 3)))
    a.text(ANTHROPIC_ENSEMBLE_AP + 0.01, 0.5, "Anthropic 3-model\nensemble 0.66*", color=PURPLE,
           fontsize=7.6, va="center", ha="left", fontweight="bold")
    for i, v in zip(y, da.macro_ap): a.text(v + 0.012, i, f"{v:.2f}", va="center", fontsize=8, color=INK)
    a.set_yticks(list(y)); a.set_yticklabels(da.label, fontsize=9)
    a.set_xlim(0, max(0.9, da.macro_ap.max() + 0.12)); a.set_xlabel("macro-AP  (within-target avg precision)")
    a.set_title("Q1  Identify binders  (classification)", loc="left")

    # Panel B: KD ranking within-target rho (+ physics), sorted, with CIs
    kdf = df.dropna(subset=["kd_rho"]).copy()
    extra = []
    if phys:
        extra = [dict(label="physics ΔG_bind\n(tmol/Rosetta)", kd_rho=phys[0], kd_lo=phys[1], kd_hi=phys[2])]
    kall = pd.concat([kdf[["label", "kd_rho", "kd_lo", "kd_hi"]], pd.DataFrame(extra)], ignore_index=True)
    kall = kall.sort_values("kd_rho")
    yb = range(len(kall))
    xerrb = np.vstack([kall.kd_rho - kall.kd_lo, kall.kd_hi - kall.kd_rho])
    bars_k = [GREEN if "physics" in str(lb) else (AMBER if lb == "ipSAE_min" else TEAL) for lb in kall.label]
    b.barh(list(yb), kall.kd_rho, color=bars_k, height=0.66, zorder=3)
    b.errorbar(kall.kd_rho, list(yb), xerr=xerrb, fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=4)
    b.axvline(0, color=INK, lw=1.1)
    b.axvline(ANTHROPIC_KD_RHO, color=PURPLE, lw=1.4, ls=(0, (4, 3)))
    b.text(ANTHROPIC_KD_RHO + 0.008, 0.55, "Anthropic within-\ntarget ρ=0.25", color=PURPLE,
           fontsize=7.6, va="center", ha="left", fontweight="bold")
    for i, v in zip(yb, kall.kd_rho): b.text(v + (0.006 if v >= 0 else -0.006), i, f"{v:+.2f}",
           va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    b.set_yticks(list(yb)); b.set_yticklabels(kall.label, fontsize=9)
    b.set_xlim(min(-0.1, kall.kd_lo.min() - 0.03), max(0.45, kall.kd_hi.max() + 0.05))
    b.set_xlabel("within-target Spearman ρ  (direction-corrected)")
    b.set_title("Q2  Rank affinity / KD  (among binders)", loc="left")

    for ax in (a, b):
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ax.tick_params(length=0); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)

    fig.suptitle("Was ipSAE the best metric? — Anthropic de novo set, consensus over 10 co-folding models",
                 x=0.5, y=0.975, fontsize=14, fontweight="bold", color=INK, ha="center")
    fig.text(0.5, 0.055,
        f"n={n_all} designs ({n_bind} binders / {n_all-n_bind} non-binders).  Q1: the ipSAE family sits in the top tie for identifying binders — ipSAE_max/ipSAE_min/PAE_iface_min/LIS/iPTM all overlap (95% CIs), consistent with\n"
        "Anthropic ranking on ipSAE_min (iPTM/LIS as shadow metrics).  Q2: NO metric ranks KD well — every metric ties at ρ≈0.25, ipSAE_min and iPTM indistinguishable, matching Anthropic's own within-target ρ=0.25; physics ΔG\n"
        "(tmol/Rosetta, not tested by Anthropic) is on par.  *Anthropic's 0.66 macro-AP is on the fuller Overath benchmark; the released set is post-selection so its range is compressed (their Claude-rank AP=0.52 lands in our band).",
        fontsize=8.4, color=SUB, ha="center")
    out = os.path.join(HERE, "figures", "metric_analysis.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
