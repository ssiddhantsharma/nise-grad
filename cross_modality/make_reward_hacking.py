#!/usr/bin/env python
"""Reward-hacking signature on de novo protein binders (Anthropic release).

The paper's small-molecule finding: gradient ascent maximises the oracle's own
score (interface pTM) while real, held-out binding does not follow -- the oracle
is gameable. The protein echo, on 1,320 released designs (354 binders / 966
confirmed non-binders): a high consensus interface pTM does not imply binding.
Non-binders reach the same oracle ceiling as real binders, so selecting purely by
the oracle's confidence -- what a gradient maximises -- keeps a large fraction of
non-binders. Reports the top-selection purity and writes reward_hacking.csv +
plots_anthropic/reward_hacking.png.

Data: /tmp/{cofold,design_summary}.parquet (HF Anthropic/claude-protein-binder-design).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps

ps.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
COFOLD, SUMMARY = "/tmp/cofold.parquet", "/tmp/design_summary.parquet"
ORACLE = "iptm_pae"   # the quantity the paper's gradient maximises


def consensus():
    cf = pd.read_parquet(COFOLD)
    g = cf.groupby(["full_name", "cofolding_model"])[ORACLE].max().reset_index()
    cons = g.groupby("full_name")[ORACLE].mean()
    ds = pd.read_parquet(SUMMARY).set_index("full_name")[["binder_final"]]
    d = cons.to_frame("oracle").join(ds, how="inner")
    d = d[d.binder_final.isin([True, False])].copy()
    d["y"] = d.binder_final.astype(int)
    return d.dropna(subset=["oracle"])


def main():
    d = consensus()
    b, nb = d[d.y == 1].oracle, d[d.y == 0].oracle
    base = d.y.mean()
    print(f"designs={len(d)} binders={len(b)} non={len(nb)} base_rate={base:.3f}")

    med_b = b.median()
    frac_nb_over_med = float((nb > med_b).mean())
    nb95, b_max = float(nb.quantile(0.95)), float(b.max())
    print(f"median binder oracle={med_b:.3f}; non-binders above it={frac_nb_over_med:.1%}")
    print(f"95th-pct non-binder oracle={nb95:.3f} vs max binder={b_max:.3f}")

    # precision @ top-K by oracle score = the hit-rate of selecting on the oracle alone
    order = d.sort_values("oracle", ascending=False)
    fracs = [0.05, 0.10, 0.20, 0.30]
    rows = []
    for f in fracs:
        k = max(1, int(round(f * len(order))))
        prec = order.y.iloc[:k].mean()
        rows.append({"top_fraction": f, "k": k, "binder_precision": prec,
                     "lift_over_base": prec / base})
        print(f"top {f:.0%} by oracle (n={k}): binder purity={prec:.1%} (base {base:.1%})")
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "reward_hacking.csv"), index=False)

    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.9))
    bins = np.linspace(min(d.oracle), max(d.oracle), 30)
    a1.hist(b, bins=bins, color=ps.BLUE_MAIN, alpha=0.75, label=f"binders (n={len(b)})", density=True)
    a1.hist(nb, bins=bins, color=ps.RED_STRONG, alpha=0.55, label=f"non-binders (n={len(nb)})", density=True)
    a1.axvline(med_b, color=ps.INK, ls=":", lw=1.3)
    a1.set_xlabel("consensus interface pTM (the oracle)")
    a1.set_ylabel("density")
    a1.legend()
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)

    xs = np.linspace(0.02, 1.0, 60)
    prec = [order.y.iloc[:max(1, int(round(f * len(order))))].mean() for f in xs]
    a2.plot(xs * 100, prec, color=ps.BLUE_MAIN, lw=2.2)
    a2.axhline(base, color=ps.INK, ls=":", lw=1.3)
    a2.text(60, base + 0.02, f"base rate {base:.0%}", color=ps.INK, fontsize=9)
    a2.set_xlabel("designs kept, ranked by oracle (%)")
    a2.set_ylabel("fraction that actually bind")
    a2.set_ylim(0, 1)
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)

    fig.suptitle("Maximising the oracle does not exclude non-binders", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, "plots_anthropic", "reward_hacking.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
