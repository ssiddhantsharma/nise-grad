"""How the ligand-aware-regularizer recipe was arrived at, in three measurements.

The numbers below were measured on an A6000 (30aa binder vs benzoic acid, jligandmpnn +
Boltz-2). This script just plots them into figures/recipe_convergence.png; the code that
produced each block is noted so it can be regenerated. Run: `python scripts/recipe_convergence.py`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1) Backbone physicality of the differentiable structure, bonds in 0.9-1.8 A (physical = 119).
#    oracle.pbind_and_output(..., recycling_steps=r, num_sampling_steps=n); count backbone bonds.
#    Sampling steps do not help; recycling does.
STEPS = [2, 25, 50, 100, 200]
BONDS_RECYCLE0 = [0, 10, 19, 21, 8]      # recycling_steps=0 (the oracle default)
BONDS_RECYCLE3 = (100, 119)               # recycling_steps=3, one point (nss=100): fully physical

# 2) Gradient magnitudes at the start point (why a small weight is ignored).
#    ||grad(-pbind)|| vs ||grad(mpnn_nll)|| wrt the sequence logits.
GRAD_PBIND, GRAD_MPNN = 1.195, 0.042      # ratio 28.6x -> parity weight ~29

# 3) Design outcome vs regularizer weight, on the frozen physical scaffold.
#    optimize_pbind(..., mpnn=ligand_reg, mpnn_weight=w); hydrophobic fraction of the result.
WEIGHTS = [2, 10, 30, 60, 100]
HYDROPHOBIC = [0.97, 0.97, 0.77, 0.70, 0.40]   # collapse -> realistic as weight passes ~29
REALISTIC = 0.45                                # LigandMPNN's own preference for the scaffold

C0, C3 = "#D55E00", "#009E73"   # Okabe-Ito: vermillion / bluish-green
CB, CM = "#0072B2", "#E69F00"   # blue / orange


def main():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.2))

    # Panel 1: physicality
    ax1.axhline(119, ls="--", lw=1, color="0.6")
    ax1.text(205, 119, "physical\n(119/119)", va="center", fontsize=8, color="0.4")
    ax1.plot(STEPS, BONDS_RECYCLE0, "-o", color=C0, lw=2, ms=6, label="recycling = 0 (default)")
    ax1.plot(*BONDS_RECYCLE3, "D", color=C3, ms=11, label="recycling = 3")
    ax1.set_xlabel("diffusion sampling steps")
    ax1.set_ylabel("backbone bonds found (of 119)")
    ax1.set_title("1. Structure is non-physical\nuntil you recycle")
    ax1.set_ylim(-5, 130)
    ax1.legend(fontsize=8, loc="center right")
    ax1.grid(alpha=0.25)

    # Panel 2: gradient balance
    ax2.bar(["-P(bind)", "LigandMPNN\nNLL"], [GRAD_PBIND, GRAD_MPNN], color=[CB, CM], width=0.6)
    ax2.set_ylabel("||gradient|| wrt sequence")
    ax2.set_title(f"2. P(bind) gradient dominates\n{GRAD_PBIND / GRAD_MPNN:.0f}x (parity weight ~29)")
    for i, v in enumerate([GRAD_PBIND, GRAD_MPNN]):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax2.set_ylim(0, 1.35)
    ax2.grid(alpha=0.25, axis="y")

    # Panel 3: design outcome
    ax3.axhspan(0.9, 1.0, color=C0, alpha=0.10)
    ax3.axhline(REALISTIC, ls="--", lw=1, color=C3)
    ax3.text(2.3, REALISTIC - 0.05, "realistic (~0.45)", fontsize=8, color=C3)
    ax3.text(2.3, 0.94, "hydrophobic collapse", fontsize=8, color=C0)
    ax3.axvline(29, ls=":", lw=1, color="0.5")
    ax3.text(30, 0.55, "gradient parity", rotation=90, fontsize=7.5, color="0.4", va="center")
    ax3.plot(WEIGHTS, HYDROPHOBIC, "-o", color=C3, lw=2, ms=7)
    ax3.set_xscale("log")
    ax3.set_xticks(WEIGHTS)
    ax3.set_xticklabels(WEIGHTS)
    ax3.set_xlabel("regularizer weight")
    ax3.set_ylabel("hydrophobic fraction of design")
    ax3.set_title("3. Weighted ligand prior\nprevents the collapse")
    ax3.set_ylim(0.3, 1.02)
    ax3.grid(alpha=0.25)

    fig.suptitle("Converging to the recipe: freeze a recycled (physical) scaffold, "
                 "then weight the ligand-aware prior above the P(bind) gradient",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "recipe_convergence.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
