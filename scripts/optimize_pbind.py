"""Optimize a binder sequence to maximize P(bind) for a ligand, MPNN-regularized so
the sequence stays protein-like; save the curve."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mosaic.proteinmpnn.mpnn import load_mpnn
from mosaic.losses.protein_mpnn import ProteinMPNNLoss

from nisegrad.oracle import PbindOracle
from nisegrad.optimize import optimize_pbind, decode, sigmoid

BINDER_LEN = 30
LIGAND = "c1ccc(cc1)C(=O)O"  # benzoic acid
STEPS = 30
MPNN_WEIGHT = 2.0


def main():
    oracle = PbindOracle(num_sampling_steps=2)
    feats = oracle.features_for("G" * BINDER_LEN, LIGAND)
    mpnn = ProteinMPNNLoss(mpnn=load_mpnn(), num_samples=1)

    logits, traj = optimize_pbind(
        oracle, feats, BINDER_LEN, mpnn=mpnn, mpnn_weight=MPNN_WEIGHT, steps=STEPS, lr=0.05)

    print("sequence:", decode(logits))
    print(f"P(bind) {sigmoid(traj[0]):.4f} -> {sigmoid(traj[-1]):.4f}")

    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot([sigmoid(t) for t in traj], marker="o", ms=3)
    plt.xlabel("gradient step")
    plt.ylabel("P(bind)")
    plt.title(f"nise-grad: {BINDER_LEN}aa binder vs benzoic acid (MPNN-regularized)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "pbind_optimization.png", dpi=130)
    print("saved figures/pbind_optimization.png")


if __name__ == "__main__":
    main()
