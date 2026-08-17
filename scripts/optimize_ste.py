"""Straight-through gradient design: a real discrete binder in ~25 steps.

The reported P(bind) is the discrete design's (forward pass folds the argmax sequence), so the
trajectory is honest. Env: NISEGRAD_BOLTZ_CACHE.
"""

from nisegrad.optimize import decode, optimize_pbind, sigmoid
from nisegrad.oracle import PbindOracle

BINDER_LEN = 30
LIGAND = "c1ccc(cc1)C(=O)O"  # benzoic acid


def main():
    oracle = PbindOracle(num_sampling_steps=25)   # nss>=25 for physical geometry
    feats = oracle.features_for("G" * BINDER_LEN, LIGAND)
    logits, traj = optimize_pbind(
        oracle, feats, BINDER_LEN, steps=25, lr=0.1,
        recycling_steps=3, straight_through=True)
    seq = decode(logits)
    hyd = sum(c in "AVLIMFWC" for c in seq) / len(seq)
    print(f"design: {seq}")
    print(f"discrete P(bind) {sigmoid(traj[0]):.2f} -> {sigmoid(traj[-1]):.2f}  "
          f"hydrophobic fraction {hyd:.2f}")


if __name__ == "__main__":
    main()
