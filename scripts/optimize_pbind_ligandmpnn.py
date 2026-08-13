"""Optimize a 30aa binder vs benzoic acid with the LIGAND-AWARE (jligandmpnn) regularizer --
the ligand-aware counterpart to optimize_pbind.py's ligand-blind ProteinMPNN baseline.

The question this answers: does conditioning the sequence prior on the ligand (LigandMPNN)
change the P(bind) reward-hacking that a ligand-blind prior (ProteinMPNN) could not fix?

Env:
  LIGANDMPNN_CKPT    path to ligandmpnn_v_32_010_25.pt
  LIGMPNN_MODEL_DIR  dir with the LigandMPNN torch module importable as `ligmpnn_model`
                     (used once to load weights via from_torch)
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.environ["LIGMPNN_MODEL_DIR"])
import ligmpnn_model as ref
from jligandmpnn.model import LigandMPNN

from nisegrad.boltz_ligand import build_boltz_regularizer
from nisegrad.optimize import decode, optimize_pbind, sigmoid
from nisegrad.oracle import PbindOracle

BINDER_LEN = 30
LIGAND = "c1ccc(cc1)C(=O)O"  # benzoic acid
STEPS = 30
MPNN_WEIGHT = float(os.environ.get("MPNN_WEIGHT", "2.0"))


def load_jlig(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = ref.ProteinMPNN(model_type="ligand_mpnn", k_neighbors=ck["num_edges"],
                        atom_context_num=ck["atom_context_num"])
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return LigandMPNN.from_torch(m)


def main():
    jmodel = load_jlig(os.environ["LIGANDMPNN_CKPT"])
    oracle = PbindOracle(num_sampling_steps=2)  # fast per-step fold for the P(bind) gradient
    feats = oracle.features_for("G" * BINDER_LEN, LIGAND)

    # Fold a PHYSICAL scaffold ONCE with recycling (no gradient) and freeze it. The recycling=0
    # per-step structure is noise (~0/119 backbone bonds); recycling=3 gives 119/119. LigandMPNN
    # is an inverse-folding prior, so scoring the sequence against a fixed physical backbone is
    # the correct use -- and it makes the geometry gate below pass.
    key = jax.random.PRNGKey(0)
    soft0 = jax.nn.softmax(0.1 * jax.random.normal(key, (BINDER_LEN, 20)), axis=-1)
    _, scaffold = oracle.pbind_and_output(
        soft0, feats, key, recycling_steps=3, num_sampling_steps=100)
    reg = build_boltz_regularizer(jmodel, feats, frozen_output=scaffold)

    # sanity: confirm the frozen scaffold is real geometry before optimizing
    X, Y = reg.struct_from_output(scaffold)
    N, CA, C = X[0, :, 0], X[0, :, 1], X[0, :, 2]
    d_nca = float(jnp.linalg.norm(CA - N, axis=-1).mean())
    d_cac = float(jnp.linalg.norm(C - CA, axis=-1).mean())
    print(f"SANITY backbone: N-CA {d_nca:.3f} A (exp ~1.46), CA-C {d_cac:.3f} A (exp ~1.52); "
          f"X {X.shape} ligand-Y {Y.shape}")
    assert 1.2 < d_nca < 1.8 and 1.3 < d_cac < 1.8, "extracted backbone geometry is wrong"

    logits, traj = optimize_pbind(
        oracle, feats, BINDER_LEN, mpnn=reg, mpnn_weight=MPNN_WEIGHT, steps=STEPS, lr=0.05)
    seq = decode(logits)
    hyd = sum(c in "AVLIMFWC" for c in seq) / len(seq)
    print(f"sequence: {seq}  (hydrophobic fraction {hyd:.2f}; collapse ~1.0, realistic ~0.45)")
    print(f"P(bind) {sigmoid(traj[0]):.4f} -> {sigmoid(traj[-1]):.4f}")

    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot([sigmoid(t) for t in traj], marker="o", ms=3)
    plt.xlabel("gradient step")
    plt.ylabel("P(bind)")
    plt.title(f"nise-grad: {BINDER_LEN}aa vs benzoic acid (LigandMPNN-regularized)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "pbind_optimization_ligandmpnn.png", dpi=130)
    print("saved figures/pbind_optimization_ligandmpnn.png")


if __name__ == "__main__":
    main()
