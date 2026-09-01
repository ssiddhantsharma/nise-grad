"""Rescue test: design a fresh sequence on the real apx1049 pocket backbone.

De-novo gradient design plateaus because it never builds the pocket. Here we supply the pocket:
fold the real crystal binder apx1049 (recycling=3) to get its physical backbone, freeze it, then
design a fresh random-init sequence to fit that frozen structure by gradient-descending the
LigandMPNN NLL(soft | backbone, ligand). The designed sequence is refolded independently on
Protenix. If the held-out score jumps toward the real binder while de-novo STE stays at ~0.45, the
bottleneck is the backbone, not the sequence layer. One design per process (a second jit leaks a
JAX tracer). Env: LIGANDMPNN_CKPT, LIGMPNN_MODEL_DIR.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import torch

from nisegrad.boltz_ligand import build_boltz_regularizer
from nisegrad.optimize import AA_ORDER, composition_kl, decode, sigmoid
from nisegrad.oracle import PbindOracle


def load_ligandmpnn(ckpt, ref_dir):
    sys.path.insert(0, ref_dir)
    import ligmpnn_model as ref
    from jligandmpnn.model import LigandMPNN
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = ref.ProteinMPNN(model_type="ligand_mpnn", k_neighbors=ck["num_edges"],
                        atom_context_num=ck["atom_context_num"])
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return LigandMPNN.from_torch(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-seq", required=True, help="real binder sequence whose backbone to freeze")
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--comp-weight", type=float, default=0.0,
                    help="composition_kl penalty weight; >0 stops the MPNN-NLL descent from "
                         "collapsing to poly-alanine (the panel run at 0.0 collapsed on 10/14 ligands)")
    ap.add_argument("--out", default="rescue.json")
    a = ap.parse_args()

    oracle = PbindOracle(num_sampling_steps=25)
    mpnn = load_ligandmpnn(os.environ["LIGANDMPNN_CKPT"], os.environ["LIGMPNN_MODEL_DIR"])
    L = len(a.ref_seq)
    feats = oracle.features_for("G" * L, a.ligand)
    key = jax.random.PRNGKey(0)

    # freeze the REAL pocket backbone (fold apx1049 with recycling for physical geometry)
    ref_oh = jax.nn.one_hot(jnp.asarray([AA_ORDER.index(c) for c in a.ref_seq]), 20)
    _, backbone = oracle.pbind_and_output(ref_oh, feats, key, recycling_steps=3)
    reg = build_boltz_regularizer(mpnn, feats, frozen_output=backbone)

    # design a FRESH sequence (random init per seed) to fit that backbone
    logits = 0.1 * jax.random.normal(jax.random.PRNGKey(a.seed), (L, 20))
    opt = optax.adam(0.1)
    state = opt.init(logits)
    nll_fn = jax.jit(jax.value_and_grad(
        lambda x: reg(jax.nn.softmax(x, -1), backbone, key)[0]
        + a.comp_weight * composition_kl(jax.nn.softmax(x, -1))))
    for i in range(a.steps):
        nll, grad = nll_fn(logits)
        updates, state = opt.update(grad, state)
        logits = optax.apply_updates(logits, updates)
        print(f"step {i:3d}  MPNN NLL {float(nll):+.3f}", flush=True)
    seq = decode(logits)
    pbind, _ = oracle.pbind_and_output(
        jax.nn.one_hot(jnp.argmax(logits, -1), 20), feats, key, recycling_steps=3)

    out = Path(a.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    rows.append({"method": "rescue_realbb", "seed": a.seed, "budget": a.steps,
                 "ligand": a.ligand, "seq": seq, "boltz_pbind": sigmoid(float(pbind))})
    out.write_text(json.dumps(rows, indent=2))
    print(f"WROTE rescue_realbb seed {a.seed}  P(bind)={sigmoid(float(pbind)):.2f}  {seq}", flush=True)


if __name__ == "__main__":
    main()
