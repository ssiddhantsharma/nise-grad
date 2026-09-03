"""LigandMPNN-as-projector: late projection of a design onto the foldable manifold.

DBMol's optimize-then-project idea via jigandmpnn's differentiable LigandMPNN (encode+decode, no
sampler). Fold the design, freeze the structure S, then gradient-descend the LigandMPNN
NLL(soft | S, ligand) over the sequence logits, pulling a (poly-Q) design toward a sequence that
would fold to S. The projected sequence is written out for held-out (Protenix) scoring vs the
original. One design per process (a second jit/optimize in the same process leaks a JAX tracer).

Env: LIGANDMPNN_CKPT.
"""

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from nisegrad.boltz_ligand import build_boltz_regularizer
from nisegrad.optimize import AA_ORDER, decode, sigmoid
from nisegrad.oracle import PbindOracle


def load_ligandmpnn(ckpt):
    # jigandmpnn (Boyd) vendors the torch LigandMPNN reference and does the torch->JAX
    # conversion, so only the checkpoint is needed (no LIGMPNN_MODEL_DIR).
    from jigandmpnn import _load_model
    return _load_model(Path(ckpt), "ligand_mpnn")


def project(oracle, mpnn, feats, seq, steps=40, lr=0.1, bias=2.0):
    key = jax.random.PRNGKey(0)
    onehot = jax.nn.one_hot(jnp.asarray([AA_ORDER.index(c) for c in seq]), 20)
    _, structure = oracle.pbind_and_output(onehot, feats, key, recycling_steps=3)  # freeze S
    reg = build_boltz_regularizer(mpnn, feats, frozen_output=structure)
    logits = bias * onehot
    opt = optax.adam(lr)
    state = opt.init(logits)
    nll_fn = jax.jit(jax.value_and_grad(lambda x: reg(jax.nn.softmax(x, -1), structure, key)[0]))
    for _ in range(steps):
        _, grad = nll_fn(logits)
        updates, state = opt.update(grad, state)
        logits = optax.apply_updates(logits, updates)
    proj = decode(logits)
    pbind, _ = oracle.pbind_and_output(
        jax.nn.one_hot(jnp.argmax(logits, -1), 20), feats, key, recycling_steps=3)
    return proj, sigmoid(float(pbind))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="design sequence to project")
    ap.add_argument("--ligand", default="c1ccc(cc1)C(=O)O")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="proj.json")
    a = ap.parse_args()

    oracle = PbindOracle(num_sampling_steps=25)
    mpnn = load_ligandmpnn(os.environ["LIGANDMPNN_CKPT"])
    feats = oracle.features_for("G" * len(a.seq), a.ligand)
    proj, pbind = project(oracle, mpnn, feats, a.seq)

    out = Path(a.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    rows.append({"method": "ste_proj", "seed": a.seed, "budget": 25, "ligand": a.ligand,
                 "seq": proj, "boltz_pbind": pbind})
    out.write_text(json.dumps(rows, indent=2))
    print(f"PROJECTED seed {a.seed}  P(bind)={pbind:.2f}  {a.seq[:24]}... -> {proj[:24]}...", flush=True)


if __name__ == "__main__":
    main()
