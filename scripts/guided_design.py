"""Guided-diffusion pipeline: guided_sample forms a pocket around the ligand (poly-A conditioning,
no real binder seed), we freeze that backbone, design a fresh sequence to fit it (LigandMPNN NLL),
and write it for an independent Protenix refold. One design/process (a second jit leaks a tracer).
Env: LIGANDMPNN_CKPT, LIGMPNN_MODEL_DIR."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
import torch
from mosaic.losses.boltz2 import boltz2_trunk, set_binder_sequence

from nisegrad.boltz_ligand import build_boltz_regularizer
from nisegrad.guided_diffusion import atom_masks, guided_sample, pocket_potential
from nisegrad.optimize import decode, sigmoid
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


def guided_backbone(oracle, feats, scale, steps, key):
    """Guided-diffusion pocket backbone -> (frozen output with structure/backbone coords, potential)."""
    init_emb, trunk = boltz2_trunk(oracle.model, feats, recycling_steps=3, deterministic=True, key=key)
    q, c, to_keys, aeb, adb, ttb = oracle.model.diffusion_conditioning(
        trunk.s, trunk.z, init_emb.relative_position_encoding, feats)
    cond = {
        "s_trunk": trunk.s, "s_inputs": init_emb.s_inputs, "feats": feats, "multiplicity": 1,
        "diffusion_conditioning": {"q": q, "c": c, "to_keys": to_keys, "atom_enc_bias": aeb,
                                "atom_dec_bias": adb, "token_trans_bias": ttb}}
    bmask, lmask = atom_masks(feats)
    pot = lambda x: pocket_potential(x, bmask, lmask)
    coords = guided_sample(oracle.model.structure_module, feats["atom_pad_mask"], steps,
                           key=jax.random.fold_in(key, 2), guidance_fn=pot, guidance_scale=scale, **cond)
    # extract N,CA,C,O backbone from the guided all-atom coords (mosaic's first_atom_idx trick)
    a2t = jnp.asarray(feats["atom_to_token"])[0]                       # [A, N_token]
    first = jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(a2t.T)  # [N_token]
    allatom = coords[0]                                                # [A, 3]
    bb = jnp.stack([allatom[first + i] for i in range(4)], -2)         # [N_token, 4, 3]
    return SimpleNamespace(structure_coordinates=coords, backbone_coordinates=bb), float(pot(coords))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--binder-len", type=int, default=120)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--design-steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="guided.json")
    a = ap.parse_args()

    oracle = PbindOracle(num_sampling_steps=a.steps)
    mpnn = load_ligandmpnn(os.environ["LIGANDMPNN_CKPT"], os.environ["LIGMPNN_MODEL_DIR"])
    feats = oracle.features_for("G" * a.binder_len, a.ligand)
    feats = set_binder_sequence(jax.nn.one_hot(jnp.zeros(a.binder_len, int), 20), feats)
    key = jax.random.PRNGKey(a.seed)

    frozen, pot = guided_backbone(oracle, feats, a.scale, a.steps, key)
    reg = build_boltz_regularizer(mpnn, feats, frozen_output=frozen)

    # design a fresh sequence to fit the guided pocket
    logits = 0.1 * jax.random.normal(jax.random.fold_in(key, 7), (a.binder_len, 20))
    opt = optax.adam(0.1)
    state = opt.init(logits)
    nll_fn = jax.jit(jax.value_and_grad(lambda x: reg(jax.nn.softmax(x, -1), frozen, key)[0]))
    for _ in range(a.design_steps):
        _nll, grad = nll_fn(logits)
        updates, state = opt.update(grad, state)
        logits = optax.apply_updates(logits, updates)
    seq = decode(logits)
    pbind, _ = oracle.pbind_and_output(
        jax.nn.one_hot(jnp.argmax(logits, -1), 20), feats, key, recycling_steps=3)

    out = Path(a.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    rows.append({"method": "guided_design", "seed": a.seed, "budget": a.design_steps,
                 "ligand": a.ligand, "seq": seq, "boltz_pbind": sigmoid(float(pbind)),
                 "pocket_potential": pot})
    out.write_text(json.dumps(rows, indent=2))
    print(f"WROTE guided_design seed {a.seed}  pocket={pot:.2f}  P(bind)={sigmoid(float(pbind)):.2f}  {seq}",
          flush=True)


if __name__ == "__main__":
    main()
