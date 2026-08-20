"""Smoke test for gradient-guided diffusion: does geometric guidance form a pocket?

Runs the guided sampler with guidance_scale 0 (stock joltz sampler) vs >0 on apixaban + a generic
poly-A binder, and reports the pocket potential (ligand-burial). If guidance drives the potential
down, the mechanism works and we build the full guided -> jligandmpnn -> Protenix pipeline. If it
does nothing or NaNs, the hook is wrong and we fix it before spending more. GPU; one config/process.
"""

import argparse

import jax
import jax.numpy as jnp
from mosaic.losses.boltz2 import boltz2_trunk, set_binder_sequence

from nisegrad.guided_diffusion import atom_masks, guided_sample, pocket_potential
from nisegrad.oracle import PbindOracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--binder-len", type=int, default=120)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=25)
    a = ap.parse_args()

    oracle = PbindOracle(num_sampling_steps=a.steps)
    feats = oracle.features_for("G" * a.binder_len, a.ligand)
    soft = jax.nn.one_hot(jnp.zeros(a.binder_len, int), 20)      # generic poly-A conditioning
    feats = set_binder_sequence(soft, feats)
    key = jax.random.PRNGKey(0)

    init_emb, trunk = boltz2_trunk(oracle.model, feats, recycling_steps=3, deterministic=True, key=key)
    q, c, to_keys, aeb, adb, ttb = oracle.model.diffusion_conditioning(
        trunk.s, trunk.z, init_emb.relative_position_encoding, feats)
    cond = dict(
        s_trunk=trunk.s, s_inputs=init_emb.s_inputs, feats=feats, multiplicity=1,
        diffusion_conditioning={"q": q, "c": c, "to_keys": to_keys, "atom_enc_bias": aeb,
                                "atom_dec_bias": adb, "token_trans_bias": ttb})

    bmask, lmask = atom_masks(feats)
    pot = lambda x: pocket_potential(x, bmask, lmask)
    sk = jax.random.fold_in(key, 2)

    coords0 = guided_sample(oracle.model.structure_module, feats["atom_pad_mask"], a.steps,
                            key=sk, guidance_scale=0.0, **cond)
    coordsG = guided_sample(oracle.model.structure_module, feats["atom_pad_mask"], a.steps,
                            key=sk, guidance_fn=pot, guidance_scale=a.scale, **cond)

    p0, pg = float(pot(coords0)), float(pot(coordsG))
    print(f"pocket potential  unguided={p0:.3f}  guided(scale={a.scale})={pg:.3f}  drop={p0 - pg:+.3f}")
    print("MECHANISM WORKS" if pg < p0 - 0.05 else "no meaningful drop (fix hook / scale)")


if __name__ == "__main__":
    main()
