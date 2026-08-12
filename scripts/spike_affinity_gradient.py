"""Feasibility spike for nise-grad.

Question: with a small-molecule ligand present, does the gradient of P(bind) flow
back to the (soft) binder sequence through the differentiable structure model?

If yes, the core primitive works and the rest of nise-grad is assembly. This runs
the Boltz-2 structure model (joltz) to fold a soft sequence + fixed ligand, attaches
the Boltz-2 affinity head (loaded from the separate affinity checkpoint) to score
P(bind), and takes a gradient of that scalar w.r.t. the soft sequence.

Needs: mosaic + the joltz branch with the affinity head, and the two Boltz-2
checkpoints (boltz2_conf.ckpt for structure, boltz2_aff.ckpt for the affinity head)
plus the CCD/mols cache in ~/.boltz. GPU strongly preferred.
"""

import os
from pathlib import Path

import boltz.model.modules.affinity as boltz_affinity
import jax
import jax.numpy as jnp
import joltz
import numpy as np
import torch
from boltz.data import const
from mosaic.losses.boltz2 import (
    boltz2_forward_from_trunk,
    boltz2_trunk,
    load_boltz2,
    load_features_and_structure_writer,
    set_binder_sequence,
)

CONF_CKPT = Path.home() / ".boltz" / "boltz2_conf.ckpt"
AFF_CKPT = Path.home() / ".boltz" / "boltz2_aff.ckpt"
# Boltz featurization cache (must contain ccd.pkl + mols/); override via env var.
BOLTZ_CACHE = Path(os.environ.get("NISEGRAD_BOLTZ_CACHE", str(Path.home() / ".boltz")))

BINDER = "GSHMKEIAQLKQKIEALEKKNAALKEKNQALKYG"  # 34-mer placeholder binder
YAML = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {BINDER}
      msa: empty
  - ligand:
      id: B
      smiles: 'c1ccc(cc1)C(=O)O'
"""


def load_affinity_head():
    """Load Boltz-2 affinity_module1 from the affinity checkpoint into JAX/joltz."""
    ck = torch.load(AFF_CKPT, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    args = dict(hp["affinity_model_args1"])
    args["pairformer_args"] = {**args["pairformer_args"], "use_trifast": False,
                               "activation_checkpointing": False}
    args["transformer_args"] = {**args["transformer_args"], "activation_checkpointing": False}
    module = boltz_affinity.AffinityModule(hp["token_s"], hp["token_z"], **args)
    prefix = "affinity_module1."
    sub = {k[len(prefix):]: v for k, v in ck["state_dict"].items() if k.startswith(prefix)}
    module.load_state_dict(sub, strict=False)
    module.eval()
    return joltz.from_torch(module)


def main():
    joltz2 = load_boltz2(CONF_CKPT)
    affinity = load_affinity_head()
    features, _ = load_features_and_structure_writer(YAML, cache=BOLTZ_CACHE)

    # The affinity head wants an affinity_token_mask; for a single-ligand complex it
    # is just the ligand (NONPOLYMER) tokens.
    features = dict(features)
    mol_type = jnp.asarray(features["mol_type"])
    features["affinity_token_mask"] = (
        mol_type == const.chain_type_ids["NONPOLYMER"]
    ).astype(jnp.float32)

    key = jax.random.PRNGKey(0)

    def pbind_logit(soft_sequence):
        feats = set_binder_sequence(soft_sequence, features)
        init_emb, trunk = boltz2_trunk(
            joltz2, feats, recycling_steps=0, deterministic=True, key=key)
        out = boltz2_forward_from_trunk(
            joltz2, feats, init_emb, trunk,
            num_sampling_steps=25, deterministic=True, key=key)
        aff = affinity(
            init_emb.s_inputs,
            trunk.z,
            out.structure_coordinates,
            feats,
            multiplicity=1,
            key=key,
            deterministic=True,
        )
        return aff["affinity_logits_binary"].sum()

    soft = jax.nn.softmax(
        jax.random.normal(jax.random.PRNGKey(1), (len(BINDER), 20)), axis=-1)
    with jax.default_matmul_precision("float32"):
        value, grad = jax.value_and_grad(pbind_logit)(soft)
    grad = np.asarray(grad)

    print(f"P(bind) logit: {float(value):.4f}")
    print(f"grad wrt sequence: finite={np.isfinite(grad).all()} "
          f"max|grad|={np.abs(grad).max():.3e} nonzero_frac={(np.abs(grad) > 0).mean():.2f}")
    assert np.isfinite(grad).all(), "non-finite gradient"
    assert np.abs(grad).max() > 0, "zero gradient (no signal to the sequence)"
    print("SPIKE OK: P(bind) is differentiable w.r.t. the sequence through structure + affinity.")


if __name__ == "__main__":
    main()
