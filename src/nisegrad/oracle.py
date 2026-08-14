"""Differentiable P(bind): (soft binder sequence, fixed ligand) -> binding logit."""

from __future__ import annotations

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

DEFAULT_CONF_CKPT = Path.home() / ".boltz" / "boltz2_conf.ckpt"
DEFAULT_AFF_CKPT = Path.home() / ".boltz" / "boltz2_aff.ckpt"
DEFAULT_CACHE = Path(os.environ.get("NISEGRAD_BOLTZ_CACHE", str(Path.home() / ".boltz")))

_YAML = """version: 1
sequences:
  - protein:
      id: A
      sequence: {sequence}
      msa: empty
  - ligand:
      id: B
      smiles: '{smiles}'
"""


def load_affinity_head(aff_ckpt: Path = DEFAULT_AFF_CKPT):
    """Load Boltz-2 affinity_module1 from the affinity checkpoint into JAX/joltz."""
    ck = torch.load(aff_ckpt, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    args = dict(hp["affinity_model_args1"])
    args["pairformer_args"] = {**args["pairformer_args"], "use_trifast": False,
                               "activation_checkpointing": False}
    args["transformer_args"] = {**args["transformer_args"], "activation_checkpointing": False}
    module = boltz_affinity.AffinityModule(hp["token_s"], hp["token_z"], **args)
    prefix = "affinity_module1."
    sub = {k[len(prefix):]: v for k, v in ck["state_dict"].items() if k.startswith(prefix)}
    missing, unexpected = module.load_state_dict(sub, strict=False)
    assert not missing and not unexpected, (missing[:3], unexpected[:3])
    module.eval()
    aff = joltz.from_torch(module)
    # from_torch stores weights as numpy; make them jax arrays so the distogram
    # embedding (weight[traced_tokens]) works under jit -- numpy[tracer] does not.
    return jax.tree.map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, aff)


class PbindOracle:
    """Boltz-2 structure + affinity head, loaded once; pbind_and_output is differentiable in the sequence."""

    def __init__(self, conf_ckpt: Path = DEFAULT_CONF_CKPT,
                 aff_ckpt: Path = DEFAULT_AFF_CKPT,
                 cache: Path = DEFAULT_CACHE,
                 num_sampling_steps: int = 10):
        self.model = load_boltz2(conf_ckpt)
        self.affinity = load_affinity_head(aff_ckpt)
        self.cache = cache
        self.num_sampling_steps = num_sampling_steps

    def features_for(self, placeholder_sequence: str, ligand_smiles: str) -> dict:
        """Featurize the protein+ligand complex; placeholder only sets the binder length."""
        yaml = _YAML.format(sequence=placeholder_sequence, smiles=ligand_smiles)
        features, _ = load_features_and_structure_writer(yaml, cache=self.cache)
        features = dict(features)
        mol_type = jnp.asarray(features["mol_type"])
        features["affinity_token_mask"] = (
            mol_type == const.chain_type_ids["NONPOLYMER"]
        ).astype(jnp.float32)
        return features

    def pbind_and_output(self, soft_sequence, features: dict, key,
                         recycling_steps: int = 0, num_sampling_steps: int | None = None):
        """(binding logit, StructureModelOutput) for soft_sequence against the ligand.

        recycling_steps=0 (default) is the fast per-gradient-step setting, but it yields
        non-physical geometry; use recycling_steps=3 for a physical structure (e.g. a
        frozen scaffold for a structure-based prior)."""
        feats = set_binder_sequence(soft_sequence, features)
        init_emb, trunk = boltz2_trunk(
            self.model, feats, recycling_steps=recycling_steps, deterministic=True, key=key)
        output = boltz2_forward_from_trunk(
            self.model, feats, init_emb, trunk,
            num_sampling_steps=num_sampling_steps or self.num_sampling_steps,
            deterministic=True, key=key)
        aff = self.affinity(
            init_emb.s_inputs, trunk.z, output.structure_coordinates, feats,
            multiplicity=1, key=key, deterministic=True)
        return aff["affinity_logits_binary"].reshape(()), output
