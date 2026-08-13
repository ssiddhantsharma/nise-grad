"""Build a ligand-aware LigandMPNN regularizer from a Boltz-2 feature dict + output.

Grounded in the verified Boltz feature layout (checked on a real benzoic-acid complex):
  - tokens split by `mol_type`: PROTEIN (binder) vs NONPOLYMER (ligand)
  - `atom_to_token` (one-hot) maps padded atoms -> tokens; `atom_pad_mask` drops padding
  - `ref_element.argmax(-1)` is the atomic number == LigandMPNN's Y_t
  - `output.backbone_coordinates` is [T,4,3] in N,CA,C,O order (asserted in mosaic)
  - `output.structure_coordinates[0]` is the [A,3] all-atom coords

For a small-molecule ligand (M <= atom_context_num) every binder residue sees the whole
ligand, so Y is the same M atoms broadcast per residue -- exactly the pre-limited context
the use_side_chains=False model expects, no per-residue topk needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from boltz.data import const

from .ligand_mpnn_reg import LigandMPNNRegularizer


def build_boltz_regularizer(model, features: dict, *, frozen_output=None, randn_seed: int = 0):
    """Return a LigandMPNNRegularizer wired to this complex's fixed featurization.

    frozen_output: a StructureModelOutput to freeze the backbone + ligand geometry from
    (fold once with recycling_steps>=3 for a physical scaffold). When given, the regularizer
    scores every step's soft sequence against this fixed physical structure -- the correct way
    to use an inverse-folding prior. When None, it reads each step's live (recycling=0) output,
    which is non-physical and only kept for comparison.
    """
    mol_type = np.asarray(features["mol_type"])[0]                    # [T]
    a2t = np.asarray(features["atom_to_token"])[0].argmax(-1)         # [A] token per atom
    apm = np.asarray(features["atom_pad_mask"])[0] > 0.5              # [A]
    elem = np.asarray(features["ref_element"])[0].argmax(-1)         # [A] atomic number

    binder_tok = mol_type == const.chain_type_ids["PROTEIN"]         # [T]
    lig_atom = (mol_type[a2t] == const.chain_type_ids["NONPOLYMER"]) & apm  # [A]
    L, M = int(binder_tok.sum()), int(lig_atom.sum())
    if L == 0 or M == 0:
        raise ValueError(f"expected a protein binder + a ligand; got L={L}, M={M}")

    binder_idx = jnp.asarray(np.nonzero(binder_tok)[0])              # [L]
    lig_idx = jnp.asarray(np.nonzero(lig_atom)[0])                   # [M]

    # fixed (non-coordinate) LigandMPNN inputs, batched [1, ...]
    Yt = jnp.broadcast_to(jnp.asarray(elem[lig_atom], jnp.float32), (L, M))[None]
    feats = {
        "mask": jnp.ones((1, L)),
        "chain_mask": jnp.ones((1, L)),                              # all residues designed
        "Y_m": jnp.ones((1, L, M)),
        "Y_t": Yt,
        "R_idx": jnp.asarray(np.asarray(features["residue_index"])[0][binder_tok],
                             jnp.float32)[None],
        "chain_labels": jnp.asarray(np.asarray(features["asym_id"])[0][binder_tok],
                                    jnp.float32)[None],
        # fixed decoding order (np.random keeps it off the jax key; seeded for reproducibility)
        "randn": jnp.asarray(np.random.RandomState(randn_seed).randn(1, L), jnp.float32),
    }

    def extract(output):
        X = output.backbone_coordinates[binder_idx][None]           # [1,L,4,3] N,CA,C,O
        Y_atoms = output.structure_coordinates[0][lig_idx]          # [M,3]
        Y = jnp.broadcast_to(Y_atoms, (L, M, 3))[None]              # [1,L,M,3]
        return X, Y

    if frozen_output is not None:
        X0, Y0 = jax.lax.stop_gradient(extract(frozen_output)[0]), \
            jax.lax.stop_gradient(extract(frozen_output)[1])
        def struct_from_output(_output):
            return X0, Y0
    else:
        struct_from_output = extract

    return LigandMPNNRegularizer(model, feats, struct_from_output, jnp.ones(L))
