"""Ligand-aware sequence regularizer, backed by jigandmpnn (Boyd's JAX/Equinox LigandMPNN).

The binder's soft sequence is scored by LigandMPNN given the predicted backbone AND the
ligand context, so the negative log-likelihood keeps the design both protein-like and
ligand-compatible, unlike a ligand-blind ProteinMPNN regularizer, which was blind to the
very interaction P(bind) is reward-hacking.

The two pure functions (af20_to_mpnn20, nll_from_logprobs) carry the bug-prone logic and are
unit-tested. Extracting the backbone from the Boltz StructureModelOutput is injected via
`coords_from_output`; the end-to-end loop is validated on a real Boltz run (GPU).
"""

from __future__ import annotations

import jax.numpy as jnp

# nise-grad soft sequences are in AlphaFold/Boltz order; LigandMPNN uses its own alphabet
# (data_utils.restype_str_to_int). Both verified against primary sources.
AF_ORDER = "ARNDCQEGHILKMFPSTWYV"
MPNN_ORDER = "ACDEFGHIKLMNPQRSTVWY"  # first 20 LigandMPNN restypes (index 20 = X, dropped)
# column j selects the AF index of the residue that is MPNN-index j
_AF_TO_MPNN = jnp.asarray([AF_ORDER.index(a) for a in MPNN_ORDER])


def af20_to_mpnn20(soft_af):
    """Reorder a soft sequence [..., 20] from AF order to LigandMPNN order."""
    return soft_af[..., _AF_TO_MPNN]


def nll_from_logprobs(soft_mpnn, log_probs, binder_mask):
    """Mean cross-entropy NLL of soft_mpnn [L,20] under log_probs [L,21] over binder
    positions (binder_mask [L], 1 = designed). Both in LigandMPNN order."""
    per_pos = -(soft_mpnn * log_probs[:, :20]).sum(-1)  # [L]
    return (per_pos * binder_mask).sum() / jnp.maximum(binder_mask.sum(), 1.0)


class LigandMPNNRegularizer:
    """Callable matching optimize_pbind's `mpnn(soft, output, key) -> (nll, aux)` interface.

    model  : jigandmpnn ProteinMPNN, ligand model (jigandmpnn._load_model of a checkpoint)
    feats  : dict of the FIXED LigandMPNN inputs (mask, Y_m, Y_t, R_idx, chain_labels,
             chain_mask, randn), each batched [1, ...], everything but the coordinates
    struct_from_output : output -> (X [1,L,4,3] binder backbone N/Ca/C/O, Y [1,L,M,3]
             ligand-atom context), both differentiable, taken from the predicted structure
    binder_mask : [L] float, 1.0 on designed binder residues
    """

    def __init__(self, model, feats: dict, struct_from_output, binder_mask):
        self.model = model
        self.feats = feats
        self.struct_from_output = struct_from_output
        self.binder_mask = binder_mask

    def __call__(self, soft_af, output, key):
        f = self.feats
        X, Y = self.struct_from_output(output)
        soft_mpnn = af20_to_mpnn20(soft_af)  # [L,20] in LigandMPNN order
        # jigandmpnn.decode wants a soft [.,21] (index 20 = X); pad an all-zero X column.
        soft21 = jnp.concatenate(
            [soft_mpnn, jnp.zeros((*soft_mpnn.shape[:-1], 1), soft_mpnn.dtype)], -1
        )[None]  # [1,L,21]
        # Encode the frozen backbone + ligand context (key=None -> no augment_eps noise,
        # deterministic), then teacher-force the soft sequence through decode.
        h_V, h_E, E_idx = self.model.encode(
            X=X, mask=f["mask"], residue_idx=f["R_idx"],
            chain_encoding_all=f["chain_labels"],
            Y=Y, Y_t=f["Y_t"], Y_m=f["Y_m"], key=None)
        log_probs = self.model.decode(
            S=soft21, h_V=h_V, h_E=h_E, E_idx=E_idx,
            mask=f["mask"], decoding_order=f["randn"])[0]  # [L,21]
        return nll_from_logprobs(soft_mpnn, log_probs, self.binder_mask), None
