"""Per-step geometric guidance inside Boltz's EDM sampler: push each denoised x0 estimate down the
gradient of a pocket potential so the trajectory forms a pocket around the ligand. Guidance is
geometric, not the learned ipTM head. Reimplements the joltz sampler loop; keep in sync."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from boltz.data import const


def atom_masks(feats):
    """Atom-level (binder, ligand) boolean masks from token mol_type via atom_to_token."""
    mol_type = jnp.asarray(feats["mol_type"])[0]
    a2t = jnp.asarray(feats["atom_to_token"])[0]
    tok = a2t.argmax(-1) if a2t.ndim == 2 else a2t
    ligand = mol_type[tok] == const.chain_type_ids["NONPOLYMER"]
    return (~ligand), ligand


def pocket_potential(coords, binder_mask, ligand_mask, contact=4.0):
    """Ligand-burial: mean softplus(nearest-binder-distance - contact) over ligand atoms. Lower =
    more enclosed. Masked all-pairs keeps it jit-safe; differentiable in coords."""
    x = coords[0]
    d = jnp.linalg.norm(x[:, None, :] - x[None, :, :] + 1e-6, axis=-1)
    d = jnp.where(ligand_mask[:, None] & binder_mask[None, :], d, 1e6)
    softmin = -0.5 * jax.nn.logsumexp(-2.0 * d, axis=-1)
    return (jax.nn.softplus(softmin - contact) * ligand_mask).sum() / jnp.maximum(ligand_mask.sum(), 1.0)


def guided_sample(structure_module, atom_mask, num_sampling_steps, *, key,
                  guidance_fn=None, guidance_scale=0.0, **network_condition_kwargs):
    """joltz EDM sampler; after each x0 estimate, step it down guidance_fn's gradient (scaled by
    t_hat^2). guidance_fn: coords [1,M,3] -> scalar. guidance_scale=0 = stock sampler."""
    import joltz

    sm = structure_module
    shape = (*atom_mask.shape, 3)
    sigmas = sm.sample_schedule(num_sampling_steps)
    gammas = jnp.where(sigmas > sm.gamma_min, sm.gamma_0, 0.0)
    step_scale = sm.step_scale

    @jax.checkpoint
    def body(carry, inp):
        sigma_tm, sigma_t, gamma = inp
        atom_coords, key = carry
        random_R, random_tr = joltz.compute_random_augmentation(key=key)
        key = jax.random.fold_in(key, 1)
        atom_coords = atom_coords - atom_coords.mean(axis=-2, keepdims=True)
        atom_coords = jnp.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr

        t_hat = sigma_tm * (1 + gamma)
        noise_var = sm.noise_scale**2 * (t_hat**2 - sigma_tm**2)
        eps = jnp.sqrt(noise_var) * jax.random.normal(shape=shape, key=key)
        key = jax.random.fold_in(key, 1)
        noisy = atom_coords + eps
        denoised = sm.preconditioned_network_forward(
            noisy, t_hat, network_condition_kwargs=dict(**network_condition_kwargs), key=key)

        if guidance_fn is not None and guidance_scale:
            g = jax.grad(guidance_fn)(denoised)
            denoised = denoised - guidance_scale * (t_hat**2) * g

        if sm.alignment_reverse_diff:
            noisy = joltz.weighted_rigid_align(noisy, denoised, atom_mask, atom_mask)

        denoised_over_sigma = (noisy - denoised) / t_hat
        atom_coords_next = noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
        return (atom_coords_next, jax.random.fold_in(key, 0)), None

    init = (sigmas[0] * jax.random.normal(shape=shape, key=key), jax.random.fold_in(key, 1))
    (atom_coords, _), _ = jax.lax.scan(body, init, (sigmas[:-1], sigmas[1:], gammas[1:]))
    return atom_coords


def guided_forward_from_trunk(model, features, initial_embedding, trunk_state, *,
                              num_sampling_steps, guidance_fn, guidance_scale, key):
    """boltz2_forward_from_trunk with guided_sample swapped in; returns a full StructureModelOutput
    (so ifsr and the pae/plddt binding terms work on the guided structure). Mirrors mosaic; keep in sync."""
    from mosaic.losses.boltz2 import (BOLTZ2_DISTOGRAM_BINS, PAE_BINS, StructureModelOutput,
                                      _BOLTZ_TOKATOM_TO_ATOM37, ref_atoms, scatter_atom37)

    distogram_logits = model.distogram_module(trunk_state.z)[0, :, :, 0, :]
    q, c, to_keys, aeb, adb, ttb = model.diffusion_conditioning(
        trunk_state.s, trunk_state.z, initial_embedding.relative_position_encoding, features)
    with jax.default_matmul_precision("float32"):
        structure_coordinates = guided_sample(
            model.structure_module, features["atom_pad_mask"], num_sampling_steps,
            key=jax.random.fold_in(key, 2), guidance_fn=guidance_fn, guidance_scale=guidance_scale,
            s_trunk=trunk_state.s, s_inputs=initial_embedding.s_inputs, feats=features, multiplicity=1,
            diffusion_conditioning={"q": q, "c": c, "to_keys": to_keys, "atom_enc_bias": aeb,
                                    "atom_dec_bias": adb, "token_trans_bias": ttb})
    confidence = model.confidence_module(
        s_inputs=initial_embedding.s_inputs, s=trunk_state.s, z=trunk_state.z,
        x_pred=structure_coordinates, feats=features, pred_distogram_logits=distogram_logits[None],
        key=jax.random.fold_in(key, 5), deterministic=True)

    fu = jax.tree.map(lambda x: x[0], features)
    assert ref_atoms["UNK"][:4] == ["N", "CA", "C", "O"]
    first = jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(fu["atom_to_token"].T)
    all_atom = structure_coordinates[0]
    backbone = jnp.stack([all_atom[first + i] for i in range(4)], -2)
    n_tokens = fu["res_type"].shape[0]
    res_slot = fu["res_type"].argmax(-1)
    a2t = fu["atom_to_token"].argmax(-1)
    tokatom_idx = jnp.arange(a2t.shape[0]) - first[a2t]
    atom37_idx = jnp.asarray(_BOLTZ_TOKATOM_TO_ATOM37)[res_slot[a2t], tokatom_idx]
    atom37_idx = jnp.where(fu["atom_pad_mask"] > 0.5, atom37_idx, jnp.int32(-1))
    atom37_coords, atom37_mask = scatter_atom37(all_atom, a2t, atom37_idx, n_tokens)

    return StructureModelOutput(
        distogram_logits=distogram_logits, distogram_bins=BOLTZ2_DISTOGRAM_BINS,
        plddt=confidence.plddt[0], pae=confidence.pae[0], pae_logits=confidence.pae_logits[0],
        pae_bins=PAE_BINS, structure_coordinates=structure_coordinates, backbone_coordinates=backbone,
        full_sequence=features["res_type"][0][:, 2:22], asym_id=features["asym_id"][0],
        residue_idx=features["residue_index"][0], atom37_coords=atom37_coords, atom37_mask=atom37_mask)
