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
