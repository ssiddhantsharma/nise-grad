"""Gradient ascent on the binder sequence: maximize the differentiable P(bind)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

# AlphaFold / Boltz standard-20 residue order (matches set_binder_sequence's [N,20]).
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

# SwissProt background amino-acid frequencies in AA_ORDER; the target for composition_kl.
_AA_BG = jnp.asarray([8.25, 5.53, 4.06, 5.46, 1.38, 3.93, 6.72, 7.07, 2.27, 5.91,
                      9.66, 5.80, 2.41, 3.86, 4.74, 6.56, 5.34, 1.09, 2.92, 6.87])
_AA_BG = _AA_BG / _AA_BG.sum()


def sigmoid(x) -> float:
    return float(1.0 / (1.0 + np.exp(-np.asarray(x))))


def decode(logits) -> str:
    """Argmax soft logits -> amino-acid string (AA_ORDER)."""
    return "".join(AA_ORDER[i] for i in np.asarray(logits).argmax(-1))


def interface_pae(output):
    """Mean binder-to-ligand predicted aligned error (lower = more confident interface)."""
    asym = output.asym_id
    binder = (asym == asym[0]).astype(output.pae.dtype)   # first chain is the binder
    ligand = 1.0 - binder
    w = binder[:, None] * ligand[None, :]                  # binder -> ligand token pairs
    return (output.pae * w).sum() / jnp.maximum(w.sum(), 1.0)


def foldability(output):
    """1 - mean pLDDT (lower = better folded)."""
    return 1.0 - output.plddt.mean()


def composition_kl(soft):
    """KL(mean amino-acid usage || SwissProt background). Zero when the composition matches natural
    frequencies, large when it collapses onto a few residues (poly-M/poly-K). Minimizing it fights
    the homopolymer collapse without pushing toward an equally-unnatural uniform composition, which
    a plain max-entropy reward would do."""
    usage = soft.mean(0)
    return (usage * (jnp.log(usage + 1e-9) - jnp.log(_AA_BG))).sum()


def repetition(soft):
    """Mean overlap between adjacent-position distributions. High = homopolymer runs (AAAA). This
    catches only adjacent (k=2) repetition, i.e. R_hpoly-style homopolymers, not periodic motifs
    (AGAG, which score ~0 here); it is an anti-homopolymer term, not a general anti-repeat one."""
    return (soft[:-1] * soft[1:]).sum(-1).mean()


def ptm_energy(output):
    """pTMEnergy (Nori et al. 2025, Eq 8): pAE logits as a LogSumExp energy, pTM-kernel weighted,
    over inter-chain pairs. Dense gradients, unlike the max-based affinity head. Lower better."""
    logits = output.pae_logits                             # [N, N, Bins]
    # N is the full token count (binder residues + ligand atoms), the AF3 pTM convention. An
    # interface-only N would rescale d0 slightly; kept as token count to match the predictor.
    n = logits.shape[0]
    d0 = jnp.maximum(1.24 * (n - 15) ** (1.0 / 3.0) - 1.8, 1e-3)
    log_g = jnp.log(1.0 / (1.0 + (output.pae_bins / d0) ** 2))            # log g(d_b), [Bins]
    e = -jax.nn.logsumexp(logits + log_g[None, None, :], axis=-1)         # E_ij = -log Σ g·e^ℓ
    asym = output.asym_id
    inter = (asym[:, None] != asym[None, :]).astype(e.dtype)              # inter-chain pairs
    return (e * inter).sum() / jnp.maximum(inter.sum(), 1.0)


def optimize_pbind(oracle, features, binder_len, *, steps=40, lr=0.05, seed=0, key_seed=0,
                   recycling_steps=0, straight_through=False, confidence_weight=0.0):
    """Ascend P(bind); returns (final_logits, per-step P(bind) logits). recycling_steps>0 folds a
    physical structure each step. straight_through folds the discrete argmax but passes gradients
    through the softmax (STE). confidence_weight adds the interface-PAE term.

    Legacy single-run path (used by scripts/optimize_ste.py). The campaign path with the decoupled
    STE and the contact/pocket/composition levers is matched_budget.ste_sweep; keep the two in sync
    if you change the estimator."""
    key = jax.random.PRNGKey(key_seed)

    def loss_fn(logits):
        soft = jax.nn.softmax(logits, axis=-1)
        seq = soft
        if straight_through:
            hard = jax.nn.one_hot(jnp.argmax(soft, -1), soft.shape[-1])
            seq = soft + jax.lax.stop_gradient(hard - soft)   # forward=hard, backward=soft
        pbind, output = oracle.pbind_and_output(
            seq, features, key, recycling_steps=recycling_steps)
        loss = -pbind
        if confidence_weight:
            loss = loss + confidence_weight * interface_pae(output)
        return loss, pbind

    logits = 0.1 * jax.random.normal(jax.random.PRNGKey(seed), (binder_len, 20))
    opt = optax.adam(lr)
    opt_state = opt.init(logits)
    step_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    trajectory = []
    for i in range(steps):
        (_, pbind), grad = step_fn(logits)
        updates, opt_state = opt.update(grad, opt_state)
        logits = optax.apply_updates(logits, updates)
        trajectory.append(float(pbind))
        print(f"step {i:3d}  P(bind) logit {float(pbind):+.3f}")
    return logits, trajectory
