"""Gradient ascent on the binder sequence: P(bind), or on-minus-off selectivity."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

# AlphaFold / Boltz standard-20 residue order (matches set_binder_sequence's [N,20]).
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"


def sigmoid(x) -> float:
    return float(1.0 / (1.0 + np.exp(-np.asarray(x))))


def decode(logits) -> str:
    """Argmax soft logits -> amino-acid string (AA_ORDER)."""
    return "".join(AA_ORDER[i] for i in np.asarray(logits).argmax(-1))


def _run(loss_fn, binder_len, *, steps, lr, seed, label):
    logits = 0.1 * jax.random.normal(jax.random.PRNGKey(seed), (binder_len, 20))
    opt = optax.adam(lr)
    opt_state = opt.init(logits)
    step_fn = jax.jit(jax.value_and_grad(loss_fn))

    trajectory = []
    for i in range(steps):
        loss, grad = step_fn(logits)
        updates, opt_state = opt.update(grad, opt_state)
        logits = optax.apply_updates(logits, updates)
        objective = float(-loss)  # we minimize the negative objective
        trajectory.append(objective)
        print(f"step {i:3d}  {label} {objective:+.3f}")
    return logits, trajectory


def interface_pae(output):
    """Mean predicted aligned error between the binder chain and the ligand (lower = more
    confident interface). A second, structure-based Boltz head, differentiable in the sequence."""
    asym = output.asym_id
    binder = (asym == asym[0]).astype(output.pae.dtype)   # first chain is the binder
    ligand = 1.0 - binder
    w = binder[:, None] * ligand[None, :]                  # binder -> ligand token pairs
    return (output.pae * w).sum() / jnp.maximum(w.sum(), 1.0)


def optimize_pbind(oracle, features, binder_len, *, mpnn=None, mpnn_weight=1.0,
                   steps=40, lr=0.05, seed=0, key_seed=0, recycling_steps=0,
                   straight_through=False, confidence_weight=0.0):
    """Ascend P(bind), optionally regularized by an MPNN sequence log-likelihood so the
    binder stays protein-like. Returns (final_logits, per-step P(bind) logits).

    recycling_steps>0 folds a physical structure each step (slower, more memory), so both
    P(bind) and a live structure prior see real geometry instead of recycling=0 noise.

    straight_through=True feeds the oracle the DISCRETE argmax sequence in the forward pass
    (so the reported P(bind) is the real discrete number, not the soft optimum) while passing
    the gradient through the soft distribution -- optimizes the discrete objective directly.

    confidence_weight>0 also minimizes the binder-ligand interface PAE (a second Boltz head),
    so the design must form a confident predicted interface, not just fool the affinity head."""
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
        if mpnn is not None:
            mpnn_nll, _ = mpnn(seq, output, key)   # negative log-likelihood
            loss = loss + mpnn_weight * mpnn_nll
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


def optimize_selectivity(oracle, on_features, off_features, binder_len,
                         *, steps=40, lr=0.1, seed=0, key_seed=0):
    """Ascend P(bind|on) - P(bind|off). Returns (final_logits, per-step on-off gap)."""
    key = jax.random.PRNGKey(key_seed)

    def neg_selectivity(logits):
        soft = jax.nn.softmax(logits, axis=-1)
        on = oracle.pbind_logit(soft, on_features, key)
        off = oracle.pbind_logit(soft, off_features, key)
        return -(on - off)

    return _run(neg_selectivity, binder_len, steps=steps, lr=lr, seed=seed,
                label="selectivity (on-off)")
