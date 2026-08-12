"""Pure-logic tests for the ligand-aware regularizer (no torch / Boltz / GPU)."""

import jax
import jax.numpy as jnp
import numpy as np

from nisegrad.ligand_mpnn_reg import (
    AF_ORDER,
    MPNN_ORDER,
    af20_to_mpnn20,
    nll_from_logprobs,
)


def test_reorder_is_correct_permutation():
    # one-hot of every AA in AF order must map to that AA's one-hot in MPNN order
    for aa in AF_ORDER:
        soft_af = jax.nn.one_hot(AF_ORDER.index(aa), 20)
        got = np.asarray(af20_to_mpnn20(soft_af))
        assert got.argmax() == MPNN_ORDER.index(aa), aa
        assert np.isclose(got.sum(), 1.0)  # probability mass preserved


def test_reorder_preserves_distribution():
    rng = np.random.RandomState(0)
    soft = jax.nn.softmax(jnp.asarray(rng.randn(7, 20)), axis=-1)
    out = af20_to_mpnn20(soft)
    assert np.allclose(np.asarray(out).sum(-1), 1.0)  # still a distribution per position
    # it is a permutation: multiset of values per row is unchanged
    assert np.allclose(np.sort(np.asarray(soft), -1), np.sort(np.asarray(out), -1))


def test_nll_matches_hand_computation():
    L = 5
    rng = np.random.RandomState(1)
    log_probs = jax.nn.log_softmax(jnp.asarray(rng.randn(L, 21)), axis=-1)
    soft = jax.nn.one_hot(jnp.asarray(rng.randint(0, 20, L)), 20)
    binder_mask = jnp.asarray([1.0, 1.0, 0.0, 1.0, 0.0])

    got = float(nll_from_logprobs(soft, log_probs, binder_mask))
    # hand: mean over binder positions of -log_prob at the chosen AA
    chosen = -(np.asarray(soft) * np.asarray(log_probs)[:, :20]).sum(-1)
    bm = np.asarray(binder_mask)
    expected = (chosen * bm).sum() / bm.sum()
    assert np.isclose(got, expected, atol=1e-6)


def test_nll_grad_flows():
    L = 6
    rng = np.random.RandomState(2)
    log_probs = jax.nn.log_softmax(jnp.asarray(rng.randn(L, 21)), axis=-1)
    binder_mask = jnp.ones(L)

    def loss(logits):
        soft = jax.nn.softmax(logits, -1)
        return nll_from_logprobs(soft, log_probs, binder_mask)

    g = jax.grad(loss)(jnp.asarray(rng.randn(L, 20)))
    assert bool(jnp.isfinite(g).all()) and bool((g != 0).any())


if __name__ == "__main__":
    test_reorder_is_correct_permutation()
    test_reorder_preserves_distribution()
    test_nll_matches_hand_computation()
    test_nll_grad_flows()
    print("LIGAND REG TESTS OK")
