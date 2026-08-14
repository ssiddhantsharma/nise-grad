"""Pure-logic tests for the STE core (no Boltz/GPU): the straight-through identity and decode."""

import jax
import jax.numpy as jnp
import numpy as np

from nisegrad.optimize import AA_ORDER, decode, sigmoid


def _ste(logits):
    soft = jax.nn.softmax(logits, -1)
    hard = jax.nn.one_hot(jnp.argmax(soft, -1), soft.shape[-1])
    return soft + jax.lax.stop_gradient(hard - soft)


def test_ste_forward_is_hard():
    logits = jnp.array([[2.0, 0.0, -1.0] + [0.0] * 17])
    hard = jax.nn.one_hot(jnp.argmax(jax.nn.softmax(logits, -1), -1), 20)
    np.testing.assert_allclose(np.asarray(_ste(logits)), np.asarray(hard), atol=1e-6)


def test_ste_backward_is_soft():
    logits = jnp.array([[2.0, 0.0, -1.0] + [0.0] * 17])
    g_ste = jax.jacobian(_ste)(logits)
    g_soft = jax.jacobian(lambda x: jax.nn.softmax(x, -1))(logits)
    np.testing.assert_allclose(np.asarray(g_ste), np.asarray(g_soft), atol=1e-6)


def test_decode_roundtrip():
    logits = np.full((6, 20), -9.0)
    for r, c in enumerate("ACDEFG"):
        logits[r, AA_ORDER.index(c)] = 9.0
    assert decode(logits) == "ACDEFG"


def test_sigmoid_monotone():
    assert sigmoid(-10) < sigmoid(0) < sigmoid(10)
    assert abs(sigmoid(0) - 0.5) < 1e-9
