"""Tests for constrained Kalman filtering."""

import jax.numpy as jnp
from ckf import kf


def test_kf_works():
    phi = jnp.atleast_2d(1.0)
    q = jnp.atleast_2d(1.0)
    latent = (phi, q)

    c = jnp.atleast_2d(1.0)
    f = jnp.atleast_2d(0)
    observe = (c, f)

    m0 = jnp.atleast_1d(1.0)
    c0 = jnp.atleast_2d(1.0)
    init = (m0, c0)

    value = 716151.214
    data = value * jnp.ones((10, 1), dtype=float)
    kalman = kf.kalman_filter()
    (m_all, c_all) = kalman(data, init=init, observe=observe, latent=latent)
    assert m_all.shape == data.shape
    assert c_all.shape == (*data.shape, *m0.shape)

    assert jnp.allclose(m_all, value)
    assert jnp.allclose(c_all, 0.)

