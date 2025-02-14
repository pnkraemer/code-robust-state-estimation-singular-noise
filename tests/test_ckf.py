"""Tests for constrained Kalman filtering."""

import jax.numpy as jnp
from ckf import ckf


def test_kf_works():
    phi = jnp.asarray([[1., 0.1], [0., 1.]])
    q = jnp.asarray([[1., 0.1], [0.1, 10]])
    latent = (phi, q)

    c = jnp.asarray([[1., 0.]])
    f = jnp.asarray([[0.]])
    observe = (c, f)

    m0 = jnp.asarray([1., 1.])
    c0 = jnp.asarray([[1., 0.], [0., 1.]])
    init = (m0, c0)

    value = 716151.214
    data = value * jnp.ones((10, 1), dtype=float)
    kalman = ckf.kalman_filter()
    (m_all, c_all) = kalman(data, init=init, observe=observe, latent=latent)
    assert m_all.shape == (10, 2)
    assert c_all.shape == (10, 2, 2)


    assert jnp.allclose(m_all[:, 0], value)
    assert jnp.allclose(c_all[:, 0, :], 0., atol=1e-5)
    assert jnp.allclose(c_all[:, :, 0], 0., atol=1e-5)

