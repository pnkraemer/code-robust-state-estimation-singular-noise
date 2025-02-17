"""Tests for constrained Kalman filtering."""

import jax.numpy as jnp
from ckf import ckf
import jax


def test_filter_one_step_works(z_dim=1, x_dim=7, y_dim=5):
    z, x_mid_z, y_mid_x = _model(z_shape=(z_dim,), x_shape=(x_dim,), y_shape=(y_dim,))

    x = ckf.marginal(prior=z, trafo=x_mid_z)
    _y, x_mid_y = ckf.condition(prior=x, trafo=y_mid_x)

    assert x_mid_y.mean.shape == (x_dim,)
    assert x_mid_y.cov.shape == (x_dim, x_dim)

    small_value = jnp.sqrt(jnp.finfo(x_mid_y.mean.dtype).eps)
    assert jnp.allclose(x_mid_y.mean[z_dim:y_dim], -y_mid_x.bias[z_dim:])
    assert jnp.allclose(x_mid_y.cov[z_dim:y_dim, :], 0.0, atol=small_value)
    assert jnp.allclose(x_mid_y.cov[:, z_dim:y_dim], 0.0, atol=small_value)


def test_model_align(z_dim=1, x_dim=7, y_dim=5):
    z, x_mid_z, y_mid_x = _model(z_shape=(z_dim,), x_shape=(x_dim,), y_shape=(y_dim,))
    y = jnp.zeros((y_dim,))
    out = ckf.model_reduce(y, y_mid_x=y_mid_x, x_mid_z=x_mid_z, z=z)
    z_small, x2_mid_z_small, y1_mid_x2_small, x1_value = out

    # Assert shapes
    print(jax.tree.map(jnp.shape, z_small))
    print()
    print(jax.tree.map(jnp.shape, x2_mid_z_small))
    print()
    print(jax.tree.map(jnp.shape, y1_mid_x2_small))
    print()
    print(jax.tree.map(jnp.shape, x1_value))

    assert z_small.mean.shape == (1,)
    assert z_small.cov.shape == (1, 1)

    x1_dim, x2_dim = y_dim - z_dim, x_dim - (y_dim - z_dim)
    assert x2_mid_z_small.linop.shape == (x2_dim, 1)
    assert x2_mid_z_small.bias.shape == (x2_dim,)
    assert x2_mid_z_small.cov.shape == (x2_dim, x2_dim)

    assert y1_mid_x2_small.linop.shape == (1, x2_dim)
    assert y1_mid_x2_small.bias.shape == (1,)
    assert y1_mid_x2_small.cov.shape == (1, 1)

    x1_dim = y_dim - z_dim
    assert x1_value.shape == (x1_dim,)
    print(x1_value)
    print(y_mid_x.bias)
    assert jnp.allclose(x1_value, value)


def _model(*, z_shape, x_shape, y_shape):
    key = jax.random.PRNGKey(seed=2)

    key, k1, k2 = jax.random.split(key, num=3)
    m0 = jax.random.normal(k1, shape=z_shape)
    c0 = jax.random.normal(k2, shape=(*z_shape, *z_shape))
    z = ckf.RandVar(m0, c0 @ c0.T)

    key, k1, k2, k3 = jax.random.split(key, num=4)
    linop = jax.random.normal(k1, shape=(*x_shape, *z_shape))
    bias = jax.random.normal(k2, shape=x_shape)
    cov = jax.random.normal(k3, shape=(*x_shape, *x_shape))
    x_mid_z = ckf.Trafo(linop, bias, cov @ cov.T)

    bias = jax.random.normal(key, shape=y_shape)
    linop = jnp.eye(*y_shape, *x_shape)
    cov = jnp.eye(*y_shape, *z_shape)
    y_mid_x = ckf.Trafo(linop, bias, cov @ cov.T)
    return z, x_mid_z, y_mid_x
