"""Tests for constrained Kalman filtering."""

import jax.numpy as jnp
from ckf import ckf
import jax

def test_one_step_condition_works(z_dim=1, x_dim=7, y_dim=5):
    z, x_given_z, y_given_x = _model(z_shape=(z_dim,), x_shape=(x_dim,), y_shape=(y_dim,))

    filter_ = ckf.filter_one_step()
    x_given_y = filter_(z=z, x_given_z=x_given_z, y_given_x=y_given_x)

    assert x_given_y.mean.shape == (x_dim,)
    assert x_given_y.cov.shape == (x_dim, x_dim)

    small_value = jnp.sqrt(jnp.finfo(x_given_y.mean.dtype).eps)
    assert jnp.allclose(x_given_y.mean[z_dim:y_dim], -y_given_x.bias[z_dim:])
    assert jnp.allclose(x_given_y.cov[z_dim:y_dim, :], 0.0, atol=small_value)
    assert jnp.allclose(x_given_y.cov[:, z_dim:y_dim], 0.0, atol=small_value)



def test_model_reduce(value=716151.214):
    z, x_given_z, y_given_x = _model(value=value)

    # todo: implement the recursion from the paper draft
    # (but ignore all z1's and thus think of 'z2' as 'z')
    out = model_reduce(z=z, x_given_z=z_to_x, y_given_x=x_to_y)
    y2_given_z, x2_given_z, y1_given_x2, x_from_xs = out

    # Assert shapes

    assert z_to_y2.linop.shape == (1, 1)
    assert z_to_y2.bias.shape == (1,)
    assert z_to_y2.cov.shape == (1, 1)

    assert z_to_x2.linop.shape == (1, 1)
    assert z_to_x2.bias.shape == (1,)
    assert z_to_x2.cov.shape == (1, 1)

    assert x2_to_y1.linop.shape == (1, 1)
    assert x2_to_y1.bias.shape == (1,)
    assert x2_to_y1.cov.shape == (1, 1)

    assert x2_value.shape == (1,)
    assert jnp.allclose(x2_value, value)

    # Assert values:
    z_given_y2, y2 = condition(likelihood=y2_given_z, prior=z)

    # From here on, we filter
    # Predict:
    x2 = marginal(x2_given_z, z_given_y2)

    # Update
    x2_given_y1, y1 = condition(likelihood=y_given_x2, prior=x2)

    # x2 is the new z. Get new models and repeat:
    # z = x2_given_y1
    # x_given_z = merge(x_given_z, x_from_xs)
    # Repeat...

    # Reconstruct x:
    x = marginal(x_from_xs, x2_given_y1)

    # Compare this x to that of the traditional Kalman filter
    assert jnp.allclose(x.mean, x_ref.mean)
    assert jnp.allclose(x.cov, x_ref.cov)


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
    x_given_z = ckf.Trafo(linop, bias, cov @ cov.T)

    bias = jax.random.normal(key, shape=y_shape)
    linop = jnp.eye(*y_shape, *x_shape)
    cov = jnp.eye(*y_shape, *z_shape)
    y_given_x = ckf.Trafo(linop, bias, cov @ cov.T)
    return z, x_given_z, y_given_x
