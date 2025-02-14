"""Tests for constrained Kalman filtering."""

import jax.numpy as jnp
from ckf import ckf


def test_one_step_condition_works(    value = 716151.214
):
    m0 = jnp.asarray([1.0])
    c0 = jnp.asarray([[1.0]])
    z = ckf.RandVar(m0, c0)

    linop = jnp.asarray([[1.0], [1.0]])
    bias = jnp.asarray([1.0])
    cov = jnp.asarray([[1.0, 0.1], [0.1, 10]])
    z_to_x = ckf.Trafo(linop, bias, cov)

    linop = jnp.asarray([[1.0, 0.], [0., 1.]])
    bias = jnp.asarray([-value, 0.])
    cov = jnp.asarray([[0., 0.], [0., 1.]])
    x_to_y = ckf.Trafo(linop, bias, cov)

    condition = ckf.condition_one_step()
    x_given_y = condition(z=z, z_to_x=z_to_x, x_to_y=x_to_y)
    assert x_given_y.mean.shape == z.mean.shape
    assert x_given_y.cov.shape == z.cov.shape

    assert jnp.allclose(x_given_y.mean[0], value)
    assert jnp.allclose(x_given_y.cov[0, :], 0., atol=1e-5)
    assert jnp.allclose(x_given_y.cov[:, 0], 0., atol=1e-5)



def test_model_reduce(    value = 716151.214
):
    m0 = jnp.asarray([1.0])
    c0 = jnp.asarray([[1.0]])
    z = ckf.RandVar(m0, c0)

    linop = jnp.asarray([[1.0], [1.0]])
    bias = jnp.asarray([1.0])
    cov = jnp.asarray([[1.0, 0.1], [0.1, 10]])
    z_to_x = ckf.Trafo(linop, bias, cov)

    linop = jnp.asarray([[1.0, 0.], [0., 1.]])
    bias = jnp.asarray([-value, 0.])
    cov = jnp.asarray([[0., 0.], [0., 1.]])
    x_to_y = ckf.Trafo(linop, bias, cov)

    out = model_reduce(z=z, z_to_x=z_to_x, x_to_y=x_to_y)
    z_to_y2, z_to_x2, x2_to_y1, x1_value = out

    assert z_to_y2.linop.shape == (1, 2)
    assert z_to_y2.bias.shape == (1,)
    assert z_to_y2.cov.shape == (1, 1)

    assert z_to_x2.linop.shape == (2, 1)
    assert z_to_x2.bias.shape == (2, )
    assert z_to_x2.cov.shape == (2, 2)

    assert x2_to_y1.linop.shape == (1, 2)
    assert x2_to_y1.bias.shape == (1,)
    assert x2_to_y1.cov.shape == (1, 1)

    assert x2_value.shape == (1,)
