"""Tests for constrained Kalman filtering."""

import jax.numpy as jnp
from ckf import ckf


def test_kf_works(    value = 716151.214
):
    linop = jnp.asarray([[1.0, 0.1], [0.0, 1.0]])
    bias = jnp.asarray([1.0, 0.1])
    cov = jnp.asarray([[1.0, 0.1], [0.1, 10]])
    z_to_x = ckf.Trafo(linop, bias, cov)

    linop = jnp.asarray([[1.0, 0.]])
    bias = jnp.asarray([-value])
    cov = jnp.asarray([[0.]])
    x_to_y = ckf.Trafo(linop, bias, cov)

    m0 = jnp.asarray([1.0, 1.0])
    c0 = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    z = ckf.RandVar(m0, c0)

    condition = ckf.condition_one_step()
    x_given_y = condition(z=z, z_to_x=z_to_x, x_to_y=x_to_y)
    assert x_given_y.mean.shape == z.mean.shape
    assert x_given_y.cov.shape == z.cov.shape


#
# def test_constraining():
#     phi = jnp.asarray([[1.0, 0.1], [0.0, 1.0]])
#     q = jnp.asarray([[1.0, 0.1], [0.1, 10]])
#     z_to_x = (phi, q)
#
#     c = jnp.asarray([[1.0, 0.0]])
#     f = jnp.asarray([[0.0]])
#     x_to_y = (c, f)
#
#     m0 = jnp.asarray([1.0, 1.0])
#     c0 = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
#     z_marg = (m0, c0)
#
#     # Transformations are _all_ like: (A x + b, C C^*)
#     # Marginals are _all_ like (m, C C^*)
#     # values are values
#     z_to_y1, z_to_x2, x2_to_y1, x1_value = model_reduce(z_marg, z_to_x, x_to_y)
#
#
