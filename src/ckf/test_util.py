"""Testing and benchmarking utilities."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ckf import ckf


class DimCfg(NamedTuple):
    x: int
    y_sing: int
    y_nonsing: int


def model_random(key, *, dim: DimCfg, impl: ckf.Impl):
    """A randomly populated state-space model with random data."""

    key, k1, k2 = jax.random.split(key, num=3)
    m0 = jax.random.normal(k1, shape=(dim.x,))
    c0 = jax.random.normal(k2, shape=(dim.x, dim.x))
    z = impl.rv_from_cholesky(m0, c0)

    key, k1, k2, k3 = jax.random.split(key, num=4)
    linop = jax.random.normal(k1, shape=(dim.x, dim.x))
    bias = jax.random.normal(k2, shape=(dim.x,))
    cov = jax.random.normal(k3, shape=(dim.x, dim.x))
    noise = impl.rv_from_cholesky(bias, cov)
    x_mid_z = ckf.AffineCond(linop, noise)

    key, k1, k2, k3 = jax.random.split(key, num=4)
    dim_y_total = dim.y_sing + dim.y_nonsing
    assert dim.x >= dim_y_total

    linop = jax.random.normal(k1, shape=(dim_y_total, dim.x))
    bias = jax.random.normal(k2, shape=(dim_y_total,))
    F = jax.random.normal(k3, shape=(dim_y_total, dim.y_nonsing))
    noise = impl.rv_from_cholesky(bias, F)
    y_mid_x = ckf.AffineCond(linop, noise)

    data = jax.random.normal(key, shape=(dim_y_total,))
    return (z, x_mid_z, y_mid_x), F, data


# todo: give this model nontrivial dynamics and more control over dimensions?
def model_interpolation(key, *, dim: DimCfg, impl: ckf.Impl):
    """Like model_random, but with an identity observation operator and F=0."""
    key, k1, k2 = jax.random.split(key, num=3)
    m0 = jax.random.normal(k1, shape=(dim.x,)) / dim.x
    c0 = jax.random.normal(k2, shape=(dim.x, dim.x)) / dim.x
    z = impl.rv_from_cholesky(m0, c0)

    key, k1, k2, k3 = jax.random.split(key, num=4)
    linop = jax.random.normal(k1, shape=(dim.x, dim.x)) / dim.x
    bias = jax.random.normal(k2, shape=(dim.x,)) / dim.x
    cov = jax.random.normal(k3, shape=(dim.x, dim.x)) / dim.x
    noise = impl.rv_from_cholesky(bias, cov)
    x_mid_z = ckf.AffineCond(linop, noise)

    key, k1, k2, k3 = jax.random.split(key, num=4)
    dim_y_total = dim.y_sing + dim.y_nonsing
    assert dim.x >= dim_y_total

    linop = jnp.eye(dim_y_total, dim.x)
    bias = jnp.zeros((dim_y_total,))
    F = jax.random.normal(k3, shape=(dim_y_total, dim.y_nonsing))
    noise = impl.rv_from_cholesky(bias, F)
    y_mid_x = ckf.AffineCond(linop, noise)
    return (z, x_mid_z, y_mid_x), F


def model_ivpsolve(*, dim, impl):
    m0 = jnp.zeros(shape=(dim.x,))
    c0 = 1e-2*jax.scipy.linalg.hilbert(dim.x)
    z = impl.rv_from_cholesky(m0, c0)

    linop = jnp.eye(dim.x)
    bias = jnp.zeros((dim.x,))
    cov = 1e-2*jax.scipy.linalg.hilbert(dim.x)
    noise = impl.rv_from_cholesky(bias, cov)
    x_mid_z = ckf.AffineCond(linop, noise)

    dim_y_total = dim.y_sing + dim.y_nonsing
    linop = jnp.eye(dim_y_total, dim.x)
    bias = jnp.zeros((dim_y_total,))
    F = jnp.zeros((dim_y_total, 0))
    noise = impl.rv_from_cholesky(bias, F)
    y_mid_x = ckf.AffineCond(linop, noise)
    return (z, x_mid_z, y_mid_x)
