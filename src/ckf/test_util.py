"""Testing and benchmarking utilities."""

import jax.numpy as jnp
from ckf import ckf, test_util
import jax
import pytest_cases

from typing import NamedTuple


class DimCfg(NamedTuple):
    z: int
    x: int
    y_sing: int
    y_nonsing: int


def model_interpolation(impl: ckf.Impl):
    m0 = jnp.zeros((2,))
    c0 = jnp.eye(2)
    z = impl.rv_from_cholesky(m0, c0)

    linop = jnp.eye(2)
    bias = jnp.zeros((2,))
    cov = jnp.eye(2)
    noise = impl.rv_from_cholesky(bias, cov)
    x_mid_z = ckf.AffineCond(linop, noise)

    linop = jnp.eye(1, 2)
    bias = jnp.zeros((1,))
    cholesky = jnp.zeros((1, 0))  # zero-rank
    noise = impl.rv_from_cholesky(bias, cholesky)
    y_mid_x = ckf.AffineCond(linop, noise)

    return (z, x_mid_z, y_mid_x), cholesky


def model_random(*, dim: DimCfg, impl: ckf.Impl):
    key = jax.random.PRNGKey(seed=3)

    key, k1, k2 = jax.random.split(key, num=3)
    m0 = jax.random.normal(k1, shape=(dim.z,))
    c0 = jax.random.normal(k2, shape=(dim.z, dim.z))
    z = impl.rv_from_cholesky(m0, c0)

    key, k1, k2, k3 = jax.random.split(key, num=4)
    linop = jax.random.normal(k1, shape=(dim.x, dim.z))
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
