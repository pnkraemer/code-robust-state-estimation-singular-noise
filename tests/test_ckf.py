"""Tests for constrained Kalman filtering."""

import jax.numpy as jnp
from ckf import ckf
import jax
import pytest_cases

from typing import NamedTuple


def case_impl_cov_based():
    return ckf.impl_cov_based()


def case_impl_cholesky_based():
    return ckf.impl_cholesky_based()


class DimCfg(NamedTuple):
    z: int
    x: int
    y_sing: int
    y_nonsing: int

# todo: test that cov-based and chol-based yield the same values

def case_dim_base():
    return DimCfg(1, 5, 2, 3)


def case_dim_sing_zero():
    return DimCfg(1, 5, 2, 0)


def case_dim_nonsing_zero():
    return DimCfg(1, 5, 0, 3)


def case_dim_sing_and_nonsing_zero():
    return DimCfg(1, 1, 0, 0)


@pytest_cases.parametrize_with_cases("dim", cases=".", prefix="case_dim_")
@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_model_align_shapes(impl, dim):
    y, (z, x_mid_z, y_mid_x), F = _model_random(dim=dim, impl=impl)

    reduced = ckf.model_reduce(y_mid_x=y_mid_x, x_mid_z=x_mid_z, F=F, impl=impl)
    x2_mid_data, x_mid_x2 = ckf.model_reduced_apply(y, z=z, reduced=reduced, impl=impl)

    x2_dim = dim.x - dim.y_sing
    assert x2_mid_data.mean.shape == (x2_dim,)
    assert x2_mid_data.cov_dense().shape == (x2_dim, x2_dim)


@pytest_cases.parametrize_with_cases("dim", cases=".", prefix="case_dim_")
@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_model_align_values(dim, impl):
    y, (z, x_mid_z, y_mid_x), F = _model_random(dim=dim, impl=impl)

    # Reference:
    ref_x = impl.rv_marginal(prior=z, trafo=x_mid_z)
    _y, ref_backward = impl.rv_condition(prior=ref_x, trafo=y_mid_x)
    ref_x_mid_y = impl.trafo_evaluate(y, trafo=ref_backward)

    # Reduced model:
    reduced = ckf.model_reduce(y_mid_x=y_mid_x, x_mid_z=x_mid_z, F=F, impl=impl)
    x2_mid_data, x_mid_x2 = ckf.model_reduced_apply(y, z=z, reduced=reduced, impl=impl)

    x_mid_data = impl.rv_marginal(prior=x2_mid_data, trafo=x_mid_x2)

    tol = jnp.sqrt(jnp.finfo(y.dtype).eps)
    print(x_mid_data.mean)
    assert jnp.allclose(x_mid_data.mean, ref_x_mid_y.mean, rtol=tol, atol=tol)
    assert False

@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_kalman_filter(impl):
    (z, x_mid_z, y_mid_x), F = _model_interpolation(impl=impl)

    data_in = jnp.linspace(0, 1, num=20)
    data_out = data_in + 0.1 + jnp.sin(data_in**2)

    means, covs = [], []
    x = z

    for d in data_out:
        x = impl.rv_marginal(prior=x, trafo=x_mid_z)
        _, x = impl.rv_condition(prior=x, trafo=y_mid_x)

        x = impl.trafo_evaluate(jnp.atleast_1d(d), trafo=x)
        means.append(x.mean)
        covs.append(x.cov_dense())

    assert jnp.allclose(jnp.stack(means)[:, 0], data_out)
    assert jnp.allclose(jnp.stack(covs)[:, 0, :], 0.0, atol=1e-5)
    assert jnp.allclose(jnp.stack(covs)[:, :, 0], 0.0, atol=1e-5)

    means, covs = [], []

    # todo: make square-root
    # todo: track marginal likelihoods

    for d in data_out:
        d = jnp.atleast_1d(d)
        reduced = ckf.model_reduce(y_mid_x=y_mid_x, x_mid_z=x_mid_z, F=F, impl=impl)
        x2, x_mid_x2 = ckf.model_reduced_apply(d, z=z, reduced=reduced, impl=impl)
        z = impl.rv_marginal(prior=x2, trafo=x_mid_x2)

        means.append(z.mean)
        covs.append(z.cov_dense())


    assert jnp.allclose(jnp.stack(means)[:, 0], data_out)
    assert jnp.allclose(jnp.stack(covs)[:, 0, :], 0.0, atol=1e-5)
    assert jnp.allclose(jnp.stack(covs)[:, :, 0], 0.0, atol=1e-5)


def _model_interpolation(impl):
    m0 = jnp.zeros((2,))
    c0 = jnp.eye(2)
    z = impl.rv_from_cholesky(m0, c0)

    linop = jnp.eye(2)
    bias = jnp.zeros((2,))
    cov = jnp.eye(2)
    noise = impl.rv_from_cholesky(bias, cov)
    x_mid_z = ckf.Trafo(linop, noise)

    linop = jnp.eye(1, 2)
    bias = jnp.zeros((1,))
    cov = jnp.zeros((1, 1))
    noise = impl.rv_from_cholesky(bias, cov)
    y_mid_x = ckf.Trafo(linop, noise)

    return (z, x_mid_z, y_mid_x), cov


def _model_random(*, dim: DimCfg, impl):
    key = jax.random.PRNGKey(seed=3)

    key, k1, k2 = jax.random.split(key, num=3)
    m0 = jax.random.normal(k1, shape=(dim.z,))
    c0 = jax.random.normal(k2, shape=(dim.z, dim.z))
    z = impl.rv_from_cholesky(m0, c0)

    # z = ckf.RandVar(m0, c0 @ c0.T)

    key, k1, k2, k3 = jax.random.split(key, num=4)
    linop = jax.random.normal(k1, shape=(dim.x, dim.z))
    bias = jax.random.normal(k2, shape=(dim.x,))
    cov = jax.random.normal(k3, shape=(dim.x, dim.x))
    noise = impl.rv_from_cholesky(bias, cov)
    x_mid_z = ckf.Trafo(linop, noise)

    key, k1, k2, k3 = jax.random.split(key, num=4)
    dim_y_total = dim.y_sing + dim.y_nonsing
    assert dim.x >= dim_y_total

    linop = jax.random.normal(k1, shape=(dim_y_total, dim.x))
    bias = jax.random.normal(k2, shape=(dim_y_total,))
    F = jax.random.normal(k3, shape=(dim_y_total, dim.y_nonsing))
    noise = impl.rv_from_cholesky(bias, F)
    y_mid_x = ckf.Trafo(linop, noise)

    data = jax.random.normal(key, shape=(dim_y_total,))
    return data, (z, x_mid_z, y_mid_x), F
