"""Tests for constrained Kalman filtering."""

import jax.numpy as jnp
from ckf import ckf, test_util
import jax
import pytest_cases

from typing import NamedTuple


def case_impl_cov_based():
    return ckf.impl_cov_based()


def case_impl_cholesky_based():
    return ckf.impl_cholesky_based()



def case_dim_base():
    return test_util.DimCfg(1, 5, 2, 3)


def case_dim_sing_zero():
    return test_util.DimCfg(1, 5, 2, 0)


def case_dim_nonsing_zero():
    return test_util.DimCfg(1, 5, 0, 3)


def case_dim_sing_and_nonsing_zero():
    return test_util.DimCfg(1, 1, 0, 0)


@pytest_cases.parametrize_with_cases("dim", cases=".", prefix="case_dim_")
@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_model_reduce_shapes(impl, dim):
    y, (z, x_mid_z, y_mid_x), F = test_util.model_random(dim=dim, impl=impl)

    reduced = ckf.model_reduce(y_mid_x=y_mid_x, x_mid_z=x_mid_z, F=F, impl=impl)
    x2_mid_data, x_mid_x2 = ckf.model_reduced_apply(y, z=z, reduced=reduced, impl=impl)

    x2_dim = dim.x - dim.y_sing
    assert x2_mid_data.mean.shape == (x2_dim,)
    assert x2_mid_data.cov_dense().shape == (x2_dim, x2_dim)


@pytest_cases.parametrize_with_cases("dim", cases=".", prefix="case_dim_")
@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_model_reduce_values(dim, impl):
    y, (z, x_mid_z, y_mid_x), F = test_util.model_random(dim=dim, impl=impl)

    # Reference:
    ref_x = impl.rv_marginal(prior=z, trafo=x_mid_z)
    _y, ref_backward = impl.rv_condition(prior=ref_x, trafo=y_mid_x)
    ref_x_mid_y = impl.trafo_evaluate(y, trafo=ref_backward)

    # Reduced model:
    reduced = ckf.model_reduce(y_mid_x=y_mid_x, x_mid_z=x_mid_z, F=F, impl=impl)
    x2_mid_data, x_mid_x2 = ckf.model_reduced_apply(y, z=z, reduced=reduced, impl=impl)

    x_mid_data = impl.rv_marginal(prior=x2_mid_data, trafo=x_mid_x2)

    tol = jnp.sqrt(jnp.finfo(y.dtype).eps)
    assert jnp.allclose(x_mid_data.mean, ref_x_mid_y.mean, rtol=tol, atol=tol)


@pytest_cases.parametrize_with_cases("dim", cases=".", prefix="case_dim_")
@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_model_reduce_values(dim, impl):
    y, (z, x_mid_z, y_mid_x), F = test_util.model_random(dim=dim, impl=impl)

    # Reference:
    ref_x = impl.rv_marginal(prior=z, trafo=x_mid_z)
    _y, ref_backward = impl.rv_condition(prior=ref_x, trafo=y_mid_x)
    ref_x_mid_y = impl.trafo_evaluate(y, trafo=ref_backward)

    # Reduced model:
    reduced = ckf.model_reduce(y_mid_x=y_mid_x, x_mid_z=x_mid_z, F=F, impl=impl)
    x2_mid_data, x_mid_x2 = ckf.model_reduced_apply(y, z=z, reduced=reduced, impl=impl)

    x_mid_data = impl.rv_marginal(prior=x2_mid_data, trafo=x_mid_x2)

    tol = jnp.sqrt(jnp.finfo(y.dtype).eps)
    assert jnp.allclose(x_mid_data.mean, ref_x_mid_y.mean, rtol=tol, atol=tol)


@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_kalman_filter(impl):
    (z, x_mid_z, y_mid_x), F = test_util.model_interpolation(impl=impl)

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

