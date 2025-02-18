"""Tests for constrained Kalman filtering."""

import jax.numpy as jnp
from ckf import ckf, test_util
import jax
import pytest_cases

from typing import NamedTuple


def case_impl_cov_based() -> ckf.Impl:
    return ckf.impl_cov_based()


def case_impl_cholesky_based() -> ckf.Impl:
    return ckf.impl_cholesky_based()


def case_dim_base() -> test_util.DimCfg:
    return test_util.DimCfg(z=1, x=5, y_sing=2, y_nonsing=3)


def case_dim_nonsing_zero() -> test_util.DimCfg:
    return test_util.DimCfg(z=1, x=5, y_sing=2, y_nonsing=0)


def case_dim_sing_zero() -> test_util.DimCfg:
    return test_util.DimCfg(z=1, x=5, y_sing=0, y_nonsing=3)


def case_dim_sing_and_nonsing_zero() -> test_util.DimCfg:
    return test_util.DimCfg(z=1, x=5, y_sing=0, y_nonsing=0)


@pytest_cases.parametrize_with_cases("dim", cases=".", prefix="case_dim_")
@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_model_reduce_shapes(impl, dim):
    (z, x_mid_z, y_mid_x), F, y = test_util.model_random(dim=dim, impl=impl)

    model_prepare, model_reduce = ckf.model_reduction(F_rank=dim.y_nonsing, impl=impl)
    prepared = model_prepare(y_mid_x=y_mid_x, x_mid_z=x_mid_z)
    y1, (z, x2_mid_z, y1_mid_x2), info = model_reduce(y, z=z, prepared=prepared)

    # Run a single filter-condition step
    x2 = impl.rv_marginal(z, x2_mid_z)
    _, bwd = impl.rv_condition(x2, y1_mid_x2)
    x2_mid_data = impl.trafo_evaluate(y1, bwd)

    # Assert the shapes are as expected
    x2_dim = dim.x - dim.y_sing
    assert x2_mid_data.mean.shape == (x2_dim,)
    assert x2_mid_data.cov_dense().shape == (x2_dim, x2_dim)


@pytest_cases.parametrize_with_cases("dim", cases=".", prefix="case_dim_")
@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_model_reduce_values(dim, impl):
    (z, x_mid_z, y_mid_x), F, y = test_util.model_random(dim=dim, impl=impl)

    # Reference:
    ref_x = impl.rv_marginal(prior=z, trafo=x_mid_z)
    _y, ref_backward = impl.rv_condition(prior=ref_x, trafo=y_mid_x)
    ref_x_mid_y = impl.trafo_evaluate(y, trafo=ref_backward)

    # Reduced model:
    model_prepare, model_reduce = ckf.model_reduction(F_rank=dim.y_nonsing, impl=impl)
    prepared = model_prepare(y_mid_x=y_mid_x, x_mid_z=x_mid_z)
    y1, (z, x2_mid_z, y1_mid_x2), (x_mid_x2, _) = model_reduce(
        y, z=z, prepared=prepared
    )

    # Run a single filter-condition step
    x2 = impl.rv_marginal(z, x2_mid_z)
    _, bwd = impl.rv_condition(x2, y1_mid_x2)
    x2_mid_data = impl.trafo_evaluate(y1, bwd)

    tol = jnp.sqrt(jnp.finfo(y.dtype).eps)
    x_mid_data = impl.rv_marginal(prior=x2_mid_data, trafo=x_mid_x2)
    assert jnp.allclose(x_mid_data.mean, ref_x_mid_y.mean, rtol=tol, atol=tol)


@pytest_cases.parametrize_with_cases("dim", cases=".", prefix="case_dim_")
@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_model_reduce_logpdf(dim, impl):
    (z, x_mid_z, y_mid_x), F, y = test_util.model_random(dim=dim, impl=impl)

    # Reference:
    ref_x = impl.rv_marginal(prior=z, trafo=x_mid_z)
    y_marg, ref_backward = impl.rv_condition(prior=ref_x, trafo=y_mid_x)
    logpdf1 = impl.rv_logpdf(y, y_marg)

    # Reduced model:
    model_prepare, model_reduce = ckf.model_reduction(F_rank=dim.y_nonsing, impl=impl)
    prepared = model_prepare(y_mid_x=y_mid_x, x_mid_z=x_mid_z)
    y1, (z, x2_mid_z, y1_mid_x2), (x_mid_x2, pdf2) = model_reduce(
        y, z=z, prepared=prepared
    )

    # Run a single filter-condition step
    x2 = impl.rv_marginal(z, x2_mid_z)
    y1_marg, _bwd = impl.rv_condition(x2, y1_mid_x2)
    pdf1 = impl.rv_logpdf(y1, y1_marg)
    logpdf2 = pdf1 + pdf2

    assert jnp.allclose(logpdf1, logpdf2)


@pytest_cases.parametrize_with_cases("dim", cases=".", prefix="case_dim_")
def test_logpdfs_consistent_across_impls(dim):
    impl = ckf.impl_cov_based()
    (z, x_mid_z, y_mid_x), F, y = test_util.model_random(dim=dim, impl=impl)
    ref_x = impl.rv_marginal(prior=z, trafo=x_mid_z)
    y_marg, ref_backward = impl.rv_condition(prior=ref_x, trafo=y_mid_x)
    logpdf1 = impl.rv_logpdf(y, y_marg)

    impl = ckf.impl_cholesky_based()
    (z, x_mid_z, y_mid_x), F, y = test_util.model_random(dim=dim, impl=impl)
    ref_x = impl.rv_marginal(prior=z, trafo=x_mid_z)
    y_marg, ref_backward = impl.rv_condition(prior=ref_x, trafo=y_mid_x)
    logpdf2 = impl.rv_logpdf(y, y_marg)

    assert jnp.allclose(logpdf1, logpdf2)


@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_kalman_filter(impl):
    (z, x_mid_z, y_mid_x), F = test_util.model_interpolation(impl=impl)

    data_in = jnp.linspace(0, 1, num=20)
    data_out = data_in + 0.1 + jnp.sin(data_in**2)

    means, covs = [], []
    x = z

    logpdf_ref = 0.0
    for d in data_out:
        x = impl.rv_marginal(prior=x, trafo=x_mid_z)
        y, x = impl.rv_condition(prior=x, trafo=y_mid_x)
        x = impl.trafo_evaluate(jnp.atleast_1d(d), trafo=x)
        logpdf_ref += impl.rv_logpdf(jnp.atleast_1d(d), y)

        means.append(x.mean)
        covs.append(x.cov_dense())

    assert jnp.allclose(jnp.stack(means)[:, 0], data_out)
    assert jnp.allclose(jnp.stack(covs)[:, 0, :], 0.0, atol=1e-5)
    assert jnp.allclose(jnp.stack(covs)[:, :, 0], 0.0, atol=1e-5)

    means, covs = [], []

    model_prepare, model_reduce = ckf.model_reduction(F_rank=F.shape[1], impl=impl)

    prepared = model_prepare(y_mid_x=y_mid_x, x_mid_z=x_mid_z)

    logpdf_reduced = 0.0
    for d in data_out:
        d = jnp.atleast_1d(d)
        y1, reduced, (x_mid_x2, pdf2) = model_reduce(d, z=z, prepared=prepared)
        (z, x2_mid_z, y1_mid_x2) = reduced

        # Run a single filter-condition step
        x2 = impl.rv_marginal(z, x2_mid_z)
        y1_marg, bwd = impl.rv_condition(x2, y1_mid_x2)
        x2 = impl.trafo_evaluate(y1, bwd)
        pdf1 = impl.rv_logpdf(y1, y1_marg)

        # Get the next 'z' to restart
        z = impl.rv_marginal(prior=x2, trafo=x_mid_x2)

        logpdf_reduced += pdf1 + pdf2
        means.append(z.mean)
        covs.append(z.cov_dense())

    assert jnp.allclose(logpdf_reduced, logpdf_ref)
    assert jnp.allclose(jnp.stack(means)[:, 0], data_out)
    assert jnp.allclose(jnp.stack(covs)[:, 0, :], 0.0, atol=1e-5)
    assert jnp.allclose(jnp.stack(covs)[:, :, 0], 0.0, atol=1e-5)
