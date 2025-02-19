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

    # Start reducing the model
    reduction = ckf.model_reduction(F_rank=dim.y_nonsing, impl=impl)
    prepared = reduction.prepare_init(y_mid_x=y_mid_x, x=z)
    y1, (x2, y1_mid_x2), (x_mid_x2, _) = reduction.reduce_init(y, prepared=prepared)

    # Condition on the initial data
    _, bwd = impl.rv_condition(x2, cond=y1_mid_x2)
    x2_mid_data = impl.cond_evaluate(y1, bwd)

    # Continue reducing the model
    prepared = reduction.prepare(y_mid_x=y_mid_x, x_mid_z=x_mid_z)
    y1, (z, x2_mid_z, y1_mid_x2), info = reduction.reduce(
        y, hidden=x2_mid_data, z_mid_hidden=x_mid_x2, prepared=prepared
    )

    # Run a single filter-condition step
    x2 = impl.rv_marginal(z, x2_mid_z)
    _, bwd = impl.rv_condition(x2, y1_mid_x2)
    x2_mid_data = impl.cond_evaluate(y1, bwd)

    # Assert the shapes are as expected
    x2_dim = dim.x - dim.y_sing
    assert x2_mid_data.mean.shape == (x2_dim,)
    assert x2_mid_data.cov_dense().shape == (x2_dim, x2_dim)


@pytest_cases.parametrize_with_cases("dim", cases=".", prefix="case_dim_")
@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_model_reduce_values(dim, impl):
    (x, x_mid_z, y_mid_x), F, y = test_util.model_random(dim=dim, impl=impl)

    # Reference (initialisation):
    _, bwd = impl.rv_condition(x, cond=y_mid_x)
    x_init = impl.cond_evaluate(y, cond=bwd)

    # Reference (step)
    ref_x = impl.rv_marginal(rv=x_init, cond=x_mid_z)
    _y, ref_backward = impl.rv_condition(rv=ref_x, cond=y_mid_x)
    ref_x_mid_y = impl.cond_evaluate(y, cond=ref_backward)

    # Start reducing the model
    reduction = ckf.model_reduction(F_rank=dim.y_nonsing, impl=impl)
    prepared = reduction.prepare_init(y_mid_x=y_mid_x, x=x)
    y1, (x2, y1_mid_x2), (x_mid_x2, _) = reduction.reduce_init(y, prepared=prepared)

    # Condition on the initial data
    _, bwd = impl.rv_condition(x2, cond=y1_mid_x2)
    x2_mid_data = impl.cond_evaluate(y1, bwd)

    # Continue reducing the model
    prepared = reduction.prepare(y_mid_x=y_mid_x, x_mid_z=x_mid_z)
    y1, (z, x2_mid_z, y1_mid_x2), info = reduction.reduce(
        y, hidden=x2_mid_data, z_mid_hidden=x_mid_x2, prepared=prepared
    )

    # Run a single filter-condition step
    x2 = impl.rv_marginal(z, x2_mid_z)
    _, bwd = impl.rv_condition(x2, y1_mid_x2)
    x2_mid_data = impl.cond_evaluate(y1, bwd)

    tol = jnp.sqrt(jnp.finfo(y.dtype).eps)
    x_mid_data = impl.rv_marginal(rv=x2_mid_data, cond=x_mid_x2)
    assert jnp.allclose(x_mid_data.mean, ref_x_mid_y.mean, rtol=tol, atol=tol)


@pytest_cases.parametrize_with_cases("dim", cases=".", prefix="case_dim_")
@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_model_reduce_logpdf(dim, impl):
    (x, x_mid_z, y_mid_x), F, y = test_util.model_random(dim=dim, impl=impl)

    # Reference (initialisation):
    y_marg, bwd = impl.rv_condition(x, cond=y_mid_x)
    logpdf1 = impl.rv_logpdf(y, y_marg)
    x_init = impl.cond_evaluate(y, cond=bwd)

    # Reference (step)
    ref_x = impl.rv_marginal(rv=x_init, cond=x_mid_z)
    y_marg, _ = impl.rv_condition(rv=ref_x, cond=y_mid_x)
    logpdf2 = impl.rv_logpdf(y, y_marg)

    # Start reducing the model
    reduction = ckf.model_reduction(F_rank=dim.y_nonsing, impl=impl)
    prepared = reduction.prepare_init(y_mid_x=y_mid_x, x=x)
    y1, (x2, y1_mid_x2), (x_mid_x2, pdf2) = reduction.reduce_init(y, prepared=prepared)

    # Condition on the initial data
    y1_marg, bwd = impl.rv_condition(x2, cond=y1_mid_x2)
    x2_mid_data = impl.cond_evaluate(y1, bwd)
    pdf1 = impl.rv_logpdf(y1, y1_marg)

    # Continue reducing the model
    prepared = reduction.prepare(y_mid_x=y_mid_x, x_mid_z=x_mid_z)
    y1, (z, x2_mid_z, y1_mid_x2), (_, pdf3) = reduction.reduce(
        y, hidden=x2_mid_data, z_mid_hidden=x_mid_x2, prepared=prepared
    )

    # Run a single filter-condition step
    x2 = impl.rv_marginal(z, x2_mid_z)
    y1_marg2, bwd = impl.rv_condition(x2, y1_mid_x2)
    pdf4 = impl.rv_logpdf(y1, y1_marg2)

    assert jnp.allclose(logpdf1 + logpdf2, pdf1 + pdf2 + pdf3 + pdf4)


@pytest_cases.parametrize_with_cases("dim", cases=".", prefix="case_dim_")
def test_logpdfs_consistent_across_impls(dim):
    impl = ckf.impl_cov_based()
    (z, x_mid_z, y_mid_x), F, y = test_util.model_random(dim=dim, impl=impl)
    ref_x = impl.rv_marginal(rv=z, cond=x_mid_z)
    y_marg, ref_backward = impl.rv_condition(rv=ref_x, cond=y_mid_x)
    logpdf1 = impl.rv_logpdf(y, y_marg)

    impl = ckf.impl_cholesky_based()
    (z, x_mid_z, y_mid_x), F, y = test_util.model_random(dim=dim, impl=impl)
    ref_x = impl.rv_marginal(rv=z, cond=x_mid_z)
    y_marg, ref_backward = impl.rv_condition(rv=ref_x, cond=y_mid_x)
    logpdf2 = impl.rv_logpdf(y, y_marg)

    tol = jnp.sqrt(jnp.finfo(y.dtype).eps)
    assert jnp.allclose(logpdf1, logpdf2, atol=tol, rtol=tol)


@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_kalman_filter(impl):
    (z, x_mid_z, y_mid_x), F = test_util.model_interpolation(impl=impl)

    data_in = jnp.linspace(0, 1, num=20)
    data_out = data_in + 0.1 + jnp.sin(data_in**2)

    means, covs = [], []
    x = z

    logpdf_ref = 0.0
    for d in data_out:
        x = impl.rv_marginal(rv=x, cond=x_mid_z)
        y, x = impl.rv_condition(rv=x, cond=y_mid_x)
        x = impl.cond_evaluate(jnp.atleast_1d(d), cond=x)
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

        # Run a single filter-condition step.
        #
        # To turn this into a smoother,
        # run rv_condition() instead of rv_marginal
        # which (due to some absorbing-logic)
        # yields a x2-to-x2 conditional (small)
        # instead of a x-to-x conditional (larger)
        # todo: make this logic a bit simpler
        x2 = impl.rv_marginal(z, x2_mid_z)
        y1_marg, bwd = impl.rv_condition(x2, y1_mid_x2)
        x2 = impl.cond_evaluate(y1, bwd)
        pdf1 = impl.rv_logpdf(y1, y1_marg)

        # Get the next 'z' to restart.
        # Note how we don't parametrise z with a marginal distribution
        # but by combining p(x2) and p(x | x2). Why? Because
        # this way, we can keep all conditionals in "x2-space"
        # and never do marginalisation or smoothing arithmetic in "x-space"
        z = (x2, x_mid_x2)

        xx = impl.rv_marginal(rv=x2, cond=x_mid_x2)
        logpdf_reduced += pdf1 + pdf2
        means.append(xx.mean)
        covs.append(xx.cov_dense())

    assert jnp.allclose(logpdf_reduced, logpdf_ref)
    assert jnp.allclose(jnp.stack(means)[:, 0], data_out)
    assert jnp.allclose(jnp.stack(covs)[:, 0, :], 0.0, atol=1e-5)
    assert jnp.allclose(jnp.stack(covs)[:, :, 0], 0.0, atol=1e-5)
