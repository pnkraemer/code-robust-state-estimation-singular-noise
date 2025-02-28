"""Tests for constrained Kalman filtering."""

import jax
import jax.numpy as jnp
import pytest_cases

from ckf import ckf, test_util


def case_impl_cov_based() -> ckf.Impl:
    return ckf.impl_cov_based(solve_fun=jnp.linalg.solve)


def case_impl_cholesky_based() -> ckf.Impl:
    return ckf.impl_cholesky_based()


def case_dim_base() -> test_util.DimCfg:
    # The base-case is (7, 2, 4) because we have arrays
    # and matrices with shapes (2, 4, 7-2, 7-(4-2)),
    # and we want all of those to be larger than 1
    # because this way, all covariance matrices are
    # "proper matrices" and we get punished for
    # incorrect transposing, for example.
    return test_util.DimCfg(x=7, y_sing=2, y_nonsing=4)


def case_dim_nonsing_zero() -> test_util.DimCfg:
    return test_util.DimCfg(x=7, y_sing=2, y_nonsing=4)


def case_dim_sing_zero() -> test_util.DimCfg:
    return test_util.DimCfg(x=7, y_sing=0, y_nonsing=4)


def case_dim_sing_and_nonsing_zero() -> test_util.DimCfg:
    return test_util.DimCfg(x=7, y_sing=0, y_nonsing=0)


@pytest_cases.parametrize_with_cases("dim", cases=".", prefix="case_dim_")
@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_model_reduce_shapes(impl, dim):
    key = jax.random.PRNGKey(seed=3)
    tmp = test_util.model_random_time_invariant(key, dim=dim, impl=impl)
    (z, x_mid_z, y_mid_x), F, y = tmp

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
    key = jax.random.PRNGKey(seed=3)
    tmp = test_util.model_random_time_invariant(key, dim=dim, impl=impl)
    (x, x_mid_z, y_mid_x), F, y = tmp

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
    key = jax.random.PRNGKey(seed=3)
    tmp = test_util.model_random_time_invariant(key, dim=dim, impl=impl)
    (x, x_mid_z, y_mid_x), F, y = tmp

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
    impl = ckf.impl_cov_based(solve_fun=jnp.linalg.solve)
    key = jax.random.PRNGKey(seed=3)
    tmp = test_util.model_random_time_invariant(key, dim=dim, impl=impl)
    (z, x_mid_z, y_mid_x), F, y = tmp

    ref_x = impl.rv_marginal(rv=z, cond=x_mid_z)
    y_marg, ref_backward = impl.rv_condition(rv=ref_x, cond=y_mid_x)
    logpdf1 = impl.rv_logpdf(y, y_marg)

    impl = ckf.impl_cholesky_based()
    tmp = test_util.model_random_time_invariant(key, dim=dim, impl=impl)
    (z, x_mid_z, y_mid_x), F, y = tmp
    ref_x = impl.rv_marginal(rv=z, cond=x_mid_z)
    y_marg, ref_backward = impl.rv_condition(rv=ref_x, cond=y_mid_x)
    logpdf2 = impl.rv_logpdf(y, y_marg)

    tol = jnp.sqrt(jnp.finfo(y.dtype).eps)
    assert jnp.allclose(logpdf1, logpdf2, atol=tol, rtol=tol)


def _allclose(a, b):
    tol = jnp.sqrt(jnp.finfo(a.dtype).eps)
    return jnp.allclose(a, b, atol=tol, rtol=tol)


@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_kalman_filter(impl):
    key = jax.random.PRNGKey(1)
    dim = test_util.DimCfg(x=3, y_sing=2, y_nonsing=0)
    (z, x_mid_z, y_mid_x), F = test_util.model_hilbert(dim=dim, impl=impl)
    data_out = jax.random.normal(key, shape=(20, dim.y_sing))

    # Assemble a Kalman filter
    kalman = ckf.kalman_filter(impl=impl)

    # Initialise
    x0_ref, logpdf_ref = kalman.init(data_out[0], x=z, y_mid_x=y_mid_x)
    assert _allclose(x0_ref.mean[: dim.y_sing], data_out[0])
    assert _allclose(x0_ref.cov_dense()[:, : dim.y_sing], 0.0)
    assert _allclose(x0_ref.cov_dense()[: dim.y_sing, :], 0.0)

    # Kalman filter iteration
    def step(x, data):
        x, logpdf = kalman.step(data, z=x, x_mid_z=x_mid_z, y_mid_x=y_mid_x)
        return x, (x, logpdf)

    _, (xs_ref, logpdfs_ref) = jax.lax.scan(step, xs=data_out[1:], init=x0_ref)
    assert _allclose(xs_ref.mean[:, : dim.y_sing], data_out[1:])
    assert _allclose(xs_ref.cov_dense()[:, :, : dim.y_sing], 0.0)
    assert _allclose(xs_ref.cov_dense()[:, : dim.y_sing, :], 0.0)

    # Reduce the model
    reduction = ckf.model_reduction(F_rank=F.shape[1], impl=impl)
    prepared_init = reduction.prepare_init(y_mid_x=y_mid_x, x=z)
    prepared = reduction.prepare(y_mid_x=y_mid_x, x_mid_z=x_mid_z)

    # Initialise
    y1, (x2, y1_mid_x2), (x_mid_x2, pdf2) = reduction.reduce_init(
        data_out[0], prepared=prepared_init
    )
    x2, pdf1 = kalman.init(y1, x=x2, y_mid_x=y1_mid_x2)
    logpdf_reduced = pdf1 + pdf2
    x0 = impl.rv_marginal(x2, cond=x_mid_x2)  # only for testing
    assert _allclose(x0.mean, x0_ref.mean)
    assert _allclose(x0.cov_dense(), x0_ref.cov_dense())
    assert _allclose(logpdf_reduced, logpdf_ref)

    # Kalman filter iteration

    def step(x2_and_cond, data):
        x2_, x_mid_x2_ = x2_and_cond
        y1_, (z_, x2_mid_z_, y1_mid_x2_), (x_mid_x2_, pdf2_) = reduction.reduce(
            data, hidden=x2_, z_mid_hidden=x_mid_x2_, prepared=prepared
        )

        # Run a single filter-condition step.
        x2_, pdf1_ = kalman.step(y1_, z=z_, x_mid_z=x2_mid_z_, y_mid_x=y1_mid_x2_)

        # Save some quantities (not part of the simulation, just for testing)
        xx = impl.rv_marginal(rv=x2_, cond=x_mid_x2_)
        logpdf_ = pdf1_ + pdf2_
        return (x2_, x_mid_x2_), (xx, logpdf_)

    _, (xs, logpdfs) = jax.lax.scan(step, xs=data_out[1:], init=(x2, x_mid_x2))
    assert _allclose(xs.mean, xs_ref.mean)
    assert _allclose(xs.cov_dense(), xs_ref.cov_dense())
    assert _allclose(logpdfs, logpdfs_ref)


@pytest_cases.parametrize_with_cases("impl", cases=".", prefix="case_impl_")
def test_rts_smoother(impl, num_data=5):
    key = jax.random.PRNGKey(1)
    dim = test_util.DimCfg(x=3, y_sing=2, y_nonsing=0)
    (z, x_mid_z, y_mid_x), F = test_util.model_hilbert(dim=dim, impl=impl)
    data_out = jax.random.normal(key, shape=(num_data, dim.y_sing))

    # Assemble a RTS smoother
    kalman = ckf.rts_smoother(impl=impl)

    # Kalman smoother iteration

    def step(x, data):
        x, logpdf, bwd = kalman.step(data, z=x, x_mid_z=x_mid_z, y_mid_x=y_mid_x)
        return x, (logpdf, bwd)

    x0_ref, logpdf_ref = kalman.init(data_out[0], x=z, y_mid_x=y_mid_x)
    xx_ref, (logpdfs_ref, smoothing) = jax.lax.scan(step, xs=data_out[1:], init=x0_ref)
    assert _allclose(xx_ref.mean[: dim.y_sing], data_out[-1])
    assert _allclose(xx_ref.cov_dense()[:, : dim.y_sing], 0.0)
    assert _allclose(xx_ref.cov_dense()[: dim.y_sing, :], 0.0)

    def smoothing_step(x, bwd):
        x = impl.rv_marginal(x, bwd)
        return x, x

    _, xs_ref = jax.lax.scan(smoothing_step, xs=smoothing, init=xx_ref, reverse=True)
    assert _allclose(xs_ref.mean[:, : dim.y_sing], data_out[:-1])
    assert _allclose(xs_ref.cov_dense()[:, :, : dim.y_sing], 0.0)
    assert _allclose(xs_ref.cov_dense()[:, : dim.y_sing, :], 0.0)

    # Reduce the model
    reduction = ckf.model_reduction(F_rank=F.shape[1], impl=impl)
    prepared_init = reduction.prepare_init(y_mid_x=y_mid_x, x=z)
    prepared = reduction.prepare(y_mid_x=y_mid_x, x_mid_z=x_mid_z)

    # Initialise
    y1, (x2, y1_mid_x2), (x_mid_x2, pdf2) = reduction.reduce_init(
        data_out[0], prepared=prepared_init
    )
    x2, pdf1 = kalman.init(y1, x=x2, y_mid_x=y1_mid_x2)
    logpdf_reduced = pdf1 + pdf2
    assert _allclose(logpdf_reduced, logpdf_ref)

    # Kalman filter iteration

    def step(x2_and_cond, data):
        x2_, x_mid_x2_ = x2_and_cond
        save = x_mid_x2_  # careful which x_mid_x2 is returned!
        y1_, (z_, x2_mid_z_, y1_mid_x2_), (x_mid_x2_, pdf2_) = reduction.reduce(
            data, hidden=x2_, z_mid_hidden=x_mid_x2_, prepared=prepared
        )

        # Run a single filter-condition step.
        x2_, pdf1_, smooth_ = kalman.step(
            y1_, z=z_, x_mid_z=x2_mid_z_, y_mid_x=y1_mid_x2_
        )

        logpdf_ = pdf1_ + pdf2_
        return (x2_, x_mid_x2_), (logpdf_, smooth_, save)

    (x2, x_mid_x2), (logpdfs, smoothing, x_mid_x2_all) = jax.lax.scan(
        step, xs=data_out[1:], init=(x2, x_mid_x2)
    )

    xx = impl.rv_marginal(x2, x_mid_x2)
    assert _allclose(xx.mean, xx_ref.mean)
    assert _allclose(xx.cov_dense(), xx_ref.cov_dense())
    assert _allclose(logpdfs, logpdfs_ref)

    def smoothing_step(x2_, smooth_and_cond):
        smooth, x_mid_x2_ = smooth_and_cond
        x2_ = impl.rv_marginal(x2_, smooth)
        xx_ = impl.rv_marginal(x2_, x_mid_x2_)  # mainly for testing
        return x2_, xx_

    _, (xs) = jax.lax.scan(
        smoothing_step, xs=(smoothing, x_mid_x2_all), init=x2, reverse=True
    )

    assert _allclose(xs.mean, xs_ref.mean)
    assert _allclose(xs.cov_dense(), xs_ref.cov_dense())
