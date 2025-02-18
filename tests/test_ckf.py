"""Tests for constrained Kalman filtering."""

import jax.numpy as jnp
from ckf import ckf
import jax


def test_model_align_shapes(z_dim=3, x_dim=15, y_dim=(2, 7)):
    impl = ckf.impl_cov_based()
    y, (z, x_mid_z, y_mid_x), F = _model_random(
        dim_z=z_dim, dim_x=x_dim, dim_y=y_dim, impl=impl
    )

    reduced = ckf.model_reduce(y_mid_x=y_mid_x, x_mid_z=x_mid_z, F=F, impl=impl)
    x2_mid_data, x_mid_x2 = ckf.model_reduced_apply(y, z=z, reduced=reduced, impl=impl)

    x2_dim = x_dim - y_dim[0]
    assert x2_mid_data.mean.shape == (x2_dim,)
    assert x2_mid_data.cov.shape == (x2_dim, x2_dim)


def test_model_align_values(z_dim=1, x_dim=6, y_dim=(1, 2)):
    impl = ckf.impl_cov_based()
    y, (z, x_mid_z, y_mid_x), F = _model_random(
        dim_z=z_dim, dim_x=x_dim, dim_y=y_dim, impl=impl
    )

    # Reference:
    ref_x = impl.marginal(prior=z, trafo=x_mid_z)
    _y, ref_backward = impl.condition(prior=ref_x, trafo=y_mid_x)
    ref_x_mid_y = impl.evaluate_conditional(y, trafo=ref_backward)

    # Reduced model:
    reduced = ckf.model_reduce(y_mid_x=y_mid_x, x_mid_z=x_mid_z, F=F, impl=impl)
    x2_mid_data, x_mid_x2 = ckf.model_reduced_apply(y, z=z, reduced=reduced, impl=impl)

    x_mid_data = impl.marginal(prior=x2_mid_data, trafo=x_mid_x2)

    assert jnp.allclose(x_mid_data.mean, ref_x_mid_y.mean)


def test_kalman_filter():
    impl = ckf.impl_cov_based()
    (z, x_mid_z, y_mid_x), F = _model_interpolation(impl=impl)

    data_in = jnp.linspace(0, 1, num=20)
    data_out = data_in + 0.1 + jnp.sin(data_in**2)

    means, covs = [], []
    x = z

    for d in data_out:
        x = impl.marginal(prior=x, trafo=x_mid_z)
        _, x = impl.condition(prior=x, trafo=y_mid_x)

        x = impl.evaluate_conditional(jnp.atleast_1d(d), trafo=x)
        means.append(x.mean)
        covs.append(x.cov)

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
        z = impl.marginal(prior=x2, trafo=x_mid_x2)

        means.append(z.mean)
        covs.append(z.cov)

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


def _model_random(*, dim_z, dim_x, dim_y, impl):
    key = jax.random.PRNGKey(seed=3)

    key, k1, k2 = jax.random.split(key, num=3)
    m0 = jax.random.normal(k1, shape=(dim_z,))
    c0 = jax.random.normal(k2, shape=(dim_z, dim_z))
    z = impl.rv_from_cholesky(m0, c0)

    # z = ckf.RandVar(m0, c0 @ c0.T)

    key, k1, k2, k3 = jax.random.split(key, num=4)
    linop = jax.random.normal(k1, shape=(dim_x, dim_z))
    bias = jax.random.normal(k2, shape=(dim_x,))
    cov = jax.random.normal(k3, shape=(dim_x, dim_x))
    noise = impl.rv_from_cholesky(bias, cov)
    x_mid_z = ckf.Trafo(linop, noise)

    key, k1, k2, k3 = jax.random.split(key, num=4)
    dim_y_sing, dim_y_nonsing = dim_y
    dim_y_total = dim_y_sing + dim_y_nonsing
    assert dim_x >= dim_y_total

    linop = jax.random.normal(k1, shape=(dim_y_total, dim_x))
    bias = jax.random.normal(k2, shape=(dim_y_total,))
    F = jax.random.normal(k3, shape=(dim_y_total, dim_y_nonsing))
    noise = impl.rv_from_cholesky(bias, F)
    y_mid_x = ckf.Trafo(linop, noise)

    data = jax.random.normal(key, shape=(dim_y_total,))
    return data, (z, x_mid_z, y_mid_x), F
