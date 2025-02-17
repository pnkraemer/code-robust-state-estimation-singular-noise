"""Tests for constrained Kalman filtering."""

import jax.numpy as jnp
from ckf import ckf
import jax

jax.config.update("jax_enable_x64", True)


def test_kalman_filter():
    (z, x_mid_z, y_mid_x) = _model_interpolation()

    data_in = jnp.linspace(0, 1, num=20)
    data_out = data_in + 0.1 + jnp.sin(data_in**2)

    means, covs = [], []
    x = z

    for d in data_out:
        x = ckf.marginal(prior=x, trafo=x_mid_z)
        _, x = ckf.condition(prior=x, trafo=y_mid_x)

        x = ckf.evaluate_conditional(jnp.atleast_1d(d), trafo=x)
        means.append(x.mean)
        covs.append(x.cov)

    assert jnp.allclose(jnp.stack(means)[:, 0], data_out)
    assert jnp.allclose(jnp.stack(covs)[:, 0, :], 0.0, atol=1e-5)
    assert jnp.allclose(jnp.stack(covs)[:, :, 0], 0.0, atol=1e-5)

    means, covs = [], []

    # todo: make square-root
    # todo: track marginal likelihoods
    x_mid_z_general = x_mid_z
    for d in data_out:
        d = jnp.atleast_1d(d)
        reduced = ckf.model_reduce(y_mid_x=y_mid_x, x_mid_z=x_mid_z, z=z)
        x2_mid_data, x_mid_x2 = ckf.model_reduced_apply(d, reduced=reduced)

        # x2_mid_data is the new "z", and the new "x_mid_z" involves
        # the reconstruction of the original state
        z = x2_mid_data
        x_mid_z = ckf.combine(outer=x_mid_z_general, inner=x_mid_x2)

        # Reconstruct (for plotting reasons) then save
        reconstructed = ckf.marginal(prior=x2_mid_data, trafo=x_mid_x2)

        means.append(reconstructed.mean)
        covs.append(reconstructed.cov)

    assert jnp.allclose(jnp.stack(means)[:, 0], data_out)
    assert jnp.allclose(jnp.stack(covs)[:, 0, :], 0.0, atol=1e-5)
    assert jnp.allclose(jnp.stack(covs)[:, :, 0], 0.0, atol=1e-5)


def test_filter_one_step_works(z_dim=1, x_dim=9, y_dim=(2, 5)):
    data, (z, x_mid_z, y_mid_x) = _model_random(dim_z=z_dim, dim_x=x_dim, dim_y=y_dim)

    x = ckf.marginal(prior=z, trafo=x_mid_z)
    _y, backward = ckf.condition(prior=x, trafo=y_mid_x)
    x_mid_y = ckf.evaluate_conditional(data, trafo=backward)

    assert x_mid_y.mean.shape == (x_dim,)
    assert x_mid_y.cov.shape == (x_dim, x_dim)

    rank = x_dim - y_dim[0]  # xdim - singular dim
    assert jnp.linalg.matrix_rank(x_mid_y.cov) == rank


def test_model_align_shapes(z_dim=3, x_dim=15, y_dim=(2, 7)):
    y, (z, x_mid_z, y_mid_x) = _model_random(dim_z=z_dim, dim_x=x_dim, dim_y=y_dim)
    # out = ckf.model_reduce(y, y_mid_x=y_mid_x, x_mid_z=x_mid_z, z=z)
    # z_small, x2_mid_z_small, y1_mid_x2_small, y1, x_from_x2 = out
    reduced = ckf.model_reduce(y_mid_x=y_mid_x, x_mid_z=x_mid_z, z=z)
    x2_mid_data, x_mid_x2 = ckf.model_reduced_apply(y, reduced=reduced)

    x2_dim = x_dim - y_dim[0]
    assert x2_mid_data.mean.shape == (x2_dim,)
    assert x2_mid_data.cov.shape == (x2_dim, x2_dim)



def test_model_align_values(z_dim=1, x_dim=7, y_dim=(2, 3)):
    y, (z, x_mid_z, y_mid_x) = _model_random(dim_z=z_dim, dim_x=x_dim, dim_y=y_dim)

    # Reference:
    ref_x = ckf.marginal(prior=z, trafo=x_mid_z)
    _y, ref_backward = ckf.condition(prior=ref_x, trafo=y_mid_x)
    ref_x_mid_y = ckf.evaluate_conditional(y, trafo=ref_backward)

    # Reduced model:
    reduced = ckf.model_reduce(y_mid_x=y_mid_x, x_mid_z=x_mid_z, z=z)
    x2_mid_data, x_mid_x2 = ckf.model_reduced_apply(y, reduced=reduced)

    x_mid_data = ckf.marginal(prior=x2_mid_data, trafo=x_mid_x2)
    assert jnp.allclose(x_mid_data.mean, ref_x_mid_y.mean)


def _model_interpolation():
    m0 = jnp.zeros((2,))
    c0 = jnp.eye(2)
    z = ckf.RandVar(m0, c0)

    linop = jnp.eye(2)
    bias = jnp.zeros((2,))
    cov = jnp.eye(2)
    x_mid_z = ckf.Trafo(linop, bias, cov)

    linop = jnp.eye(1, 2)
    bias = jnp.zeros((1,))
    cov = jnp.zeros((1, 1))
    y_mid_x = ckf.Trafo(linop, bias, cov)

    return (z, x_mid_z, y_mid_x)


def _model_random(*, dim_z, dim_x, dim_y):
    key = jax.random.PRNGKey(seed=2)

    key, k1, k2 = jax.random.split(key, num=3)
    m0 = jax.random.normal(k1, shape=(dim_z,))
    c0 = jax.random.normal(k2, shape=(dim_z, dim_z))
    z = ckf.RandVar(m0, c0 @ c0.T)

    key, k1, k2, k3 = jax.random.split(key, num=4)
    linop = jax.random.normal(k1, shape=(dim_x, dim_z))
    bias = jax.random.normal(k2, shape=(dim_x,))
    cov = jax.random.normal(k3, shape=(dim_x, dim_x))
    x_mid_z = ckf.Trafo(linop, bias, cov @ cov.T)

    key, k1, k2, k3 = jax.random.split(key, num=4)
    dim_y_sing, dim_y_nonsing = dim_y
    dim_y_total = dim_y_sing + dim_y_nonsing
    assert dim_x >= dim_y_total
    linop = jax.random.normal(k1, shape=(dim_y_total, dim_x))
    bias = jax.random.normal(k2, shape=(dim_y_total,))
    cov = jax.random.normal(k3, shape=(dim_y_total, dim_y_nonsing))
    y_mid_x = ckf.Trafo(linop, bias, cov @ cov.T)

    data = jax.random.normal(key, shape=(dim_y_total,))
    return data, (z, x_mid_z, y_mid_x)
