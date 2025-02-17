import jax.numpy as jnp
from ckf import ckf
import jax

import time


def main(d=2):
    (z, x_mid_z, y_mid_x), F = _model_interpolation(d=d)

    data_in = jnp.linspace(0, 1, num=1000)
    data_out = data_in + 0.1 + jnp.sin(data_in**2)
    data_out = data_out[..., None] * jnp.ones((1, d))

    def step(x, data):
        x = ckf.marginal(prior=x, trafo=x_mid_z)
        _, x = ckf.condition(prior=x, trafo=y_mid_x)

        x = ckf.evaluate_conditional(jnp.atleast_1d(data), trafo=x)
        return x, (x.mean, x.cov)

    _, (means, covs) = jax.lax.scan(step, xs=data_out, init=z)
    means.block_until_ready()
    covs.block_until_ready()

    best_yet = 1000
    for _ in range(100):
        t0 = time.perf_counter()
        _, (means, covs) = jax.lax.scan(step, xs=data_out, init=z)
        means.block_until_ready()
        covs.block_until_ready()
        t1 = time.perf_counter()
        if t1 - t0 < best_yet:
            best_yet = t1 - t0

    print(best_yet)

    # assert jnp.allclose(jnp.stack(means)[:, 0], data_out)
    # assert jnp.allclose(jnp.stack(covs)[:, 0, :], 0.0, atol=1e-5)
    # assert jnp.allclose(jnp.stack(covs)[:, :, 0], 0.0, atol=1e-5)

    # Why is the below massively faster???

    def step(x, data):
        reduced = ckf.model_reduce(y_mid_x=y_mid_x, x_mid_z=x_mid_z, F=F)

        data = jnp.atleast_1d(data)
        x2, x_mid_x2 = ckf.model_reduced_apply(data, z=z, reduced=reduced)
        x = ckf.marginal(prior=x2, trafo=x_mid_x2)
        return x, (x.mean, x.cov)

    _, (means, covs) = jax.lax.scan(step, xs=data_out, init=z)
    means.block_until_ready()
    covs.block_until_ready()

    best_yet = 1000
    for _ in range(100):
        t0 = time.perf_counter()
        _, (means, covs) = jax.lax.scan(step, xs=data_out, init=z)
        means.block_until_ready()
        covs.block_until_ready()
        t1 = time.perf_counter()
        if t1 - t0 < best_yet:
            best_yet = t1 - t0

    print(best_yet)

    # assert jnp.allclose(jnp.stack(means)[:, 0], data_out)
    # assert jnp.allclose(jnp.stack(covs)[:, 0, :], 0.0, atol=1e-5)
    # assert jnp.allclose(jnp.stack(covs)[:, :, 0], 0.0, atol=1e-5)


def _model_interpolation(d=5):
    eye_d = jnp.eye(d)

    m0 = jnp.concatenate([jnp.zeros((2,))] * d, axis=0)
    c0 = jnp.kron(jnp.eye(2), eye_d)
    z = ckf.RandVar(m0, c0)

    linop = jnp.kron(jnp.eye(2), eye_d)
    bias = jnp.concatenate([jnp.zeros((2,))] * d, axis=0)
    cov = jnp.kron(jnp.eye(2), eye_d)
    x_mid_z = ckf.Trafo(linop, bias, cov)

    linop = jnp.kron(jnp.eye(1, 2), eye_d)
    bias = jnp.concatenate([jnp.zeros((1,))] * d, axis=0)
    cov = jnp.kron(jnp.zeros((1, 1)), eye_d)
    y_mid_x = ckf.Trafo(linop, bias, cov)

    return (z, x_mid_z, y_mid_x), cov


if __name__ == "__main__":
    main()
