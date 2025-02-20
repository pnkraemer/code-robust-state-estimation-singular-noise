import jax
import jax.numpy as jnp
import pytest_cases

from ckf import ckf, test_util
import optax
import tqdm
import matplotlib.pyplot as plt


def main():
    key = jax.random.PRNGKey(1)
    impl = ckf.impl_cholesky_based()

    key, subkey = jax.random.split(key, num=2)
    theta_true = jax.random.normal(subkey, shape=(3,)) / 10
    (z, x_mid_z, y_mid_x) = model(theta_true, impl=impl)

    key, subkey = jax.random.split(key, num=2)
    data_out = sample(key, z, x_mid_z, y_mid_x, impl=impl, num_samples=100)

    xs = jnp.linspace(0, 1, num=data_out.shape[0])

    likelihood = marginal_likelihood(impl=impl)
    loss = jax.jit(jax.value_and_grad(likelihood, has_aux=True))

    theta = jnp.zeros((3,))
    optimizer = optax.adam(0.125)
    opt_state = optimizer.init(theta)

    layout = [["before", "after"]]
    fig, ax = plt.subplot_mosaic(layout, sharex=True, sharey=True)

    (val, ms), grads = loss(theta, data_out)
    ax["before"].set_title("Before optimisation")
    ax["before"].plot(xs, data_out, ".", color="gray")
    ax["before"].plot(xs[1:], ms)

    progressbar = tqdm.tqdm(range(1000))
    progressbar.set_description(f"{1e-4:1e}")
    for _ in progressbar:
        (val, ms), grads = loss(theta, data_out)
        updates, opt_state = optimizer.update(grads, opt_state)
        theta = optax.apply_updates(theta, updates)
        progressbar.set_description(f"{val:1e}")

    ax["after"].set_title("After optimisation")
    ax["after"].plot(xs, data_out, ".", color="gray")
    ax["after"].plot(xs[1:], ms)
    plt.show()
    assert False

    print()
    print(theta)
    print()
    print(theta_true)

def sample(key, z, x_mid_z, y_mid_x, impl, num_samples=10):
    key, subkey = jax.random.split(key, num=2)
    x0 = impl.rv_sample(subkey, z)

    key, subkey = jax.random.split(key, num=2)
    y0 = impl.cond_evaluate(x0, y_mid_x)
    y0_sample = impl.rv_sample(subkey, y0)
    samples = [y0_sample]
    for _ in range(num_samples - 1):
        key, subkey = jax.random.split(key, num=2)
        x = impl.cond_evaluate(x0, x_mid_z)
        x0 = impl.rv_sample(subkey, x)

        key, subkey = jax.random.split(key, num=2)
        y0 = impl.cond_evaluate(x0, y_mid_x)
        y0_sample = impl.rv_sample(subkey, y0)
        samples.append(y0_sample)
    return jnp.stack(samples)


def marginal_likelihood(impl):
    @jax.jit
    def evaluate(theta, data_out):
        kalman = ckf.kalman_filter(impl=impl)

        (z, x_mid_z, y_mid_x) = model(theta, impl=impl)

        # Reduce the model
        reduction = ckf.model_reduction(F_rank=1, impl=impl)
        prepared_init = reduction.prepare_init(y_mid_x=y_mid_x, x=z)
        prepared = reduction.prepare(y_mid_x=y_mid_x, x_mid_z=x_mid_z)

        # Initialise
        y1, (x2, y1_mid_x2), (x_mid_x2, pdf2) = reduction.reduce_init(
            data_out[0], prepared=prepared_init
        )
        x2, pdf1 = kalman.init(y1, x=x2, y_mid_x=y1_mid_x2)
        logpdf_reduced = pdf1 + pdf2

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
            return (x2_, x_mid_x2_), (xx.mean, logpdf_)

        _, (ms, logpdfs) = jax.lax.scan(step, xs=data_out[1:], init=(x2, x_mid_x2))
        logpdf = logpdf_reduced + jnp.sum(logpdfs)
        return -logpdf, ms

    return evaluate


def model(theta, impl):
    m0 = jnp.zeros(shape=(3,))
    c0 = jnp.eye(3)
    z = impl.rv_from_cholesky(m0, c0)

    linop = jnp.array([[1, theta[0], theta[1]], [0, 1, theta[2]], [0, 0, 1]])
    bias = jnp.zeros((3,))
    cov = 1e-2*jnp.linalg.cholesky(jax.scipy.linalg.hilbert(3))
    noise = impl.rv_from_cholesky(bias, cov)
    x_mid_z = ckf.AffineCond(linop, noise)

    linop = jnp.eye(2, 3)
    bias = jnp.zeros((2,))
    F = jnp.array([[2.], [0.]])
    noise = impl.rv_from_cholesky(bias, F)
    y_mid_x = ckf.AffineCond(linop, noise)
    return z, x_mid_z, y_mid_x


if __name__ == "__main__":
    main()
