import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tqdm
from tueplots import axes, bundles

from ckf import ckf


def main():
    plt.rcParams.update(bundles.icml2022(usetex=True, column="half"))
    plt.rcParams.update(axes.lines())
    plt.rcParams.update(axes.legend())
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
    optimizer = optax.adam(0.25)
    opt_state = optimizer.init(theta)

    fig, ax = plt.subplots(dpi=200)

    (val, ms), grads = loss(theta, data_out)

    ax.plot(
        xs,
        data_out,
        ".",
        markeredgewidth=0.75,
        markeredgecolor="black",
        markerfacecolor="white",
        alpha=0.5,
        label="Observations",
    )
    ax.plot(
        xs[1:],
        ms[:, :2],
        alpha=0.8,
        linestyle="dashed",
        color="C0",
        label="Before optimisation",
    )
    progressbar = tqdm.tqdm(range(5_000))
    progressbar.set_description(f"{1e4:1e}")
    for _ in progressbar:
        try:
            (val, ms), grads = loss(theta, data_out)
            updates, opt_state = optimizer.update(grads, opt_state)
            theta = optax.apply_updates(theta, updates)
            progressbar.set_description(f"{val:1e}")
        except KeyboardInterrupt:
            break
    print()
    print(theta)
    print()
    print(theta_true)

    ax.plot(
        xs[1:],
        ms[:, :2],
        alpha=0.8,
        linestyle="solid",
        color="C1",
        label="After optimisation",
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_ylabel(r"Observations \& mean estimate")
    ax.set_xlabel("Time $t$")

    path = pathlib.Path(__file__).parent.resolve()
    plt.savefig(f"{path}/figure.pdf")
    plt.show()


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
    cov = jnp.linalg.cholesky(1e-2 * jax.scipy.linalg.hilbert(3))
    noise = impl.rv_from_cholesky(bias, cov)
    x_mid_z = ckf.AffineCond(linop, noise)

    linop = jnp.eye(2, 3)
    bias = jnp.zeros((2,))
    F = jnp.array([[1.0], [0.0]])
    noise = impl.rv_from_cholesky(bias, F)
    y_mid_x = ckf.AffineCond(linop, noise)
    return z, x_mid_z, y_mid_x


if __name__ == "__main__":
    main()
