import jax.numpy as jnp
from ckf import ckf, test_util
import jax

import time
import pickle
import pathlib
QR_CONST = 2 / 3


def main(seed=1, num_data=50, num_runs=1):

    key = jax.random.PRNGKey(seed)
    impl = ckf.impl_cholesky_based()

    data = {}
    for n in [4, 8]:  # Loop over the columns of the to-be-assembled table
        data[f"$n={n}$"] = {}
        cfgs = [
            ("$m=n-1$, $r=0$", test_util.DimCfg(n, n-1, 0)),
            (r"$m=\frac{n}{2}$, $r=0$", test_util.DimCfg(n, n // 2, 0)),
            (r"$m=\frac{n}{2}$, $r=\frac{n}{4}$",test_util.DimCfg(n, n // 4, n // 4)),
            (r"$m=\frac{n}{4}$, $r=0$", test_util.DimCfg(n, n // 4, 0)),
            (r"$m=\frac{n}{4}$, $r=\frac{n}{8}$", test_util.DimCfg(n, n // 8, n // 8)),
            (r"$m=\frac{n}{4}$, $r=\frac{n}{4}$", test_util.DimCfg(n, 0, n // 2)),  # for context
        ]
        for idx, dim in cfgs:
            print(dim)
            key, subkey = jax.random.split(key, num=2)
            model_random = test_util.model_random(subkey, dim=dim, impl=impl)
            (z, x_mid_z, y_mid_x), F, _y = model_random

            # Assemble all Kalman filters
            unreduced = _filter_unreduced(impl=impl, z=z, x_mid_z=x_mid_z, y_mid_x=y_mid_x)
            reduced = _filter_reduced(
                impl=impl, z=z, x_mid_z=x_mid_z, y_mid_x=y_mid_x, F_rank=F.shape[1]
            )

            # Benchmark both filters
            results = {}
            for alg, name in [(unreduced, "Unreduced"), (reduced, "Reduced")]:
                alg = jax.jit(alg)
                key, subkey = jax.random.split(key, num=2)
                data_out = jax.random.normal(
                    subkey, shape=(num_data, dim.y_sing + dim.y_nonsing)
                )

                x0, xs, pdf = alg(data_out)
                x0.mean.block_until_ready()
                xs.mean.block_until_ready()
                x0.cholesky.block_until_ready()
                xs.cholesky.block_until_ready()
                ts = []
                for _ in range(num_runs):
                    key, subkey = jax.random.split(key, num=2)
                    data_out = jax.random.normal(
                        subkey, shape=(num_data, dim.y_sing + dim.y_nonsing)
                    )

                    t0 = time.perf_counter()
                    x0, xs, pdf = alg(data_out)
                    x0.mean.block_until_ready()
                    xs.mean.block_until_ready()
                    x0.cholesky.block_until_ready()
                    xs.cholesky.block_until_ready()

                    t1 = time.perf_counter()
                    ts.append(t1 - t0)

                # Select the summary statistic from the runtimes (eg, fastest run)
                ts = jnp.asarray(ts)
                results[name] = float(jnp.amin(ts))
                print(f"\t{name}: \t {jnp.amin(ts):.1e}s \t (lower is better)")

            # Save the ratio of runtimes
            ratio = results["Reduced"] / results["Unreduced"]
            data[f"$n={n}$"][idx] = ratio

            print(f"\tRatio: \t\t {ratio:.2f} \t\t (lower is better)")

    # n and the cfgs are still in the namespace, and we use the most recent
    # values to make predictions. The predictions don't depend on precise
    # values of n, since we only look at the ratios.
    data["Predicted"] = {}
    for idx, dim in cfgs:
        n, m, r = dim.x, dim.y_sing + dim.y_nonsing, dim.y_nonsing
        reduced = flops_reduced(m=m, n=n, r=r)
        unreduced = flops_unreduced(m=m, n=n)
        ratio = reduced / unreduced
        data["Predicted"][idx] = ratio


    path = pathlib.Path(__file__).parent.resolve()
    with open(f"{path}/data_runtimes.pkl", 'wb') as f:
        pickle.dump(data, f)



def flops_reduced(m, n, r):
    qr = n ** 3 + 2 * (n - m + r) ** 3 + (n - m + 2 * r) ** 3
    bwd_subst = (n-m+r) * (m - r) ** 2 + r ** 3
    return QR_CONST * qr + bwd_subst

def flops_unreduced(m, n):
    qr = 2 * n ** 3 + (n + m) ** 3
    bwd_subst = n * m ** 2
    return QR_CONST * qr + bwd_subst

def _filter_unreduced(impl, z, x_mid_z, y_mid_x):
    kalman = ckf.kalman_filter(impl=impl)

    @jax.jit
    def apply(data_out):
        # Initialise
        x0_ref, logpdf_ref = kalman.init(data_out[0], x=z, y_mid_x=y_mid_x)

        # Kalman filter iteration
        def step(x, data):
            x, logpdf = kalman.step(data, z=x, x_mid_z=x_mid_z, y_mid_x=y_mid_x)
            return x, (x, logpdf)

        _, (xs_ref, logpdfs_ref) = jax.lax.scan(step, xs=data_out[1:], init=x0_ref)

        return x0_ref, xs_ref, jnp.sum(logpdfs_ref) + logpdf_ref

    return apply


def _filter_reduced(impl, z, x_mid_z, y_mid_x, F_rank):
    kalman = ckf.kalman_filter(impl=impl)

    reduction = ckf.model_reduction(F_rank=F_rank, impl=impl)
    prepared_init = reduction.prepare_init(y_mid_x=y_mid_x, x=z)
    prepared = reduction.prepare(y_mid_x=y_mid_x, x_mid_z=x_mid_z)

    @jax.jit
    def apply(data_out):
        # Initialise
        y1, (x2, y1_mid_x2), (x_mid_x2, pdf2) = reduction.reduce_init(
            data_out[0], prepared=prepared_init
        )
        x2, pdf1 = kalman.init(y1, x=x2, y_mid_x=y1_mid_x2)
        logpdf = pdf1 + pdf2

        # Kalman filter iteration

        def step(x2_and_cond, data):
            x2_, x_mid_x2_ = x2_and_cond
            y1_, (z_, x2_mid_z_, y1_mid_x2_), (x_mid_x2_, pdf2_) = reduction.reduce(
                data, hidden=x2_, z_mid_hidden=x_mid_x2_, prepared=prepared
            )

            # Run a single filter-condition step.
            x2_, pdf1_ = kalman.step(y1_, z=z_, x_mid_z=x2_mid_z_, y_mid_x=y1_mid_x2_)

            # Save some quantities (not part of the simulation, just for testing)
            logpdf_ = pdf1_ + pdf2_
            return (x2_, x_mid_x2_), (x2_, logpdf_)

        _, (xs, logpdfs) = jax.lax.scan(step, xs=data_out[1:], init=(x2, x_mid_x2))

        return x2, xs, jnp.sum(logpdfs) + logpdf

    return apply


if __name__ == "__main__":
    main()
