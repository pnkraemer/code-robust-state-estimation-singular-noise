import pathlib
import pickle
import time
from typing import Callable

import jax
import jax.numpy as jnp

from ckf import ckf, test_util


def main(seed=1, num_data=50, num_runs=3):
    key = jax.random.PRNGKey(seed)
    impl = ckf.impl_cholesky_based()

    data = {}
    # todo: run until size 2048 (or at least 1024)
    for n in [
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    ]:  # Loop over the columns of the to-be-assembled table
        data[f"$n={n}$"] = {}
        cfgs = [
            # These are the interesting configs:
            ("$m=n-1$, $r=0$", test_util.DimCfg(n, n - 1, 0)),
            (r"$m=\frac{n}{2}$, $r=0$", test_util.DimCfg(n, n // 2, 0)),
            (r"$m=\frac{n}{2}$, $r=\frac{n}{4}$", test_util.DimCfg(n, n // 4, n // 4)),
            (r"$m=\frac{n}{4}$, $r=0$", test_util.DimCfg(n, n // 4, 0)),
            (r"$m=\frac{n}{4}$, $r=\frac{n}{8}$", test_util.DimCfg(n, n // 8, n // 8)),
            # For context: config where there is nothing to gain
            (r"$m=\frac{n}{4}$, $r=\frac{n}{4}$", test_util.DimCfg(n, 0, n // 2)),
        ]
        for idx, dim in cfgs:
            print(dim)
            key, subkey = jax.random.split(key, num=2)
            model_random = test_util.model_random(subkey, dim=dim, impl=impl)
            (z, x_mid_z, y_mid_x), F, _y = model_random

            data_shape = (num_data, dim.y_sing + dim.y_nonsing)
            key, subkey = jax.random.split(key, num=2)
            data_out = jax.random.normal(subkey, shape=data_shape)

            # Assemble all Kalman filters
            unreduced = filter_unreduced(
                impl=impl, z=z, x_mid_z=x_mid_z, y_mid_x=y_mid_x
            )
            reduced = filter_reduced(
                impl=impl, z=z, x_mid_z=x_mid_z, y_mid_x=y_mid_x, F_rank=F.shape[1]
            )

            # Benchmark both filters
            results = {}
            for alg, name in [(unreduced, "unreduced"), (reduced, "reduced")]:
                ts = benchmark_filter(data_out, alg=alg, num_runs=num_runs)

                # Select the summary statistic from the runtimes (eg, fastest run)
                results[name] = float(jnp.amin(ts))
                print(f"\t{name}: \t {jnp.amin(ts):.1e}s \t (lower is better)")

            # Save the ratio of runtimes
            ratio = results["reduced"] / results["unreduced"]
            data[f"$n={n}$"][idx] = ratio

            print(f"\tRatio: \t\t {ratio:.2f} \t\t (lower is better)")

    # n and the cfgs are still in the namespace, and we use the most recent
    # values to make predictions. The predictions don't depend on precise
    # values of n, since we only look at the ratios.
    data["Prediction"] = {}
    for idx, dim in cfgs:
        n, m, r = dim.x, dim.y_sing + dim.y_nonsing, dim.y_nonsing
        reduced = flops_reduced(m=m, n=n, r=r)
        unreduced = flops_unreduced(m=m, n=n)
        ratio = reduced / unreduced
        data["Prediction"][idx] = ratio

    path = pathlib.Path(__file__).parent.resolve()
    with open(f"{path}/data_runtimes.pkl", "wb") as f:
        pickle.dump(data, f)


def benchmark_filter(data_out, alg, num_runs):
    # Execute once to compile
    alg = jax.jit(alg)
    x0, xs, pdf = alg(data_out)
    x0.mean.block_until_ready()
    xs.mean.block_until_ready()
    x0.cholesky.block_until_ready()
    xs.cholesky.block_until_ready()

    # Execute repeatedly, measuring all runtimes
    ts = []
    for _ in range(num_runs):
        t0 = time.perf_counter()

        x0, xs, pdf = alg(data_out)
        x0.mean.block_until_ready()
        xs.mean.block_until_ready()
        x0.cholesky.block_until_ready()
        xs.cholesky.block_until_ready()

        t1 = time.perf_counter()
        ts.append(t1 - t0)

    # Return a stack of runtimes
    return jnp.asarray(ts)


def flops_reduced(m: int, n: int, r: int) -> float:
    qr = n**3 + 2 * (n - m + r) ** 3 + (n - m + 2 * r) ** 3
    bwd_subst = (n - m + r) * (m - r) ** 2 + r**3
    return 2 / 3 * qr + bwd_subst


def flops_unreduced(m: int, n: int) -> float:
    qr = 2 * n**3 + (n + m) ** 3
    bwd_subst = n * m**2
    return 2 / 3 * qr + bwd_subst


def filter_unreduced(impl, z, x_mid_z, y_mid_x) -> Callable:
    kalman = ckf.kalman_filter(impl=impl)

    def apply(data_out):
        x0_ref, logpdf_ref = kalman.init(data_out[0], x=z, y_mid_x=y_mid_x)
        _, (xs_ref, logpdfs_ref) = jax.lax.scan(step, xs=data_out[1:], init=x0_ref)
        return x0_ref, xs_ref, jnp.sum(logpdfs_ref) + logpdf_ref

    def step(x, data):
        x, logpdf = kalman.step(data, z=x, x_mid_z=x_mid_z, y_mid_x=y_mid_x)
        return x, (x, logpdf)

    return apply


def filter_reduced(impl, z, x_mid_z, y_mid_x, F_rank) -> Callable:
    kalman = ckf.kalman_filter(impl=impl)

    # Prepare the reduction
    reduction = ckf.model_reduction(F_rank=F_rank, impl=impl)
    prepared_init = reduction.prepare_init(y_mid_x=y_mid_x, x=z)
    prepared = reduction.prepare(y_mid_x=y_mid_x, x_mid_z=x_mid_z)

    def apply(data_out):
        reduced_init = reduction.reduce_init(data_out[0], prepared=prepared_init)
        y1, (x2, y1_mid_x2), (x_mid_x2, pdf2) = reduced_init

        x2, pdf1 = kalman.init(y1, x=x2, y_mid_x=y1_mid_x2)
        logpdf = pdf1 + pdf2

        # Kalman filter iteration
        _, (xs, logpdfs) = jax.lax.scan(step, xs=data_out[1:], init=(x2, x_mid_x2))
        return x2, xs, jnp.sum(logpdfs) + logpdf

    def step(x2_and_cond, data):
        x2, x_mid_x2 = x2_and_cond

        reduced = reduction.reduce(
            data, hidden=x2, z_mid_hidden=x_mid_x2, prepared=prepared
        )
        y1, (z, x2_mid_z, y1_mid_x2), (x_mid_x2, pdf2) = reduced

        x2, pdf1 = kalman.step(y1, z=z, x_mid_z=x2_mid_z, y_mid_x=y1_mid_x2)
        logpdf = pdf1 + pdf2
        return (x2, x_mid_x2), (x2, logpdf)

    return apply


if __name__ == "__main__":
    main()
