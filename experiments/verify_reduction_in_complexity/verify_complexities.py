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

    ns_all = [4, 8]
    data = {r"$\ell$": [], "$r$": []}
    for n in ns_all:
        data[f"$n={n}$"] = []

    data["Prediction"] = []

    cfgs = setup_configs()

    for ell, r, dim_fun in cfgs:  # loop over rows
        data[r"$\ell$"].append(ell)
        data["$r$"].append(r)

        for n in ns_all:
            dim = dim_fun(n)
            print(dim)
            key, subkey = jax.random.split(key, num=2)
            model_random = test_util.model_random(subkey, dim=dim, impl=impl)
            (z, x_mid_z, y_mid_x), F, _y = model_random

            key, subkey = jax.random.split(key, num=2)
            sample_data = ckf.ssm_sample(impl=impl, num_data=num_data)
            data_out = sample_data(key, z, x_mid_z, y_mid_x)

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
                print(f"\t{name}: \t {jnp.amin(ts):.1e}s")

            # Save the ratio of runtimes
            ratio = results["reduced"] / results["unreduced"]
            data[f"$n={n}$"].append(ratio)

            print(f"\tRatio: \t\t {ratio:.2f}")

        # n and the cfgs are still in the namespace, and we use the most recent
        # values to make predictions. The predictions don't depend on precise
        # values of n, since we only look at the ratios.
        n, m, r = dim.x, dim.y_sing + dim.y_nonsing, dim.y_nonsing
        reduced = flops_reduced(m=m, n=n, r=r)
        unreduced = flops_unreduced(m=m, n=n)
        ratio = reduced / unreduced
        data["Prediction"].append(ratio)

    path = pathlib.Path(__file__).parent.resolve()
    with open(f"{path}/data_runtimes.pkl", "wb") as f:
        pickle.dump(data, f)


def setup_configs():
    def dim_n_minus_1_0(s):
        return test_util.DimCfg(x=s, y_sing=s - 1, y_nonsing=0)

    def dim_n2_0(s):
        return test_util.DimCfg(x=s, y_sing=s // 2, y_nonsing=0)

    def dim_n4_n4(s):
        return test_util.DimCfg(x=s, y_sing=s // 4, y_nonsing=s // 4)

    def dim_n4_0(s):
        return test_util.DimCfg(x=s, y_sing=s // 4, y_nonsing=0)

    def dim_n8_n8(s):
        return test_util.DimCfg(x=s, y_sing=s // 8, y_nonsing=s // 8)

    def dim_nogain_n0_n4(s):
        return test_util.DimCfg(x=s, y_sing=0, y_nonsing=s // 4)

    return [
        ("$n-1$", "$0$", dim_n_minus_1_0),
        ("$n/2$", "$0$", dim_n2_0),
        ("$n/4$", "$n/4$", dim_n4_n4),
        ("$n/4$", "$0$", dim_n4_0),
        ("$n/8$", "$n/8$", dim_n8_n8),
        # For context: config where there is nothing to gain
        ("$0$", "$n/4$", dim_nogain_n0_n4),
    ]


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
