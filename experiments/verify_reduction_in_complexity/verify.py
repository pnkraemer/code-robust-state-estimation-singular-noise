import pathlib
import pickle
import time
from typing import Callable

import jax
import jax.numpy as jnp

from ckf import ckf, test_util

import os

# disable constant folding because it raises a bunch of warnings
# during compilation
os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=constant_folding"

# https://docs.jax.dev/en/latest/_autosummary/jax.clear_caches.html#jax.clear_caches
jax.config.update("jax_enable_compilation_cache", False)


def main(seed=1, num_data=50, num_runs=3):
    key = jax.random.PRNGKey(seed)
    impl = ckf.impl_cholesky_based()

    ns_all = [10, 100]
    data = {r"$\ell$": [], "$r$": []}
    for n in ns_all:
        data[f"$n={n}$"] = []

    data["Prediction"] = []

    cfgs = setup_configs()

    for ell, r, dim_fun in cfgs:  # loop over rows
        print("---------------------")
        data[r"$\ell$"].append(ell)
        data["$r$"].append(r)

        # Predict ratio
        dim = dim_fun(512)  # value cancels out so doesn't matter
        reduced = flops_reduced(n=dim.x, ell=dim.y_sing, r=dim.y_nonsing)
        unreduced = flops_unreduced(n=dim.x, ell=dim.y_sing, r=dim.y_nonsing)
        predicted = reduced / unreduced
        data["Prediction"].append(predicted)

        for n in ns_all:
            dim = dim_fun(n)
            print(dim)
            key, subkey = jax.random.split(key, num=2)
            tmp = test_util.model_random_time_varying(
                subkey, dim=dim, impl=impl, num_data=num_data
            )
            (z, x_mid_z, y_mid_x), F, data_out = tmp

            # Assemble all Kalman filters
            unreduced = filter_unreduced(
                impl=impl, z=z, x_mid_z=x_mid_z, y_mid_x=y_mid_x
            )
            reduced = filter_reduced(
                impl=impl, z=z, x_mid_z=x_mid_z, y_mid_x=y_mid_x, F_rank=F.shape[-1]
            )

            # Benchmark both filters
            results = {}
            for alg, name in [(unreduced, "unreduced"), (reduced, "reduced")]:
                ts = benchmark_filter(data_out, alg=alg, num_runs=num_runs)

                # Select the summary statistic from the runtimes (eg, fastest run)
                results[name] = float(jnp.amin(ts))
                print(f"\t{name}: \t {jnp.amin(ts):.1e} sec")

            # Save the ratio of runtimes
            ratio = results["reduced"] / results["unreduced"]
            data[f"$n={n}$"].append(ratio)
            print(f"\tRatio: \t\t {ratio:.4f}")
            print(f"\tPredicted: \t {predicted:.4f}")

    path = pathlib.Path(__file__).parent.resolve()
    with open(f"{path}/data_runtimes.pkl", "wb") as f:
        pickle.dump(data, f)


def setup_configs():
    def dim_n2_0(s):
        _assert_n(s)
        return test_util.DimCfg(x=s, y_sing=s // 2, y_nonsing=0)

    def dim_n4_0(s):
        _assert_n(s)
        return test_util.DimCfg(x=s, y_sing=s // 4, y_nonsing=0)

    def dim_n4_n4(s):
        _assert_n(s)
        return test_util.DimCfg(x=s, y_sing=s // 4, y_nonsing=s // 4)

    def dim_n8_n8(s):
        _assert_n(s)
        return test_util.DimCfg(x=s, y_sing=s // 8, y_nonsing=s // 8)

    def _assert_n(s):
        pass

    return [
        # (ell, r, ...)
        ("$n/2$", "$0$", dim_n2_0),
        ("$n/4$", "$0$", dim_n4_0),
        ("$n/4$", "$n/4$", dim_n4_n4),
        ("$n/8$", "$n/8$", dim_n8_n8),
    ]


def benchmark_filter(data_out, alg, num_runs):
    # Clear caches: https://github.com/jax-ml/jax/issues/10828
    jax.clear_caches()

    # Compile
    alg = jax.jit(alg)

    data_out.block_until_ready()

    # Execute once to compile
    x1, pdf = alg(data_out)
    x1.mean.block_until_ready()
    x1.cholesky.block_until_ready()
    pdf.block_until_ready()

    # Execute repeatedly, measuring all runtimes
    ts = []
    for _ in range(num_runs):
        t0 = time.perf_counter()

        x1, pdf = alg(data_out)
        x1.mean.block_until_ready()
        x1.cholesky.block_until_ready()
        pdf.block_until_ready()

        t1 = time.perf_counter()
        ts.append(t1 - t0)

    # Return a stack of runtimes
    return jnp.asarray(ts)


def flops_reduced(*, ell: int, n: int, r: int) -> float:
    qr = n**3 + (n - ell) ** 3 + (n - ell + r) ** 3
    bwd_subst = (n - ell) * ell**2 + r**3
    return qr + bwd_subst


def flops_unreduced(*, ell: int, r: int, n: int) -> float:
    m = ell + r
    qr = n**3 + (n + m) ** 3
    bwd_subst = n * m**2
    return qr + bwd_subst


def filter_unreduced(impl, z, x_mid_z, y_mid_x) -> Callable:
    kalman = ckf.kalman_filter(impl=impl)
    y_mid_x0 = jax.tree.map(lambda s: s[0], y_mid_x)
    y_mid_x1 = jax.tree.map(lambda s: s[1:], y_mid_x)
    del y_mid_x

    def apply(data_out):
        x0_ref, logpdf_ref = kalman.init(data_out[0], x=z, y_mid_x=y_mid_x0)

        scan_over = data_out[1:], x_mid_z, y_mid_x1
        x1_ref, logpdfs_ref = jax.lax.scan(step, xs=scan_over, init=x0_ref)
        return x1_ref, jnp.sum(logpdfs_ref) + logpdf_ref

    def step(x, scan_over):
        data, x_mid_z_, y_mid_x_ = scan_over
        x, logpdf = kalman.step(data, z=x, x_mid_z=x_mid_z_, y_mid_x=y_mid_x_)
        return x, logpdf

    return apply


def filter_reduced(impl, z, x_mid_z, y_mid_x, F_rank) -> Callable:
    kalman = ckf.kalman_filter(impl=impl)

    y_mid_x0 = jax.tree.map(lambda s: s[0], y_mid_x)
    y_mid_x1 = jax.tree.map(lambda s: s[1:], y_mid_x)
    del y_mid_x

    # Prepare the reduction
    reduction = ckf.model_reduction(F_rank=F_rank, impl=impl)
    prepared_init = reduction.prepare_init(y_mid_x=y_mid_x0, x=z)
    prepared = jax.vmap(reduction.prepare)(y_mid_x=y_mid_x1, x_mid_z=x_mid_z)

    def apply(data_out):
        reduced_init = reduction.reduce_init(data_out[0], prepared=prepared_init)
        y1, (x2, y1_mid_x2), (x_mid_x2, pdf2) = reduced_init

        x2, pdf1 = kalman.init(y1, x=x2, y_mid_x=y1_mid_x2)
        logpdf = pdf1 + pdf2

        # Kalman filter iteration
        scan_over = (data_out[1:], prepared)
        (x2, _), logpdfs = jax.lax.scan(step, xs=scan_over, init=(x2, x_mid_x2))
        return x2, jnp.sum(logpdfs) + logpdf

    def step(x2_and_cond, data_and_model):
        x2, x_mid_x2 = x2_and_cond
        data_i, prepared_i = data_and_model
        reduced = reduction.reduce(
            data_i, hidden=x2, z_mid_hidden=x_mid_x2, prepared=prepared_i
        )
        y1, (z, x2_mid_z, y1_mid_x2), (x_mid_x2, pdf2) = reduced

        x2, pdf1 = kalman.step(y1, z=z, x_mid_z=x2_mid_z, y_mid_x=y1_mid_x2)
        logpdf = pdf1 + pdf2
        return (x2, x_mid_x2), logpdf

    return apply


if __name__ == "__main__":
    main()
