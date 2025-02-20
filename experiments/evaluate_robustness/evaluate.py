import jax.numpy as jnp


import jax
import jax.numpy as jnp
import pytest_cases
import pathlib

from ckf import ckf, test_util
import pickle

import tqdm

jax.config.update("jax_enable_x64", True)


def main(num_data=500):
    results = {}
    for name, impl_test in [
        ("Cholesky-based", ckf.impl_cholesky_based()),
        ("Cov-based (LU-solve)", ckf.impl_cov_based(solve_fun=jnp.linalg.solve)),
        (
            "Cov-based (Cholesky-solve)",
            ckf.impl_cov_based(solve_fun=ckf.solve_fun_cholesky()),
        ),
    ]:
        print()
        results[name] = {}
        for n in range(1, 12, 1):
            dim = test_util.DimCfg(x=n, y_sing=n // 2, y_nonsing=0)

            key = jax.random.PRNGKey(seed=1)
            data_out = jax.random.normal(key, shape=(num_data, dim.y_sing))

            # Reference:
            impl_ref = ckf.impl_cholesky_based()
            (z, x_mid_z, y_mid_x) = test_util.model_ivpsolve(dim=dim, impl=impl_ref)
            unreduced = smoother_unreduced(z, x_mid_z, y_mid_x, impl=impl_ref)
            ref = unreduced(data_out)

            # Compute test solver
            (z, x_mid_z, y_mid_x) = test_util.model_ivpsolve(dim=dim, impl=impl_test)
            reduced = smoother_reduced(z, x_mid_z, y_mid_x, impl=impl_test, F_rank=0)
            x0 = jax.jit(reduced)(data_out)
            mae_mean = jnp.abs(x0.mean - ref.mean).mean()
            mae_cov = jnp.abs(x0.cov_dense() - ref.cov_dense()).mean()
            mae = mae_mean + mae_cov
            print(f"n = {n} \tm = {dim.y_sing} \t{name}: \t{jnp.log10(mae):.1f}")
            results[name][f"$n={n}$, $m={dim.y_sing}$"] = float(jnp.log10(mae))

    path = pathlib.Path(__file__).parent.resolve()
    with open(f"{path}/data_errors.pkl", "wb") as f:
        pickle.dump(results, f)


def smoother_unreduced(z, x_mid_z, y_mid_x, *, impl):
    def smooth(data_out):
        kalman = ckf.rts_smoother(impl=impl)

        # Kalman smoother iteration

        def step(x, data):
            x, logpdf, bwd = kalman.step(data, z=x, x_mid_z=x_mid_z, y_mid_x=y_mid_x)
            return x, (logpdf, bwd)

        x0_ref, logpdf_ref = kalman.init(data_out[0], x=z, y_mid_x=y_mid_x)
        xx_ref, (logpdfs_ref, smoothing) = jax.lax.scan(
            step, xs=data_out[1:], init=x0_ref
        )

        def smoothing_step(x, bwd):
            x = impl.rv_marginal(x, bwd)
            return x, x

        x_, _ = jax.lax.scan(smoothing_step, xs=smoothing, init=xx_ref, reverse=True)
        return x_

    return smooth


def smoother_reduced(z, x_mid_z, y_mid_x, *, impl, F_rank):
    def smooth(data_out):
        kalman = ckf.rts_smoother(impl=impl)

        # Reduce the model
        reduction = ckf.model_reduction(F_rank=F_rank, impl=impl)
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

        def smoothing_step(x2_, smooth_and_cond):
            smooth, x_mid_x2_ = smooth_and_cond
            x2_, _ = x2_
            x2_ = impl.rv_marginal(x2_, smooth)
            xx_ = impl.rv_marginal(x2_, x_mid_x2_)  # mainly for testing
            return (x2_, xx_), None

        xx = impl.rv_marginal(x2, x_mid_x2)
        (_, xx), _ = jax.lax.scan(
            smoothing_step, xs=(smoothing, x_mid_x2_all), init=(x2, xx), reverse=True
        )
        return xx

    return smooth


if __name__ == "__main__":
    main()
