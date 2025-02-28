import pathlib
import pickle

import jax
import jax.numpy as jnp

from ckf import ckf, test_util

jax.config.update("jax_enable_x64", True)


def main(num_data=500):
    solve_lu, solve_chol = jnp.linalg.solve, ckf.solve_fun_cholesky()
    conventional_lu = ckf.impl_cov_based(solve_fun=solve_lu)
    conventional_chol = ckf.impl_cov_based(solve_fun=solve_chol)
    ours = ckf.impl_cholesky_based()
    impls = [
        ("Ours", ours),
        ("Conventional (LU)", conventional_lu),
        ("Conventional (Chol.)", conventional_chol),
    ]

    results = {"$n$": [], r"$\ell$": []}
    for name, impl_test in impls:
        results[name] = []

    for n in range(3, 12, 1):
        dim = test_util.DimCfg(x=n, y_sing=n // 2, y_nonsing=0)
        results["$n$"].append(n)
        results[r"$\ell$"].append(dim.y_sing)
        print()
        for name, impl_test in impls:
            # Generate data
            key = jax.random.PRNGKey(seed=1)
            impl_data = ckf.impl_cholesky_based()
            (z, x_mid_z, y_mid_x), _ = test_util.model_hilbert(dim=dim, impl=impl_data)
            sample = ckf.ssm_sample_time_invariant(impl=impl_data, num_data=num_data)
            data_out = sample(key, z, x_mid_z, y_mid_x)

            # Reference:
            impl_ref = ckf.impl_cholesky_based()
            (z, x_mid_z, y_mid_x), _ = test_util.model_hilbert(dim=dim, impl=impl_ref)
            unreduced = smoother_fixpt_unreduced(z, x_mid_z, y_mid_x, impl=impl_ref)
            ref = unreduced(data_out)

            # Compute test solver
            (z, x_mid_z, y_mid_x), _ = test_util.model_hilbert(dim=dim, impl=impl_test)
            reduced = smoother_reduced(z, x_mid_z, y_mid_x, impl=impl_test, F_rank=0)
            x0 = jax.jit(reduced)(data_out)

            # Evaluate error
            mae_mean = jnp.abs(x0.mean - ref.mean).mean()
            mae_cov = jnp.abs(x0.cov_dense() - ref.cov_dense()).mean()
            mae = mae_mean + mae_cov
            print(f"n = {n} \tm = {dim.y_sing} \t{name}: \t{jnp.log10(mae):.1f}")
            results[name].append(float(jnp.log10(mae)))
    print(results)

    path = pathlib.Path(__file__).parent.resolve()
    with open(f"{path}/data_errors.pkl", "wb") as f:
        pickle.dump(results, f)


def smoother_fixpt_unreduced(z, x_mid_z, y_mid_x, *, impl):
    kalman = ckf.rts_smoother(impl=impl)

    def smooth(data_out):
        d0, ds = data_out[0], data_out[1:]
        x0, _ = kalman.init(d0, x=z, y_mid_x=y_mid_x)

        init = (x0, _identity_like(x0))
        (x1, bwd_density), _ = jax.lax.scan(step_fwd, xs=ds, init=init)

        return impl.rv_marginal(x1, bwd_density)

    def _identity_like(x):
        (n,) = x.mean.shape
        eye = jnp.eye(n)
        zero_noise = impl.rv_from_cholesky(jnp.zeros((n,)), jnp.zeros((n, n)))
        return ckf.AffineCond(eye, zero_noise)

    def step_fwd(x_and_bwd, data):
        x, bwd_previous = x_and_bwd
        x, logpdf, bwd = kalman.step(data, z=x, x_mid_z=x_mid_z, y_mid_x=y_mid_x)
        bwd_new = impl.cond_combine(outer=bwd_previous, inner=bwd)
        return (x, bwd_new), None

    return smooth


def smoother_reduced(z, x_mid_z, y_mid_x, *, impl, F_rank):
    # Reduce the model (before seeing data)
    kalman = ckf.rts_smoother(impl=impl)
    reduction = ckf.model_reduction(F_rank=F_rank, impl=impl)
    prepared_init = reduction.prepare_init(y_mid_x=y_mid_x, x=z)
    prepared = reduction.prepare(y_mid_x=y_mid_x, x_mid_z=x_mid_z)

    def smooth(data_out):
        # Reduce and initialise
        reduced_init = reduction.reduce_init(data_out[0], prepared=prepared_init)
        y1, (x2, y1_mid_x2), (x_mid_x2_init, _) = reduced_init
        x2, _ = kalman.init(y1, x=x2, y_mid_x=y1_mid_x2)

        # Kalman filter iteration
        init = (x2, x_mid_x2_init)
        (x2, _), bwd_densities = jax.lax.scan(step_fwd, xs=data_out[1:], init=init)

        x_smoothed, _ = jax.lax.scan(step_bwd, xs=bwd_densities, init=x2, reverse=True)
        return impl.rv_marginal(x_smoothed, x_mid_x2_init)

    def step_fwd(x2_and_cond, data):
        x2, x_mid_x2 = x2_and_cond

        # Continue reducing
        tmp = reduction.reduce(
            data, hidden=x2, z_mid_hidden=x_mid_x2, prepared=prepared
        )
        y1, (z_red, x2_mid_z_red, y1_mid_x2), (x_mid_x2, _) = tmp

        # Run a single filter-condition step.
        x2, _, bwd_dens = kalman.step(
            y1, z=z_red, x_mid_z=x2_mid_z_red, y_mid_x=y1_mid_x2
        )

        return (x2, x_mid_x2), bwd_dens

    def step_bwd(x1, backward_density):
        x0 = impl.rv_marginal(x1, backward_density)
        return x0, None

    return smooth


if __name__ == "__main__":
    main()
