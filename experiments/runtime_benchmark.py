import jax.numpy as jnp
from ckf import ckf, test_util
import jax

import time


def main(seed=1, num_data=50):
    cfgs = []
    for d in [32]:
        cfgs.append((d, d // 2, 0))
        cfgs.append((d, d // 2, d // 4))
        cfgs.append((d, d // 2, d // 2))

    for x, y_sing, y_nonsing in cfgs:
        dim = test_util.DimCfg(x=x, y_sing=y_sing, y_nonsing=y_nonsing)
        print()
        print(dim)
        for impl, impl_name in [
            (ckf.impl_cov_based(), "Cov-based"),
            (ckf.impl_cholesky_based(), "Cholesky-based"),
        ]:
            print()
            print("\t", impl_name)
            print("\t =================")

            key = jax.random.PRNGKey(seed)
            key, subkey = jax.random.split(key, num=2)
            (z, x_mid_z, y_mid_x), F, _y = test_util.model_random(
                subkey, dim=dim, impl=impl
            )

            # Assemble all Kalman filters

            filter_conv = _filter_conventional(
                impl=impl, z=z, x_mid_z=x_mid_z, y_mid_x=y_mid_x
            )
            filter_redu = _filter_reduced(
                impl=impl, z=z, x_mid_z=x_mid_z, y_mid_x=y_mid_x, F_rank=F.shape[1]
            )

            for alg, name in [(filter_conv, "Conventional"), (filter_redu, "Reduced")]:
                alg = jax.jit(alg)

                key, subkey = jax.random.split(key, num=2)
                data_out = jax.random.normal(
                    subkey, shape=(num_data, dim.y_sing + dim.y_nonsing)
                )

                x0, xs, pdf = alg(data_out)
                x0.mean.block_until_ready()
                xs.mean.block_until_ready()
                if impl_name == "Cov-based":
                    x0.cov.block_until_ready()
                    xs.cov.block_until_ready()
                else:
                    x0.cholesky.block_until_ready()
                    xs.cholesky.block_until_ready()
                ts = []
                t0_total = time.perf_counter()
                while time.perf_counter() - t0_total < 1:
                    key, subkey = jax.random.split(key, num=2)
                    data_out = jax.random.normal(
                        subkey, shape=(num_data, dim.y_sing + dim.y_nonsing)
                    )

                    t0 = time.perf_counter()
                    x0, xs, pdf = alg(data_out)
                    x0.mean.block_until_ready()
                    xs.mean.block_until_ready()
                    if impl_name == "Cov-based":
                        x0.cov.block_until_ready()
                        xs.cov.block_until_ready()
                    else:
                        x0.cholesky.block_until_ready()
                        xs.cholesky.block_until_ready()

                    t1 = time.perf_counter()
                    ts.append(t1 - t0)
                ts = jnp.asarray(ts)
                print(f"\t {name} \t {jnp.mean(ts):.5f} \t (lower is better)")


def _filter_conventional(impl, z, x_mid_z, y_mid_x):
    kalman = ckf.rts_smoother(impl=impl)

    @jax.jit
    def apply(data_out):
        # Initialise
        x0_ref, logpdf_ref = kalman.init(data_out[0], x=z, y_mid_x=y_mid_x)

        # Kalman filter iteration
        def step(x, data):
            x, logpdf, smooth = kalman.step(data, z=x, x_mid_z=x_mid_z, y_mid_x=y_mid_x)
            return x, (x, logpdf, smooth)

        x1_ref, (xs_ref, logpdfs_ref, smoothing) = jax.lax.scan(step, xs=data_out[1:], init=x0_ref)

        def smoothing_step(x, bwd):
            x = impl.rv_marginal(x, bwd)
            return x, x

        _, xs_ref = jax.lax.scan(smoothing_step, xs=smoothing, init=x1_ref, reverse=True)

        return x1_ref, xs_ref, jnp.sum(logpdfs_ref) + logpdf_ref

    return apply


def _filter_reduced(impl, z, x_mid_z, y_mid_x, F_rank):
    kalman = ckf.rts_smoother(impl=impl)

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
            save = x_mid_x2_
            y1_, (z_, x2_mid_z_, y1_mid_x2_), (x_mid_x2_, pdf2_) = reduction.reduce(
                data, hidden=x2_, z_mid_hidden=x_mid_x2_, prepared=prepared
            )

            # Run a single filter-condition step.
            x2_, pdf1_, smooth = kalman.step(y1_, z=z_, x_mid_z=x2_mid_z_, y_mid_x=y1_mid_x2_)

            # Save some quantities (not part of the simulation, just for testing)
            logpdf_ = pdf1_ + pdf2_
            return (x2_, x_mid_x2_), (x2_, logpdf_, smooth,save)

        (x2, _), (xs, logpdfs, smoothing, x2_mids) = jax.lax.scan(step, xs=data_out[1:], init=(x2, x_mid_x2))

        def smoothing_step(x2_, smooth_and_cond):
            smooth, x_mid_x2_ = smooth_and_cond
            x2_ = impl.rv_marginal(x2_, smooth)
            return x2_, x2_

        _, xs = jax.lax.scan(
            smoothing_step, xs=(smoothing, x2_mids), init=x2, reverse=True
        )

        return x2, xs, jnp.sum(logpdfs) + logpdf

    return apply


if __name__ == "__main__":
    main()
