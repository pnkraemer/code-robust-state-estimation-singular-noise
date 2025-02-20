"""Constrained Kalman filtering (-> "ckf") and Rauch--Tung-Striebel smoothing."""

import jax.numpy as jnp
import jax
import dataclasses

from typing import Callable, Generic, TypeVar


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class CovNormal:
    mean: jax.Array
    cov: jax.Array

    def __rmatmul__(self, other: jax.Array):
        """Implement Array @ AffineCond"""
        mean = other @ self.mean
        cov = other @ self.cov @ other.T
        return CovNormal(mean, cov)

    def __add__(self, other: jax.Array):
        return CovNormal(self.mean + other, self.cov)

    def cov_dense(self):
        return self.cov


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class CholeskyNormal:
    mean: jax.Array
    cholesky: jax.Array

    def __rmatmul__(self, other: jax.Array):
        """Implement Array @ AffineCond"""
        mean = other @ self.mean
        cholesky = other @ self.cholesky
        return CholeskyNormal(mean, cholesky)

    def __add__(self, other: jax.Array):
        return CholeskyNormal(self.mean + other, self.cholesky)

    def cov_dense(self):
        return CholeskyNormal._cov_dense_static(self.cholesky)

    @staticmethod
    def _cov_dense_static(chol):
        if chol.ndim > 2:
            return jax.vmap(CholeskyNormal._cov_dense_static)(chol)
        return chol @ chol.T


T = TypeVar("T", bound=(CholeskyNormal | CovNormal))


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class AffineCond(Generic[T]):
    """Affine conditional distribution."""

    linop: jax.Array
    noise: T

    def __matmul__(self, other: jax.Array):
        """Implement AffineCond @ Array"""
        return AffineCond(linop=self.linop @ other, noise=self.noise)

    def __rmatmul__(self, other: jax.Array):
        """Implement Array @ AffineCond"""
        return AffineCond(linop=other @ self.linop, noise=other @ self.noise)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SplitAffineCond(Generic[T]):
    """Affine conditional with a linear operator that's split in two parts."""

    linop1: jax.Array
    linop2: jax.Array
    noise: T


@dataclasses.dataclass
class Impl(Generic[T]):
    rv_from_cholesky: Callable[[jax.Array, jax.Array], T]
    rv_condition: Callable[[T, AffineCond[T]], tuple[T, AffineCond[T]]]
    rv_marginal: Callable[[T, AffineCond[T]], T]
    rv_logpdf: Callable[[jax.Array, T], jax.Array]

    rv_factorise: Callable[[T, int], tuple[T, AffineCond[T]]]
    cond_evaluate: Callable[[jax.Array, AffineCond[T]], T]

    get_F: Callable

    def split_cond_fix_x1(self, x1, split_cond):
        cond = AffineCond(split_cond.linop1, split_cond.noise)
        noise = self.cond_evaluate(x1, cond=cond)

        linop = split_cond.linop2
        return AffineCond(linop, noise)

    def cond_combine(self, outer, inner):
        linop = outer.linop @ inner.linop
        noise = self.rv_marginal(inner.noise, outer)
        return AffineCond(linop, noise)

    def cond_combine_outer_det(self, outer, inner):
        linop = outer.linop @ inner.linop

        noise = outer.linop @ inner.noise + outer.noise.mean
        return AffineCond(linop, noise)

    def cond_combine_inner_det(self, outer, inner):
        linop = outer.linop @ inner.linop
        noise = self.cond_evaluate(inner.noise.mean, outer)
        return AffineCond(linop, noise)

    def cond_factorise(
        self, cond: AffineCond, index: int
    ) -> tuple[CovNormal, AffineCond]:
        x1, x2_mid_x1 = self.rv_factorise(cond.noise, index=index)

        linop1, linop2 = jnp.split(cond.linop, indices_or_sections=[index], axis=0)
        x1_mid_z = AffineCond(linop1, x1)
        linop_z = linop2 - x2_mid_x1.linop @ linop1
        x2_mid_x1_and_z = SplitAffineCond(x2_mid_x1.linop, linop_z, x2_mid_x1.noise)
        return x1_mid_z, x2_mid_x1_and_z


def impl_cholesky_based() -> Impl[CholeskyNormal]:
    def rv_from_cholesky(m, c):
        return CholeskyNormal(mean=m, cholesky=c)

    def rv_condition(rv, cond):
        # Make the observation Cholesky factor square because otherwise
        # the rank of the conditional Cholesky factor varies with iteration counts.
        # This would be fine on paper, but does not go with JAX's JIT'ing mechanics.
        L_YX = jnp.zeros((cond.noise.cholesky.shape[0], cond.noise.cholesky.shape[0]))
        L_YX = L_YX.at[:, : cond.noise.cholesky.shape[1]].set(cond.noise.cholesky)
        R_YX = L_YX.T

        R_X = rv.cholesky.T
        R_X_F = rv.cholesky.T @ cond.linop.T
        R_y, (R_xy, G) = _revert_conditional(R_X_F=R_X_F, R_X=R_X, R_YX=R_YX)

        s = cond.linop @ rv.mean + cond.noise.mean
        mean_new = rv.mean - G @ s

        marg = CholeskyNormal(s, R_y.T)
        cond = AffineCond(G, CholeskyNormal(mean_new, R_xy.T))
        return marg, cond

    def _revert_conditional(R_X_F: jax.Array, R_X: jax.Array, R_YX: jax.Array):
        # Taken from:
        # https://github.com/pnkraemer/probdiffeq/blob/main/probdiffeq/util/cholesky_util.py

        R = jnp.block([[R_YX, jnp.zeros((R_YX.shape[0], R_X.shape[1]))], [R_X_F, R_X]])
        # print(f"condition-QR of {R.shape}")
        R = jnp.linalg.qr(R, mode="r")

        # ~R_{Y}
        d_out = R_YX.shape[1]
        R_Y = R[:d_out, :d_out]

        # something like the cross-covariance
        R12 = R[:d_out, d_out:]

        # Implements G = R12.T @ np.linalg.inv(R_Y.T) in clever:
        # print(f"condition-Bwd.-Subst. of {R_Y.shape}")
        G = jax.scipy.linalg.solve_triangular(R_Y, R12, lower=False).T

        # ~R_{X \mid Y}
        R_XY = R[d_out:, d_out:]
        return R_Y, (R_XY, G)

    def rv_marginal(rv, cond):
        mean = cond.linop @ rv.mean + cond.noise.mean
        mtrx = jnp.concatenate([rv.cholesky.T @ cond.linop.T, cond.noise.cholesky.T])
        # print(f"marginal-QR of {mtrx.shape}")
        R = jnp.linalg.qr(mtrx, mode="r")
        return CholeskyNormal(mean, R.T)

    def rv_factorise(
        rv: CholeskyNormal, index: int
    ) -> tuple[CholeskyNormal, AffineCond[CholeskyNormal]]:
        # print(f"factorise-QR of {rv.cholesky.T.shape}")
        R = jnp.linalg.qr(rv.cholesky.T, mode="r")
        R1 = R[:index, :index]
        R12 = R[:index, index:]
        R2 = R[index:, index:]
        # print(f"factorise-Bwd.-Subst. of {R1.shape}")
        G = jax.scipy.linalg.solve_triangular(R1, R12, lower=False).T

        bias1, bias2 = jnp.split(rv.mean, indices_or_sections=[index], axis=0)

        x1 = CholeskyNormal(bias1, R1.T)
        noise = CholeskyNormal(bias2 - G @ bias1, R2.T)
        x1_mid_x2 = AffineCond(G, noise)
        return x1, x1_mid_x2

    def cond_evaluate(x, /, cond):
        return CholeskyNormal(cond.linop @ x + cond.noise.mean, cond.noise.cholesky)

    def get_F(cond, F_rank):
        del F_rank
        return cond.noise.cholesky

    def rv_logpdf(u, /, rv):
        # Ensure that the Cholesky factor is triangular
        # (it should be, but there is no quarantee).
        # print(f"logpdf-QR of {rv.cholesky.T.shape}")
        cholesky = jnp.linalg.qr(rv.cholesky.T, mode="r").T
        diagonal = jnp.diagonal(cholesky, axis1=-1, axis2=-2)
        slogdet = jnp.sum(jnp.log(jnp.abs(diagonal)))

        dx = u - rv.mean
        # print(f"logpdf-Bwd.-Subst. of {cholesky.T.shape}")
        residual_white = jax.scipy.linalg.solve_triangular(cholesky.T, dx, trans="T")
        x1 = jnp.dot(residual_white, residual_white)
        x2 = 2.0 * slogdet
        x3 = u.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    return Impl(
        rv_from_cholesky=rv_from_cholesky,
        rv_condition=rv_condition,
        rv_marginal=rv_marginal,
        rv_factorise=rv_factorise,
        cond_evaluate=cond_evaluate,
        get_F=get_F,
        rv_logpdf=rv_logpdf,
    )


def cholesky_solve():
    def solve(A, b):
        cho_factor = jax.scipy.linalg.cho_factor(A)
        return jax.scipy.linalg.cho_solve(cho_factor, b)

    return solve


def impl_cov_based(solve_fun=cholesky_solve()) -> Impl[CovNormal]:
    def rv_from_cholesky(m, c):
        return CovNormal(m, c @ c.T)

    def rv_factorise(rv, index: int) -> tuple[AffineCond, SplitAffineCond]:
        # Factorise ignoring the linops
        cov1, cov2 = jnp.split(rv.cov, indices_or_sections=[index], axis=0)
        C1, C21 = jnp.split(cov1, indices_or_sections=[index], axis=1)
        C12, C2 = jnp.split(cov2, indices_or_sections=[index], axis=1)

        G = solve_fun(C1.T, C12.T).T
        Z = C2 - G @ C1 @ G.T

        bias1, bias2 = jnp.split(rv.mean, indices_or_sections=[index], axis=0)

        x1 = CovNormal(bias1, C1)
        noise2 = CovNormal(bias2 - G @ bias1, Z)
        x2_mid_x1 = AffineCond(G, noise2)
        return x1, x2_mid_x1

    def cond_evaluate(data, cond: AffineCond[CovNormal]) -> CovNormal:
        mean = cond.linop @ data + cond.noise.mean
        return CovNormal(mean=mean, cov=cond.noise.cov)

    def rv_condition(rv: CovNormal, cond: AffineCond):
        z = cond.linop @ rv.mean + cond.noise.mean
        S = cond.linop @ rv.cov @ cond.linop.T + cond.noise.cov
        marg = CovNormal(z, S)

        K = solve_fun(S.T, cond.linop @ rv.cov).T
        m = rv.mean - K @ z
        C = rv.cov - K @ cond.linop @ rv.cov
        cond = AffineCond(linop=K, noise=CovNormal(mean=m, cov=C))
        return marg, cond

    def rv_marginal(rv: CovNormal, cond: AffineCond) -> CovNormal:
        m = cond.linop @ rv.mean + cond.noise.mean
        C = cond.linop @ rv.cov @ cond.linop.T + cond.noise.cov
        return CovNormal(m, C)

    def get_F(cond: AffineCond, F_rank):
        eigh = jnp.linalg.eigh(cond.noise.cov)

        i = jnp.flip(jnp.argsort(eigh.eigenvalues))[:F_rank]
        vals = eigh.eigenvalues[i]
        vecs = eigh.eigenvectors[:, i]
        return vecs @ jnp.diag(jnp.sqrt(vals))

    def rv_logpdf(u, rv):
        cholesky = jnp.linalg.cholesky(rv.cov)

        diagonal = jnp.diagonal(cholesky, axis1=-1, axis2=-2)
        slogdet = jnp.sum(jnp.log(jnp.abs(diagonal)))

        dx = u - rv.mean
        residual_white = jax.scipy.linalg.solve_triangular(cholesky.T, dx, trans="T")
        x1 = jnp.dot(residual_white, residual_white)
        x2 = 2.0 * slogdet
        x3 = u.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)
        # logpdf = jax.scipy.stats.multivariate_normal.logpdf
        # return logpdf(u, rv.mean, rv.cov)

    return Impl(
        rv_from_cholesky=rv_from_cholesky,
        rv_condition=rv_condition,
        rv_marginal=rv_marginal,
        rv_factorise=rv_factorise,
        cond_evaluate=cond_evaluate,
        get_F=get_F,
        rv_logpdf=rv_logpdf,
    )


def cond_split(cond: AffineCond, index: int) -> SplitAffineCond:
    linop1, linop2 = jnp.split(cond.linop, indices_or_sections=[index], axis=1)
    return SplitAffineCond(linop1, linop2, cond.noise)


def cond_invert_dirac(y: jax.Array, /, dirac_cond) -> jax.Array:
    A = dirac_cond.linop
    b = y - dirac_cond.noise.mean
    # print(f"invert_dirac-Bwd.-Subst. of {A.shape}")
    # A is lower-triangular because it is A=S.T and S comes from a QR decomp.
    return jax.scipy.linalg.solve_triangular(A, b, lower=True)


@dataclasses.dataclass
class ModelReduction:
    prepare_init: Callable
    reduce_init: Callable
    prepare: Callable
    reduce: Callable


def model_reduction(F_rank, impl) -> ModelReduction:
    # todo: reduce duplication between the two prepare's and reduce's
    #  because currently, they're almost identical
    # todo: remove rmatmul from trafo, because I think there is a bunch of unnecessary computation?
    def prepare_init(y_mid_x: AffineCond, x: T):
        """Like prepare(), but for Phi=0 which simplifies some of the calls."""
        # Extract F
        F = impl.get_F(y_mid_x, F_rank=F_rank)

        # First QR iteration
        # print(f"prepare-init-QR of {F.shape}")
        V, R = jnp.linalg.qr(F, mode="complete")
        V1, V2 = jnp.split(V, indices_or_sections=[F_rank], axis=1)
        y1_mid_x = V1.T @ y_mid_x
        y2_mid_x = V2.T @ y_mid_x

        # Second QR iteration
        # print(f"prepare-init-QR of {y2_mid_x.linop.T.shape}")
        W, S = jnp.linalg.qr(y2_mid_x.linop.T, mode="complete")
        W1, W2 = jnp.split(W, indices_or_sections=[len(V2.T)], axis=1)

        # Factorise the z-to-x conditional
        x1_and_x2 = W.T @ x
        x1, x2_mid_x1 = impl.rv_factorise(rv=x1_and_x2, index=len(V2.T))

        # y2 | x1 is deterministic (ie zero cov) and y2 is independent of x2 given x1
        y2_mid_x1 = y2_mid_x @ W1

        # y1 now depends on both x1 and x2; we implement this as a split' conditional,
        #  which is a conditional with two linops
        y1_mid_x1_and_x2 = cond_split(cond=(y1_mid_x @ W), index=len(V2.T))

        # We need to memorise how to turn x1/x2 back into x
        zero_noise = impl.rv_from_cholesky(
            jnp.zeros((len(W),)), jnp.zeros((len(W), len(W)))
        )
        cond = AffineCond(W, zero_noise)
        x_mid_x1_and_x2 = cond_split(cond=cond, index=len(V2.T))

        # We only care about y2 | z, not about x1 | z, so we combine transformations
        y2 = y2_mid_x1.linop @ x1 + y2_mid_x1.noise.mean

        # Return values:
        reduced_model = (y2, x2_mid_x1, y1_mid_x1_and_x2)
        info_transform_back = x_mid_x1_and_x2
        info_identify_constraint = y2_mid_x1
        info_split_data = (V1, V2)
        info = (info_transform_back, info_identify_constraint, info_split_data)
        return reduced_model, info

    def reduce_init(y: jax.Array, prepared):
        """Like prepare(), but for Phi=0 which simplifies some of the calls."""
        # Read off prepared values
        prepared_models, info = prepared
        (info_transform_back, info_identify_constraint, info_split_data) = info
        (y2_marg, x2_mid_x1, y1_mid_x1_and_x2) = prepared_models
        x_mid_x1_and_x2 = info_transform_back
        y2_mid_x1 = info_identify_constraint
        (V1, V2) = info_split_data

        # Split the data data
        y1, y2 = V1.T @ y, V2.T @ y

        # Fix y2 (via x1) in remaining conditionals.
        #  Recall that by construction of the QR decompositions,
        #  y2_mid_x1 has zero covariance.
        x1_value = cond_invert_dirac(y2, dirac_cond=y2_mid_x1)
        x2 = impl.cond_evaluate(x1_value, cond=x2_mid_x1)
        y1_mid_x2 = impl.split_cond_fix_x1(x1_value, split_cond=y1_mid_x1_and_x2)
        x_mid_x2 = impl.split_cond_fix_x1(x1_value, split_cond=x_mid_x1_and_x2)

        # Fix y2 in the "prior" distribution
        logpdf_y2 = impl.rv_logpdf(y2, y2_marg)

        # Now we have z, x2_mid_z, and y1_mid_x2
        # which is a "complete model" (just smaller than the previous one)
        # and we can run the usual estimation (eg Kalman filter)
        return y1, (x2, y1_mid_x2), (x_mid_x2, logpdf_y2)

    def prepare(y_mid_x: AffineCond, x_mid_z: AffineCond):
        """Carry out as much reduction as possible without seeing data."""

        # Extract F
        F = impl.get_F(y_mid_x, F_rank=F_rank)

        # First QR iteration
        # todo: don't always do that?
        # print(f"prepare-QR of {F.shape}")
        V, R = jnp.linalg.qr(F, mode="complete")
        V1, V2 = jnp.split(V, indices_or_sections=[F_rank], axis=1)
        y1_mid_x = V1.T @ y_mid_x
        y2_mid_x = V2.T @ y_mid_x

        # Second QR iteration
        # todo: don't always do that?
        # print(f"prepare-QR of {y2_mid_x.linop.T.shape}")
        W, S = jnp.linalg.qr(y2_mid_x.linop.T, mode="complete")
        W1, W2 = jnp.split(W, indices_or_sections=[len(V2.T)], axis=1)

        # Factorise the z-to-x conditional
        x1_and_x2_mid_z = W.T @ x_mid_z
        x1_mid_z, x2_mid_x1_and_z = impl.cond_factorise(
            cond=x1_and_x2_mid_z, index=len(V2.T)
        )

        # y2 | x1 is deterministic (ie zero cov) and y2 is independent of x2 given x1
        y2_mid_x1 = y2_mid_x @ W1

        # y1 now depends on both x1 and x2; we implement this as a split' conditional,
        #  which is a conditional with two linops
        y1_mid_x1_and_x2 = cond_split(cond=(y1_mid_x @ W), index=len(V2.T))

        # We need to memorise how to turn x1/x2 back into x
        n = len(W)
        zero_noise = impl.rv_from_cholesky(jnp.zeros((n,)), jnp.zeros((n, n)))
        cond = AffineCond(W, zero_noise)
        x_mid_x1_and_x2 = cond_split(cond=cond, index=len(V2.T))

        # We only care about y2 | z, not about x1 | z, so we combine transformations
        y2_mid_z = impl.cond_combine_outer_det(outer=y2_mid_x1, inner=x1_mid_z)

        # Return values:
        reduced_model = (y2_mid_z, x2_mid_x1_and_z, y1_mid_x1_and_x2)
        info_transform_back = x_mid_x1_and_x2
        info_identify_constraint = y2_mid_x1
        info_split_data = (V1, V2)
        info = (info_transform_back, info_identify_constraint, info_split_data)
        return reduced_model, info

    def reduce_(y: jax.Array, hidden, z_mid_hidden, prepared):
        """Reduce the model to its 'minimal' version using data."""

        # Read off prepared values
        prepared_models, info = prepared
        (info_transform_back, info_identify_constraint, info_split_data) = info
        (y2_mid_z, x2_mid_x1_and_z, y1_mid_x1_and_x2) = prepared_models
        x_mid_x1_and_x2 = info_transform_back
        y2_mid_x1 = info_identify_constraint
        (V1, V2) = info_split_data

        # Split the data data
        y1, y2 = V1.T @ y, V2.T @ y

        # Fix y2 (via x1) in remaining conditionals.
        #  Recall that by construction of the QR decompositions,
        #  y2_mid_x1 has zero covariance.
        x1_value = cond_invert_dirac(y2, dirac_cond=y2_mid_x1)
        x2_mid_z = impl.split_cond_fix_x1(x1_value, split_cond=x2_mid_x1_and_z)
        y1_mid_x2 = impl.split_cond_fix_x1(x1_value, split_cond=y1_mid_x1_and_x2)
        x_mid_x2 = impl.split_cond_fix_x1(x1_value, split_cond=x_mid_x1_and_x2)

        # If "z" is not a distribution, but a conditional distribution
        #  (which is the case when stepping through a sequence of iterations)
        #  then we need to absorb this conditional into z before proceeding.
        #  why? because this way, all x2_mid_z and y2_mid_z are actually
        #  x2_mid_x2prev, y2_mid_x2prev, which means
        #  that the model remains in "small space"
        # todo: z_mid_hidden is deterministic so we can save compute
        x2_mid_z = impl.cond_combine_inner_det(outer=x2_mid_z, inner=z_mid_hidden)
        y2_mid_z = impl.cond_combine_inner_det(outer=y2_mid_z, inner=z_mid_hidden)
        z = hidden

        # Fix y2 in the "prior" distribution
        y2_marg, z_mid_y2 = impl.rv_condition(rv=z, cond=y2_mid_z)
        z = impl.cond_evaluate(y2, cond=z_mid_y2)
        logpdf_y2 = impl.rv_logpdf(y2, y2_marg)

        # Now we have z, x2_mid_z, and y1_mid_x2
        # which is a "complete model" (just smaller than the previous one)
        # and we can run the usual estimation (eg Kalman filter)
        return y1, (z, x2_mid_z, y1_mid_x2), (x_mid_x2, logpdf_y2)

    return ModelReduction(
        prepare_init=prepare_init,
        reduce_init=reduce_init,
        reduce=reduce_,
        prepare=prepare,
    )


@dataclasses.dataclass
class Estimator:
    init: Callable
    step: Callable


def kalman_filter(impl) -> Estimator:
    def init(data, x, y_mid_x):
        y_marg, bwd = impl.rv_condition(x, y_mid_x)
        logpdf_ref = impl.rv_logpdf(data, y_marg)
        x0_ref = impl.cond_evaluate(data, bwd)
        return x0_ref, logpdf_ref

    def step(data, z, x_mid_z, y_mid_x):
        x = impl.rv_marginal(rv=z, cond=x_mid_z)
        y_marg, bwd = impl.rv_condition(rv=x, cond=y_mid_x)
        x_cond = impl.cond_evaluate(data, cond=bwd)
        logpdf = impl.rv_logpdf(data, y_marg)
        return x_cond, logpdf

    return Estimator(init=init, step=step)


def rts_smoother(impl) -> Estimator:
    def init(data, x, y_mid_x):
        y_marg, bwd = impl.rv_condition(x, y_mid_x)
        logpdf_ref = impl.rv_logpdf(data, y_marg)
        x0_ref = impl.cond_evaluate(data, bwd)
        return x0_ref, logpdf_ref

    def step(data, z, x_mid_z, y_mid_x):
        x, smoothing = impl.rv_condition(rv=z, cond=x_mid_z)
        y_marg, bwd = impl.rv_condition(rv=x, cond=y_mid_x)
        x_cond = impl.cond_evaluate(data, cond=bwd)
        logpdf = impl.rv_logpdf(data, y_marg)
        return x_cond, logpdf, smoothing

    return Estimator(init=init, step=step)
