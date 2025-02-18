import jax.numpy as jnp
import jax
import dataclasses

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Trafo(Generic[T]):
    """Affine transformation."""

    linop: jax.Array
    noise: T

    def __matmul__(self, other: jax.Array):
        """Implement Trafo @ Array"""
        return Trafo(linop=self.linop @ other, noise=self.noise)

    def __rmatmul__(self, other: jax.Array):
        """Implement Array @ Trafo"""
        return Trafo(linop=other @ self.linop, noise=other @ self.noise)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SplitTrafo(Generic[T]):
    """Affine transformation with a split' linear operator."""

    linop1: jax.Array
    linop2: jax.Array
    noise: T


@dataclasses.dataclass
class Impl[T]:
    rv_from_cholesky: Callable[[jax.Array, jax.Array], T]
    rv_condition: Callable[[T, Trafo[T]], tuple[T, Trafo[T]]]
    rv_marginal: Callable[[T, Trafo[T]], T]

    trafo_factorise: Callable[[Trafo[T]], tuple[Trafo[T], SplitTrafo[T]]]
    trafo_combine: Callable[[Trafo[T], Trafo[T]], Trafo[T]]
    trafo_evaluate: Callable[[jax.Array, Trafo[T]], T]

    split_trafo_fix_x1: Callable[[jax.Array, SplitTrafo[T]], Trafo[T]]


def impl_cholesky_based() -> Impl:
    @jax.tree_util.register_dataclass
    @dataclasses.dataclass
    class RandVar:
        mean: jax.Array
        cholesky: jax.Array

        def __rmatmul__(self, other: jax.Array):
            """Implement Array @ Trafo"""
            mean = other @ self.mean
            cholesky = other @ self.cholesky
            return RandVar(mean, cholesky)

        def cov_dense(self):
            return self.cholesky @ self.cholesky.T

    def rv_from_cholesky(m, c):
        return RandVar(mean=m, cholesky=c)

    def rv_condition(prior, trafo):
        R_YX = trafo.noise.cholesky.T
        R_X = prior.cholesky.T
        R_X_F = prior.cholesky.T @ trafo.linop.T
        R_y, (R_xy, G) = _revert_conditional(R_X_F=R_X_F, R_X=R_X, R_YX=R_YX)

        s = trafo.linop @ prior.mean + trafo.noise.mean
        mean_new = prior.mean - G @ s
        return RandVar(s, R_y.T), Trafo(G, RandVar(mean_new, R_xy.T))

    def rv_marginal(prior, trafo):
        mean = trafo.linop @ prior.mean + trafo.noise.mean

        mtrx = jnp.concatenate(
            [prior.cholesky.T @ trafo.linop.T, trafo.noise.cholesky.T]
        )
        R = jnp.linalg.qr(mtrx, mode="r")
        return RandVar(mean, R.T)

    def trafo_factorise(*, trafo, index):
        R = jnp.linalg.qr(trafo.noise.cholesky.T, mode="r")
        R1 = R[:index, :index]
        R12 = R[:index, index:]
        R2 = R[index:, index:]
        G = jax.scipy.linalg.solve_triangular(R1, R12, lower=False).T

        bias1, bias2 = jnp.split(trafo.noise.mean, indices_or_sections=[index], axis=0)
        linop1, linop2 = jnp.split(trafo.linop, indices_or_sections=[index], axis=0)

        x1_mid_z = Trafo(linop1, RandVar(bias1, R1.T))

        noise = RandVar(bias2 - G @ bias1, R2.T)
        x1_mid_x2_and_z = SplitTrafo(G, linop2 - G @ linop1, noise)
        return x1_mid_z, x1_mid_x2_and_z

    def trafo_combine(*, outer, inner):
        linop = outer.linop @ inner.linop
        bias = outer.linop @ inner.noise.mean + outer.noise.mean

        mtrx = jnp.concatenate(
            [inner.noise.cholesky.T @ outer.linop.T, outer.noise.cholesky.T], axis=0
        )

        R = jnp.linalg.qr(mtrx, mode="r")
        return Trafo(linop, RandVar(bias, R.T))

    def trafo_evaluate(x, /, *, trafo):
        return RandVar(trafo.linop @ x + trafo.noise.mean, trafo.noise.cholesky)

    def split_trafo_fix_x1(x1, *, split_trafo):
        trafo = Trafo(split_trafo.linop1, split_trafo.noise)
        noise = trafo_evaluate(x1, trafo=trafo)

        linop = split_trafo.linop2
        return Trafo(linop, noise)

    def _revert_conditional(*, R_X_F: jax.Array, R_X: jax.Array, R_YX: jax.Array):
        # Taken from:
        # https://github.com/pnkraemer/probdiffeq/blob/main/probdiffeq/util/cholesky_util.py

        R = jnp.block([[R_YX, jnp.zeros((R_YX.shape[0], R_X.shape[1]))], [R_X_F, R_X]])
        R = jnp.linalg.qr(R, mode="r")

        # ~R_{Y}
        d_out = R_YX.shape[1]
        R_Y = R[:d_out, :d_out]

        # something like the cross-covariance
        R12 = R[:d_out, d_out:]

        # Implements G = R12.T @ np.linalg.inv(R_Y.T) in clever:
        G = jax.scipy.linalg.solve_triangular(R_Y, R12, lower=False).T

        # ~R_{X \mid Y}
        R_XY = R[d_out:, d_out:]
        return R_Y, (R_XY, G)

    return Impl(
        rv_from_cholesky=rv_from_cholesky,
        rv_condition=rv_condition,
        rv_marginal=rv_marginal,
        trafo_factorise=trafo_factorise,
        trafo_combine=trafo_combine,
        trafo_evaluate=trafo_evaluate,
        split_trafo_fix_x1=split_trafo_fix_x1,
    )


def impl_cov_based() -> Impl:
    @jax.tree_util.register_dataclass
    @dataclasses.dataclass
    class RandVar:
        mean: jax.Array
        cov: jax.Array

        def __rmatmul__(self, other: jax.Array):
            """Implement Array @ Trafo"""
            mean = other @ self.mean
            cov = other @ self.cov @ other.T
            return RandVar(mean, cov)

        def cov_dense(self):
            return self.cov

    def rv_from_cholesky(m, c):
        return RandVar(m, c @ c.T)

    def trafo_factorise(*, trafo: Trafo, index: int) -> tuple[Trafo, SplitTrafo]:
        cov1, cov2 = jnp.split(trafo.noise.cov, indices_or_sections=[index], axis=0)
        C1, C21 = jnp.split(cov1, indices_or_sections=[index], axis=1)
        C12, C2 = jnp.split(cov2, indices_or_sections=[index], axis=1)

        G = jnp.linalg.solve(C1.T, C12.T).T
        Z = C2 - G @ C1 @ G.T

        bias1, bias2 = jnp.split(trafo.noise.mean, indices_or_sections=[index], axis=0)
        linop1, linop2 = jnp.split(trafo.linop, indices_or_sections=[index], axis=0)

        x1_mid_z = Trafo(linop1, RandVar(bias1, C1))
        x1_mid_x2_and_z = SplitTrafo(
            G, linop2 - G @ linop1, RandVar(bias2 - G @ bias1, Z)
        )
        return x1_mid_z, x1_mid_x2_and_z

    def trafo_combine(*, outer: Trafo[T], inner: Trafo[T]) -> Trafo[T]:
        linop = outer.linop @ inner.linop
        bias = outer.linop @ inner.noise.mean + outer.noise.mean
        cov = outer.linop @ inner.noise.cov @ outer.linop.T + outer.noise.cov
        return Trafo(linop, RandVar(bias, cov))

    def trafo_evaluate(data, *, trafo: Trafo[T]) -> T:
        mean = trafo.linop @ data + trafo.noise.mean
        return RandVar(mean=mean, cov=trafo.noise.cov)

    def rv_condition(*, prior: T, trafo: Trafo):
        z = trafo.linop @ prior.mean + trafo.noise.mean
        S = trafo.linop @ prior.cov @ trafo.linop.T + trafo.noise.cov
        marg = RandVar(z, S)

        K = jnp.linalg.solve(S.T, trafo.linop @ prior.cov).T
        m = prior.mean - K @ z
        C = prior.cov - K @ trafo.linop @ prior.cov
        cond = Trafo(linop=K, noise=RandVar(mean=m, cov=C))
        return marg, cond

    def rv_marginal(*, prior: T, trafo: Trafo) -> T:
        m = trafo.linop @ prior.mean + trafo.noise.mean
        C = trafo.linop @ prior.cov @ trafo.linop.T + trafo.noise.cov
        return RandVar(m, C)

    def split_trafo_fix_x1(x1: jax.Array, /, *, split_trafo):
        trafo = Trafo(split_trafo.linop1, split_trafo.noise)
        noise = trafo_evaluate(x1, trafo=trafo)

        linop = split_trafo.linop2
        return Trafo(linop, noise)

    return Impl(
        rv_from_cholesky=rv_from_cholesky,
        rv_condition=rv_condition,
        rv_marginal=rv_marginal,
        trafo_factorise=trafo_factorise,
        trafo_combine=trafo_combine,
        trafo_evaluate=trafo_evaluate,
        split_trafo_fix_x1=split_trafo_fix_x1,
    )


def trafo_split(*, trafo: Trafo, index: int) -> SplitTrafo:
    linop1, linop2 = jnp.split(trafo.linop, indices_or_sections=[index], axis=1)
    return SplitTrafo(linop1, linop2, trafo.noise)


def trafo_invert_dirac(y: jax.Array, /, *, dirac_trafo) -> jax.Array:
    A = dirac_trafo.linop
    b = y - dirac_trafo.noise.mean
    return jnp.linalg.solve(A, b)


def model_reduction(F, impl):
    def model_reduce(*, y_mid_x: Trafo, x_mid_z: Trafo):
        # First QR iteration
        _, ndim = F.shape
        V, R = jnp.linalg.qr(F, mode="complete")
        V1, V2 = jnp.split(V, indices_or_sections=[ndim], axis=1)
        y1_mid_x = V1.T @ y_mid_x
        y2_mid_x = V2.T @ y_mid_x

        # Second QR iteration
        W, S = jnp.linalg.qr(y2_mid_x.linop.T, mode="complete")
        W1, W2 = jnp.split(W, indices_or_sections=[len(V2.T)], axis=1)

        # Factorise the z-to-x conditional
        x1_and_x2_mid_z = W.T @ x_mid_z
        x1_mid_z, x2_mid_x1_and_z = impl.trafo_factorise(
            trafo=x1_and_x2_mid_z, index=len(V2.T)
        )

        # y2 | x1 is deterministic (ie zero cov) and y2 is independent of x2 given x1
        y2_mid_x1 = y2_mid_x @ W1

        # y1 now depends on both x1 and x2; we implement this as a split' conditional,
        #  which is a conditional with two linops
        y1_mid_x1_and_x2 = trafo_split(trafo=(y1_mid_x @ W), index=len(V2.T))

        # We need to memorise how to turn x1/x2 back into x
        trafo = Trafo(
            W, impl.rv_from_cholesky(jnp.zeros((len(W),)), jnp.zeros((len(W), len(W))))
        )
        x_mid_x1_and_x2 = trafo_split(trafo=trafo, index=len(V2.T))

        # We only care about y2 | z, not about x1 | z, so we combine transformations
        y2_mid_z = impl.trafo_combine(outer=y2_mid_x1, inner=x1_mid_z)

        # Return values:
        reduced_model = (y2_mid_z, x2_mid_x1_and_z, y1_mid_x1_and_x2)
        info_transform_back = x_mid_x1_and_x2
        info_identify_constraint = y2_mid_x1
        info_split_data = (V1, V2)
        info = (info_transform_back, info_identify_constraint, info_split_data)
        return reduced_model, info


    def model_reduced_apply(y: jax.Array, *, z, reduced):
        # todo: make the rank of F an argument somewhere
        # todo: make all solves into solve_triangular etc.
        # todo: marginal likelihood
        # todo: test that cov-based and chol-based yield the same values (by testing that all marginal likelihood configs are identical?)

        # Read off prepared values
        reduced_model, info = reduced
        (info_transform_back, info_identify_constraint, info_split_data) = info
        (y2_mid_z, x2_mid_x1_and_z, y1_mid_x1_and_x2) = reduced_model
        x_mid_x1_and_x2 = info_transform_back
        y2_mid_x1 = info_identify_constraint
        (V1, V2) = info_split_data

        # Split the data data
        y1, y2 = V1.T @ y, V2.T @ y

        # Fix y2 (via x1) in remaining conditionals.
        #  Recall that by construction of the QR decompositions,
        #  y2_mid_x1 has zero covariance.
        x1_value = trafo_invert_dirac(y2, dirac_trafo=y2_mid_x1)
        x2_mid_z = impl.split_trafo_fix_x1(x1_value, split_trafo=x2_mid_x1_and_z)
        y1_mid_x2 = impl.split_trafo_fix_x1(x1_value, split_trafo=y1_mid_x1_and_x2)
        x_mid_x2 = impl.split_trafo_fix_x1(x1_value, split_trafo=x_mid_x1_and_x2)

        # Fix y2 in the "prior" distribution
        _y2, z_mid_y2 = impl.rv_condition(prior=z, trafo=y2_mid_z)
        z = impl.trafo_evaluate(y2, trafo=z_mid_y2)

        # Now we have z, x2_mid_z, and y1_mid_x2
        # which is a "complete model" (just smaller than the previous one)
        # and we can run the usual estimation
        x2 = impl.rv_marginal(prior=z, trafo=x2_mid_z)
        _y1, backward = impl.rv_condition(prior=x2, trafo=y1_mid_x2)
        x2_mid_y1 = impl.trafo_evaluate(y1, trafo=backward)
        return x2_mid_y1, x_mid_x2
    return model_reduce, model_reduced_apply