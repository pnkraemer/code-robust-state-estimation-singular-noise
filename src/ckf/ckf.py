import jax.numpy as jnp
import jax
import dataclasses

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclasses.dataclass
class Impl:
    rv_from_cholesky: Callable
    trafo_factorise: Callable
    combine: Callable
    evaluate_conditional: Callable
    condition: Callable
    marginal: Callable


def impl_cov_based():
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

    def rv_from_cholesky(m, c):
        return RandVar(m, c @ c.T)

    def trafo_factorise(*, trafo: Trafo, index: int) -> tuple[Trafo, SplitTrafo]:
        cov1, cov2 = jnp.split(trafo.noise.cov, indices_or_sections=[index], axis=0)
        C1, C21 = jnp.split(cov1, indices_or_sections=[index], axis=1)
        C12, C2 = jnp.split(cov2, indices_or_sections=[index], axis=1)

        G = C12 @ jnp.linalg.inv(C1)
        Z = C2 - G @ C1 @ G.T

        bias1, bias2 = jnp.split(trafo.noise.mean, indices_or_sections=[index], axis=0)
        linop1, linop2 = jnp.split(trafo.linop, indices_or_sections=[index], axis=0)

        x1_mid_z = Trafo(linop1, RandVar(bias1, C1))
        x1_mid_x2_and_z = SplitTrafo(
            G, linop2 - G @ linop1, RandVar(bias2 - G @ bias1, Z)
        )
        return x1_mid_z, x1_mid_x2_and_z

    def combine(*, outer: Trafo[T], inner: Trafo[T]) -> Trafo[T]:
        linop = outer.linop @ inner.linop
        bias = outer.linop @ inner.noise.mean + outer.noise.mean
        cov = outer.linop @ inner.noise.cov @ outer.linop.T + outer.noise.cov
        return Trafo(linop, RandVar(bias, cov))

    def evaluate_conditional(data, *, trafo: Trafo[T]) -> T:
        mean = trafo.linop @ data + trafo.noise.mean
        return RandVar(mean=mean, cov=trafo.noise.cov)

    def condition(*, prior: T, trafo: Trafo):
        z = trafo.linop @ prior.mean + trafo.noise.mean
        S = trafo.linop @ prior.cov @ trafo.linop.T + trafo.noise.cov
        marg = RandVar(z, S)

        K = prior.cov @ trafo.linop.T @ jnp.linalg.inv(S)
        m = prior.mean - K @ z
        C = prior.cov - K @ trafo.linop @ prior.cov
        cond = Trafo(linop=K, noise=RandVar(mean=m, cov=C))
        return marg, cond

    def marginal(*, prior: T, trafo: Trafo) -> T:
        m = trafo.linop @ prior.mean + trafo.noise.mean
        C = trafo.linop @ prior.cov @ trafo.linop.T + trafo.noise.cov
        return RandVar(m, C)

    return Impl(
        rv_from_cholesky=rv_from_cholesky,
        trafo_factorise=trafo_factorise,
        combine=combine,
        evaluate_conditional=evaluate_conditional,
        condition=condition,
        marginal=marginal,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Trafo(Generic[T]):
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
    linop1: jax.Array
    linop2: jax.Array
    noise: T


def trafo_split(*, trafo: Trafo, index: int) -> SplitTrafo:
    linop1, linop2 = jnp.split(trafo.linop, indices_or_sections=[index], axis=1)
    return SplitTrafo(linop1, linop2, trafo.noise)


def invert_deterministic(y: jax.Array, /, *, deterministic_trafo) -> jax.Array:
    A = deterministic_trafo.linop
    b = y - deterministic_trafo.noise.mean
    return jnp.linalg.solve(A, b)


def fix_x1(x1: jax.Array, /, *, split_trafo, impl):
    linop = split_trafo.linop2

    trafo = Trafo(split_trafo.linop1, split_trafo.noise)
    noise = impl.evaluate_conditional(x1, trafo=trafo)
    return Trafo(linop, noise)


def model_reduce(*, y_mid_x: Trafo, x_mid_z: Trafo, F, impl):
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
    y2_mid_z = impl.combine(outer=y2_mid_x1, inner=x1_mid_z)

    # Return values:
    reduced_model = (y2_mid_z, x2_mid_x1_and_z, y1_mid_x1_and_x2)
    info_transform_back = x_mid_x1_and_x2
    info_identify_constraint = y2_mid_x1
    info_split_data = (V1, V2)
    info = (info_transform_back, info_identify_constraint, info_split_data)
    return reduced_model, info


def model_reduced_apply(y: jax.Array, *, z, reduced, impl):
    # todo: isolate all covariance logic:
    #  - separate covariance-affected computation from the other
    #  - make the rank of F an argument somewhere
    #  - test corner cases of fully constrained and unconstrained estimation
    #  - square-root versions of everything
    # todo: marginal likelihood

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
    x1_value = invert_deterministic(y2, deterministic_trafo=y2_mid_x1)
    x2_mid_z = fix_x1(x1_value, split_trafo=x2_mid_x1_and_z, impl=impl)
    y1_mid_x2 = fix_x1(x1_value, split_trafo=y1_mid_x1_and_x2, impl=impl)
    x_mid_x2 = fix_x1(x1_value, split_trafo=x_mid_x1_and_x2, impl=impl)

    # Fix y2 in the "prior" distribution
    _y2, z_mid_y2 = impl.condition(prior=z, trafo=y2_mid_z)
    z = impl.evaluate_conditional(y2, trafo=z_mid_y2)

    # Now we have z, x2_mid_z, and y1_mid_x2
    # which is a "complete model" and we can run the usual estimation
    x2 = impl.marginal(prior=z, trafo=x2_mid_z)
    _y1, backward = impl.condition(prior=x2, trafo=y1_mid_x2)
    x2_mid_y1 = impl.evaluate_conditional(y1, trafo=backward)
    return x2_mid_y1, x_mid_x2
