import jax.numpy as jnp
import jax
import dataclasses

from typing import Callable


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RandVar:
    mean: jax.Array
    cov: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Trafo:
    linop: jax.Array
    bias: jax.Array
    cov: jax.Array

    def cov_to_cholesky(self):
        eigh = jnp.linalg.eigh(self.cov)
        cholesky = eigh.eigenvectors @ jnp.sqrt(jnp.diag(eigh.eigenvalues))
        return jnp.linalg.qr(cholesky.T, mode="r").T

    def cov_to_low_rank(self):
        eigh = jnp.linalg.eigh(self.cov)
        cholesky = eigh.eigenvectors @ jnp.sqrt(jnp.diag(eigh.eigenvalues))

        small = jnp.finfo(cholesky.dtype).eps
        return cholesky[:, eigh.eigenvalues > small]

    def __matmul__(self, other: jax.Array):
        return Trafo(
            linop=self.linop @ other,
            bias=self.bias,
            cov=self.cov,
        )

    def __rmatmul__(self, other: jax.Array):
        return Trafo(
            linop=other @ self.linop,
            bias=other @ self.bias,
            cov=other @ self.cov @ other.T,
        )

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TrafoDuo:
    linop1: jax.Array
    linop2: jax.Array
    bias: jax.Array
    cov: jax.Array



def marginal(*, prior: RandVar, trafo: Trafo) -> RandVar:
    m = trafo.linop @ prior.mean + trafo.bias
    C = trafo.linop @ prior.cov @ trafo.linop.T + trafo.cov
    return RandVar(m, C)


def condition(*, prior: RandVar, trafo: Trafo):
    z = trafo.linop @ prior.mean + trafo.bias
    S = trafo.linop @ prior.cov @ trafo.linop.T + trafo.cov
    marg = RandVar(z, S)

    K = prior.cov @ trafo.linop.T @ jnp.linalg.inv(S)
    m = prior.mean - K @ z
    C = prior.cov - K @ trafo.linop @ prior.cov
    cond = Trafo(linop=K, bias=m, cov=C)
    return marg, cond


def evaluate_conditional(data, *, trafo: Trafo) -> RandVar:
    return RandVar(mean=trafo.linop @ data + trafo.bias, cov=trafo.cov)


def combine(*, outer: Trafo, inner: Trafo) -> Trafo:
    linop = outer.linop @ inner.linop
    bias = outer.linop @ inner.bias + outer.bias
    cov = outer.linop @ inner.cov @ outer.linop.T + outer.cov
    return Trafo(linop, bias, cov)


def model_reduce(*, y_mid_x: Trafo, x_mid_z: Trafo, F):
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
    x1_mid_z, x2_mid_x1_and_z = factorise_conditional(x1_and_x2_mid_z, len(V2.T))

    # y2 | x1 is deterministic (and y2 is independent of x2 given x1)
    y2_mid_x1 = y2_mid_x @ W1  # deterministic (ie zero cov)

    # y1 now depends on both x1 and x2; we implement this as a "split' conditional"
    y1_mid_x1_and_x2 = split_conditional(y1_mid_x @ W, len(V2.T))

    # We need to memorise how to turn x1/x2 back into x
    x_mid_x1_and_x2 = split_conditional(Trafo(W, 0., 0.), len(V2.T))

    # We only care about y2 | z, not about x1 | z
    y2_mid_z = combine(outer=y2_mid_x1, inner=x1_mid_z)

    # Return values:
    reduced_model = (y2_mid_z, x2_mid_x1_and_z, y1_mid_x1_and_x2)
    info_transform_back = x_mid_x1_and_x2
    info_identify_constraint = y2_mid_x1
    info_split_data = (V1, V2)
    return reduced_model, info_transform_back, info_identify_constraint, info_split_data


def model_reduced_apply(y: jax.Array, *, z, reduced):
    # Read off prepared values
    reduced_model, info_transform_back, info_identify_constraint, info_split_data = reduced
    (y2_mid_z, x2_mid_x1_and_z, y1_mid_x1_and_x2) = reduced_model
    x_mid_x1_and_x2 = info_transform_back
    y2_mid_x1 = info_identify_constraint
    (V1, V2) = info_split_data


    # Split the data data
    y1, y2 = V1.T @ y, V2.T @ y

    # Fix y2 (via x1) in remaining conditionals
    x1_value = condition_deterministic(y2, y2_mid_x1)
    x2_mid_z = fix_x1(x1_value, x2_mid_x1_and_z)
    y1_mid_x2 = fix_x1(x1_value, y1_mid_x1_and_x2)
    x_mid_x2 = fix_x1(x1_value, x_mid_x1_and_x2)

    # Fix y2 in the "prior" distribution
    _y2, z_mid_y2 = condition(prior=z, trafo=y2_mid_z)
    z = evaluate_conditional(y2, trafo=z_mid_y2)

    # Now we have z, x2_mid_z, and y1_mid_x2
    # which is a "complete model" and we can run the usual estimation
    x2 = marginal(prior=z, trafo=x2_mid_z)
    _y1, backward = condition(prior=x2, trafo=y1_mid_x2)
    x2_mid_y1 = evaluate_conditional(y1, trafo=backward)
    return x2_mid_y1, x_mid_x2


def split_conditional(cond, index):
    linop1, linop2 = jnp.split(cond.linop, indices_or_sections=[index], axis=1)
    return TrafoDuo(linop1, linop2, cond.bias, cond.cov)

def factorise_conditional(cond, index):
    cov1, cov2 = jnp.split(cond.cov, indices_or_sections=[index], axis=0)
    C1, C21 = jnp.split(cov1, indices_or_sections=[index], axis=1)
    C12, C2 = jnp.split(cov2, indices_or_sections=[index], axis=1)

    G = C12 @ jnp.linalg.inv(C1)
    Z = C2 - G @ C1 @ G.T

    bias1, bias2 = jnp.split(cond.bias, indices_or_sections=[index], axis=0)
    linop1, linop2 = jnp.split(cond.linop, indices_or_sections=[index], axis=0)

    x1_mid_z = Trafo(linop1, bias1, C1)
    x1_mid_x2_and_z = TrafoDuo(G, linop2 - G @ linop1, bias2 - G @ bias1, Z)
    return x1_mid_z, x1_mid_x2_and_z

def condition_deterministic(y, trafo):
    return jnp.linalg.solve(trafo.linop, y - trafo.bias)

def fix_x1(x1, duo_trafo):
    return Trafo(duo_trafo.linop2, duo_trafo.linop1 @ x1 + duo_trafo.bias, duo_trafo.cov)
