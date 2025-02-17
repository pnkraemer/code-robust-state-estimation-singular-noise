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

    def __rmatmul__(self, other: jax.Array):
        return Trafo(
            linop=other @ self.linop,
            bias=other @ self.bias,
            cov=other @ self.cov @ other.T,
        )


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


def model_reduce(*, y_mid_x: Trafo, x_mid_z: Trafo):
    # First QR iteration
    F = y_mid_x.cov_to_low_rank()
    _, ndim = F.shape
    V, R = jnp.linalg.qr(F, mode="complete")
    V1, V2 = jnp.split(V, indices_or_sections=[ndim], axis=1)
    R1, zeros = jnp.split(R, indices_or_sections=[ndim], axis=0)

    # Second QR iteration
    y2_mid_x = V2.T @ y_mid_x
    W, S = jnp.linalg.qr(y2_mid_x.linop.T, mode="complete")
    W1, W2 = jnp.split(W, indices_or_sections=[len(V2.T)], axis=1)
    S1, zeros = jnp.split(S, indices_or_sections=[len(V2.T)], axis=0)

    # Parametrise x2_mid_x1
    Q = x_mid_z.cov
    C = y_mid_x.linop
    x1_mid_z_raw = W1.T @ x_mid_z
    x2_mid_z_raw = W2.T @ x_mid_z
    G = W2.T @ Q @ W1 @ jnp.linalg.inv(x1_mid_z_raw.cov)
    Z = x2_mid_z_raw.cov - G @ x1_mid_z_raw.cov @ G.T

    # Parametrise y2_mid_z
    linop = S1 @ x1_mid_z_raw.linop
    bias = S1 @ x1_mid_z_raw.bias + y2_mid_x.bias
    cov = S1 @ W1.T @ x_mid_z.cov @ W1 @ S1.T
    y2_mid_z = Trafo(linop, bias, cov)

    linop = x2_mid_z_raw.linop - G @ x1_mid_z_raw.linop
    bias = x2_mid_z_raw.bias - G @ x1_mid_z_raw.bias
    cov = Z
    x2_mid_z_no_x1 = Trafo(linop, bias, cov)

    reduced = ((V1, V2), (W1, W2), S1, R1, (G, Z), y_mid_x, x2_mid_z_no_x1, C, y2_mid_z)
    return reduced


def model_reduced_apply(y: jax.Array, *, z, reduced, extra_trafo):
    ((V1, V2), (W1, W2), S1, R1, (G, Z), y_mid_x, x2_mid_z_no_x1, C, y2_mid_z) = reduced

    # Split measurement model and data
    y1_mid_x = V1.T @ y_mid_x
    y2_mid_x = V2.T @ y_mid_x

    y1, y2 = V1.T @ y, V2.T @ y
    x1_value = jnp.linalg.solve(S1, y2 - y2_mid_x.bias)

    # Condition z on y2
    if extra_trafo is not None:
        y2_mid_z = combine(outer=y2_mid_z, inner=extra_trafo)
    _y2, z_mid_y2 = condition(prior=z, trafo=y2_mid_z)
    z = evaluate_conditional(y2, trafo=z_mid_y2)

    # Marginalise to the "next future" x2
    x2_mid_z = Trafo(
        x2_mid_z_no_x1.linop, x2_mid_z_no_x1.bias + G @ x1_value, x2_mid_z_no_x1.cov
    )
    if extra_trafo is not None:
        x2_mid_z = combine(outer=x2_mid_z, inner=extra_trafo)
    x2 = marginal(prior=z, trafo=x2_mid_z)

    # Condition x2 on y2
    linop = V1.T @ C @ W2
    bias = V1.T @ C @ W1 @ x1_value + y1_mid_x.bias
    cov = R1 @ R1.T
    y1_mid_x2 = Trafo(linop, bias, cov)
    _y1, backward = condition(prior=x2, trafo=y1_mid_x2)
    x2_mid_y1 = evaluate_conditional(y1, trafo=backward)

    x_mid_x2 = Trafo(linop=W2, bias=W1 @ x1_value, cov=jnp.zeros((len(W1), len(W1))))
    return x2_mid_y1, x_mid_x2
