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
        return cholesky[:, eigh.eigenvalues > 0.0]

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


def condition(data, *, prior: RandVar, trafo: Trafo) -> RandVar:
    z = trafo.linop @ prior.mean + trafo.bias
    S = trafo.linop @ prior.cov @ trafo.linop.T + trafo.cov
    marg = RandVar(z, S)

    K = prior.cov @ trafo.linop.T @ jnp.linalg.inv(S)
    m = prior.mean - K @ (z - data)
    C = prior.cov - K @ trafo.linop @ prior.cov
    cond = RandVar(m, C)
    return marg, cond


def model_reduce(y: jax.Array, *, y_mid_x: Trafo, x_mid_z: Trafo, z: RandVar):
    # First QR iteration
    F = y_mid_x.cov_to_low_rank()
    _, ndim = F.shape
    V, R = jnp.linalg.qr(F, mode="complete")
    V1, V2 = jnp.split(V, indices_or_sections=[ndim], axis=1)
    R1, zeros = jnp.split(R, indices_or_sections=[ndim], axis=0)

    # Split measurement model and data
    y1_mid_x = V1.T @ y_mid_x
    y2_mid_x = V2.T @ y_mid_x
    y1, y2 = V1.T @ y, V2.T @ y

    # Second QR iteration
    W, S = jnp.linalg.qr(y2_mid_x.linop.T, mode="complete")
    W1, W2 = jnp.split(W, indices_or_sections=[len(y2)], axis=1)
    S1, zeros = jnp.split(S, indices_or_sections=[len(y2)], axis=0)

    Q = x_mid_z.cov
    C = y_mid_x.linop
    x1_value = jnp.linalg.solve(S1, y2 - y2_mid_x.bias)
    x1_mid_z_raw = W1.T @ x_mid_z
    x2_mid_z_raw = W2.T @ x_mid_z
    G = W2.T @ Q @ W1 @ jnp.linalg.inv(x1_mid_z_raw.cov)
    Z = x2_mid_z_raw.cov - G @ x1_mid_z_raw.cov @ G.T

    x2_mid_z = Trafo(
        linop=x2_mid_z_raw.linop - G @ x1_mid_z_raw.linop,
        bias=x2_mid_z_raw.bias - G @ x1_mid_z_raw.bias + G @ x1_value,
        cov=Z,
    )
    y1_mid_x2 = Trafo(linop=V1.T @ C @ W2, bias=V1.T @ C @ W1 @ x1_value + y1_mid_x.bias, cov=R1 @ R1.T)


    # Handle the z-to-y2 relation
    y2_mid_z = Trafo(
        linop=S1 @ x1_mid_z_raw.linop,
        bias=S1 @ x1_mid_z_raw.bias + y2_mid_x.bias,
        cov=S1 @ W1.T @ x_mid_z.cov @ W1 @ S1.T,
    )
    y2, z_mid_y2 = condition(y2, prior=z, trafo=y2_mid_z)

    return z_mid_y2, x2_mid_z, y1_mid_x2, x1_value, y1, W1, W2
    # assert False
    # # Collect terms:
    #
    # # Data
    # y1, y2 = V1.T @ y, V2.T @ y
    # x1_value = jnp.linalg.solve(S1, y2)
    # assert False
    # y2_mid_z = Trafo(linop=S1 @ W1.T @ x_mid_z.bias, bias=S1 @ W1.T @ x_mid_z.bias, cov=S1 @ W1.T @ x_mid_z.cov @ W1 @ S1.T)
    # z_mid_y2, y2 = condition(prior=z, trafo=y2_mid_z)
    #
    # x2_mid_z = 0 # ...
    # y1_mid_z = 0 # ...
    #
    # return z_mid_y2, x2_mid_z, y1_mid_x2
    #
    # linop1 = W1.T @ x_mid_z.linop
    # bias1 = W1.T @ x_mid_z.bias
    # cov1 = W1.T @ x_mid_z.cov @ W1  # nonzero
    # x1_mid_z = Trafo(linop1, bias1, cov1)
    #
    # linop2 = W2.T @ x_mid_z.linop
    # bias2 = W2.T @ x_mid_z.bias
    # cov2 = W2.T @ x_mid_z.cov @ W2  # nonzero
    # x2_mid_z = Trafo(linop2, bias2, cov2)
    #
    # G = W2 @ Q @ W1.T @ jnp.linalg.inv(W1 @ Q @ W1.T)
    #
    # return model1, model2
