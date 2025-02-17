import jax.numpy as jnp
import jax
import dataclasses

from typing import Callable


@dataclasses.dataclass
class RandVar:
    mean: jax.Array
    cov: jax.Array


@dataclasses.dataclass
class Trafo:
    linop: jax.Array
    bias: jax.Array
    cov: jax.Array


def filter_one_step() -> Callable:
    def cond(*, z: RandVar, x_given_z: Trafo, y_given_x: Trafo) -> RandVar:
        x = predict(prior=z, trafo=x_given_z)
        x_given_y = update(prior=x, trafo=y_given_x)
        return x_given_y

    def predict(*, prior: RandVar, trafo: Trafo) -> RandVar:
        m = trafo.linop @ prior.mean + trafo.bias
        C = trafo.linop @ prior.cov @ trafo.linop.T + trafo.cov
        return RandVar(m, C)

    def update(*, prior: RandVar, trafo: Trafo) -> RandVar:
        z = trafo.linop @ prior.mean + trafo.bias
        S = trafo.linop @ prior.cov @ trafo.linop.T + trafo.cov
        K = prior.cov @ trafo.linop.T @ jnp.linalg.inv(S)

        m = prior.mean - K @ z
        C = prior.cov - K @ trafo.linop @ prior.cov
        return RandVar(m, C)

    return cond
