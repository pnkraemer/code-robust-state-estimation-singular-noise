import jax.numpy as jnp
import jax
import dataclasses

from typing import  Callable

@dataclasses.dataclass
class RandVar:
    mean: jax.Array
    cov: jax.Array

@dataclasses.dataclass
class Trafo:
    linop: jax.Array
    bias: jax.Array
    cov: jax.Array



def condition_one_step() -> Callable:
    def cond(*, z: RandVar, z_to_x: Trafo, x_to_y: Trafo) -> RandVar:
        x = predict(z, z_to_x)
        x_given_y = update(x, x_to_y)
        return x_given_y

    def predict(rv: RandVar, model: Trafo) -> RandVar:
        m = model.linop @ rv.mean + model.bias
        C = model.linop @ rv.cov @ model.linop.T + model.cov
        return RandVar(m, C)

    def update(rv: RandVar, model: Trafo) -> RandVar:
        print(model.linop.shape, rv.mean.shape, model.bias.shape)
        z = model.linop @ rv.mean + model.bias
        S = model.linop @ rv.cov @ model.linop.T + model.cov
        K = rv.cov @ model.linop.T @ jnp.linalg.inv(S)

        m = rv.mean - K @ z
        C = rv.cov - K @ model.linop @ rv.cov
        return RandVar(m, C)

    return cond

