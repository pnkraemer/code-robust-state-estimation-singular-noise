import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Transition(NamedTuple):
    linop: jax.Array
    cov: jax.Array


class Trafo(NamedTuple):
    full: jax.Array
    perp: jax.Array
    para: jax.Array


class RV(NamedTuple):
    mean: jax.Array
    cov: jax.Array


def main(N=10, D=3, d=1):
    print(f"D={D}, d={d}, N={N}")

    # Set up config
    jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(3)

    # Set up SSM
    prior = model_prior(D=D)
    constraint = model_constraint(D=D, d=d)
    init = model_init(D=D, constraint=constraint)
    data = jnp.zeros((N, d))

    solution = run_kf(init=init, prior=prior, constraint=constraint, data=data)

    # Assert constraint is satisfied
    for s1, s2 in zip(solution.mean, data):
        eps = jnp.sqrt(jnp.finfo(jnp.dtype(s1)).eps)
        assert jnp.allclose(constraint.linop @ s1, s2, rtol=eps, atol=eps)




def model_prior(*, D) -> Transition:
    Phi = jnp.eye(D)
    Sigma = jnp.eye(D)
    return Transition(Phi, Sigma)


def model_constraint(*, D, d) -> Transition:
    A = jnp.eye(d, D)
    R = jnp.zeros((d, d))
    return Transition(A, R)


def model_init(*, D, constraint) -> RV:
    key = jax.random.PRNGKey(2141)
    mean = jax.random.normal(key, shape=(D,))
    chol = jax.random.normal(key, shape=(D, D))
    cov = chol @ chol.T + jnp.eye(D)
    init = RV(mean, cov)
    init_ = filter_condition(init, constraint, data=0.0)
    return init_


def run_kf(*, prior, constraint, init, data):
    step = functools.partial(filter_step, prior, constraint)
    _, solution = jax.lax.scan(step, init=init, xs=data)
    return solution


def filter_step(prior: Transition, constraint: Transition, rv: RV, data):
    rv = filter_predict(rv, prior)
    rv = filter_condition(rv, constraint, data)
    return rv, rv


def filter_predict(rv: RV, prior: Transition):
    m, C = rv
    m = prior.linop @ m
    C = prior.linop @ C @ prior.linop.T + prior.cov
    return RV(m, C)


def filter_condition(rv, constraint, data):
    S = constraint.linop @ rv.cov @ constraint.linop.T + constraint.cov
    K = jnp.linalg.solve(S.T, constraint.linop @ rv.cov.T).T
    m = rv.mean - K @ (constraint.linop @ rv.mean - data)
    C = rv.cov - K @ S @ K.T
    return RV(m, C)


if __name__ == "__main__":
    main()
