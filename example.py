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


def main(N=4, D=3, d=1):
    # Set up config
    jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(3)

    key = jax.random.PRNGKey(1)
    key1, key2, key3 = jax.random.split(key, num=3)

    # Set up SSM
    prior = model_prior(key1, D=D)
    constraint = model_constraint(key2, D=D, d=d)
    init = model_init(key3, D=D, constraint=constraint)
    data = jnp.zeros((N, d))

    # Run KF
    step = functools.partial(filter_step, prior, constraint)
    _, solution = jax.lax.scan(step, init=init, xs=data)

    # Assert constraint is satisfied
    for s1, s2 in zip(solution.mean, data):
        eps = jnp.sqrt(jnp.finfo(jnp.dtype(s1)).eps)
        assert jnp.allclose(constraint.linop @ s1, s2, rtol=eps, atol=eps)
    print()
    print()
    print("-----------------------")
    print()
    print()
    # Change coordinates
    Q, _R = jnp.linalg.qr(constraint.linop.T, mode="complete")
    Q = Trafo(full=Q, para=Q[:, :d], perp=Q[:, d:])
    init, prior, constraint = coordinate_change(init, prior, constraint, Q=Q)

    step = functools.partial(filter_step, prior, constraint)
    _, solution_new = jax.lax.scan(step, init=init, xs=data)

    # Assert constraint is satisfied
    for m1, m2 in zip(solution_new.mean, solution.mean):
        eps = jnp.sqrt(jnp.finfo(jnp.dtype(m1)).eps)
        assert jnp.allclose(Q.full @ m1, m2, rtol=eps, atol=eps)


def model_prior(key, *, D) -> Transition:
    Phi = jax.random.normal(key, shape=(D, D)) / jnp.sqrt(D)
    Sigma = Phi @ Phi.T + jnp.eye(D)
    return Transition(Phi, Sigma)


def model_constraint(key, *, D, d) -> Transition:
    A = jax.random.normal(key, shape=(d, D))
    R = jnp.zeros((d, d))
    return Transition(A, R)


def model_init(key, *, D, constraint) -> RV:
    mean = jax.random.normal(key, shape=(D,))
    chol = jax.random.normal(key, shape=(D, D))
    cov = chol @ chol.T + jnp.eye(D)
    init = RV(mean, cov)
    return filter_condition(init, constraint, data=0.0)


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


def coordinate_change(init, prior, constraint, *, Q: Trafo):
    m, C = init
    init = RV(Q.full.T @ m, Q.full.T @ C @ Q.full)

    linop, cov = prior
    linop = Q.full.T @ linop @ Q.full
    cov = Q.full.T @ cov @ Q.full
    prior = Transition(linop, cov)

    linop, cov = constraint
    linop = linop @ Q.full
    constraint = Transition(linop, cov)
    return init, prior, constraint


if __name__ == "__main__":
    main()
