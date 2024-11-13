import jax
import jax.numpy as jnp
import functools
from typing import NamedTuple

# todo: take the Kalman filter and move more slowly.
# 1. Take the coordinate change induced by QR(S)
# 2. Reparametrise the whole system
# 3. The states according to image of S should be zero
#    The remaining states should be independent of that?
# 4. Can we ditch states?
# 5. Does this match what we do in maths?
# 6. What about time-varying systems?


class Transition(NamedTuple):
    linop: jax.Array
    cov: jax.Array


class RV(NamedTuple):
    mean: jax.Array
    cov: jax.Array


def main(N=4, D=3, d=2):
    # Set up config
    jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(1)
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
    for s, d in zip(solution.mean, data):
        eps = jnp.sqrt(jnp.finfo(jnp.dtype(s)).eps)
        assert jnp.allclose(constraint.linop @ s, d, rtol=eps, atol=eps)


    # Change coordinates
    Q, R = jnp.linalg.qr(constraint.linop.T, mode="complete")
    init, prior, constraint = coordinate_change(init, prior, constraint, Q=Q)
    step = functools.partial(filter_step, prior, constraint)
    _, solution_new = jax.lax.scan(step, init=init, xs=data)

    # Assert constraint is satisfied
    for m1, m2 in zip(solution_new.mean, solution.mean):
        eps = jnp.sqrt(jnp.finfo(jnp.dtype(s)).eps)
        assert jnp.allclose(Q @ m1, m2, rtol=eps, atol=eps)

    print(constraint)
    print(solution_new.cov)

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
    K = rv.cov @ constraint.linop.T @ jnp.linalg.inv(S)
    m = rv.mean - K @ (constraint.linop @ rv.mean - data)
    C = rv.cov - K @ S @ K.T
    return RV(m, C)

def coordinate_change(init, prior, constraint, *, Q):
    m, C = init 
    init = RV(Q.T @ m, Q.T @ C @ Q)

    linop, cov = prior
    linop = Q.T @ linop @ Q 
    cov = Q.T @ cov @ Q
    prior = Transition(linop, cov)

    linop, cov = constraint 
    linop = linop @ Q
    constraint = Transition(linop, cov)
    return init, prior, constraint


if __name__ == "__main__":
    main()
