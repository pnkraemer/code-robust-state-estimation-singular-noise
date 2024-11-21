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


def main(N=2, D=3, d=1):
    print(f"D={D}, d={d}, N={N}")

    # Set up config
    jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(3)

    # Set up SSM
    prior = model_prior(D=D)
    constraint = model_constraint(D=D, d=d)
    init = model_init(D=D, constraint=constraint)
    data = jnp.zeros((N, d))

    solution_kf = run_kf(init=init, prior=prior, constraint=constraint, data=data, D=D)
    solution_ortho, Q = run_ortho(
        init=init, prior=prior, constraint=constraint, data=data, D=D, d=d
    )
    solution_small, Q = run_small(
        init=init, prior=prior, constraint=constraint, data=data, D=D, d=d
    )

    print(solution_kf.mean, solution_kf.cov)
    print(solution_small.mean, solution_small.cov)



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


def run_kf(*, prior, constraint, init, data, D):
    step = functools.partial(filter_step, jnp.eye(D), prior, constraint)
    _, solution = jax.lax.scan(step, init=init, xs=data)
    # Assert constraint is satisfied
    for s1, s2 in zip(solution.mean, data):
        eps = jnp.sqrt(jnp.finfo(jnp.dtype(s1)).eps)
        assert jnp.allclose(constraint.linop @ s1, s2, rtol=eps, atol=eps)
    return solution


def run_ortho(*, prior, constraint, init, data, D, d):
    Q, _R = jnp.linalg.qr(constraint.linop.T, mode="complete")
    Q = Trafo(full=Q, para=Q[:, :d], perp=Q[:, d:])
    init, prior, constraint = coordinate_change_big(init, prior, constraint, Q=Q)

    step = functools.partial(filter_step, Q.full, prior, constraint)
    _, solution_ortho = jax.lax.scan(step, init=init, xs=data)
    return solution_ortho, Q


def coordinate_change_big(init, prior, constraint, *, Q: Trafo):
    E_perp = Q.perp.T @ Q.full
    E_para = Q.para.T @ Q.full

    m, C = init
    m = Q.full.T @ m
    C = Q.full.T @ C @ Q.full
    init = RV(m, C)

    linop, cov = prior
    linop = Q.full.T @ linop @ Q.full
    cov = Q.full.T @ cov @ Q.full
    prior_ = Transition(linop, cov)

    linop, cov = constraint
    constraint = Transition(linop @ Q.full, cov)
    return init, prior_, constraint


def run_small(*, prior, constraint, init, data, D, d):
    # Change coordinates with reduction
    prior_small = prior
    constraint_small = constraint
    init_small = init
    data_small = data

    Q, _R = jnp.linalg.qr(constraint_small.linop.T, mode="complete")
    Q = Trafo(full=Q, para=Q[:, :d], perp=Q[:, d:])
    init_small, prior_small, constraint_small = coordinate_change_small(
        init_small, prior_small, constraint_small, Q=Q
    )
    step = functools.partial(filter_step, Q.perp, prior_small, constraint_small)
    _, solution_small = jax.lax.scan(step, init=init_small, xs=data_small)
    return solution_small, Q


def coordinate_change_small(init, prior, constraint, *, Q: Trafo):
    E_perp = Q.perp.T @ Q.full
    E_para = Q.para.T @ Q.full


    # Transform initial condition
    m, C = init
    m = Q.perp.T @ m
    C = Q.perp.T @ C @ Q.perp
    init = RV(m, C)

    # Transform rest
    linop, cov = prior
    linop = Q.full.T @ linop @ Q.full
    cov = Q.full.T @ cov @ Q.full
    S = E_para @ cov @ E_para.T
    K = E_perp @ cov @ E_para.T @ jnp.linalg.inv(S)
    cov = E_perp @ cov @ E_perp.T - K @ S @ K.T

    Phi_perp = Q.perp.T @ prior.linop @ Q.perp
    Phi_para = Q.para.T @ prior.linop @ Q.perp
    prior_ = Transition(Phi_perp - K @ Phi_para, cov)
    constraint = Transition(Phi_para, S)
    return init, prior_, constraint


def filter_step(Q, prior: Transition, constraint: Transition, rv: RV, data):
    rv = filter_predict(rv, prior)
    # print("predicted", Q @ rv.mean)
    rv = filter_condition(rv, constraint, data)
    # print("updated", Q @ rv.mean)
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
