import jax
import jax.numpy as jnp
import functools
from typing import NamedTuple


class Transition(NamedTuple):
    linop: jax.Array
    cov: jax.Array


class RV(NamedTuple):
    mean: jax.Array
    cov: jax.Array


class Trafo(NamedTuple):
    para: jax.Array
    perp: jax.Array


def main(N=10, D=3, d=2):
    jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(5)
    key = jax.random.PRNGKey(1)

    Phi = jax.random.normal(key, shape=(D, D)) / jnp.sqrt(D)
    Sigma = Phi @ Phi.T + 1e5*jnp.eye(D)
    Phi = jnp.eye(D)
    # Sigma = jnp.eye(D)
    prior = Transition(Phi, Sigma)

    A = jax.random.normal(key, shape=(d, D))
    R = jnp.zeros((d, d))
    constraint = Transition(A, R)

    mean = jax.random.normal(key, shape=(D,))
    chol = jax.random.normal(key, shape=(D, D))
    cov = chol @ chol.T
    init = RV(mean, cov)
    init = condition(init, constraint, data=0.0)
    init_ref = init

    data = jnp.zeros((N, d))
    step = functools.partial(filter_step, prior, constraint)
    _, solution = jax.lax.scan(step, init=init, xs=data)
    solution_ref = solution

    for s in solution.mean:
        eps = jnp.sqrt(jnp.finfo(jnp.dtype(s)).eps)
        assert jnp.allclose(constraint.linop @ s, 0.0, rtol=eps, atol=eps)

    init, Q = reduce_rv(init, constraint.linop)

    assert jnp.allclose(Q.perp @ init.mean, init_ref.mean)
    assert jnp.allclose(Q.perp @ init.cov @ Q.perp.T, init_ref.cov)

    prior, constraint, _ = reduce_ssm(prior, constraint)

    data = jnp.zeros((N, d))
    step = functools.partial(filter_step, prior, constraint)
    _, solution = jax.lax.scan(step, init=init, xs=data)

    for m, c, m_ref, c_ref in zip(*solution, *solution_ref):
        print(Q.perp @ m)
        print(m_ref)
        print()
        print(Q.perp @ c @ Q.perp.T)
        print(c_ref)
        print()
        print()
        assert jnp.allclose(Q.perp @ m, m_ref)
        assert jnp.allclose(Q.perp @ c @ Q.perp.T, c_ref)


def filter_step(prior: Transition, constraint: Transition, rv: RV, data):
    m, C = rv

    m = prior.linop @ m
    C = prior.linop @ C @ prior.linop.T + prior.cov

    S = constraint.linop @ C @ constraint.linop.T + constraint.cov
    K = C @ constraint.linop.T @ jnp.linalg.inv(S)

    m = m - K @ (constraint.linop @ m - data)
    C = C - K @ S @ K.T
    return RV(m, C), RV(m, C)


def condition(rv, constraint, data):
    S = constraint.linop @ rv.cov @ constraint.linop.T + constraint.cov
    K = rv.cov @ constraint.linop.T @ jnp.linalg.inv(S)
    m = rv.mean - K @ (constraint.linop @ rv.mean - data)
    C = rv.cov - K @ S @ K.T
    return RV(m, C)


def reduce_rv(rv: RV, linop: jax.Array):
    D, d = jnp.shape(linop.T)

    U, S = jnp.linalg.qr(linop.T, mode="complete")

    U_para = U[:, :d]
    U_perp = U[:, d:]

    m, C = rv
    m0_new = U_perp.T @ m
    C0_new = U_perp.T @ C @ U_perp

    return RV(m0_new, C0_new), Trafo(perp=U_perp, para=U_para)


def reduce_ssm(prior: Transition, constraint: Transition):
    D, d = jnp.shape(constraint.linop.T)

    _, S = jnp.linalg.qr(constraint.linop.T)
    U, _ = jnp.linalg.qr(constraint.linop.T, mode="complete")

    U_para = U[:, :d]
    U_perp = U[:, d:]

    E_para = U_para.T @ U
    E_perp = U_perp.T @ U

    chol = jnp.linalg.inv(jnp.linalg.cholesky(prior.cov))
    V, R = jnp.linalg.qr(chol.T @ U)

    R_perp = E_perp @ R @ E_perp.T
    R_para = E_para @ R @ E_para.T
    R_star = E_perp @ R @ E_para.T

    K = jnp.linalg.inv(R_perp) @ R_star

    Phi_perp = U_perp.T @ prior.linop @ U_perp
    Phi_para = U_para.T @ prior.linop @ U_perp

    A_new = Phi_perp + K @ Phi_para
    D_new = Phi_para

    R_prior = jnp.linalg.inv(R_perp.T @ R_perp)
    R_obs = jnp.linalg.inv(R_para.T @ R_para)

    prior = Transition(Phi_perp, R_prior)
    constraint = Transition(Phi_para, R_obs)
    return prior, constraint, U_perp


if __name__ == "__main__":
    main()
