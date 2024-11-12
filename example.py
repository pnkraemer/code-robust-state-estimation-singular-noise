import jax
import jax.numpy as jnp
import functools


def main(N=10, n=5, d=2):
    key = jax.random.PRNGKey(1)

    As = jax.random.normal(key, shape=(N, n, n))
    Qs = jax.random.normal(key, shape=(N, n, n))
    Qs = jax.vmap(lambda v: jnp.dot(v, v.T))(Qs)


    D = jax.random.normal(key, shape=(d, n))
    Ds = jnp.stack([D]*N)

    Rs = jnp.zeros((N, d, d))
    m0 = jax.random.normal(key, shape=(n,))
    C0 = jax.random.normal(key, shape=(n, n))

    init = (m0, C0)
    xs = (As, Qs, Ds, Rs)

    terminal, intermediate = jax.lax.scan(step, xs=xs, init=init)
    print(terminal[0])
    print(Ds[-1] @ terminal[0])

    # todo: assert constraint is satisfied


    # print(intermediate[1])

    init = init_reduce(*init, Ds[0])
    xs, (Q_para, Q_perp) = ssm_reduce(xs)

    terminal, intermediate = jax.lax.scan(step, xs=xs, init=init)

    # todo: assert it matches filter result from above

    print(Q_para.shape)
    print(Q_perp.shape)
    # print(terminal[0].shape)
    print(Q_perp[-1] @ terminal[0])


def init_reduce(m, C, D):
    _, S = jnp.linalg.qr(D.T)
    U, _ = jnp.linalg.qr(D.T, mode="complete")

    U_para = U[:, : len(S)]
    U_perp = U[:, len(S) :]

    m0_new = U_perp.T @ m
    C0_new = U_perp.T @ C @ U_perp
    return m0_new, C0_new


@jax.vmap
def ssm_reduce(inputs):
    # m, C = rv
    A, Q, D, _R = inputs

    n, p = D.shape

    _, S = jnp.linalg.qr(D.T)
    U, _ = jnp.linalg.qr(D.T, mode="complete")

    U_para = U[:, : len(S)]
    U_perp = U[:, len(S) :]

    E_para = U_para.T @ U
    E_perp = U_perp.T @ U 


    V, R = jnp.linalg.qr(jnp.linalg.inv(jnp.linalg.cholesky(Q).T) @ U)
    ndim = R.shape[0]
    nobs = S.shape[0]
    R_perp = E_perp @ R @ E_perp.T
    R_para = E_para @ R @ E_para.T
    R_star = E_perp @ R @ E_para.T
    K = jnp.linalg.inv(R_perp) @ R_star


    Phi_perp = E_perp @ A @E_perp.T
    Phi_para = E_para @ A @ E_perp.T

    A_new = Phi_perp + K @ Phi_para
    D_new = Phi_para

    R_prior = jnp.linalg.inv(R_perp.T @ R_perp)
    R_obs = jnp.linalg.inv(R_para.T @ R_para)

    return (A_new, R_prior, D_new, R_obs), (U_para, U_perp)


def step(rv, inputs):
    A, Q, D, R = inputs
    m, C = rv

    print(A.shape, Q.shape, D.shape, C.shape)
    mp = A @ m
    Cp = A @ C @ A.T + Q

    S = D @ Cp @ D.T + R
    K = Cp @ D.T @ jnp.linalg.inv(S)

    m = mp + K @ D @ mp

    C = Cp - K @ S @ K.T
    return (m, C), (m, C)


if __name__ == "__main__":
    main()
