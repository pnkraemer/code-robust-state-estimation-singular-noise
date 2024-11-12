import jax
import jax.numpy as jnp
import functools


def main(N=10, n=3, d=2):
    key = jax.random.PRNGKey(1)

    As = jax.random.normal(key, shape=(N, n, n))
    Qs = jax.random.normal(key, shape=(N, n, n))
    Qs = jax.vmap(lambda v: jnp.dot(v, v.T))(Qs)

    Ds = jax.random.normal(key, shape=(N, n, d))
    Ds = jax.vmap(lambda v: jnp.linalg.qr(v)[0].T)(Ds)
    Rs = jnp.zeros((N, d, d))
    m0 = jax.random.normal(key, shape=(n,))
    C0 = jax.random.normal(key, shape=(n, n))

    init = (m0, C0)
    xs = (As, Qs, Ds, Rs)

    terminal, intermediate = jax.lax.scan(step, xs=xs, init=init)

    print(intermediate[1])

    init = init_reduce(*init, Ds[0])
    xs = ssm_reduce(xs)

    terminal, intermediate = jax.lax.scan(step, xs=xs, init=init)
    print(intermediate[1])


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

    V, R = jnp.linalg.qr(jnp.linalg.cholesky(Q).T @ U)
    ndim = R.shape[0]
    nobs = S.shape[0]
    R_perp = R[: (ndim - nobs), : (ndim - nobs)]
    R_para = R[(ndim - nobs) :, (ndim - nobs) :]
    R_star = R[: (ndim - nobs), (ndim - nobs) :]
    K = jnp.linalg.inv(R_perp) @ R_star

    U_para = U[:, : len(S)]
    U_perp = U[:, len(S) :]
    Phi_perp = U_perp.T @ U @ A @ U.T @ U_perp
    Phi_para = U_para.T @ U @ A @ U.T @ U_perp

    A_new = Phi_perp + K @ Phi_para
    D_new = Phi_para

    R_prior = jnp.linalg.inv(R_perp.T @ R_perp)
    R_obs = jnp.linalg.inv(R_para.T @ R_para)

    return (A_new, R_prior, D_new, R_obs)


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
