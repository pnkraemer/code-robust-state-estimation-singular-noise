import jax
import jax.numpy as jnp


def main(N=10, n=3, d=3):
    key = jax.random.PRNGKey(1)

    As = jax.random.normal(key, shape=(N, n, n))
    Qs = jax.random.normal(key, shape=(N, n, n))
    Qs = jax.vmap(lambda v: jnp.dot(v, v.T))(Qs)

    Ds = jax.random.normal(key, shape=(N, n, d))
    Ds = jax.vmap(lambda v: jnp.linalg.qr(v)[0].T)(Ds)

    m0 = jax.random.normal(key, shape=(n,))
    C0 = jax.random.normal(key, shape=(n, n))

    init = (m0, C0)
    xs = (As, Qs, Ds)

    terminal, intermediate = jax.lax.scan(step, xs=xs, init=init)

    print(intermediate[1])


def step(rv, inputs):
    A, Q, D = inputs
    m, C = rv

    print(A.shape, Q.shape, D.shape, C.shape)
    mp = A @ m
    Cp = A @ C @ A.T + Q

    S = D @ Cp @ D.T
    K = Cp @ D.T @ jnp.linalg.inv(S)

    m = mp + K @ D @ mp

    C = Cp - K @ S @ K.T
    return (m, C), (m, C)


if __name__ == "__main__":
    main()
