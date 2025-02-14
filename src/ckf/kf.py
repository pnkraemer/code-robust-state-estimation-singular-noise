import jax.numpy as jnp


def kalman_filter():
    def kf(data, /, *, init, latent, observe):
        n, d = data.shape
        ms = jnp.zeros((n, d))
        cs = jnp.zeros((n, d, d))

        rv = init
        for i, d in enumerate(data):
            rv = predict(rv, latent)
            rv = update(rv, d, observe)

            ms = ms.at[i].set(rv[0])
            cs = cs.at[i].set(rv[1])
        return (ms, cs)

    return kf



def predict(rv, model):
    A, Q = model
    m, C = rv

    m = A @ m
    C = A @ C @ A.T + Q
    return m, C

def update(rv, data, model):
    A, Q = model
    m, C = rv

    z = A @ m
    S = A @ C @ A.T + Q
    K = C @ A.T @ jnp.linalg.inv(S)

    m = m - K @ (z - data)
    C = C - K @ A @ C
    return (m, C)