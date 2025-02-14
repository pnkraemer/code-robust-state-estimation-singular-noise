import jax.numpy as jnp


def kalman_filter():
    def kf(data, /, *, init, latent, observe):
        n, d = data.shape

        ms = jnp.zeros((n, d))
        cs = jnp.zeros((n, d, d))
        return (ms, cs)

    return kf
