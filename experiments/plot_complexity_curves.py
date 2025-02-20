import jax.numpy as jnp
import matplotlib.pyplot as plt

QR_CONST = 2 / 3
MATVEC_PROP = 0

def main():
    n = 100  # should cancel out
    m = n // 2
    r = n // 4

    reduced = flops_reduced(m=m, n=n, r=r)
    unreduced = flops_unreduced(m=m, n=n)
    print(reduced / unreduced)


def flops_reduced(m, n, r):
    qr = n**3 + 2 * (n - m + r) ** 3 + (n - m + 2 * r) ** 3
    bwd_subst = r * (m - r) ** 2 + r**3
    return QR_CONST * qr + bwd_subst


def flops_unreduced(m, n):
    qr = 2 * n**3 + (n + m) ** 3
    bwd_subst = n * m**2
    return QR_CONST * qr + bwd_subst


if __name__ == "__main__":
    main()
