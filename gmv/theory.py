"""Population quantities and the asymptotic results of Theorems 2.1-2.2 and Corollaries 2.1-2.2."""
import numpy as np


def gmv_variance(Sigma):
    return 1 / np.linalg.solve(Sigma, np.ones(Sigma.shape[0])).sum()


def relative_loss(w, Sigma, var_gmv):
    """(w'Σw - σ²_GMV) / σ²_GMV, the paper's measure (3.2)."""
    return (w @ Sigma @ w - var_gmv) / var_gmv


def target_loss(Sigma, b=None):
    """R_b, the relative loss of the target (equal weights by default)."""
    p = Sigma.shape[0]
    b = np.full(p, 1 / p) if b is None else b
    v = gmv_variance(Sigma)
    return (b @ Sigma @ b - v) / v


def loss_traditional(c):
    """c/(1-c) for c < 1, (c²-c+1)/(c-1) for c > 1."""
    c = np.asarray(c, float)
    return np.where(c < 1, c / (1 - c), (c ** 2 - c + 1) / (c - 1))


def alpha(c, Rb):
    c = np.asarray(c, float)
    return np.where(c < 1, (1 - c) * Rb / (c + (1 - c) * Rb), (c - 1) * Rb / ((c - 1) ** 2 + c + (c - 1) * Rb))


def loss_shrinkage(c, Rb):
    a = alpha(c, Rb)
    return a ** 2 * loss_traditional(c) + (1 - a) ** 2 * Rb
