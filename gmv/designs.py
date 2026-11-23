"""Population covariance matrices for the simulations, and samples drawn from them."""
import numpy as np


def haar(p, rng):
    q, r = np.linalg.qr(rng.standard_normal((p, p)))
    return q * np.sign(np.diag(r))


def spectrum(p, kind):
    """'fig1': 20% at 3, 40% at 1, 40% at 0.5 (Figures 1-4). 'bounded': 1/9 at 2, 4/9 at 5, 4/9 at 10.
    'unbounded': as 'bounded' with the largest eigenvalue set to p (one factor)."""
    if kind == "fig1":
        k1, k2 = round(0.2 * p), round(0.4 * p)
        return np.r_[np.full(k1, 3.0), np.full(k2, 1.0), np.full(p - k1 - k2, 0.5)]
    k1, k2 = round(p / 9), round(4 * p / 9)
    lam = np.r_[np.full(k1, 2.0), np.full(k2, 5.0), np.full(p - k1 - k2, 10.0)]
    if kind == "unbounded":
        lam[-1] = float(p)
    return lam


def _roots(lam, V):
    return (V * np.sqrt(lam)) @ V.T, (V / np.sqrt(lam)) @ V.T


def make_sigma(p, kind, rng):
    """The paper's designs, Haar eigenvectors. Returns Σ, Σ^½, Σ^-½."""
    lam, V = spectrum(p, kind), haar(p, rng)
    return (V * lam) @ V.T, *_roots(lam, V)


def make_sigma_market(p, rng):
    """One-factor market: ββ' + diag(d), β ~ U(0.5, 1.5), d ~ U(1, 4)."""
    beta, d = rng.uniform(0.5, 1.5, p), rng.uniform(1, 4, p)
    Sigma = np.outer(beta, beta) + np.diag(d)
    return Sigma, *_roots(*np.linalg.eigh(Sigma))


def make_sigma_empirical(p, rng):
    """Neutral design: the sample covariance of p random S&P 500 stocks, 2013-2018, average variance 1."""
    from gmv.data import sp500_returns
    R = sp500_returns().to_numpy()
    Sigma = np.cov(R[:, rng.choice(R.shape[1], p, replace=False)], rowvar=False)
    Sigma /= np.trace(Sigma) / p
    return Sigma, *_roots(*np.linalg.eigh(Sigma))


def simulate(root, n, rng, dist="normal"):
    """n x p returns X Σ^½ with X normal or Student-t(5) at unit variance. Returns (Y, X)."""
    p = root.shape[0]
    X = rng.standard_normal((n, p)) if dist == "normal" else rng.standard_t(5, size=(n, p)) * np.sqrt(3 / 5)
    return X @ root, X
