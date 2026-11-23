"""GMV weights and covariance estimators. Equation numbers refer to Bodnar, Parolya & Schmid (2018)."""
import numpy as np


def sample_cov(Y):
    """(2.2): demeaned, divisor n; observations in rows."""
    Yc = Y - Y.mean(axis=0)
    return Yc.T @ Yc / Y.shape[0]


def gmv(Sinv):
    w = Sinv.sum(axis=1)
    return w / w.sum()


def pinv(S):
    """Moore-Penrose inverse. numpy's default cut-off inverts some of the ~1e-16 eigenvalues of a singular S
    and blows up the weights, so we cut at 1e-10 of the largest."""
    return np.linalg.pinv(S, rcond=1e-10, hermitian=True)


def inverse(S):
    """S⁻¹, or S⁺ when S is singular (c > 1)."""
    return pinv(S) if np.linalg.matrix_rank(S) < S.shape[0] else np.linalg.inv(S)


def gmv_from_cov(C):
    return gmv(np.linalg.inv(C))


def inverse_variance(var):
    w = 1 / var
    return w / w.sum()


# The paper ------------------------------------------------------------------------------------------------

def bona_fide(S, n, b=None, Sinv=None):
    """Bona fide shrinkage (2.29)-(2.31). Returns weights and the estimated intensity α."""
    p = S.shape[0]
    c = p / n
    b = np.full(p, 1 / p) if b is None else b
    Sinv = inverse(S) if Sinv is None else Sinv
    q, bSb = Sinv.sum(), b @ S @ b
    if c < 1:
        R = (1 - c) * bSb * q - 1
        alpha = (1 - c) * R / (c + (1 - c) * R)
    else:
        R = c * (c - 1) * bSb * q - 1
        alpha = (c - 1) * R / ((c - 1) ** 2 + c + (c - 1) * R)
    return alpha * gmv(Sinv) + (1 - alpha) * b, alpha


def oracle(Sinv, Sigma, b=None):
    """Oracle shrinkage (2.10)/(2.19): the best α computed with the true Σ."""
    p = Sigma.shape[0]
    b = np.full(p, 1 / p) if b is None else b
    w = gmv(Sinv)
    d = b - w
    alpha = (d @ Sigma @ b) / (d @ Sigma @ d)
    return alpha * w + (1 - alpha) * b, alpha


def s_star(X, inv_root):
    """Oracle generalised inverse (2.16). As printed it inverts n·S, not S; we include the factor n, which
    Theorem 2.3(b) needs (the GMV weights do not depend on it)."""
    return X.shape[0] * inv_root @ pinv(X.T @ X) @ inv_root


def frahm_memmel(S, n):
    """Frahm & Memmel (2010), the benchmark (3.1); c < 1 only."""
    p = S.shape[0]
    Sinv = np.linalg.inv(S)
    v = 1 / Sinv.sum()
    R = (S.sum() / p ** 2 - v) / v
    k = (p - 3) / (n - p + 2) / R
    return (1 - k) * gmv(Sinv) + k / p


def multi_target(S, n, targets, Sinv=None):
    """The sample GMV shrunk towards several targets at once; with one target it is bona_fide."""
    p = S.shape[0]
    c = p / n
    Sinv = inverse(S) if Sinv is None else Sinv
    w = gmv(Sinv)
    if c < 1:
        s, sig2 = 1 / (1 - c), 1 / ((1 - c) * Sinv.sum())
    else:
        s, sig2 = c * c / (c - 1), 1 / (c * (c - 1) * Sinv.sum())
    T = np.atleast_2d(targets)
    M = T @ S @ T.T - 2 * sig2 + s * sig2
    beta = sig2 * (s - 1) * np.linalg.solve(M, np.ones(len(T)))
    return w + beta @ (T - w)


# Ridge base with targets (BPT 2024, and our several-target version) --------------------------------------

ETA_GRID = np.logspace(-3, 2, 41)       # η relative to tr(S)/p


def ridge_terms(S, n, targets, etas=ETA_GRID, demeaned=True):
    """For each η: ridge GMV weights w_η and random-matrix estimates of w_η'Σw_η and w_η'Σb_k (derivation in
    the README). Returns η, W (E x p), w'Σw (E), w'Σb (E x K), b'Sb (K x K)."""
    p = S.shape[0]
    ne = n - 1 if demeaned else n
    St = S * n / ne
    c = p / ne
    l, V = np.linalg.eigh(St)
    l = np.clip(l, 0, None)
    eta = etas * l.mean()
    o = V.sum(axis=0)
    R = 1 / (l[None, :] + eta[:, None])
    a1 = (o ** 2 * R).sum(axis=1)
    W = (R * o) @ V.T / a1[:, None]
    m, a2, a1sq = R.mean(axis=1), (R ** 2).mean(axis=1), (o ** 2 * R ** 2).sum(axis=1)
    x = 1 - c + c * eta * m
    xp = c * m - c * eta * a2
    D = x - eta * xp
    q2 = (x * a2 - xp * m) / D
    with np.errstate(divide="ignore", invalid="ignore"):
        vww = (a1 - eta * a1sq) / D / (1 - c * (1 - 2 * eta * m + eta ** 2 * q2)) / a1 ** 2
    T = np.atleast_2d(targets) if len(targets) else np.zeros((0, p))
    ab = (R * o) @ (T @ V).T
    vwb = (1 - eta[:, None] * ab) / x[:, None] / a1[:, None]
    return eta, W, vww, vwb, T @ St @ T.T


def best_combination(W, vww, vwb, B, T):
    """Over η, the best w = w_η + Σ β_k (b_k - w_η). Returns (estimated loss, η index, β, weights)."""
    best = (np.inf, None, None, None)
    for e in range(len(vww)):
        if not np.isfinite(vww[e]) or vww[e] <= 0:
            continue
        g = vww[e] - vwb[e]
        M = B - vwb[e][:, None] - vwb[e][None, :] + vww[e]
        beta = np.linalg.lstsq(M, g, rcond=1e-12)[0] if len(g) else g
        loss = vww[e] - g @ beta
        if loss < best[0]:
            best = (loss, e, beta, W[e] + beta @ (T - W[e]) if len(g) else W[e])
    return best


def ridge_choose(terms, T, keep):
    """Ridge + the targets T[keep], from precomputed ridge_terms. Returns weights and (η index, β, loss)."""
    eta, W, vww, vwb, B = terms
    keep = list(keep)
    loss, e, beta, w = best_combination(W, vww, vwb[:, keep], B[np.ix_(keep, keep)], T[keep])
    return w, (e, beta, loss)


def ridge_multi_target(S, n, targets=(), Sigma=None, etas=ETA_GRID, demeaned=True):
    """Ridge GMV shrunk towards the targets; with the single target 1/N this is BPT 2024. With Sigma given,
    the oracle choice in the same family."""
    eta, W, vww, vwb, B = ridge_terms(S, n, targets, etas, demeaned)
    T = np.atleast_2d(targets) if len(targets) else np.zeros((0, S.shape[0]))
    if Sigma is not None:
        vww = np.einsum("ep,pq,eq->e", W, Sigma, W)
        vwb, B = W @ Sigma @ T.T, T @ Sigma @ T.T
    loss, e, beta, w = best_combination(W, vww, vwb, B, T)
    return w, (e, beta, loss)


# Ledoit-Wolf --------------------------------------------------------------------------------------------

def ledoit_wolf(Y):
    """Linear shrinkage towards a scaled identity (Ledoit & Wolf 2004)."""
    n, p = Y.shape
    Yc = Y - Y.mean(axis=0)
    S = Yc.T @ Yc / n
    mu = np.trace(S) / p
    d2 = ((S - mu * np.eye(p)) ** 2).sum() / p
    b2 = ((Yc ** 2).sum(axis=1) ** 2).sum() / n ** 2 / p - (S ** 2).sum() / n / p
    k = min(max(b2, 0), d2) / d2
    return k * mu * np.eye(p) + (1 - k) * S


def ledoit_wolf_weighted(Y, w):
    """ledoit_wolf with observation weights w summing to 1."""
    p = Y.shape[1]
    Yc = Y - w @ Y
    S = Yc.T @ (w[:, None] * Yc)
    mu = np.trace(S) / p
    d2 = ((S - mu * np.eye(p)) ** 2).sum() / p
    q = (Yc ** 2).sum(axis=1)
    ySy = np.einsum("sp,pq,sq->s", Yc, S, Yc)
    b2 = (w ** 2 * (q ** 2 - 2 * ySy + (S ** 2).sum())).sum() / p
    k = min(max(b2, 0), d2) / d2
    return k * mu * np.eye(p) + (1 - k) * S


def lw_nonlinear(Y):
    """Analytical nonlinear shrinkage (Ledoit & Wolf 2020, Annals of Statistics 48(5), Section 4)."""
    Yc = Y - Y.mean(axis=0)
    n = Y.shape[0] - 1
    return lw_nonlinear_cov(Yc.T @ Yc / n, n)


def lw_nonlinear_cov(S, n):
    """The same from a covariance matrix with n effective observations."""
    p = S.shape[0]
    n = int(n)
    lam, U = np.linalg.eigh(S)
    k = min(p, n)
    lam = np.clip(lam[p - k:], 1e-300, None)
    h = n ** (-1 / 3)
    H = h * lam[None, :]
    x = (lam[:, None] - lam[None, :]) / H
    f = 3 / (4 * np.sqrt(5)) * np.mean(np.maximum(1 - x ** 2 / 5, 0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hf = -3 / (10 * np.pi) * x + 3 / (4 * np.sqrt(5) * np.pi) * (1 - x ** 2 / 5) * np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
    edge = np.isclose(np.abs(x), np.sqrt(5))
    hf[edge] = (-3 / (10 * np.pi) * x)[edge]
    Hf = np.mean(hf / H, axis=1)
    if p <= n:
        r = p / n
        d = lam / ((np.pi * r * lam * f) ** 2 + (1 - r - np.pi * r * lam * Hf) ** 2)
    else:
        Hf0 = (1 / np.pi) * (3 / (10 * h ** 2) + 3 / (4 * np.sqrt(5) * h) * (1 - 1 / (5 * h ** 2))
                             * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))) * np.mean(1 / lam)
        d0 = 1 / (np.pi * (p - n) / n * Hf0)
        d1 = lam / (np.pi ** 2 * lam ** 2 * (f ** 2 + Hf ** 2))
        d = np.r_[np.full(p - n, d0), d1]
    return (U * d) @ U.T


# Exponential weighting and volatility -------------------------------------------------------------------

def ew_weights(L, halflife):
    """Weights of the last L observations, oldest first, summing to 1."""
    w = 0.5 ** (np.arange(L)[::-1] / halflife)
    return w / w.sum()


def weighted_cov(Y, w):
    Yc = Y - w @ Y
    return Yc.T @ (w[:, None] * Yc)


def ewma_vol(R, lam, init=63):
    """RiskMetrics volatility: σ²[t] = λσ²[t-1] + (1-λ)r²[t-1], from data before t only. T + 1 rows, NaN before init."""
    from scipy.signal import lfilter
    T, p = R.shape
    s2 = np.full((T + 1, p), np.nan)
    s2[init] = (R[:init] ** 2).mean(axis=0)
    s2[init + 1:] = lfilter([1.0], [1.0, -lam], (1 - lam) * R[init:] ** 2, axis=0, zi=lam * s2[init][None, :])[0]
    return np.sqrt(s2)
