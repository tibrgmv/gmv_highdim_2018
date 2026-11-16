"""
Shared helpers for the studies.

  make_sigma(p, kind)        population covariance matrices used in the paper's simulations
  simulate(Sigma, n, dist)   an n x p sample of returns with covariance Sigma (normal or Student-t, 5 df)
  sample_cov(Y)              the paper's sample covariance (2.2): divisor n, demeaned
  traditional(S)             sample GMV weights; Moore-Penrose inverse when S is singular (c > 1)
  bona_fide(S, n, b)         the paper's bona fide shrinkage estimator (2.29)-(2.31)
  oracle(S_inv, Sigma, b)    the oracle shrinkage estimator (2.10)/(2.19), given the inverse it should use
  frahm_memmel(S, n)         the benchmark (3.1), defined for c < 1 only
  relative_loss(w, Sigma)    (w'Σw - σ²_GMV) / σ²_GMV, the paper's performance measure (3.2)
  theory_*                   the asymptotic results of Theorems 2.1 and 2.2
  plot_style()               chart style shared by all figures
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")

# Chart style: matches the presentation (warm white, ink, one red accent, handwritten notes)
SURFACE, INK, MUTED, GRIDC, EDGE = "#FAFAF8", "#1E1E1E", "#6B6862", "#ECE8E0", "#CFC9BE"
RED, NOTE, WARM, DARK, LIGHT = "#B5232A", "#C8372D", "#A39C90", "#5E5A55", "#CBC4B8"
GREY = WARM
PALETTE = [INK, RED, WARM, DARK]
# One colour per estimator, the same in every figure: the paper's estimator is the only red one
COLOURS = {"traditional": WARM, "Frahm-Memmel": DARK, "oracle": INK, "bona fide": RED, "equal weights": LIGHT}
STYLES = {"traditional": "-", "Frahm-Memmel": "-", "oracle": "-", "bona fide": "--", "equal weights": ":"}
SLIDE = (12, 5.4)            # one chart per slide: the slide's image area is 2.25 : 1
FONTS = os.path.join(ROOT, "fonts")


def plot_style():
    """Caladea for text, Caveat for handwritten notes (both in fonts/, SIL Open Font License)."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from cycler import cycler
    for f in os.listdir(FONTS):
        if f.endswith(".ttf"):
            font_manager.fontManager.addfont(os.path.join(FONTS, f))
    plt.rcParams.update({
        "font.family": "Caladea", "mathtext.fontset": "stix", "font.size": 14,
        "axes.prop_cycle": cycler(color=PALETTE), "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": EDGE, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 1.0, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.titlesize": 17, "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 14,
        "lines.linewidth": 2.6, "lines.markersize": 7, "legend.frameon": False, "legend.fontsize": 13,
    })


def note(ax, x, y, text, size=24, ha="left", **kw):
    """A handwritten red note, as on the slides."""
    from matplotlib.font_manager import FontProperties
    fp = FontProperties(fname=os.path.join(FONTS, "Caveat.ttf"), size=size)
    return ax.text(x, y, text, fontproperties=fp, color=NOTE, ha=ha, va="center", rotation=-2, linespacing=0.95, **kw)


def arrow(ax, start, end, rad=0.3):
    """A curved hand-drawn arrow from a note to what it points at."""
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", color=NOTE, lw=1.8,
                                 arrowstyle="-|>,head_length=8,head_width=4.5", shrinkA=3, shrinkB=5))


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.15)


# ----------------------------------------------------------------------------
# Population covariance matrices
# ----------------------------------------------------------------------------
def haar(p, rng):
    """A random orthogonal matrix from the Haar measure (QR of a Gaussian matrix with the sign fix)."""
    q, r = np.linalg.qr(rng.standard_normal((p, p)))
    return q * np.sign(np.diag(r))


def spectrum(p, kind):
    """Eigenvalues of the paper's designs.
    'fig1'      : 20% equal to 3, 40% to 1, 40% to 0.5 (Figures 1-4)
    'bounded'   : 1/9 equal to 2, 4/9 to 5, 4/9 to 10 (Figure 5, Section 3.1)
    'unbounded' : as 'bounded', but the largest eigenvalue is p (Section 3.2, a one-factor structure)"""
    if kind == "fig1":
        k1, k2 = round(0.2 * p), round(0.4 * p)
        return np.r_[np.full(k1, 3.0), np.full(k2, 1.0), np.full(p - k1 - k2, 0.5)]
    k1, k2 = round(p / 9), round(4 * p / 9)
    lam = np.r_[np.full(k1, 2.0), np.full(k2, 5.0), np.full(p - k1 - k2, 10.0)]
    if kind == "unbounded":
        lam[-1] = float(p)
    return lam


def make_sigma(p, kind, rng):
    """Σ = V diag(λ) V' with Haar eigenvectors V. Returns Σ, Σ^{1/2}, Σ^{-1/2}."""
    lam, V = spectrum(p, kind), haar(p, rng)
    Sigma = (V * lam) @ V.T
    root = (V * np.sqrt(lam)) @ V.T
    inv_root = (V / np.sqrt(lam)) @ V.T
    return Sigma, root, inv_root


def simulate(root, n, rng, dist="normal"):
    """Returns Y (n x p) = X Σ^{1/2}, with X i.i.d. zero mean, unit variance: normal, or t with 5 df rescaled."""
    p = root.shape[0]
    if dist == "normal":
        X = rng.standard_normal((n, p))
    else:
        X = rng.standard_t(5, size=(n, p)) * np.sqrt(3 / 5)
    return X @ root, X


# ----------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------
def sample_cov(Y):
    """The paper's (2.2): (1/n) Y'(I - 11'/n) Y, for Y with observations in rows."""
    Yc = Y - Y.mean(axis=0)
    return Yc.T @ Yc / Y.shape[0]


def gmv(Sinv):
    """Weights Sinv 1 / 1'Sinv 1 for a given (generalised) inverse."""
    w = Sinv.sum(axis=1)
    return w / w.sum()


def pinv(S):
    """Moore-Penrose inverse. A singular S has eigenvalues of order 1e-16 that are zero in exact arithmetic;
    numpy's default cut-off can treat some of them as non-zero and invert them, which blows up the weights.
    We cut at 1e-10 of the largest eigenvalue."""
    return np.linalg.pinv(S, rcond=1e-10, hermitian=True)


def inverse(S, n=None):
    """The inverse the bona fide estimator uses: S^{-1} if S is invertible (n > p), else the Moore-Penrose inverse (2.29)."""
    p = S.shape[0]
    if n is not None:
        return np.linalg.inv(S) if n > p + 1 else pinv(S)
    return pinv(S) if np.linalg.matrix_rank(S) < p else np.linalg.inv(S)


def traditional(S):
    return gmv(inverse(S))


def bona_fide(S, n, b=None, Sinv=None):
    """The bona fide optimal shrinkage estimator (2.29)-(2.31). Returns weights and the estimated intensity."""
    p = S.shape[0]
    c = p / n
    b = np.full(p, 1 / p) if b is None else b
    Sinv = inverse(S) if Sinv is None else Sinv
    one_Sinv_one = Sinv.sum()
    bSb = b @ S @ b
    if c < 1:
        R = (1 - c) * bSb * one_Sinv_one - 1
        alpha = (1 - c) * R / (c + (1 - c) * R)
    else:
        R = c * (c - 1) * bSb * one_Sinv_one - 1
        alpha = (c - 1) * R / ((c - 1) ** 2 + c + (c - 1) * R)
    return alpha * gmv(Sinv) + (1 - alpha) * b, alpha


def oracle(Sinv, Sigma, b=None):
    """The oracle optimal shrinkage estimator: intensity (2.10) or (2.19) computed with the true Σ."""
    p = Sigma.shape[0]
    b = np.full(p, 1 / p) if b is None else b
    w = gmv(Sinv)
    d = b - w
    alpha = (d @ Sigma @ b) / (d @ Sigma @ d)
    return alpha * w + (1 - alpha) * b, alpha


def s_star(X, inv_root):
    """The oracle generalised inverse (2.16), X with observations in rows.
    The paper writes Σ^{-1/2} (X'X)^+ Σ^{-1/2}; that is a generalised inverse of n·S, not of S (S* S S* = S*/n).
    We include the factor n, which Theorem 2.3(b) needs; the GMV weights do not depend on it."""
    return X.shape[0] * inv_root @ pinv(X.T @ X) @ inv_root


def frahm_memmel(S, n):
    """The dominating estimator (3.1) of Frahm and Memmel (2010); only defined when S is invertible."""
    p = S.shape[0]
    Sinv = np.linalg.inv(S)
    w = gmv(Sinv)
    var_gmv = 1 / Sinv.sum()                       # sample variance of the sample GMV portfolio
    R = (S.sum() / p ** 2 - var_gmv) / var_gmv     # estimated relative loss of the naive portfolio
    k = (p - 3) / (n - p + 2) / R
    return (1 - k) * w + k / p


def relative_loss(w, Sigma, var_gmv):
    return (w @ Sigma @ w - var_gmv) / var_gmv


def gmv_variance(Sigma):
    return 1 / np.linalg.solve(Sigma, np.ones(Sigma.shape[0])).sum()


# ----------------------------------------------------------------------------
# The asymptotic results (Theorems 2.1, 2.2 and Corollaries 2.1, 2.2)
# ----------------------------------------------------------------------------
def theory_loss_traditional(c):
    """Relative loss of the traditional estimator: c/(1-c) for c < 1, (c^2-c+1)/(c-1) for c > 1."""
    c = np.asarray(c, float)
    return np.where(c < 1, c / (1 - c), (c ** 2 - c + 1) / (c - 1))


def theory_alpha(c, Rb):
    c = np.asarray(c, float)
    return np.where(c < 1, (1 - c) * Rb / (c + (1 - c) * Rb), (c - 1) * Rb / ((c - 1) ** 2 + c + (c - 1) * Rb))


def theory_loss_shrinkage(c, Rb):
    """α² R_S + (1-α)² R_b (Corollaries 2.1(b), 2.2(b))."""
    a = theory_alpha(c, Rb)
    return a ** 2 * theory_loss_traditional(c) + (1 - a) ** 2 * Rb


def target_loss(Sigma, b=None):
    """R_b: the relative loss of the target portfolio b (equal weights by default)."""
    p = Sigma.shape[0]
    b = np.full(p, 1 / p) if b is None else b
    v = gmv_variance(Sigma)
    return (b @ Sigma @ b - v) / v


# ----------------------------------------------------------------------------
# Study 5: a ridge base and several targets
# ----------------------------------------------------------------------------
def make_sigma_market(p, rng):
    """A one-factor market: Σ = ββ' + diag(d), betas U(0.5, 1.5), idiosyncratic variances U(1, 4).
    Unlike the Haar designs, the top eigenvector is close to equal weights (the market), the largest eigenvalue
    grows like p, and stocks differ in risk, so the equal-weight target is far from the GMV portfolio."""
    beta, d = rng.uniform(0.5, 1.5, p), rng.uniform(1, 4, p)
    Sigma = np.outer(beta, beta) + np.diag(d)
    lam, V = np.linalg.eigh(Sigma)
    return Sigma, (V * np.sqrt(lam)) @ V.T, (V / np.sqrt(lam)) @ V.T


def inverse_variance(var):
    """Inverse-variance weights from a vector of variances."""
    w = 1 / var
    return w / w.sum()


def ledoit_wolf(Y):
    """Ledoit and Wolf (2004) linear shrinkage of the covariance matrix towards a scaled identity."""
    n, p = Y.shape
    Yc = Y - Y.mean(axis=0)
    S = Yc.T @ Yc / n
    mu = np.trace(S) / p
    d2 = ((S - mu * np.eye(p)) ** 2).sum() / p
    b2 = ((Yc ** 2).sum(axis=1) ** 2).sum() / n ** 2 / p - (S ** 2).sum() / n / p
    k = min(max(b2, 0), d2) / d2
    return k * mu * np.eye(p) + (1 - k) * S


def multi_target(S, n, targets, Sinv=None):
    """Shrinkage of the sample GMV portfolio towards several targets at once: w = w_S + Σ_k β_k (b_k - w_S).
    From the proofs of Theorems 2.1 and 2.2: w_S'Σw_S ≈ s σ², w_S'Σb ≈ σ² for every target, b_k'Σb_j ≈ b_k'S b_j,
    with s = 1/(1-c) (c < 1) or c²/(c-1) (c > 1). Hence β = σ²(s-1) M⁻¹1, M = B - 2σ² + sσ².
    With one target this is exactly the paper's bona fide estimator (2.29)-(2.31)."""
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


ETA_GRID = np.logspace(-3, 2, 41)       # ridge η relative to the average eigenvalue tr(S)/p


def ridge_terms(S, n, targets, etas=ETA_GRID, demeaned=True):
    """For each η: the ridge GMV weights w_η = (S+ηI)⁻¹1 / 1'(S+ηI)⁻¹1, and random-matrix estimates of their
    out-of-sample variance w_η'Σw_η and of w_η'Σb_k, valid for c < 1 and c > 1. Derivation (deterministic
    equivalents, (S+ηI)⁻¹ ≍ (xΣ+ηI)⁻¹ with x = 1 - c + cη m, m = tr(S+ηI)⁻¹/p):
        1'AΣb  ≈ (1 - η 1'Ab) / x
        1'AΣA1 ≈ (1'A1 - η 1'A²1) / D / (1 - c(1 - 2ηm + η² q₂)),
    with A = (S+ηI)⁻¹, a₂ = tr A²/p, x' = cm - cηa₂, D = x - ηx' = 1 - c + cη²a₂, q₂ = (x a₂ - x'm)/D.
    The theory needs targets independent of the data in S. With demeaned=True (the default) we use n - 1
    observations; demeaned=False uses n and S as given, the paper's convention.
    Returns η, weights (E x p), w'Σw (E), w'Σb (E x K), B (K x K)."""
    p = S.shape[0]
    ne = n - 1 if demeaned else n
    St = S * n / ne
    c = p / ne
    l, V = np.linalg.eigh(St)
    l = np.clip(l, 0, None)
    eta = etas * l.mean()
    o = V.sum(axis=0)                                    # V'1
    R = 1 / (l[None, :] + eta[:, None])                  # E x p
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
    ab = (R * o) @ (T @ V).T                             # E x K: 1'A b_k
    vwb = (1 - eta[:, None] * ab) / x[:, None] / a1[:, None]
    B = T @ St @ T.T
    return eta, W, vww, vwb, B


def best_combination(W, vww, vwb, B, T):
    """For each η the best β (w = w_η + Σ β_k (b_k - w_η)) and its loss.
    Returns (estimated loss, η index, β, weights) at the best η."""
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
    """Our estimator from precomputed ridge_terms(S, n, T): keep = indices of the targets to use. One
    eigendecomposition serves several target sets (for example BPT 2024 = ridge + 1/N alone)."""
    eta, W, vww, vwb, B = terms
    keep = list(keep)
    loss, e, beta, w = best_combination(W, vww, vwb[:, keep], B[np.ix_(keep, keep)], T[keep])
    return w, (e, beta, loss)


def ridge_multi_target(S, n, targets=(), Sigma=None, etas=ETA_GRID, demeaned=True):
    """Our estimator: a ridge GMV portfolio shrunk towards several targets; η and β minimise the random-matrix
    estimate of the out-of-sample variance. With one target (1/N) this is the method of Bodnar, Parolya and
    Thorsén (2024, JMLR). With Sigma given, the oracle: the same family, chosen with the true Σ.
    Returns weights and (η index, β, estimated loss)."""
    eta, W, vww, vwb, B = ridge_terms(S, n, targets, etas, demeaned)
    T = np.atleast_2d(targets) if len(targets) else np.zeros((0, S.shape[0]))
    if Sigma is not None:
        vww = np.einsum("ep,pq,eq->e", W, Sigma, W)
        vwb, B = W @ Sigma @ T.T, T @ Sigma @ T.T
    loss, e, beta, w = best_combination(W, vww, vwb, B, T)
    return w, (e, beta, loss)


def lw_nonlinear(Y):
    """Analytical nonlinear shrinkage of the covariance matrix: Ledoit and Wolf (2020), "Analytical nonlinear
    shrinkage of large-dimensional covariance matrices", Annals of Statistics 48(5), 3043-3065, Section 4
    (kernel estimate of the sample spectral density and its Hilbert transform, Epanechnikov kernel, locally
    adaptive bandwidth h·λ_j with h = n^(-1/3)). Data are demeaned, so n - 1 observations."""
    n0, p = Y.shape
    Yc = Y - Y.mean(axis=0)
    n = n0 - 1
    S = Yc.T @ Yc / n
    lam, U = np.linalg.eigh(S)
    k = min(p, n)
    lam = np.clip(lam[p - k:], 1e-300, None)                # the non-zero eigenvalues, ascending
    h = n ** (-1 / 3)
    H = h * lam[None, :]                                    # bandwidth of eigenvalue j, column j
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


def make_sigma_empirical(p, rng):
    """A neutral design: Σ = the sample covariance of p random S&P 500 stocks over the full 2013-2018 sample
    (1258 daily returns), scaled to an average variance of 1. Neither ours nor the paper's construction."""
    close = pd_read_close()
    R = close.dropna(axis=1).pct_change().iloc[1:].to_numpy()
    idx = rng.choice(R.shape[1], p, replace=False)
    Sigma = np.cov(R[:, idx], rowvar=False)
    Sigma /= np.trace(Sigma) / p
    lam, V = np.linalg.eigh(Sigma)
    return Sigma, (V * np.sqrt(lam)) @ V.T, (V / np.sqrt(lam)) @ V.T


def pd_read_close():
    import pandas as pd
    return pd.read_csv(os.path.join(DATA, "sp500_close.csv"), index_col=0, parse_dates=True)
