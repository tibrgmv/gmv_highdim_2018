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
