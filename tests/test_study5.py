"""
Quick checks of the Study 5 code (about 20 seconds). Run from the repository root:
    python tests/test_study5.py          (or: python -m pytest tests)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (make_sigma, simulate, sample_cov, gmv, pinv, bona_fide, multi_target, ridge_multi_target, lw_nonlinear,
                    relative_loss, gmv_variance)
import study5_ridge_multitarget as s5


def test_ridge_one_target_small_eta_is_bona_fide():
    """With one target, η -> 0 and the paper's convention (n observations), ridge_multi_target is (2.26)."""
    rng = np.random.default_rng(1)
    _, root, _ = make_sigma(60, "bounded", rng)
    for n in (120, 80):                                      # c = 0.5, 0.75
        S = sample_cov(simulate(root, n, rng)[0])
        b = np.full(60, 1 / 60)
        w_r = ridge_multi_target(S, n, [b], etas=np.array([1e-9]), demeaned=False)[0]
        w_p = bona_fide(S, n, b)[0]
        assert np.abs(w_r - w_p).max() < 1e-6, np.abs(w_r - w_p).max()


def test_multi_target_one_target_is_bona_fide():
    rng = np.random.default_rng(2)
    _, root, _ = make_sigma(60, "bounded", rng)
    for n in (120, 30):                                      # c = 0.5 and c = 2
        S = sample_cov(simulate(root, n, rng)[0])
        b = rng.dirichlet(np.ones(60))
        assert np.abs(multi_target(S, n, [b]) - bona_fide(S, n, b)[0]).max() < 1e-12


def test_nonlinear_shrinkage_beats_sample_covariance():
    """Frobenius loss and GMV loss, p = 100, n = 200 and n = 60 (pseudo-inverse for the sample covariance)."""
    rng = np.random.default_rng(3)
    Sigma, root, _ = make_sigma(100, "fig1", rng)
    v = gmv_variance(Sigma)
    for n in (200, 60):
        fro_s, fro_nl, gl_s, gl_nl = [], [], [], []
        for _ in range(10):
            Y = simulate(root, n, rng)[0]
            Yc = Y - Y.mean(axis=0)
            S, NL = Yc.T @ Yc / (n - 1), lw_nonlinear(Y)
            fro_s.append(np.linalg.norm(S - Sigma)); fro_nl.append(np.linalg.norm(NL - Sigma))
            gl_s.append(relative_loss(gmv(pinv(S)), Sigma, v)); gl_nl.append(relative_loss(gmv(np.linalg.inv(NL)), Sigma, v))
        assert np.mean(fro_nl) < np.mean(fro_s) and np.mean(gl_nl) < np.mean(gl_s), (n, np.mean(gl_nl), np.mean(gl_s))


def test_backtest_never_reads_day_t_or_later():
    """Weights for day t are unchanged when every row from t on is replaced by NaN; reading too far back fails."""
    rng = np.random.default_rng(4)
    R = rng.standard_normal((400, 30)) * 0.01
    for n in (60, 20):                                       # c = 0.5 and c = 1.5
        t = n + s5.PRIOR + 37
        W, _ = s5.weights_at(R, t, n)
        R2 = R.copy()
        R2[t:] = np.nan
        W2, _ = s5.weights_at(R2, t, n)
        assert np.all(np.isfinite(W2)) and np.array_equal(W, W2)
    try:
        s5.Past(R, 50, 30).last(31)
    except AssertionError:
        pass
    else:
        raise AssertionError("a read beyond the allowed window was not caught")
    try:
        s5.weights_at(R, s5.PRIOR, 30)                       # not enough history before t
    except AssertionError:
        pass
    else:
        raise AssertionError("a read before the first row was not caught")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok ", name)
