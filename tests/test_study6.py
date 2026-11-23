"""Study 6: no look-ahead, the devolatilised GMV, drifting weights. python tests/test_study6.py"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gmv.backtest import hold, turnover, rule
from gmv.estimators import gmv, sample_cov, bona_fide, ewma_vol, ledoit_wolf, ledoit_wolf_weighted
import studies.study6_levers as s6


def test_backtest_never_reads_day_t_or_later():
    """Weights at a rebalance day t, including the EWMA volatility and the rule's choice, are unchanged when every
    row from t on is replaced by NaN. p = 30 with a 20-day window covers the pseudo-inverse and p > n branches."""
    rng = np.random.default_rng(5)
    T, p, windows = 620, 30, (20, 40)
    R = 0.01 * rng.standard_normal((T, p)) * np.exp(0.5 * np.sin(np.arange(T) / 40))[:, None]
    cfg = s6.configs(windows)
    t_start = s6.INIT + s6.WARM + max(windows)
    t = t_start + 14 * s6.REB
    R2 = R.copy()
    R2[t:] = np.nan
    for lam in (0.94, 0.5 ** (1 / 60)):
        assert np.array_equal(ewma_vol(R, lam)[:t + 1], ewma_vol(R2, lam)[:t + 1], equal_nan=True)
        assert np.isnan(ewma_vol(R2, lam)[t + 1]).all()
    out = [s6.backtest(X, cfg, windows, end=t + 1, workers=1) for X in (R, R2)]
    (r1, W1, D1, _), (r2, W2, D2, _) = out
    assert np.all(np.isfinite(W2)) and np.array_equal(W1, W2)
    s = t - t_start
    assert np.array_equal(r1[:, :s], r2[:, :s])
    for cands in s6.rules(cfg).values():
        c1, c2 = rule(r1, W1, D1, cands, s6.REB, s6.LOOK)[2], rule(r2, W2, D2, cands, s6.REB, s6.LOOK)[2]
        assert c1[-1] >= 0 and np.array_equal(c1, c2)


def test_devolatilised_gmv_is_the_gmv_of_DCD():
    """min v'Cv s.t. a'v = 1, a = D⁻¹1, then w = D⁻¹v: the GMV of DCD. And the paper's formulas with the vector a in
    place of 1 (and target D·1/N) give exactly the paper's estimator applied to the rescaled returns Z·D."""
    C = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 1.0]])
    d = np.array([0.5, 1.0, 2.0])
    a = 1 / d
    v = np.linalg.solve(C, a) / (a @ np.linalg.solve(C, a))
    w = v / d
    assert np.allclose(w / w.sum(), gmv(np.linalg.inv(C * np.outer(d, d))))
    rng = np.random.default_rng(6)
    for n in (40, 12):
        Z = rng.standard_normal((n, 10))
        d = rng.uniform(0.5, 2, 10)
        c = 10 / n
        a, bv = 1 / d, d / 10
        Sz = sample_cov(Z)
        Si = np.linalg.inv(Sz)
        Rb = (1 - c) * (bv @ Sz @ bv) * (a @ Si @ a) - 1
        alpha = (1 - c) * Rb / (c + (1 - c) * Rb)
        vS = Si @ a / (a @ Si @ a)
        w_a = (alpha * vS + (1 - alpha) * bv) / d
        assert np.allclose(w_a, s6.estimate("paper", Z, d, None))
        assert np.isclose(alpha, bona_fide(sample_cov(Z * d), n)[1])


def test_drifted_weights_two_assets():
    """Buy 60/40, returns (+10%, -5%) then (0%, +10%); rebalance to 50/50 afterwards."""
    W = np.array([[0.6, 0.4]])
    Rp = np.array([[0.10, -0.05], [0.0, 0.10]])
    r, end = hold(W, Rp)
    h1 = np.array([0.66, 0.38]) / 1.04
    assert np.allclose(r[0], [0.04, h1[1] * 0.10])
    assert np.allclose(end[0], np.array([0.66, 0.418]) / 1.078)
    Ws = np.array([W, [[0.5, 0.5]]])
    Dend = np.array([end, end])
    assert np.allclose(turnover(Ws, Dend)[0, 1], np.abs(np.array([0.5, 0.5]) - end[0]).sum())
    assert np.isclose(turnover(Ws, Dend)[0, 1], 2 * (0.66 / 1.078 - 0.5))


def test_weighted_ledoit_wolf_reduces_to_ledoit_wolf():
    Y = np.random.default_rng(7).standard_normal((50, 20))
    assert np.allclose(ledoit_wolf_weighted(Y, np.full(50, 1 / 50)), ledoit_wolf(Y))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok ", name)
