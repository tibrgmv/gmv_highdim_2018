"""Study 6. Which lever lowers the risk of a GMV portfolio: the estimator, the history, or the volatility model?

Every combination is backtested on Kenneth French's 100 size x book-to-market portfolios (p = 95, 1963-2026), the 49
industries (1971-2026) and S&P 500 stocks (100 random portfolios of p = 150, 2013-2018). Monthly rebalancing,
weights drift in between, returns evaluated daily, one evaluation period per data set.
    python -m studies.study6_levers check        do the paper's and BPT's formulas survive devolatilisation?
    python -m studies.study6_levers run ff100    every configuration (also ind49, sp500)
    python -m studies.study6_levers weekly       the lever configurations, rebalanced every 5 days
    python -m studies.study6_plot                tables and figures
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time
import numpy as np
import pandas as pd
from gmv.backtest import Past, hold, turnover, rule
from gmv.data import results_dir, french, sp500_returns
from gmv.estimators import (sample_cov, gmv, pinv, bona_fide, oracle, ledoit_wolf, lw_nonlinear, lw_nonlinear_cov,
                            inverse_variance, ridge_multi_target, ew_weights, weighted_cov, ledoit_wolf_weighted, ewma_vol)
from gmv.theory import relative_loss, gmv_variance

OUT = results_dir(6)
WINDOWS = (63, 126, 252, 504, 1008)
HALFLIVES = (63, 126, 252)
VOLS = {"none": None, "RiskMetrics": 0.94, "half-life 60": 0.5 ** (1 / 60)}
MAIN_VOLS = ("none", "RiskMetrics")
EST = ("1/N", "inverse variance", "sample", "paper", "BPT 2024", "LW linear", "LW nonlinear")
COV_EST = ("sample", "LW linear", "LW nonlinear")
INIT, WARM = 63, 126          # EWMA start, then warm-up before any standardised return is used
LOOK, REB = 252, 21           # the rule's look-back; rebalancing interval
SETS = {"ff100": {"windows": WINDOWS, "label": "100 size x book-to-market portfolios (p = 95)"},
        "ind49": {"windows": WINDOWS, "label": "49 industry portfolios"},
        "sp500": {"windows": WINDOWS[:-1], "label": "S&P 500 stocks, random portfolios of p = 150", "p": 150, "draws": 100}}


def configs(windows):
    """(estimator, history, volatility); history is a window length or 'EW<half-life>'."""
    cfg = [("1/N", "-", "none")]
    for v in VOLS:
        for e in EST[1:]:
            for h in list(windows) + ([f"EW{h}" for h in HALFLIVES] if e in COV_EST else []):
                cfg.append((e, h, v))
    return cfg


def label(c):
    e, h, v = c
    return "1/N" if e == "1/N" else f"{e}, {h if isinstance(h, str) else f'n = {h}'}, vol: {v}"


def estimate(e, Z, d, om):
    """Weights from the window Z (standardised if d, today's volatility, is given); om = observation weights.
    Covariance estimators forecast D C D. The paper's and BPT's estimators are applied to Z·D, which solves
    min v'S_z v subject to a'v = 1, a = D⁻¹1, with w = D⁻¹v (tests/test_study6.py)."""
    L, p = Z.shape
    dd = np.ones(p) if d is None else d
    ew = np.full(p, 1 / p)
    if e == "inverse variance":
        return inverse_variance(dd ** 2 * Z.var(axis=0))
    if e in ("paper", "BPT 2024"):
        S = sample_cov(Z * dd)
        if e == "paper":
            return bona_fide(S, L, ew, np.linalg.inv(S) if p < L else pinv(S))[0]
        return ridge_multi_target(S, L, [ew])[0]
    if om is None:
        C = {"sample": sample_cov, "LW linear": ledoit_wolf, "LW nonlinear": lw_nonlinear}[e](Z)
    elif e == "sample":
        C = weighted_cov(Z, om)
    elif e == "LW linear":
        C = ledoit_wolf_weighted(Z, om)
    else:
        C = lw_nonlinear_cov(weighted_cov(Z, om), round(1 / (om ** 2).sum()))
    Sig = C * np.outer(dd, dd)
    singular = e == "sample" and om is None and L <= p
    return gmv(pinv(Sig) if singular else np.linalg.inv(Sig))


def weights_at(R, sigs, t, cfg, windows):
    """Weights of every configuration for day t, from rows before t only."""
    maxw = max(windows)
    past = Past(R, t, maxw)
    p = R.shape[1]
    W = np.empty((len(cfg), p))
    cache = {}
    for k, (e, h, v) in enumerate(cfg):
        if e == "1/N":
            W[k] = 1 / p
            continue
        if (h, v) not in cache:
            L = h if isinstance(h, int) else maxw
            Y = past.last(L)
            if v == "none":
                cache[(h, v)] = (Y, None)
            else:
                s = sigs[v]
                assert t - L >= INIT + WARM, "volatility not warmed up"
                cache[(h, v)] = (Y / s[t - L:t], s[t])
        Z, d = cache[(h, v)]
        W[k] = estimate(e, Z, d, None if isinstance(h, int) else ew_weights(len(Z), int(h[2:])))
    return W


_G = {}


def _init(*args):
    _G["args"] = args


def _weights(t):
    R, sigs, cfg, windows = _G["args"]
    return weights_at(R, sigs, t, cfg, windows)


def backtest(R, cfg, windows, reb=REB, end=None, workers=None):
    """Returns daily returns (K x days from t_start), weights and drifted holdings at each rebalance, t_start."""
    T, p = R.shape
    sigs = {v: ewma_vol(R, lam, INIT) for v, lam in VOLS.items() if lam}
    t_start = INIT + WARM + max(windows)
    dates = list(range(t_start, T if end is None else end, reb))
    if workers == 1:
        Ws = [weights_at(R, sigs, t, cfg, windows) for t in dates]
    else:
        from multiprocessing import Pool
        with Pool(workers, initializer=_init, initargs=(R, sigs, cfg, windows)) as pool:
            Ws = pool.map(_weights, dates, chunksize=2)
    rets = np.full((len(cfg), T - t_start), np.nan)
    Dend = []
    for t, W in zip(dates, Ws):
        stop = min(t + reb, T)
        rets[:, t - t_start:stop - t_start], d = hold(W, R[t:stop])
        Dend.append(d)
    return rets, np.array(Ws), np.array(Dend), t_start


def rules(cfg):
    """Candidates of the past-data rule: per estimator (its histories, vol none or RiskMetrics), and everything."""
    out = {f"rule: {e}": [k for k, c in enumerate(cfg) if c[0] == e and c[2] in MAIN_VOLS] for e in EST[1:]}
    out["rule: everything"] = [k for k, c in enumerate(cfg) if c[2] in MAIN_VOLS]
    return out


def run_one(R, windows, reb=REB, workers=None, cfg=None):
    """Every configuration (and the rules, for the full grid), on the evaluation days: LOOK days after the first rebalance."""
    with_rules = cfg is None
    cfg = configs(windows) if cfg is None else cfg
    rets, Ws, Dend, t_start = backtest(R, cfg, windows, reb, workers=workers)
    rr, tt, names = [rets], [turnover(Ws, Dend)], [label(c) for c in cfg]
    for name, cands in (rules(cfg).items() if with_rules else []):
        r, t, _ = rule(rets, Ws, Dend, cands, reb, LOOK)
        rr.append(r[None])
        tt.append(t[None])
        names.append(name)
    keep = -(-LOOK // reb)
    return names, np.vstack(rr)[:, LOOK:], np.vstack(tt)[:, keep:], t_start + LOOK, keep * reb - LOOK


def load(name):
    return sp500_returns() if name == "sp500" else french(name)


def draws(name, N):
    if name != "sp500":
        return [np.arange(N)]
    rng = np.random.default_rng(66)
    return [np.sort(rng.choice(N, SETS[name]["p"], replace=False)) for _ in range(SETS[name]["draws"])]


def run(name, reb=REB, cfg=None, tag=""):
    t0 = time.time()
    df = load(name)
    R = df.to_numpy()
    G, U = [], []
    for i, idx in enumerate(draws(name, R.shape[1])):
        names, rets, turn, t_eval, first = run_one(R[:, idx], SETS[name]["windows"], reb, cfg=cfg)
        G.append(rets.astype(np.float32))
        U.append(turn.astype(np.float32))
        print(name, f"portfolio {i + 1}", f"{time.time() - t0:.0f} s", flush=True)
    dates = df.index[t_eval:]
    np.savez_compressed(f"{OUT}/{name}{tag}.npz", names=np.array(names), dates=np.array([str(d.date()) for d in dates]),
                        gross=np.array(G), turn=np.array(U), reb=reb, first=first)
    print(f"{name}: evaluation {dates[0].date()} to {dates[-1].date()} ({len(dates)} days)")


def check():
    """GARCH(1,1) volatility per asset with the correlations of the 100 portfolios: the paper's α and BPT's variance
    estimate against their oracles under the true covariance of the next day, with and without devolatilisation.
    Devolatilisation can break the theory because σ_t comes from the same window and is itself estimated."""
    rng = np.random.default_rng(61)
    R = load("ff100").to_numpy()
    sig = ewma_vol(R, VOLS["RiskMetrics"], INIT)[:len(R)]
    C = np.corrcoef((R / sig)[INIT + WARM:], rowvar=False)
    p = C.shape[0]
    lam, V = np.linalg.eigh(C)
    root = (V * np.sqrt(np.clip(lam, 0, None))) @ V.T
    ga, gb, burn = 0.06, 0.93, 300
    ew = np.full(p, 1 / p)
    rows = []
    for n in (63, 126, 252):
        for _ in range(100):
            Tn = INIT + WARM + n
            h = np.ones(p)
            Rs = np.empty((burn + Tn, p))
            for s in range(burn + Tn):
                Rs[s] = np.sqrt(h) * (rng.standard_normal(p) @ root)
                h = (1 - ga - gb) + ga * Rs[s] ** 2 + gb * h
            Rs = Rs[burn:]
            Sig = C * np.sqrt(np.outer(h, h))
            v = gmv_variance(Sig)
            s_hat = ewma_vol(Rs, VOLS["RiskMetrics"], INIT)
            Y = Rs[Tn - n:]
            for mode, X in (("no devolatilisation", Y), ("devolatilised (RiskMetrics)", Y / s_hat[Tn - n:Tn] * s_hat[Tn])):
                S = sample_cov(X)
                Sinv = np.linalg.inv(S) if p < n else pinv(S)
                w, a = bona_fide(S, n, ew, Sinv)
                wo, ao = oracle(Sinv, Sig, ew)
                wb, (e, beta, lhat) = ridge_multi_target(S, n, [ew])
                rows.append({"n": n, "c": round(p / n, 2), "mode": mode, "alpha estimated": a, "alpha oracle": ao,
                             "paper: loss": relative_loss(w, Sig, v), "paper oracle: loss": relative_loss(wo, Sig, v),
                             "BPT: estimated / true variance": lhat / (wb @ Sig @ wb), "BPT: loss": relative_loss(wb, Sig, v)})
    out = pd.DataFrame(rows).groupby(["n", "c", "mode"]).agg(["mean", "median"]).reset_index()
    out.columns = [" ".join(x).strip() for x in out.columns]
    out.to_csv(f"{OUT}/check.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "run":
        run(sys.argv[2])
    elif cmd == "weekly":
        run("ff100", reb=5, tag="_weekly",
            cfg=[c for c in configs(WINDOWS) if c[2] in MAIN_VOLS and (c[1] == 252 or c[0] in ("sample", "LW nonlinear", "1/N"))])
