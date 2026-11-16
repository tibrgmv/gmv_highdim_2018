"""
Study 6 — Which lever lowers the risk of a minimum-variance portfolio?

A practitioner has three levers: (1) the estimator, (2) how much history to use, (3) whether to account for
changing volatility. We backtest every combination on long data without survivorship bias (Kenneth French's
100 size x book-to-market portfolios, p = 95, 1963-2026; 49 industries, 1971-2026) and on the S&P 500 stocks of
Study 3 (random portfolios of p = 150, 2013-2018), with one evaluation period per data set.

  estimators   1/N; inverse variance; sample covariance (pseudo-inverse when n <= p); the paper's bona fide
               estimator (target 1/N); BPT 2024 (ridge + 1/N); Ledoit-Wolf linear; Ledoit-Wolf nonlinear (2020)
  history      rolling windows of 63, 126, 252, 504, 1008 days; for the covariance-matrix estimators also
               exponentially weighted estimates with half-lives 63, 126, 252 days (over the last 1008 days)
  volatility   none; EWMA devolatilisation (RiskMetrics λ = 0.94; half-life 60 days as a robustness check):
               correlation structure from z = r / σ, forecast Σ_t = D_t C D_t, D_t = diag(σ_t)
  rule         at each rebalance, the window and volatility setting (none or RiskMetrics) with the lowest realised
               variance over the previous 252 days of out-of-sample returns; per estimator, and over everything

Weights are rebalanced every 21 trading days and drift with returns in between; returns are evaluated daily.

  python study6_levers.py check        do the paper's and BPT's formulas still hold after devolatilisation?
  python study6_levers.py run SET      every configuration; SET = ff100, ind49, sp500  -> results/study6_SET.npz
  python study6_levers.py weekly       the configurations of the lever decomposition, rebalanced every 5 days
  python study6_levers.py plot         tables and figures
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")             # small matrices, many processes: one thread each
import sys
import numpy as np
import pandas as pd
from common import (RESULTS, DATA, sample_cov, gmv, pinv, bona_fide, oracle, ledoit_wolf, lw_nonlinear, lw_nonlinear_cov,
                    inverse_variance, ridge_multi_target, relative_loss, gmv_variance, ew_weights, weighted_cov,
                    ledoit_wolf_weighted, ewma_vol)
from study5_ridge_multitarget import Past, nw_tstat

OUT = os.path.join(RESULTS, "study6")
WINDOWS = (63, 126, 252, 504, 1008)
HALFLIVES = (63, 126, 252)
VOLS = {"none": None, "RiskMetrics": 0.94, "half-life 60": 0.5 ** (1 / 60)}
MAIN_VOLS = ("none", "RiskMetrics")           # the rule chooses among these; half-life 60 is a robustness check
EST = ("1/N", "inverse variance", "sample", "paper", "BPT 2024", "LW linear", "LW nonlinear")
COV_EST = ("sample", "LW linear", "LW nonlinear")
INIT, WARM = 63, 126       # EWMA started on the first 63 days, then run 126 days before any standardised return is used
LOOK, REB = 252, 21        # the rule's look-back; rebalancing interval (both fixed in advance)
BLOCK, BOOT = 21, 1000     # block bootstrap over days
COSTS = (10, 25)           # proportional costs, basis points
SETS = {"ff100": {"windows": WINDOWS, "label": "100 size x book-to-market portfolios (p = 95)"},
        "ind49": {"windows": WINDOWS, "label": "49 industry portfolios"},
        "sp500": {"windows": WINDOWS[:-1], "label": "S&P 500 stocks, random portfolios of p = 150",
                  "p": 150, "draws": 100}}
PERIODS = {"full": None, "1968-1989": ("1968-01-01", "1989-12-31"), "1990-2007": ("1990-01-01", "2007-12-31"),
           "2008-today": ("2008-01-01", "2100-01-01"), "crisis 2000-2002": ("2000-01-01", "2002-12-31"),
           "crisis 2008-2009": ("2008-01-01", "2009-12-31"), "crisis 2020": ("2020-01-01", "2020-12-31")}


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
    if e == "1/N":
        return "1/N"
    return f"{e}, {h if isinstance(h, str) else f'n = {h}'}, vol: {v}"


# ----------------------------------------------------------------------------------------------------------
def estimate(e, Z, d, om):
    """Weights of estimator e from the (possibly standardised) window Z, oldest row first; d = today's volatility
    forecast σ_t (None: no devolatilisation); om = observation weights (None: a rolling window).
    Covariance estimators: C from Z, Σ_t = D C D. The paper's and BPT 2024 output weights: they are applied to the
    returns rescaled to today's volatility, Z·D, whose sample covariance is D S_z D; their GMV problem with the
    constraint 1'w = 1 is exactly min v'S_z v subject to a'v = 1, a = D⁻¹1, with w = D⁻¹v (see check)."""
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
    """Weights of every configuration for day t. Returns are read through a checked accessor (rows t - 1008, ...,
    t - 1 at most); volatility forecasts σ[s] use rows before s only (common.ewma_vol)."""
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
    """Rebalance on days t_start, t_start + reb, ... (before `end`); hold the weights and let them drift. Returns
    daily returns (K x days from t_start), weights and drifted end-of-period holdings at each rebalance (n_reb x K x p)."""
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


def hold(W, Rp):
    """Buy weights W (K x p) and hold them over the days in Rp (days x p) without rebalancing: daily portfolio
    returns (K x days) and the drifted weights at the end (K x p)."""
    G = np.cumprod(1 + Rp, axis=0)                             # growth of each asset since the purchase
    V = W @ G.T                                                # portfolio value, K x days
    return V / np.c_[np.ones(len(W)), V[:, :-1]] - 1, W * G[-1] / V[:, -1:]


def turnover(Ws, Dend):
    """Σ|w_new - drifted old| at each rebalance (K x n_reb), NaN at the first."""
    out = np.full(Ws.shape[:2], np.nan)
    out[1:] = np.abs(Ws[1:] - Dend[:-1]).sum(axis=2)
    return out.T


def rule(rets, Ws, Dend, cands, reb=REB):
    """At each rebalance at least LOOK days after the first, choose among `cands` the configuration with the
    lowest variance of its out-of-sample returns over the previous LOOK days (all before the rebalance day).
    Returns the rule's daily returns, its turnover at each rebalance, and the index chosen (-1: not yet)."""
    n = len(Ws)
    ch = np.full(n, -1)
    out = np.full(rets.shape[1], np.nan)
    turn = np.full(n, np.nan)
    for k in range(n):
        s = k * reb
        if s < LOOK:
            continue
        c = cands[int(np.argmin(rets[cands, s - LOOK:s].var(axis=1)))]
        ch[k] = c
        out[s:s + reb] = rets[c, s:s + reb]
        if ch[k - 1] >= 0:
            turn[k] = np.abs(Ws[k, c] - Dend[k - 1, ch[k - 1]]).sum()
    return out, turn, ch


def rules(cfg):
    """Candidate sets: per estimator (its windows and EW half-lives, volatility none or RiskMetrics), and all."""
    out = {f"rule: {e}": [k for k, c in enumerate(cfg) if c[0] == e and c[2] in MAIN_VOLS] for e in EST[1:]}
    out["rule: everything"] = [k for k, c in enumerate(cfg) if c[2] in MAIN_VOLS]
    return out


def run_one(R, windows, reb=REB, workers=None, cfg=None):
    """All configurations (and, for the full grid, the rules) on one return matrix; output restricted to the
    evaluation days, which start LOOK days after the first rebalance."""
    with_rules = cfg is None
    cfg = configs(windows) if cfg is None else cfg
    rets, Ws, Dend, t_start = backtest(R, cfg, windows, reb, workers=workers)
    turn = turnover(Ws, Dend)
    names = [label(c) for c in cfg]
    rr, tt = [rets], [turn]
    for name, cands in (rules(cfg).items() if with_rules else []):
        r, t, ch = rule(rets, Ws, Dend, cands, reb)
        rr.append(r[None]); tt.append(t[None]); names.append(name)
    rets, turn = np.vstack(rr), np.vstack(tt)
    keep = -(-LOOK // reb)                                   # first rebalance inside the evaluation period
    return names, rets[:, LOOK:], turn[:, keep:], t_start + LOOK, keep * reb - LOOK


def load(name):
    if name == "sp500":
        from study5_ridge_multitarget import load_returns
        R, dates = load_returns()
        return pd.DataFrame(R, index=dates)
    return pd.read_csv(os.path.join(DATA, f"{name}_daily.csv"), index_col=0, parse_dates=True)


def run(name, reb=REB, cfg=None, tag=""):
    import time
    t0 = time.time()
    df = load(name)
    S = SETS[name]
    R = df.to_numpy()
    if name == "sp500":
        rng = np.random.default_rng(66)
        draws = [np.sort(rng.choice(R.shape[1], S["p"], replace=False)) for _ in range(S["draws"])]
    else:
        draws = [np.arange(R.shape[1])]
    G, U = [], []
    for i, idx in enumerate(draws):
        names, rets, turn, t0_eval, first = run_one(R[:, idx], S["windows"], reb, cfg=cfg)
        G.append(rets.astype(np.float32)); U.append(turn.astype(np.float32))
        print(name, f"portfolio {i + 1}/{len(draws)}", f"{time.time() - t0:.0f} s", flush=True)
    dates = df.index[t0_eval:]
    np.savez_compressed(f"{OUT}_{name}{tag}.npz", names=np.array(names), dates=np.array([str(d.date()) for d in dates]),
                        gross=np.array(G), turn=np.array(U), reb=reb, first=first)
    print(f"{name}: {len(names)} series, evaluation {dates[0].date()} to {dates[-1].date()} ({len(dates)} days), "
          f"{time.time() - t0:.0f} s")


# ----------------------------------------------------------------------------------------------------------
def check():
    """Do the paper's and BPT's random-matrix formulas still hold after devolatilisation?
    Algebra: with a = D⁻¹1, min v'Cv s.t. a'v = 1 gives v = C⁻¹a / a'C⁻¹a, and w = D⁻¹v ∝ (DCD)⁻¹1, the GMV of DCD
    (tests/test_study6.py). The formulas use 1 only as a fixed vector, so they hold with a in its place, which is
    the same as applying them to the rescaled returns Z·D (estimate). What can fail: (i) the theory needs a fixed
    or independent of the sample, but σ_t is computed from the last days of the window; (ii) Z is standardised by
    estimated, not true, volatilities. Simulation: GARCH(1,1) volatility per asset (0.06, 0.93), correlations
    = the sample correlation of the standardised 100 portfolios; for each sample, the true covariance of the next
    day D_t C D_t. We compare the estimated intensity α of the paper's estimator with its oracle (the best α for
    the same base and target, with the true covariance), and BPT's estimated variance at its chosen η with the
    true one, without and with devolatilisation."""
    rng = np.random.default_rng(61)
    R = load("ff100").to_numpy()
    sig = ewma_vol(R, VOLS["RiskMetrics"], INIT)[:len(R)]
    Z = (R / sig)[INIT + WARM:]
    C = np.corrcoef(Z, rowvar=False)
    p = C.shape[0]
    lam, V = np.linalg.eigh(C)
    root = (V * np.sqrt(np.clip(lam, 0, None))) @ V.T
    ga, gb = 0.06, 0.93
    ew = np.full(p, 1 / p)
    rows = []
    for n in (63, 126, 252):
        for rep in range(100):
            Tn = INIT + WARM + n
            burn = 300
            h = np.ones(p)
            Rs = np.empty((burn + Tn, p))
            for s in range(burn + Tn):
                Rs[s] = np.sqrt(h) * (rng.standard_normal(p) @ root)
                h = (1 - ga - gb) + ga * Rs[s] ** 2 + gb * h
            Rs = Rs[burn:]                                   # h is now the true variance of the next day, t = Tn
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
                             "BPT: estimated / true variance": lhat / (wb @ Sig @ wb),
                             "BPT: loss": relative_loss(wb, Sig, v)})
    out = pd.DataFrame(rows).groupby(["n", "c", "mode"]).agg(["mean", "median"]).reset_index()
    out.columns = [" ".join(x).strip() for x in out.columns]
    out.to_csv(f"{OUT}_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "run":
        run(sys.argv[2])
    elif cmd == "weekly":
        run("ff100", reb=5, cfg=[c for c in configs(WINDOWS) if c[2] in MAIN_VOLS and
                                 (c[1] == 252 or c[0] in ("sample", "LW nonlinear") or c[0] == "1/N")], tag="_weekly")
    else:
        from study6_plot import plot
        plot()
