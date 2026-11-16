"""
Study 5 — A ridge base and several targets, compared fairly.

The paper's estimator mixes the sample GMV portfolio (S⁻¹, or S⁺ for c > 1) with one target. We replace the base by
the ridge GMV portfolio (S + ηI)⁻¹1 / 1'(S + ηI)⁻¹1 and mix it with several targets (1/N and inverse-variance
weights); η and the target weights minimise a random-matrix estimate of the out-of-sample variance
(common.ridge_terms). Ridge with the single target 1/N is the method of Bodnar, Parolya and Thorsén (2024, JMLR),
"BPT 2024" below; what is new is several targets with a random-matrix choice of their weights.

Every comparison is made in two information regimes, on the same portfolios and the same evaluation days:
  same window   every estimator sees only the last n days. Our inverse-variance target is then built inside the
                window (from its first half: the version chosen in simulation, see validate).
  same history  every estimator may use the last n + PRIOR days: Ledoit-Wolf on n + PRIOR days, the paper's
                estimator with an inverse-variance target from the PRIOR days, ours with that target.

  python study5_ridge_multitarget.py validate     variance estimate vs truth; bias of an in-window target;
                                                  nonlinear shrinkage vs its oracle
  python study5_ridge_multitarget.py sim          relative loss, three designs, both regimes
  python study5_ridge_multitarget.py sp500 P DRAWS   S&P 500 2013-2018 backtest (p = 54: 200, p = 150: 100)
  python study5_ridge_multitarget.py plot         tables (bootstrap, costs, half-periods) and figures
"""
import os
import sys
import numpy as np
import pandas as pd
from common import (RESULTS, DATA, make_sigma, make_sigma_market, make_sigma_empirical, simulate, sample_cov, gmv, pinv,
                    bona_fide, ledoit_wolf, lw_nonlinear, inverse_variance, ridge_terms, ridge_choose, ridge_multi_target,
                    relative_loss, gmv_variance, ETA_GRID)

OUT = os.path.join(RESULTS, "study5")
CS_SIM, CS_EMP = (0.5, 0.9, 1.5, 3.0), (0.5, 0.9, 1.5, 2.0)
P_SIM, REPS = 200, 100
PRIOR = 120              # extra days available in the same-history regime (fixed in advance, not tuned)
BLOCK, BOOT = 10, 1000   # block bootstrap over days: block length and number of resamples (fixed in advance)
COSTS = (0, 5, 10)       # proportional transaction costs, basis points
OURS = "ours: ridge + 1/N + inv. variance"
OURS_H = "ours, target from prior days"
BPT = "BPT 2024: ridge + 1/N"
# name: regime ("window", "history" or "both": an estimator that uses only the last n days belongs to both)
NAMES = {"1/N": "both",
         "inverse variance": "window",
         "paper: bona fide, 1/N": "both",
         "Ledoit-Wolf linear": "both",
         "Ledoit-Wolf nonlinear": "both",
         BPT: "both",
         OURS: "window",
         "ours, target from whole window": "window",
         "inverse variance, n + 120 days": "history",
         "Ledoit-Wolf linear, n + 120 days": "history",
         "Ledoit-Wolf nonlinear, n + 120 days": "history",
         "paper: bona fide, target from prior days": "history",
         OURS_H: "history"}
EST = list(NAMES)
REGIMES = {"same window": (OURS, [k for k, v in NAMES.items() if v in ("window", "both")]),
           "same history": (OURS_H, [k for k, v in NAMES.items() if v in ("history", "both")])}
DESIGNS = {"neutral: S&P 500 covariance": make_sigma_empirical,
           "paper's bounded design": lambda p, rng: make_sigma(p, "bounded", rng),
           "one-factor market": make_sigma_market}


# ----------------------------------------------------------------------------------------------------------
class Past:
    """Read access to the rows before day t, at most `limit` of them; every read is checked."""
    def __init__(self, r, t, limit):
        assert 0 < limit <= t <= len(r), (limit, t, len(r))
        self.r, self.t, self.limit = r, t, limit

    def last(self, L):
        """Rows t - L, ..., t - 1."""
        assert 0 < L <= self.limit, f"reads {L} rows, allowed {self.limit}"
        out = self.r[self.t - L:self.t]
        assert len(out) == L
        return out


def gmv_from_cov(C):
    return gmv(np.linalg.inv(C))


def weights_at(r, t, n):
    """Weights of every estimator for day t, from rows before t only. Same-window estimators get a reader that
    cannot see more than n rows; same-history estimators one that cannot see more than n + PRIOR rows."""
    win, hist = Past(r, t, n), Past(r, t, n + PRIOR)
    Y = win.last(n)
    p = Y.shape[1]
    ew = np.full(p, 1 / p)
    S = sample_cov(Y)
    Sinv = np.linalg.inv(S) if p < n else pinv(S)
    iv_half = inverse_variance(Y[: n // 2].var(axis=0))          # the in-window target, from the first half
    iv_win = inverse_variance(Y.var(axis=0))
    H = hist.last(n + PRIOR)
    iv_prior = inverse_variance(H[:PRIOR].var(axis=0))           # rows t-n-PRIOR ... t-n-1, before the window
    T = np.array([ew, iv_half, iv_win, iv_prior])
    terms = ridge_terms(S, n, T)
    w = {"1/N": ew,
         "inverse variance": iv_win,
         "paper: bona fide, 1/N": bona_fide(S, n, ew, Sinv)[0],
         "Ledoit-Wolf linear": gmv_from_cov(ledoit_wolf(Y)),
         "Ledoit-Wolf nonlinear": gmv_from_cov(lw_nonlinear(Y)),
         BPT: ridge_choose(terms, T, [0])[0],
         OURS: ridge_choose(terms, T, [0, 1])[0],
         "ours, target from whole window": ridge_choose(terms, T, [0, 2])[0],
         "inverse variance, n + 120 days": inverse_variance(H.var(axis=0)),
         "Ledoit-Wolf linear, n + 120 days": gmv_from_cov(ledoit_wolf(H)),
         "Ledoit-Wolf nonlinear, n + 120 days": gmv_from_cov(lw_nonlinear(H)),
         "paper: bona fide, target from prior days": bona_fide(S, n, iv_prior, Sinv)[0],
         OURS_H: ridge_choose(terms, T, [0, 3])[0]}
    return np.array([w[k] for k in EST]), T


# ----------------------------------------------------------------------------------------------------------
def validate():
    rng = np.random.default_rng(55)
    # (a) the variance estimate of w_η against the truth, at fixed η, three designs
    rows = []
    for design, make in DESIGNS.items():
        Sigma, root, _ = make(P_SIM, rng)
        v = gmv_variance(Sigma)
        ew = np.full(P_SIM, 1 / P_SIM)
        for c in (0.5, 2.0):
            n = round(P_SIM / c)
            for r in range(REPS):
                Y, _ = simulate(root, n, rng)
                eta, W, vww, vwb, B = ridge_terms(sample_cov(Y), n, [ew])
                true = np.einsum("ep,pq,eq->e", W, Sigma, W)
                for k in range(len(eta)):
                    rows.append({"design": design, "c": c, "rep": r, "eta_rel": ETA_GRID[k],
                                 "estimate": vww[k] / v - 1, "truth": true[k] / v - 1})
    val = pd.DataFrame(rows)
    val.to_csv(f"{OUT}_validation.csv", index=False)
    a = val.groupby(["design", "c", "eta_rel"])[["estimate", "truth"]].mean()
    a["est / true - 1"] = a.estimate / a.truth - 1
    print("(a) bias of the variance estimate at fixed η: range over the grid")
    print(a.groupby(["design", "c"])["est / true - 1"].agg(["min", "max"]).round(3).to_string())

    # (b) the inverse-variance target built inside the window: bias of its terms, and true loss of the result
    rows = []
    for design, make in DESIGNS.items():
        for p in (54, 150):
            Sigma, root, _ = make(p, rng)
            v = gmv_variance(Sigma)
            ew = np.full(p, 1 / p)
            for c in CS_EMP:
                n = round(p / c)
                for r in range(60):
                    Y, _ = simulate(root, n, rng)
                    S = sample_cov(Y)
                    tg = {"independent (earlier sample)": inverse_variance(np.diag(sample_cov(simulate(root, PRIOR, rng)[0]))),
                          "whole window": inverse_variance(Y.var(axis=0)),
                          "first half of the window": inverse_variance(Y[: n // 2].var(axis=0))}
                    T = np.array([ew] + list(tg.values()))
                    terms = ridge_terms(S, n, T)
                    eta, W, vww, vwb, B = terms
                    w0 = ridge_choose(terms, T, [0])[0]
                    rows.append({"design": design, "p": p, "c": c, "target": "none: BPT 2024", "true loss": relative_loss(w0, Sigma, v)})
                    for k, name in enumerate(tg, start=1):
                        w, (e, beta, lhat) = ridge_choose(terms, T, [0, k])
                        b = T[k]
                        rows.append({"design": design, "p": p, "c": c, "target": name,
                                     "bias of b'Sb": B[k, k] / (b @ Sigma @ b) - 1,
                                     "bias of w'Σb": vwb[e, k] / (W[e] @ Sigma @ b) - 1,
                                     "estimated loss": lhat / v - 1, "true loss": relative_loss(w, Sigma, v)})
    tb = pd.DataFrame(rows).groupby(["design", "p", "c", "target"]).mean(numeric_only=True).reset_index()
    tb["estimated / true loss - 1"] = tb["estimated loss"] / tb["true loss"] - 1
    tb.to_csv(f"{OUT}_target_bias.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n(b) inverse-variance target inside the window (means over 60 samples)")
    print(tb.round(3).to_string(index=False))
    one = tb[tb.target.isin(["whole window", "first half of the window"])].pivot_table(
        index=["design", "p", "c"], columns="target", values="true loss")
    print("\nfirst half / whole window, true loss: mean ratio",
          round(float((one["first half of the window"] / one["whole window"]).mean()), 3))

    # (c) nonlinear shrinkage against the oracle with the same eigenvectors, and the competitors
    rows = []
    for design, make in DESIGNS.items():
        Sigma, root, _ = make(P_SIM, rng)
        v = gmv_variance(Sigma)
        for n in (400, 100):
            for r in range(20):
                Y, _ = simulate(root, n, rng)
                Yc = Y - Y.mean(axis=0)
                S = Yc.T @ Yc / (n - 1)
                lam, U = np.linalg.eigh(S)
                C = {"sample covariance": S, "Ledoit-Wolf linear": ledoit_wolf(Y), "Ledoit-Wolf nonlinear": lw_nonlinear(Y),
                     "oracle: same eigenvectors, best eigenvalues": (U * np.einsum("pi,pq,qi->i", U, Sigma, U)) @ U.T}
                for k, M in C.items():
                    Mi = pinv(M) if k == "sample covariance" else np.linalg.inv(M)
                    rows.append({"design": design, "n": n, "estimator": k, "GMV relative loss": relative_loss(gmv(Mi), Sigma, v),
                                 "Frobenius loss / p": np.linalg.norm(M - Sigma) ** 2 / P_SIM})
    nl = pd.DataFrame(rows).groupby(["design", "n", "estimator"]).mean().reset_index()
    nl.to_csv(f"{OUT}_nonlinear_check.csv", index=False)
    print(f"\n(c) nonlinear shrinkage, p = {P_SIM}, 20 samples")
    print(nl.round(3).to_string(index=False))


def sim():
    """Relative loss; each sample has n + PRIOR observations, the same-window estimators see only the last n."""
    rng = np.random.default_rng(56)
    rows = []
    for design, make in DESIGNS.items():
        Sigma, root, _ = make(P_SIM, rng)
        v = gmv_variance(Sigma)
        for dist in ("normal", "t5"):
            for c in CS_SIM:
                n = round(P_SIM / c)
                for r in range(REPS):
                    Y, _ = simulate(root, n + PRIOR, rng, dist)
                    W, T = weights_at(Y, n + PRIOR, n)
                    S = sample_cov(Y[-n:])
                    w_or = ridge_multi_target(S, n, T[:2], Sigma)[0]
                    for name, w in list(zip(EST, W)) + [("oracle: ridge + targets", w_or)]:
                        if name in ("1/N", "inverse variance", "inverse variance, n + 120 days"):
                            continue
                        rows.append({"design": design, "returns": dist, "c": c, "rep": r, "estimator": name,
                                     "loss": relative_loss(w, Sigma, v)})
                print(design, dist, "c =", c, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}_sim.csv", index=False)


# ----------------------------------------------------------------------------------------------------------
def _one_portfolio(args):
    """Daily gross returns and turnover (against the drifted weights of the day before) of every estimator."""
    R, idx, n, t0 = args
    r = R[:, idx]
    days = R.shape[0] - t0
    gross = np.empty((len(EST), days))
    turn = np.zeros((len(EST), days))
    prev = None
    for j, t in enumerate(range(t0, R.shape[0])):
        W, _ = weights_at(r, t, n)
        assert np.all(np.isfinite(W)), "non-finite weights"
        gross[:, j] = W @ r[t]                       # applied to day t, never seen by weights_at
        if prev is not None:
            drift = prev * (1 + r[t - 1])
            drift /= drift.sum(axis=1, keepdims=True)
            turn[:, j] = np.abs(W - drift).sum(axis=1)
        prev = W
    return gross, turn


def load_returns():
    close = pd.read_csv(os.path.join(DATA, "sp500_close.csv"), index_col=0, parse_dates=True)
    R = close.dropna(axis=1).pct_change().iloc[1:]
    return R.to_numpy(), R.index


def sp500(p=54, draws=200):
    from multiprocessing import Pool
    R, dates = load_returns()
    T, N = R.shape
    ns = [round(p / c) for c in CS_EMP]
    t0 = max(ns) + PRIOR                      # the same evaluation days for every c, regime and estimator
    print(f"{N} stocks with a full history, T = {T}; evaluation {dates[t0].date()} to {dates[-1].date()} "
          f"({T - t0} days), p = {p}, {draws} portfolios")
    rng = np.random.default_rng(57)
    idx = [rng.choice(N, p, replace=False) for _ in range(draws)]
    out = {"names": np.array(EST), "dates": np.array([str(d.date()) for d in dates[t0:]])}
    with Pool() as pool:
        for c, n in zip(CS_EMP, ns):
            res = pool.map(_one_portfolio, [(R, i, n, t0) for i in idx])
            out[f"gross_{c}"] = np.array([x[0] for x in res], dtype=np.float32)    # draws x estimators x days
            out[f"turn_{c}"] = np.array([x[1] for x in res], dtype=np.float32)
            print("c =", c, "done", flush=True)
    np.savez_compressed(f"{OUT}_sp500_p{p}.npz", **out)


# ----------------------------------------------------------------------------------------------------------
def nw_tstat(d):
    """t-statistic of the mean of a daily series with Newey-West standard errors."""
    T = len(d)
    L = int(4 * (T / 100) ** (2 / 9))
    e = d - d.mean()
    s = e @ e / T
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * (e[k:] @ e[:-k]) / T
    return d.mean() / np.sqrt(s / T)


def block_counts(T, rng):
    """BOOT x T matrix: how often each day enters a moving-block bootstrap resample."""
    nb = -(-T // BLOCK)
    starts = rng.integers(0, T - BLOCK + 1, size=(BOOT, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(BOOT, -1)[:, :T]
    C = np.zeros((BOOT, T))
    np.add.at(C, (np.repeat(np.arange(BOOT), T), idx.ravel()), 1)
    return C


def compare(X, names, ours, rng):
    """X: portfolios x estimators x days of (net) returns. For each estimator: median and mean variance over
    portfolios, the ratio of mean variances ours / this with a 95% block-bootstrap interval over days, the
    Newey-West t-statistic of the day-by-day difference in squared deviations (averaged over portfolios; the
    inference is over time, the only independent dimension), and the share of portfolios where ours is lower."""
    Tn = X.shape[2]
    var = X.var(axis=2, ddof=1)
    dev2 = (X - X.mean(axis=2, keepdims=True)) ** 2
    C = block_counts(Tn, rng) / Tn                          # BOOT x days, rows sum to 1
    flat = X.reshape(-1, Tn).astype(float)
    m1, m2 = flat @ C.T, (flat ** 2) @ C.T                  # (portfolios*estimators) x BOOT
    bvar = (m2 - m1 ** 2).reshape(X.shape[0], X.shape[1], -1).mean(axis=0)     # estimators x BOOT
    o = names.index(ours)
    rows = []
    for k, name in enumerate(names):
        row = {"estimator": name, "median variance (x1e-5)": 1e5 * np.median(var[:, k]),
               "mean variance (x1e-5)": 1e5 * var[:, k].mean()}
        if k != o:
            ratio_b = bvar[o] / bvar[k]
            row.update({"ratio ours / this": var[:, o].mean() / var[:, k].mean(),
                        "95% low": np.quantile(ratio_b, 0.025), "95% high": np.quantile(ratio_b, 0.975),
                        "t-stat (ours - this)": nw_tstat((dev2[:, o] - dev2[:, k]).mean(axis=0)),
                        "ours lower (share)": np.mean(var[:, o] < var[:, k])})
        rows.append(row)
    return rows


def sp500_tables(p):
    z = np.load(f"{OUT}_sp500_p{p}.npz")
    names, dates = list(z["names"]), z["dates"]
    Tn = len(dates)
    halves = {"full period": slice(0, Tn), "first half": slice(0, Tn // 2), "second half": slice(Tn // 2, Tn)}
    rng = np.random.default_rng(58)
    rows = []
    for c in CS_EMP:
        G, U = z[f"gross_{c}"].astype(float), z[f"turn_{c}"].astype(float)
        for regime, (ours, members) in REGIMES.items():
            ks = [names.index(m) for m in members]
            for period, sl in halves.items():
                for cost in (COSTS if period == "full period" else (0,)):
                    X = G[:, ks, sl] - cost * 1e-4 * U[:, ks, sl]
                    for row in compare(X, members, ours, rng):
                        k = names.index(row["estimator"])
                        rows.append({"p": p, "c": c, "regime": regime, "period": f"{period} ({dates[sl][0][:7]} to {dates[sl][-1][:7]})",
                                     "cost (bp)": cost, **row, "turnover per day": U[:, k, sl][:, 1:].mean()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import plot_style, note, arrow, save, INK, MUTED, RED, WARM, DARK, LIGHT, SLIDE, SURFACE
    plot_style()
    os.makedirs(os.path.join(RESULTS, "slides"), exist_ok=True)
    pd.set_option("display.width", 260)
    col = {"1/N": LIGHT, "paper: bona fide, 1/N": WARM, "Ledoit-Wolf linear": "#9C8F7A", "Ledoit-Wolf nonlinear": DARK,
           BPT: "#7A2A2E", OURS: RED, "ours, target from whole window": "#D9A19C", "oracle: ridge + targets": INK,
           "paper: bona fide, target from prior days": WARM, "Ledoit-Wolf linear, n + 120 days": "#9C8F7A",
           "Ledoit-Wolf nonlinear, n + 120 days": DARK, OURS_H: RED, "inverse variance": "#B9AE9C",
           "inverse variance, n + 120 days": "#B9AE9C"}

    # 1. validation ------------------------------------------------------------------------------------
    val = pd.read_csv(f"{OUT}_validation.csv")
    fig, ax = plt.subplots(2, 3, figsize=(20, 9.5))
    for j, design in enumerate(DESIGNS):
        for i, c in enumerate((0.5, 2.0)):
            a, g = ax[i, j], val[(val.design == design) & (val.c == c)].groupby("eta_rel")
            x = g.truth.mean().index
            a.fill_between(x, g.estimate.quantile(0.1), g.estimate.quantile(0.9), color=RED, alpha=0.15, lw=0,
                           label="estimate, 10-90% of samples")
            a.plot(x, g.estimate.mean(), color=RED, ls="--", label="estimate, mean")
            a.plot(x, g.truth.mean(), color=INK, label="truth, mean")
            a.set_xscale("log"); a.set_ylim(0, None); a.set_title(f"{design}, c = {c}", fontsize=14)
            if i == 1:
                a.set_xlabel(r"ridge $\eta$ / average eigenvalue")
    ax[0, 0].set_ylabel("relative loss, ridge GMV"); ax[1, 0].set_ylabel("relative loss, ridge GMV"); ax[0, 0].legend(fontsize=11)
    plt.tight_layout(); save(fig, f"{OUT}_validation.png"); plt.close(fig)

    g = val[(val.design == "neutral: S&P 500 covariance") & (val.c == 2.0)].groupby("eta_rel")
    fig, a = plt.subplots(figsize=SLIDE)
    x = g.truth.mean().index
    a.fill_between(x, g.estimate.quantile(0.1), g.estimate.quantile(0.9), color=RED, alpha=0.15, lw=0)
    a.plot(x, g.truth.mean(), color=INK, lw=3); a.plot(x, g.estimate.mean(), color=RED, lw=3, ls="--")
    a.set_xscale("log"); a.set_ylim(0, None)
    a.set_xlabel(r"ridge $\eta$ / average eigenvalue"); a.set_ylabel("relative loss, ridge GMV portfolio")
    top = g.estimate.quantile(0.9).max()
    note(a, x[1], top * 0.92, "neutral design (S&P 500 covariance), c = 2:\nthe estimate from the data (red) tracks the truth (black)")
    note(a, x[-12], top * 0.55, "band: 80% of samples;\nat the chosen η the estimate is optimistic", size=20)
    save(fig, os.path.join(RESULTS, "slides", "s5_estimate_tracks_truth.png")); plt.close(fig)

    # 2. simulation ------------------------------------------------------------------------------------
    s = pd.read_csv(f"{OUT}_sim.csv")
    tab = s.groupby(["design", "returns", "c", "estimator"]).loss.agg(["mean", "median"]).reset_index()
    tab.to_csv(f"{OUT}_sim_table.csv", index=False)
    # paired tests over the samples: mean log(loss ours / loss rival), its t-statistic, geometric-mean ratio
    piv = s.pivot_table(index=["design", "returns", "c", "rep"], columns="estimator", values="loss")
    tests = []
    for ours in (OURS, OURS_H):
        for rival in piv.columns:
            if rival == ours or (ours == OURS and NAMES.get(rival) == "history") or (ours == OURS_H and NAMES.get(rival) == "window"):
                continue
            d = np.log(piv[ours] / piv[rival]).groupby(level=["design", "returns", "c"])
            tests.append(pd.DataFrame({"ours": ours, "rival": rival, "ratio (geometric mean)": np.exp(d.mean()),
                                       "t-stat": d.mean() / (d.std() / np.sqrt(d.count()))}).reset_index())
    tests = pd.concat(tests)
    tests.to_csv(f"{OUT}_sim_tests.csv", index=False)
    print(tab.pivot_table(index=["design", "returns", "estimator"], columns="c", values="median").round(3).to_string())
    show = ["paper: bona fide, 1/N", "Ledoit-Wolf linear", "Ledoit-Wolf nonlinear", BPT, OURS, "oracle: ridge + targets"]
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.2))
    for a, design in zip(ax, DESIGNS):
        t = tab[(tab.design == design) & (tab.returns == "normal")]
        for name in show:
            v = t[t.estimator == name].set_index("c")["median"].reindex(CS_SIM)
            a.plot(CS_SIM, v, marker="o", color=col[name], ls=":" if name.startswith("oracle") else "-", label=name)
        a.set_yscale("log"); a.set_xticks(CS_SIM); a.set_xlabel("c = p / n"); a.set_title(f"{design}", fontsize=15)
    ax[0].set_ylabel(f"median relative loss, p = {P_SIM}, same window")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08), fontsize=12)
    plt.tight_layout(); save(fig, f"{OUT}_sim.png"); plt.close(fig)

    fig, a = plt.subplots(figsize=SLIDE)
    t = tab[(tab.design == "neutral: S&P 500 covariance") & (tab.returns == "normal")]
    for name in show:
        v = t[t.estimator == name].set_index("c")["median"].reindex(CS_SIM)
        a.plot(CS_SIM, v, marker="o", lw=3, ms=8, color=col[name], ls=":" if name.startswith("oracle") else "-", label=name)
    a.set_yscale("log"); a.set_xticks(CS_SIM); a.legend(loc="upper left", fontsize=12, ncol=2)
    a.set_ylim(0.3, 4); a.set_yticks([0.4, 0.6, 1, 2, 3], ["0.4", "0.6", "1", "2", "3"])
    a.set_xlabel("c = p / n"); a.set_ylabel(f"median relative loss, p = {P_SIM}")
    a.set_title(r"Neutral design: $\Sigma$ = sample covariance of 200 S&P 500 stocks; every estimator sees the same n days", fontsize=14)
    note(a, 1.7, 0.42, "ours, BPT 2024 and Ledoit-Wolf nonlinear\nwithin 7% of each other", size=20)
    save(fig, os.path.join(RESULTS, "slides", "s5_simulation_neutral.png")); plt.close(fig)

    # 3. S&P 500 ---------------------------------------------------------------------------------------
    emp = pd.concat([sp500_tables(p) for p in (54, 150) if os.path.exists(f"{OUT}_sp500_p{p}.npz")])
    emp.to_csv(f"{OUT}_sp500.csv", index=False)
    full = emp[emp.period.str.startswith("full") & (emp["cost (bp)"] == 0)]
    cols = ["p", "c", "regime", "estimator", "median variance (x1e-5)", "ratio ours / this", "95% low", "95% high",
            "t-stat (ours - this)", "ours lower (share)", "turnover per day"]
    print(full[cols].round(3).to_string(index=False))

    rivals = {"same window": [BPT, "Ledoit-Wolf nonlinear", "Ledoit-Wolf linear", "paper: bona fide, 1/N", "1/N"],
              "same history": ["Ledoit-Wolf nonlinear, n + 120 days", "Ledoit-Wolf linear, n + 120 days",
                               "paper: bona fide, target from prior days", BPT, "1/N"]}

    def forest(a, e, regime, legend=True):
        rv = rivals[regime]
        w = 0.8 / len(rv)
        for k, name in enumerate(rv):
            t = e[(e.regime == regime) & (e.estimator == name)].set_index("c").reindex(CS_EMP)
            x = np.arange(len(CS_EMP)) + (k - len(rv) / 2 + 0.5) * w
            a.errorbar(x, t["ratio ours / this"], yerr=[t["ratio ours / this"] - t["95% low"], t["95% high"] - t["ratio ours / this"]],
                       fmt="o", color=col[name], ms=8, capsize=3, lw=2, label=f"vs {name}")
        a.axhline(1, color=MUTED, lw=1.2, ls=":")
        a.set_xticks(range(len(CS_EMP)), [f"c = {c}" for c in CS_EMP]); a.grid(axis="x", visible=False)
        a.set_ylim(0.45, 1.35)

    fig, ax = plt.subplots(2, 2, figsize=(20, 11), sharey=True)
    for i, (p, e) in enumerate(full.groupby("p")):
        for j, regime in enumerate(REGIMES):
            forest(ax[i, j], e, regime)
            ax[i, j].set_title(f"p = {p}, {regime}", fontsize=15)
            ax[i, j].legend(fontsize=10, loc="lower left")
        ax[i, 0].set_ylabel("variance ratio, ours / rival (95% CI)")
    plt.tight_layout(); save(fig, f"{OUT}_sp500.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=SLIDE, sharey=True)
    for a, (p, e) in zip(ax, full.groupby("p")):
        forest(a, e, "same window")
        a.set_title(f"p = {p}: every estimator sees the same n days", fontsize=15)
    ax[0].set_ylabel("variance ratio ours / rival (95% CI)\nbelow 1: ours has lower variance", fontsize=13)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.09), fontsize=11)
    note(ax[0], 0.0, 1.27, "ours = BPT 2024 at every c", size=21)
    note(ax[1], 0.0, 1.27, "beats Ledoit-Wolf nonlinear for c ≤ 0.9", size=21)
    plt.tight_layout()
    save(fig, os.path.join(RESULTS, "slides", "s5_sp500_fair.png")); plt.close(fig)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "validate":
        validate()
    elif cmd == "sim":
        sim()
    elif cmd == "sp500":
        sp500(*(int(a) for a in sys.argv[2:4]))
    else:
        plot()
