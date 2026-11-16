"""
Study 5 — A ridge base and several targets: an estimator that also works for c >= 0.9.

The paper's estimator mixes the sample GMV portfolio (S⁻¹, or S⁺ for c > 1) with one target. Studies 1 and 3
show its two weak points: S⁺ makes the intensity biased for c > 1, and near c = 1 the sample GMV portfolio is so
noisy that 1/N beats the mixture. We replace the base by the ridge GMV portfolio (S + ηI)⁻¹1 / 1'(S + ηI)⁻¹1 and
mix it with several targets (1/N and inverse-variance weights). η and the target weights minimise a
random-matrix estimate of the out-of-sample variance that needs only the data (common.ridge_terms), so the
estimator is bona fide for every c, with no pseudo-inverse.

  python study5_ridge_multitarget.py validate     the variance estimate against the truth (simulation)
  python study5_ridge_multitarget.py sim          relative loss of all estimators (simulation)
  python study5_ridge_multitarget.py sp500 [p] [draws]   S&P 500, 2013-2018 (default p = 54, 200 portfolios)
  python study5_ridge_multitarget.py plot         tables and figures from the saved runs
"""
import os
import sys
import numpy as np
import pandas as pd
from common import (RESULTS, DATA, make_sigma, make_sigma_market, simulate, sample_cov, gmv, inverse, pinv, bona_fide,
                    multi_target, ledoit_wolf, inverse_variance, ridge_terms, ridge_multi_target, relative_loss, ETA_GRID,
                    gmv_variance)

OUT = os.path.join(RESULTS, "study5")
CS_SIM, CS_EMP = (0.5, 0.9, 1.5, 3.0), (0.5, 0.9, 1.5, 2.0)
P_SIM, REPS = 200, 100
PRIOR = 120            # days before the estimation window used for the inverse-variance target (independent of it)
EST = ("1/N", "inverse variance", "paper: bona fide (1/N)", "multi-target, S⁻¹ / S⁺", "Ledoit-Wolf GMV",
       "ridge + 1/N", "ridge + 1/N + inv. variance", "oracle: ridge + targets")


def estimators(Y, n, ew, iv, Sigma=None):
    """Weights of every estimator from one estimation window Y (n x p)."""
    S = sample_cov(Y)
    c = S.shape[0] / n
    Sinv = np.linalg.inv(S) if c < 1 else pinv(S)
    w = {"1/N": ew, "inverse variance": iv,
         "paper: bona fide (1/N)": bona_fide(S, n, ew, Sinv)[0],
         "multi-target, S⁻¹ / S⁺": multi_target(S, n, [ew, iv], Sinv),
         "Ledoit-Wolf GMV": gmv(np.linalg.inv(ledoit_wolf(Y))),
         "ridge + 1/N": ridge_multi_target(S, n, [ew])[0],
         "ridge + 1/N + inv. variance": ridge_multi_target(S, n, [ew, iv])[0]}
    if Sigma is not None:
        w["oracle: ridge + targets"] = ridge_multi_target(S, n, [ew, iv], Sigma)[0]
    return w


DESIGNS = {"paper's bounded design": lambda p, rng: make_sigma(p, "bounded", rng),
           "one-factor market": make_sigma_market}


# ----------------------------------------------------------------------------------------------------------
def validate():
    """Mean and spread of the estimated out-of-sample variance of w_η against the truth, over η."""
    rng = np.random.default_rng(55)
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
    pd.DataFrame(rows).to_csv(f"{OUT}_validation.csv", index=False)
    t = pd.DataFrame(rows).groupby(["design", "c", "eta_rel"])[["estimate", "truth"]].mean()
    print(t.iloc[::5].round(3).to_string())


def sim():
    """Relative loss of every estimator; the inverse-variance target comes from an independent earlier sample."""
    rng = np.random.default_rng(56)
    rows = []
    for design, make in DESIGNS.items():
        for dist in ("normal", "t5"):
            Sigma, root, _ = make(P_SIM, rng)
            v = gmv_variance(Sigma)
            ew = np.full(P_SIM, 1 / P_SIM)
            for c in CS_SIM:
                n = round(P_SIM / c)
                for r in range(REPS):
                    iv = inverse_variance(np.diag(sample_cov(simulate(root, PRIOR, rng, dist)[0])))
                    Y, _ = simulate(root, n, rng, dist)
                    for name, w in estimators(Y, n, ew, iv, Sigma).items():
                        rows.append({"design": design, "returns": dist, "c": c, "rep": r, "estimator": name,
                                     "loss": relative_loss(w, Sigma, v)})
                print(design, dist, "c =", c, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}_sim.csv", index=False)


# ----------------------------------------------------------------------------------------------------------
def _one_portfolio(args):
    """Daily out-of-sample returns and turnover of every estimator for one random portfolio."""
    R, idx, n, t0 = args
    r = R[:, idx]
    p = len(idx)
    ew = np.full(p, 1 / p)
    names = EST[:-1]
    rets = np.empty((len(names), R.shape[0] - t0))
    turn = np.zeros(len(names))
    prev = None
    for j, t in enumerate(range(t0, R.shape[0])):
        iv_target = inverse_variance(r[t - n - PRIOR:t - n].var(axis=0))
        w = estimators(r[t - n:t], n, ew, iv_target)
        w["inverse variance"] = inverse_variance(r[t - PRIOR:t].var(axis=0))
        W = np.array([w[k] for k in names])
        rets[:, j] = W @ r[t]
        if prev is not None:
            turn += np.abs(W - prev).sum(axis=1)
        prev = W
    return rets, turn / (rets.shape[1] - 1)


def sp500(p=54, draws=200):
    from multiprocessing import Pool
    close = pd.read_csv(os.path.join(DATA, "sp500_close.csv"), index_col=0, parse_dates=True)
    R = close.dropna(axis=1).pct_change().iloc[1:].to_numpy()
    T, N = R.shape
    ns = [round(p / c) for c in CS_EMP]
    t0 = max(ns) + PRIOR                      # one evaluation period for every c
    print(f"{N} stocks with a full history, T = {T}; out-of-sample days {T - t0}, p = {p}, {draws} portfolios")
    rng = np.random.default_rng(57)
    idx = [rng.choice(N, p, replace=False) for _ in range(draws)]
    out = {}
    with Pool() as pool:
        for c, n in zip(CS_EMP, ns):
            res = pool.map(_one_portfolio, [(R, i, n, t0) for i in idx])
            out[f"rets_{c}"] = np.array([x[0] for x in res], dtype=np.float32)    # draws x estimators x days
            out[f"turn_{c}"] = np.array([x[1] for x in res])
            print("c =", c, "done", flush=True)
    np.savez_compressed(f"{OUT}_sp500_p{p}.npz", **out, names=np.array(EST[:-1]))


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


def sp500_table(p):
    z = np.load(f"{OUT}_sp500_p{p}.npz")
    names = list(z["names"])
    ours, paper, lw = names.index("ridge + 1/N + inv. variance"), names.index("paper: bona fide (1/N)"), names.index("Ledoit-Wolf GMV")
    rows = []
    for c in CS_EMP:
        X = z[f"rets_{c}"].astype(float)                       # draws x estimators x days
        dev2 = (X - X.mean(axis=2, keepdims=True)) ** 2         # squared deviations: their mean is the variance
        var = dev2.mean(axis=2) * X.shape[2] / (X.shape[2] - 1)
        for k, name in enumerate(names):
            # does ours have a lower variance than this estimator? the day-by-day difference, averaged over
            # portfolios, with Newey-West errors: the inference is over time, the only independent dimension
            d = (dev2[:, ours] - dev2[:, k]).mean(axis=0)
            rows.append({"p": p, "c": c, "estimator": name, "median variance (x1e-5)": 1e5 * np.median(var[:, k]),
                         "ours lower (share of portfolios)": np.mean(var[:, ours] < var[:, k]) if k != ours else np.nan,
                         "t-stat, ours minus this": nw_tstat(d) if k != ours else np.nan,
                         "turnover per day": z[f"turn_{c}"][:, k].mean()})
    return pd.DataFrame(rows)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import plot_style, note, arrow, save, INK, MUTED, RED, WARM, DARK, LIGHT, SLIDE
    plot_style()
    os.makedirs(os.path.join(RESULTS, "slides"), exist_ok=True)
    pd.set_option("display.width", 250)
    col = {"1/N": LIGHT, "inverse variance": "#9C8F7A", "paper: bona fide (1/N)": WARM, "multi-target, S⁻¹ / S⁺": "#D9A19C",
           "Ledoit-Wolf GMV": DARK, "ridge + 1/N": "#7A2A2E", "ridge + 1/N + inv. variance": RED, "oracle: ridge + targets": INK}
    lab = lambda name: name.replace("S⁻¹ / S⁺", r"$S^{-1}$ / $S^{+}$").replace("ridge + 1/N + inv. variance", "ours: ridge + 1/N + inv. variance")

    # 1. validation ------------------------------------------------------------------------------------
    val = pd.read_csv(f"{OUT}_validation.csv")
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    for a, ((design, c), g) in zip(ax, val.groupby(["design", "c"], sort=False)):
        q = g.groupby("eta_rel")
        x = q.truth.mean().index
        a.fill_between(x, q.estimate.quantile(0.1), q.estimate.quantile(0.9), color=RED, alpha=0.15, lw=0,
                       label="estimate, 10-90% of samples")
        a.plot(x, q.estimate.mean(), color=RED, ls="--", label="estimate, mean")
        a.plot(x, q.truth.mean(), color=INK, label="truth, mean")
        a.set_xscale("log"); a.set_title(f"{design}, c = {c}", fontsize=14); a.set_xlabel(r"ridge $\eta$ / average eigenvalue")
        a.set_ylim(0, None)
    ax[0].set_ylabel("relative loss of the ridge GMV portfolio"); ax[0].legend(fontsize=11)
    plt.tight_layout(); save(fig, f"{OUT}_validation.png"); plt.close(fig)

    g = val[(val.design == "paper's bounded design") & (val.c == 2.0)].groupby("eta_rel")
    fig, a = plt.subplots(figsize=SLIDE)
    x = g.truth.mean().index
    a.fill_between(x, g.estimate.quantile(0.1), g.estimate.quantile(0.9), color=RED, alpha=0.15, lw=0)
    a.plot(x, g.truth.mean(), color=INK, lw=3); a.plot(x, g.estimate.mean(), color=RED, lw=3, ls="--")
    a.set_xscale("log"); a.set_xlabel(r"ridge $\eta$ / average eigenvalue"); a.set_ylabel("relative loss, ridge GMV portfolio")
    a.set_ylim(0, None)
    k = int(np.argmin(g.truth.mean().to_numpy()))
    note(a, x[2], g.truth.mean().max() * 0.5, "c = 2: the estimate from the data alone (red)\ntracks the true loss (black)")
    note(a, x[k] * 0.7, g.truth.mean().iloc[k] * 0.45, "band: 80% of samples", size=20)
    save(fig, os.path.join(RESULTS, "slides", "s5_estimate_tracks_truth.png")); plt.close(fig)

    # 2. simulation ------------------------------------------------------------------------------------
    s = pd.read_csv(f"{OUT}_sim.csv")
    s = s[~s.estimator.isin(["1/N", "inverse variance"])]
    tab = s.groupby(["design", "returns", "c", "estimator"]).loss.agg(["mean", "median"]).reset_index()
    tab.to_csv(f"{OUT}_sim_table.csv", index=False)
    print(tab.pivot_table(index=["design", "returns", "estimator"], columns="c", values="median").round(3).to_string())
    names = [e for e in EST if e not in ("1/N", "inverse variance")]
    fig, ax = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
    w = 0.8 / len(names)
    for a, design in zip(ax, DESIGNS):
        t = tab[(tab.design == design) & (tab.returns == "normal")]
        for k, name in enumerate(names):
            vals = t[t.estimator == name].set_index("c")["median"].reindex(CS_SIM)
            a.bar(np.arange(len(CS_SIM)) + (k - len(names) / 2 + 0.5) * w, vals, w * 0.92, color=col[name], label=lab(name))
        a.set_yscale("log"); a.set_xticks(range(len(CS_SIM)), [f"c = {c}" for c in CS_SIM]); a.grid(axis="x", visible=False)
        a.set_title(f"{design}, normal returns, p = {P_SIM}")
    ax[0].set_ylabel(f"median relative loss, {REPS} samples")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=13)
    plt.tight_layout(); save(fig, f"{OUT}_sim.png"); plt.close(fig)

    fig, a = plt.subplots(figsize=SLIDE)
    t = tab[(tab.design == "one-factor market") & (tab.returns == "normal")]
    for name in ("paper: bona fide (1/N)", "Ledoit-Wolf GMV", "ridge + 1/N + inv. variance", "oracle: ridge + targets"):
        v = t[t.estimator == name].set_index("c")["median"].reindex(CS_SIM)
        a.plot(CS_SIM, v, marker="o", lw=3, ms=9, color=col[name], ls=":" if name.startswith("oracle") else "-")
        a.text(CS_SIM[-1] + 0.08, v.iloc[-1], name.replace("paper: ", "paper, ").replace("ridge + 1/N + inv. variance", "ours"),
               color=col[name], fontsize=15, va="center")
    a.set_yscale("log"); a.set_xlim(0.3, 4.1); a.set_xticks(CS_SIM)
    a.set_xlabel("c = p / n"); a.set_ylabel(f"median relative loss, p = {P_SIM}")
    note(a, 0.75, 0.55, "with a market factor the paper's estimator\nand Ledoit-Wolf lose 2 to 16 times more")
    arrow(a, (1.2, 0.44), (1.42, 0.32), 0.2)
    save(fig, os.path.join(RESULTS, "slides", "s5_simulation.png")); plt.close(fig)

    # 3. S&P 500 ---------------------------------------------------------------------------------------
    tabs = [sp500_table(p) for p in (54, 150) if os.path.exists(f"{OUT}_sp500_p{p}.npz")]
    if not tabs:
        return
    emp = pd.concat(tabs)
    emp.to_csv(f"{OUT}_sp500.csv", index=False)
    print(emp.round(3).to_string(index=False))
    show = [e for e in EST[:-1]]
    for p, e in emp.groupby("p"):
        fig, a = plt.subplots(figsize=SLIDE)
        w = 0.8 / len(show)
        for k, name in enumerate(show):
            vals = e[e.estimator == name].set_index("c")["median variance (x1e-5)"].reindex(CS_EMP)
            a.bar(np.arange(len(CS_EMP)) + (k - len(show) / 2 + 0.5) * w, vals, w * 0.92, color=col[name], label=lab(name))
        a.set_xticks(range(len(CS_EMP)), [f"c = {c}" for c in CS_EMP]); a.grid(axis="x", visible=False)
        a.set_ylabel(r"median out-of-sample daily variance, $\times 10^{-5}$")
        lo = e["median variance (x1e-5)"].min()
        a.set_ylim(lo * 0.8, e[e.estimator != "paper: bona fide (1/N)"]["median variance (x1e-5)"].max() * 1.12)
        fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13), fontsize=12)
        a.set_title(f"S&P 500 stocks, 2013-2018, random portfolios of p = {p}  (one period, {e.c.nunique()} values of c)", fontsize=16)
        top = e[e.estimator == "paper: bona fide (1/N)"].set_index("c")["median variance (x1e-5)"]
        for i, c in enumerate(CS_EMP):
            if top[c] > a.get_ylim()[1]:
                a.text(i + (2 - len(show) / 2 + 0.5) * w, a.get_ylim()[1] * 0.985, f"{top[c]:.1f}", ha="center", va="top", fontsize=11, color=INK)
        save(fig, os.path.join(RESULTS, "slides", f"s5_sp500_p{p}.png")); save(fig, f"{OUT}_sp500_p{p}.png")
        plt.close(fig)


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
