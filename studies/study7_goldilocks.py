"""Study 7. Nonlinear shrinkage meets the target: w = α · w_LW-nonlinear + (1 - α) · 1/N.

Ledoit-Wolf nonlinear shrinkage fixes the eigenvalues and treats every direction alike; the paper's target 1/N uses
the one special direction of the GMV problem, the vector 1. α ∈ {0, 0.1, ..., 1} is picked at each rebalance as the
mixture with the lowest variance of realised returns over the previous 252 days (the Study 6 rule).
    python -m studies.study7_goldilocks run ff100    Study 6 framework (also ind49, sp500)
    python -m studies.study7_goldilocks table
    python -m studies.study7_goldilocks study5       the Study 5 setting: fixed c, daily rebalancing, p = 54 and 150
    python -m studies.study7_goldilocks plot
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time
import numpy as np
import pandas as pd
from gmv.backtest import Past, hold, turnover, rule, compare, net
from gmv.data import results_dir
from gmv.estimators import sample_cov, gmv_from_cov, ledoit_wolf, lw_nonlinear, ridge_multi_target
from studies.study6_levers import SETS, INIT, WARM, LOOK, REB, load, draws

OUT = results_dir(7)
ALPHAS = np.round(np.linspace(0, 1, 11), 1)
MIX = "LW nonlinear + 1/N"


def configs(windows):
    cfg = [("1/N", None, None)]
    for h in windows:
        cfg += [("BPT 2024", h, None), ("LW linear", h, None)] + [("mix", h, a) for a in ALPHAS]
    return cfg


def label(c):
    e, h, a = c
    if e == "1/N":
        return "1/N"
    if e != "mix":
        return f"{e}, n = {h}"
    return f"LW nonlinear, n = {h}" if a == 1 else f"{MIX}, n = {h}, alpha = {a:.1f}"


def weights_at(R, t, cfg, windows):
    past = Past(R, t, max(windows))
    p = R.shape[1]
    ew = np.full(p, 1 / p)
    base = {}
    for h in windows:
        Y = past.last(h)
        base[h] = {"mix": gmv_from_cov(lw_nonlinear(Y)), "LW linear": gmv_from_cov(ledoit_wolf(Y)),
                   "BPT 2024": ridge_multi_target(sample_cov(Y), h, [ew])[0]}
    W = np.empty((len(cfg), p))
    for k, (e, h, a) in enumerate(cfg):
        W[k] = ew if e == "1/N" else (a * base[h]["mix"] + (1 - a) * ew if e == "mix" else base[h][e])
    return W


_G = {}


def _init(*args):
    _G["args"] = args


def _weights(t):
    R, cfg, windows = _G["args"]
    return weights_at(R, t, cfg, windows)


def run_one(R, windows, workers=None):
    from multiprocessing import Pool
    T, p = R.shape
    cfg = configs(windows)
    t_start = INIT + WARM + max(windows)                    # the same first rebalance as Study 6
    dates = list(range(t_start, T, REB))
    with Pool(workers, initializer=_init, initargs=(R, cfg, windows)) as pool:
        Ws = np.array(pool.map(_weights, dates, chunksize=2))
    rets = np.full((len(cfg), T - t_start), np.nan)
    Dend = []
    for t, W in zip(dates, Ws):
        stop = min(t + REB, T)
        rets[:, t - t_start:stop - t_start], d = hold(W, R[t:stop])
        Dend.append(d)
    Dend = np.array(Dend)
    names = [label(c) for c in cfg]
    rr, tt, alpha = [rets], [turnover(Ws, Dend)], {}
    for h in windows:
        cands = [k for k, c in enumerate(cfg) if c[0] == "mix" and c[1] == h]
        r, tu, ch = rule(rets, Ws, Dend, cands, REB, LOOK)
        rr.append(r[None])
        tt.append(tu[None])
        names.append(f"{MIX} (alpha by rule), n = {h}")
        alpha[h] = np.where(ch >= 0, np.array([cfg[c][2] if c >= 0 else np.nan for c in ch]), np.nan)
    keep = -(-LOOK // REB)
    return (names, np.vstack(rr)[:, LOOK:], np.vstack(tt)[:, keep:], t_start + LOOK, keep * REB - LOOK,
            np.array([alpha[h][keep:] for h in windows]))


def run(name):
    t0 = time.time()
    df = load(name)
    R = df.to_numpy()
    G, U, A = [], [], []
    for i, idx in enumerate(draws(name, R.shape[1])):
        names, rets, turn, t_eval, first, alpha = run_one(R[:, idx], SETS[name]["windows"])
        G.append(rets.astype(np.float32))
        U.append(turn.astype(np.float32))
        A.append(alpha)
        print(name, f"portfolio {i + 1}", f"{time.time() - t0:.0f} s", flush=True)
    dates = df.index[t_eval:]
    np.savez_compressed(f"{OUT}/{name}.npz", names=np.array(names), dates=np.array([str(d.date()) for d in dates]),
                        gross=np.array(G), turn=np.array(U), alpha=np.array(A), windows=np.array(SETS[name]["windows"]),
                        reb=REB, first=first)
    print(f"{name}: evaluation {dates[0].date()} to {dates[-1].date()} ({len(dates)} days)")


def table():
    from studies.study6_plot import boot_var, periods_of
    rows, arows = [], []
    for name in ("ff100", "ind49", "sp500"):
        path = f"{OUT}/{name}.npz"
        if not os.path.exists(path):
            continue
        z = np.load(path)
        names, dates = list(z["names"]), pd.to_datetime(z["dates"])
        G, U = z["gross"].astype(float), z["turn"].astype(float)
        reb_days = int(z["first"]) + int(z["reb"]) * np.arange(U.shape[2])
        for wi, h in enumerate(z["windows"]):
            a = z["alpha"][:, wi]
            arows.append({"set": name, "window": int(h), "mean alpha": np.nanmean(a), "median alpha": np.nanmedian(a),
                          "share alpha = 1": np.nanmean(a == 1), "share alpha <= 0.5": np.nanmean(a <= 0.5)})
        for cost in (0, 10, 25):
            X = G if cost == 0 else net(G, U, reb_days, cost)
            for per, m in periods_of(dates).items():
                if cost and per != "full":
                    continue
                Xp = X[:, :, m]
                bv = boot_var(Xp, np.random.default_rng(7))
                vol = np.sqrt(Xp.var(axis=2, ddof=1).mean(axis=0) * 252) * 100
                for h in z["windows"]:
                    ours = names.index(f"{MIX} (alpha by rule), n = {h}")
                    fixed = [names.index(f"{MIX}, n = {h}, alpha = {a:.1f}") if a < 1 else names.index(f"LW nonlinear, n = {h}")
                             for a in ALPHAS]
                    best = fixed[int(np.argmin(vol[fixed]))]
                    rivals = {"LW nonlinear": names.index(f"LW nonlinear, n = {h}"), "BPT 2024": names.index(f"BPT 2024, n = {h}"),
                              "LW linear": names.index(f"LW linear, n = {h}"), "1/N": names.index("1/N"),
                              "best fixed alpha (ex post)": best}
                    row = {"set": name, "window": int(h), "period": per, "cost bp": cost, "days": int(m.sum()),
                           "vol ours %": vol[ours], "turnover ours": np.nanmean(U[:, ours]),
                           "best fixed alpha": ALPHAS[fixed.index(best)]}
                    for rn, j in rivals.items():
                        row[f"vol {rn} %"] = vol[j]
                        row.update({f"{rn}: {k}": v for k, v in compare(Xp, bv, ours, j).items()})
                    rows.append(row)
    S, A = pd.DataFrame(rows), pd.DataFrame(arows)
    S.to_csv(f"{OUT}/summary.csv", index=False)
    A.to_csv(f"{OUT}/alpha.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["set", "window", "vol ours %", "vol LW nonlinear %", "LW nonlinear: ratio", "LW nonlinear: 95% low",
            "LW nonlinear: 95% high", "LW nonlinear: t-stat", "BPT 2024: ratio", "1/N: ratio", "best fixed alpha"]
    print(S[(S.period == "full") & (S["cost bp"] == 0)][cols].round(3).to_string(index=False))
    print(A.round(2).to_string(index=False))


def study5():
    """The Study 5 portfolios and days, held one day each, so the mixture's return is α·r_LW-NL + (1 - α)·r_1/N.
    α is picked daily from the previous 252 days; those first 252 days are a warm-up and excluded for everyone."""
    from studies.study5_ridge import OUT as OUT5, CS_EMP, BPT, compare as compare5
    rows, rng = [], np.random.default_rng(71)
    for p in (54, 150):
        z = np.load(f"{OUT5}/sp500_p{p}.npz")
        names, dates = list(z["names"]), z["dates"]
        for c in CS_EMP:
            G = z[f"gross_{c}"].astype(float)
            rn, re = G[:, names.index("Ledoit-Wolf nonlinear")], G[:, names.index("1/N")]
            M = ALPHAS[:, None, None] * rn[None] + (1 - ALPHAS[:, None, None]) * re[None]
            Tn = G.shape[2]
            ours, chosen = np.full(rn.shape, np.nan), np.full((G.shape[0], Tn), np.nan)
            for t in range(LOOK, Tn):
                k = np.argmin(M[:, :, t - LOOK:t].var(axis=2), axis=0)
                ours[:, t] = M[k, np.arange(G.shape[0]), t]
                chosen[:, t] = ALPHAS[k]
            keep = ["Ledoit-Wolf nonlinear", BPT, "Ledoit-Wolf linear", "1/N", "paper: bona fide, 1/N"]
            X = np.concatenate([ours[:, None, LOOK:], G[:, [names.index(k) for k in keep], LOOK:]], axis=1)
            h = X.shape[2] // 2
            for period, sl in (("full", slice(None)), ("first half", slice(0, h)), ("second half", slice(h, None))):
                d = dates[LOOK:][sl]
                for row in compare5(X[:, :, sl], [MIX] + keep, MIX, rng):
                    if row["estimator"] != MIX:
                        rows.append({"p": p, "c": c, "period": period, "days": len(d), "from": d[0], "to": d[-1],
                                     "mean alpha": np.nanmean(chosen), **row})
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/study5_setting.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[["p", "c", "period", "mean alpha", "estimator", "ratio ours / this", "95% low", "95% high",
               "t-stat (ours - this)"]].round(3).to_string(index=False))


def plot():
    from gmv.style import plt, MUTED, RED, WARM, DARK, TAN, WINE, INK, SLIDE, note, save
    s5 = pd.read_csv(f"{OUT}/study5_setting.csv")
    s5 = s5[s5.period == "full"]
    rivals = {"Ledoit-Wolf nonlinear": DARK, "BPT 2024: ridge + 1/N": WINE, "Ledoit-Wolf linear": TAN,
              "paper: bona fide, 1/N": WARM}
    cs = sorted(s5.c.unique())
    fig, ax = plt.subplots(1, 2, figsize=SLIDE, sharey=True)
    for a, p in zip(ax, (54, 150)):
        w = 0.8 / len(rivals)
        for k, (name, colour) in enumerate(rivals.items()):
            t = s5[(s5.p == p) & (s5.estimator == name)].set_index("c").reindex(cs)
            x = np.arange(len(cs)) + (k - len(rivals) / 2 + 0.5) * w
            r = t["ratio ours / this"]
            a.errorbar(x, r, yerr=[r - t["95% low"], t["95% high"] - r], fmt="o", color=colour, ms=8, capsize=3, lw=2,
                       label=f"vs {name}")
        a.axhline(1, color=MUTED, lw=1.2, ls=":")
        a.set_xticks(range(len(cs)), [f"c = {c}" for c in cs])
        a.grid(axis="x", visible=False)
        a.set_title(f"S&P 500, p = {p}", fontsize=15)
    ax[0].set_ylim(0.4, 1.12)
    ax[0].set_ylabel("variance ratio, ours / rival (95% CI)\nbelow 1: ours is less risky", fontsize=13)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=12)
    note(ax[0], -0.3, 1.06, "below 1 against every rival, at every c", size=21)
    plt.tight_layout()
    save(fig, f"{OUT}/sp500.png")

    S = pd.read_csv(f"{OUT}/summary.csv")
    A = pd.read_csv(f"{OUT}/alpha.csv")
    S = S[(S.period == "full") & (S["cost bp"] == 0)]
    labels = {"ff100": "100 size x B/M portfolios", "ind49": "49 industries", "sp500": "S&P 500 stocks, p = 150"}
    colours = {"ff100": INK, "ind49": WARM, "sp500": RED}
    fig, ax = plt.subplots(1, 2, figsize=SLIDE)
    for s, colour in colours.items():
        t = S[S.set == s]
        r = t["LW nonlinear: ratio"]
        ax[0].errorbar(t.window, r, yerr=[r - t["LW nonlinear: 95% low"], t["LW nonlinear: 95% high"] - r], fmt="o-",
                       color=colour, capsize=3, lw=2, label=labels[s])
        a = A[A.set == s]
        ax[1].plot(a.window, a["mean alpha"], "o-", color=colour, lw=2.6, label=labels[s])
    for a in ax:
        a.set_xscale("log")
        a.set_xticks([63, 126, 252, 504, 1008], ["63", "126", "252", "504", "1008"])
        a.minorticks_off()
        a.set_xlabel("window, trading days")
    ax[0].axhline(1, color=MUTED, lw=1.2, ls=":")
    ax[0].set(ylabel="variance ratio vs LW nonlinear (95% CI)", title="Monthly rebalancing, Study 6 data")
    ax[1].set(ylabel=r"mean $\alpha$ chosen by the rule", ylim=(0.5, 1.02), title="Weight on LW nonlinear")
    ax[0].legend(loc="lower left", fontsize=12)
    note(ax[1], 70, 0.56, "stocks: 1/N is closer to the GMV,\nso more weight on the target", size=19)
    plt.tight_layout()
    save(fig, f"{OUT}/monthly.png")


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "table": table, "study5": study5, "plot": plot}[sys.argv[1]]()
