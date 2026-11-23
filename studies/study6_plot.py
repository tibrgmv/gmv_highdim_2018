"""Study 6, tables and figures from the saved backtests.

    summary.csv   every configuration and rule, every period: volatility, Sharpe, turnover, costs, ratios vs baselines
    levers.csv    the lever decomposition; rule.csv what the rule picks; weekly.csv weekly vs monthly rebalancing
"""
import os
import numpy as np
import pandas as pd
from gmv.backtest import block_counts, boot_var as _boot_var, compare, net
from gmv.data import risk_free
from studies.study6_levers import OUT, SETS, MAIN_VOLS, LOOK

BLOCK = 21
COSTS = (10, 25)
PERIODS = {"full": None, "1968-1989": ("1968-01-01", "1989-12-31"), "1990-2007": ("1990-01-01", "2007-12-31"),
           "2008-today": ("2008-01-01", "2100-01-01"), "crisis 2000-2002": ("2000-01-01", "2002-12-31"),
           "crisis 2008-2009": ("2008-01-01", "2009-12-31"), "crisis 2020": ("2020-01-01", "2020-12-31")}
BASE = "sample, n = 252, vol: none"
BEST_PRACTICE = "LW nonlinear, n = 252, vol: none"
LEVERS = ("estimator only", "window only", "volatility only", "all three", "past-data rule")


def load(name, tag=""):
    z = np.load(f"{OUT}/{name}{tag}.npz")
    dates = pd.to_datetime(z["dates"])
    G, U = z["gross"].astype(float), z["turn"].astype(float)
    reb_days = int(z["first"]) + int(z["reb"]) * np.arange(U.shape[2])
    return list(z["names"]), dates, G, U, reb_days, risk_free(dates)


def boot_var(X, rng):
    C = block_counts(X.shape[2], rng, BLOCK).astype(np.float32)
    C /= X.shape[2]
    return _boot_var(X, C)


def periods_of(dates):
    out = {}
    for name, span in PERIODS.items():
        m = np.ones(len(dates), bool) if span is None else (dates >= span[0]) & (dates <= span[1])
        if m.sum() > LOOK and (span is None or not m.all()):
            out[name] = m
    return out


def lever_choices(names, var_full):
    """What each lever switches to: the best by full-period variance (ex post), and the rule (ex ante)."""
    v = dict(zip(names, var_full))
    best = lambda cands: min(cands, key=lambda k: v[k])
    at252 = [n for n in names if n == "1/N" or n.endswith(", n = 252, vol: none")]
    hist = lambda e: [n for n in names if n.startswith(e + ",") and n.endswith("vol: none")]
    allc = [n for n in names if not n.startswith("rule") and (n == "1/N" or n.split("vol: ")[-1] in MAIN_VOLS)]
    return {base: {"estimator only": best(at252), "window only": best(hist(e)),
                   "volatility only": f"{e}, n = 252, vol: RiskMetrics", "all three": best(allc),
                   "past-data rule": "rule: everything",
                   "volatility only (half-life 60)": f"{e}, n = 252, vol: half-life 60",
                   f"rule: {e} (window and vol)": f"rule: {e}"}
            for base, e in ((BASE, "sample"), (BEST_PRACTICE, "LW nonlinear"))}


def tables(name):
    names, dates, G, U, reb_days, rf = load(name)
    rng = np.random.default_rng(67)
    k0, k1 = names.index(BASE), names.index(BEST_PRACTICE)
    choice = lever_choices(names, G.var(axis=2, ddof=1).mean(axis=0))
    summ, lev = [], []
    for per, m in periods_of(dates).items():
        Xg = G[:, :, m]
        bv = boot_var(Xg, rng)
        rd = np.flatnonzero(np.isin(np.arange(len(dates)), reb_days) & m)
        Ug = U[:, :, np.isin(reb_days, rd)]
        nets = {c: net(G, U, reb_days, c)[:, :, m] for c in COSTS}
        ex = Xg - rf[m]
        for k, s in enumerate(names):
            row = {"set": name, "period": per, "from": str(dates[m][0].date()), "to": str(dates[m][-1].date()),
                   "series": s, "volatility (% a year)": 100 * np.sqrt(252 * Xg[:, k].var(axis=1, ddof=1).mean()),
                   "Sharpe ratio (a year)": np.mean(np.sqrt(252) * ex[:, k].mean(axis=1) / ex[:, k].std(axis=1, ddof=1)),
                   "turnover per rebalance": np.nanmean(Ug[:, k])}
            for c in COSTS:
                row[f"volatility net of {c} bp"] = 100 * np.sqrt(252 * nets[c][:, k].var(axis=1, ddof=1).mean())
                exn = nets[c][:, k] - rf[m]
                row[f"Sharpe net of {c} bp"] = np.mean(np.sqrt(252) * exn.mean(axis=1) / exn.std(axis=1, ddof=1))
            for tag, kb in (("vs baseline", k0), ("vs best practice", k1)):
                if k != kb:
                    row.update({f"{a} {tag}": b for a, b in compare(Xg, bv, k, kb).items()})
            summ.append(row)
        for base, ch in choice.items():
            kb = names.index(base)
            for lever, s in ch.items():
                lev.append({"set": name, "period": per, "baseline": base, "lever": lever, "switch to": s,
                            **compare(Xg, bv, names.index(s), kb)})
    return pd.DataFrame(summ), pd.DataFrame(lev)


def rule_picks(name):
    """Share of rebalances at which the overall rule picks each estimator, history and volatility (first portfolio)."""
    names, dates, G, U, reb_days, rf = load(name)
    cands = [k for k, n in enumerate(names) if not n.startswith("rule") and (n == "1/N" or n.split("vol: ")[-1] in MAIN_VOLS)]
    X = G[0]
    picks = []
    for d in reb_days[reb_days >= LOOK]:
        c = cands[int(np.argmin(X[cands, d - LOOK:d].var(axis=1)))]
        assert np.allclose(X[c, d:d + 1], X[names.index("rule: everything"), d:d + 1], equal_nan=True)
        picks.append((dates[d], names[c]))
    p = pd.DataFrame(picks, columns=["date", "series"])
    p["estimator"] = p.series.str.split(",").str[0]
    p["history"] = p.series.str.split(", ").str[1].fillna("-")
    p["volatility"] = p.series.str.split("vol: ").str[1].fillna("-")
    return pd.DataFrame([{"set": name, "what": col, "value": val, "share of rebalances": share}
                         for col in ("estimator", "history", "volatility")
                         for val, share in p[col].value_counts(normalize=True).items()])


def weekly():
    out = []
    for tag in ("", "_weekly"):
        names, dates, G, U, reb_days, rf = load("ff100", tag)
        bv = boot_var(G, np.random.default_rng(68))
        for k, s in enumerate(names):
            if s.startswith("rule"):
                continue
            row = {"rebalancing": "weekly" if tag else "monthly", "series": s, "from": str(dates[0].date()),
                   "volatility (% a year)": 100 * np.sqrt(252 * G[:, k].var(axis=1, ddof=1).mean()),
                   "turnover per rebalance": np.nanmean(U[:, k]),
                   "turnover a year": np.nanmean(U[:, k]) * 252 / (5 if tag else 21)}
            for c in COSTS:
                row[f"volatility net of {c} bp"] = 100 * np.sqrt(252 * net(G, U, reb_days, c)[:, k].var(axis=1, ddof=1).mean())
            if s != BASE:
                row.update(compare(G, bv, k, names.index(BASE)))
            out.append(row)
    w = pd.DataFrame(out)
    return w[w.series.isin(set(w[w.rebalancing == "weekly"].series))]


def plot():
    from gmv.style import plt, INK, MUTED, RED, WARM, DARK, LIGHT, TAN, SAND, WINE, SLIDE, note, save
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    sets = [s for s in SETS if os.path.exists(f"{OUT}/{s}.npz")]
    S, L, P = [], [], []
    for s in sets:
        a, b = tables(s)
        S.append(a)
        L.append(b)
        P.append(rule_picks(s))
        print(s, "done", flush=True)
    S, L, P = pd.concat(S), pd.concat(L), pd.concat(P)
    S.to_csv(f"{OUT}/summary.csv", index=False)
    L.to_csv(f"{OUT}/levers.csv", index=False)
    P.to_csv(f"{OUT}/rule.csv", index=False)
    if os.path.exists(f"{OUT}/ff100_weekly.npz"):
        W = weekly()
        W.to_csv(f"{OUT}/weekly.csv", index=False)
        print(W.round(3).to_string(index=False))
    print(L.round(3).to_string(index=False))
    print(P.round(3).to_string(index=False))

    col = {"1/N": LIGHT, "inverse variance": SAND, "sample": WARM, "paper": RED, "BPT 2024": WINE,
           "LW linear": TAN, "LW nonlinear": INK}
    lcol = {"estimator only": DARK, "window only": WARM, "volatility only": TAN, "all three": INK, "past-data rule": RED}

    def vol_curve(a, s, vol, est=("sample", "paper", "BPT 2024", "LW linear", "LW nonlinear"), lw=2.6, labels=True):
        t = S[(S.set == s) & (S.period == "full")].set_index("series")["volatility (% a year)"]
        ws = SETS[s]["windows"]
        for e in est:
            a.plot(ws, [t[f"{e}, n = {w}, vol: {vol}"] for w in ws], marker="o", color=col[e], lw=lw,
                   label=e if labels else None, ls="--" if e == "LW linear" else "-")
        a.set_xscale("log")
        a.set_xticks(ws, [str(w) for w in ws])
        a.minorticks_off()

    def lever_bars(a, s, period, base, ypos, h=0.36, labels=True):
        t = L[(L.set == s) & (L.period == period) & (L.baseline == base)].set_index("lever")
        for i, lv in enumerate(LEVERS):
            r = t.loc[lv]
            red, lo, hi = 100 * (1 - r["ratio"]), 100 * (1 - r["95% high"]), 100 * (1 - r["95% low"])
            a.barh(ypos - i * h, red, h * 0.85, color=lcol[lv], label=lv if labels else None)
            a.errorbar(red, ypos - i * h, xerr=[[red - lo], [hi - red]], fmt="none", ecolor=MUTED, capsize=3, lw=1.4)

    fig, ax = plt.subplots(2, len(sets), figsize=(7 * len(sets), 11))
    ax = np.atleast_2d(ax).reshape(2, -1)
    for j, s in enumerate(sets):
        vol_curve(ax[0, j], s, "none")
        vol_curve(ax[0, j], s, "RiskMetrics", lw=1.4, labels=False)
        ax[0, j].set_title(f"{SETS[s]['label']}\nthick: no vol scaling, thin: RiskMetrics", fontsize=13)
        ax[0, j].set(xlabel="window, days", ylabel="volatility, % a year")
        lo = S[(S.set == s) & (S.period == "full") & S.series.str.contains(", n = ")]["volatility (% a year)"]
        ax[0, j].set_ylim(lo.min() * 0.9, min(lo.quantile(0.9), lo.min() * 2.2))
        for b, base in enumerate((BASE, BEST_PRACTICE)):
            lever_bars(ax[1, j], s, "full", base, -b * 2.4, labels=(b == 0))
        ax[1, j].set_yticks([-0.72, -3.12], ["from sample\ncovariance, n = 252", "from Ledoit-Wolf\nnonlinear, n = 252"])
        ax[1, j].axvline(0, color=MUTED, lw=1)
        ax[1, j].set_xlabel("variance reduction, % (95% CI)")
        ax[1, j].grid(axis="y", visible=False)
    ax[0, 0].legend(fontsize=10)
    ax[1, 0].legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    save(fig, f"{OUT}/overview.png")

    lv = L[(L.set == "ff100") & (L.period == "full")].set_index(["baseline", "lever"])
    red = lambda base, lever: 100 * (1 - lv.loc[(base, lever), "ratio"])

    fig, a = plt.subplots(figsize=SLIDE)
    lever_bars(a, "ff100", "full", BASE, 0, h=0.42)
    lever_bars(a, "ff100", "full", BEST_PRACTICE, -2.9, h=0.42, labels=False)
    a.set_yticks([-0.84, -3.74], ["starting from the\nsample covariance,\n252 days", "starting from\nLedoit-Wolf nonlinear,\n252 days"])
    a.axvline(0, color=MUTED, lw=1)
    a.grid(axis="y", visible=False)
    a.set_xlabel("reduction of out-of-sample variance, % (95% interval)")
    a.set_title("100 size x book-to-market portfolios, 1969-2026: what each lever buys", fontsize=15)
    a.legend(loc="lower right", fontsize=12)
    alone = [red(BASE, l) for l in LEVERS[:3]]
    note(a, 28.5, 0.05, f"each lever alone: {min(alone):.0f}-{max(alone):.0f}%,\nall three: {red(BASE, 'all three'):.0f}%", size=21)
    note(a, 16, -3.6, "from a good estimator,\nlittle is left", size=21)
    if red(BEST_PRACTICE, "estimator only") == 0:
        a.text(0.6, -2.9, "already the best estimator", va="center", fontsize=11, color=MUTED)
    a.set_xlim(-6, 45)
    save(fig, f"{OUT}/levers.png", dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=SLIDE, sharey=True)
    for a, vol, title in ((ax[0], "none", "no volatility scaling"), (ax[1], "RiskMetrics", "EWMA volatility scaling (RiskMetrics)")):
        vol_curve(a, "ff100", vol, lw=2.8)
        a.set_title(title, fontsize=15)
        a.set(xlabel="window, trading days", ylim=(9, 15))
    ax[0].set_ylabel("out-of-sample volatility, % a year")
    t = S[(S.set == "ff100") & (S.period == "full")].set_index("series")["volatility (% a year)"]
    ax[0].text(1000, 14.8, f"1/N: {t['1/N']:.1f}% (off the scale)", ha="right", fontsize=12, color=MUTED)
    fig.legend(*[h[:5] for h in ax[0].get_legend_handles_labels()], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07), fontsize=12)
    note(ax[0], 300, 12.4, "the sample covariance and\nthe paper's estimator need\nlong windows; shrinkage\nestimators do not", size=19)
    plt.tight_layout()
    save(fig, f"{OUT}/window.png", dpi=150)

    crises = ("crisis 2000-2002", "crisis 2008-2009", "crisis 2020")
    fig, ax = plt.subplots(1, 2, figsize=SLIDE, sharey=True)
    w = 0.16
    for a, base, title in ((ax[0], BASE, "from the sample covariance, 252 days"),
                           (ax[1], BEST_PRACTICE, "from Ledoit-Wolf nonlinear, 252 days")):
        t = L[(L.set == "ff100") & (L.baseline == base)].set_index(["period", "lever"])
        for i, lever in enumerate(LEVERS):
            x = np.arange(len(crises)) + (i - 2) * w
            r = np.array([t.loc[(c, lever), "ratio"] for c in crises])
            lo = np.array([t.loc[(c, lever), "95% high"] for c in crises])
            hi = np.array([t.loc[(c, lever), "95% low"] for c in crises])
            a.bar(x, 100 * (1 - r), w * 0.9, color=lcol[lever], label=lever)
            a.errorbar(x, 100 * (1 - r), yerr=[100 * (lo - r), 100 * (r - hi)], fmt="none", ecolor=MUTED, capsize=2, lw=1.2)
        a.axhline(0, color=MUTED, lw=1)
        a.set_xticks(range(len(crises)), ["2000-2002", "2008-2009", "2020"])
        a.grid(axis="x", visible=False)
        a.set_title(title, fontsize=15)
    ax[0].set_ylabel("variance reduction, % (95% interval)")
    if red(BEST_PRACTICE, "estimator only") == 0:
        ax[1].text(-0.32, 1.2, "0: already the best", rotation=90, va="bottom", ha="center", fontsize=10, color=MUTED)
    note(ax[1], -0.3, 40, "from a good estimator, no lever\nis reliably better in a crisis", size=20)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07), fontsize=12)
    plt.tight_layout()
    save(fig, f"{OUT}/crises.png", dpi=150)


if __name__ == "__main__":
    plot()
