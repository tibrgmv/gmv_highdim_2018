"""Study 4. The paper's S&P 500 study (Section 4, Figures 15-16), with the target 1/N added as a benchmark.

22 April 2013 - 19 March 2014 (T = 230); 1000 random portfolios of p = 54; rolling window n = p/c,
c in {0.5, 0.9, 1.5, 2}; each portfolio held for one day. Variance and Sharpe ratio as in (4.1).
    python -m studies.study4_sp500        ~5 min; delete results/study4/backtest.csv to recompute
"""
import os
import numpy as np
import pandas as pd
from gmv.data import results_dir, sp500_paper_window
from gmv.estimators import pinv, sample_cov, gmv, bona_fide, frahm_memmel
from gmv.style import plt, MUTED, WARM, DARK, COLOURS, STYLES, SLIDE, note, arrow, save

OUT = results_dir(4)
P, DRAWS, CS = 54, 1000, (0.5, 0.9, 1.5, 2.0)
ORDER = ("traditional", "Frahm-Memmel", "equal weights", "bona fide")


def backtest(r, n, c):
    T = len(r)
    rets = {"traditional": [], "Frahm-Memmel": [], "bona fide": [], "equal weights": []}
    for t in range(n, T):
        S = sample_cov(r[t - n:t])
        Sinv = np.linalg.inv(S) if c < 1 else pinv(S)
        rets["traditional"].append(gmv(Sinv) @ r[t])
        rets["bona fide"].append(bona_fide(S, n, Sinv=Sinv)[0] @ r[t])
        rets["equal weights"].append(r[t].mean())
        if c < 1:
            rets["Frahm-Memmel"].append(frahm_memmel(S, n) @ r[t])
    return rets


def run(R, draws):
    T = len(R)
    rows = []
    for c in CS:
        n = int(round(P / c))
        for k, idx in enumerate(draws):
            for name, x in backtest(R[:, idx], n, c).items():
                if not x:
                    continue
                x = np.asarray(x)
                mu = x.mean()
                var = ((x - mu) ** 2).sum() / (T - n - 1)
                rows.append({"c": c, "n": n, "draw": k, "estimator": name, "variance": var, "sharpe": mu / np.sqrt(var)})
        print("c =", c, "done", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/backtest.csv", index=False)


def summarise(res):
    """Medians, and how often the bona fide estimator wins, portfolio by portfolio."""
    rows = []
    for c, g in res.groupby("c"):
        v = g.pivot(index="draw", columns="estimator", values="variance")
        s = g.pivot(index="draw", columns="estimator", values="sharpe")
        for name in v.columns:
            other = name != "bona fide"
            rows.append({"c": c, "estimator": name, "median variance (x1e-5)": 1e5 * v[name].median(),
                         "median Sharpe": s[name].median(),
                         "bona fide has lower variance (share)": (v["bona fide"] < v[name]).mean() if other else np.nan,
                         "bona fide has higher Sharpe (share)": (s["bona fide"] > s[name]).mean() if other else np.nan})
    return pd.DataFrame(rows)


def intensity(R, draws):
    """The α the bona fide estimator actually uses (first 100 portfolios, every fifth window)."""
    rows = []
    for c in CS:
        n = int(round(P / c))
        a = []
        for idx in draws[:100]:
            r = R[:, idx]
            for t in range(n, len(R), 5):
                S = sample_cov(r[t - n:t])
                a.append(bona_fide(S, n, Sinv=np.linalg.inv(S) if c < 1 else pinv(S))[1])
        rows.append({"c": c, "mean alpha": np.mean(a), "median alpha": np.median(a), "share negative": np.mean(np.array(a) < 0)})
    return pd.DataFrame(rows)


def ecdf(a, x, name, lw=2.6):
    x = np.sort(x)
    a.plot(x, np.arange(1, len(x) + 1) / len(x), color=COLOURS[name], ls=STYLES[name], lw=lw, label=name)


def plot(res, summ):
    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    xmax = {0.5: 2.5e-4, 0.9: 1.5e-3, 1.5: 2e-3, 2.0: 2e-3}
    for i, c in enumerate(CS):
        g = res[res.c == c]
        for name in ORDER:
            h = g[g.estimator == name]
            if len(h):
                ecdf(ax[0, i], h.variance.to_numpy(), name)
                ecdf(ax[1, i], h.sharpe.to_numpy(), name)
        ax[0, i].set(xlim=(0, xmax[c]), title=f"Out-of-sample variance, c = {c}", xlabel="daily variance")
        ax[1, i].set(xlim=(-0.2, 0.25), title=f"Out-of-sample Sharpe ratio, c = {c}", xlabel="daily Sharpe ratio")
        ax[0, i].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax[0, 0].set_ylabel("share of portfolios")
    ax[1, 0].set_ylabel("share of portfolios")
    ax[0, 0].legend(loc="lower right")
    plt.tight_layout()
    save(fig, f"{OUT}/overview.png")

    g = res[res.c == 0.9]
    fig, a = plt.subplots(figsize=SLIDE)
    for name in ORDER:
        ecdf(a, 1e4 * g[g.estimator == name].variance.to_numpy(), name, lw=3)
    a.set_xscale("log")
    a.set(xlim=(0.35, 12), ylim=(0, 1.02), ylabel="share of portfolios",
          xlabel=r"out-of-sample daily variance, $\times 10^{-4}$ (log scale)")
    a.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
    a.text(2.45, 0.30, "Frahm-Memmel", color=DARK, fontsize=16)
    a.text(6.2, 0.55, "traditional", color=WARM, fontsize=16)
    note(a, 1.0, 0.13, "shrinkage beats both rivals, as in the paper")
    arrow(a, (0.98, 0.18), (0.66, 0.38), -0.3)
    note(a, 0.37, 0.88, "but plain 1/N\nis lower still")
    arrow(a, (0.5, 0.77), (0.55, 0.6), 0.2)
    save(fig, f"{OUT}/variance_c09.png")

    fig, ax = plt.subplots(1, 2, figsize=SLIDE, sharey=True)
    w = 0.26
    for a, col, title in ((ax[0], "bona fide has lower variance (share)", "lower out-of-sample variance"),
                          (ax[1], "bona fide has higher Sharpe (share)", "higher out-of-sample Sharpe ratio")):
        for k, rival in enumerate(("traditional", "Frahm-Memmel", "equal weights")):
            sub = summ[summ.estimator == rival].set_index("c")[col].reindex(CS)
            x = np.arange(len(CS)) + (k - 1) * w
            a.bar(x, 100 * sub.to_numpy(), w * 0.92, color=COLOURS[rival], label="vs 1/N" if rival == "equal weights" else f"vs {rival}")
        a.axhline(50, color=MUTED, lw=1.2, ls=":")
        a.set_xticks(np.arange(len(CS)), [f"c = {c}" for c in CS])
        a.set_title(f"Shrinkage wins: {title}", fontsize=15)
        a.grid(axis="x", visible=False)
    ax[0].set(ylabel="% of the 1000 portfolios", ylim=(0, 128), yticks=[0, 25, 50, 75, 100])
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=13)
    note(ax[1], 1.5, 116, "against 1/N, shrinkage loses 2 times out of 3", ha="center", size=22)
    save(fig, f"{OUT}/win_rates.png")


if __name__ == "__main__":
    R = sp500_paper_window().to_numpy()
    rng = np.random.default_rng(54)
    draws = [rng.choice(R.shape[1], P, replace=False) for _ in range(DRAWS)]
    if not os.path.exists(f"{OUT}/backtest.csv"):
        run(R, draws)
    res = pd.read_csv(f"{OUT}/backtest.csv")
    summ, alpha = summarise(res), intensity(R, draws)
    summ.to_csv(f"{OUT}/summary.csv", index=False)
    alpha.to_csv(f"{OUT}/alpha.csv", index=False)
    print(summ.round(3).to_string(index=False), alpha.round(3).to_string(index=False), sep="\n\n")
    plot(res, summ)
