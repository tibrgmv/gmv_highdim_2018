"""
Study 3 — The paper's empirical study (Section 4, Figures 15 and 16).
Daily returns of S&P 500 stocks, 22 April 2013 to 19 March 2014 (T = 230). Draw 1000 random portfolios of
p = 54 stocks. For each, estimate the weights on a rolling window of n = p/c days, hold them for one day,
move on. Out-of-sample variance and Sharpe ratio as in (4.1). c in {0.5, 0.9, 1.5, 2}: n = 108, 60, 36, 27.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (pinv, INK, MUTED, SURFACE, RED, WARM, DARK, LIGHT, COLOURS, STYLES, SLIDE, DATA, plot_style, note, arrow,
                    save, sample_cov, gmv, bona_fide, frahm_memmel)

plot_style()
OUT = "results/study3"
P, DRAWS, CS = 54, 1000, (0.5, 0.9, 1.5, 2.0)
R = pd.read_csv(os.path.join(DATA, "sp500_returns.csv"), index_col=0, parse_dates=True)
T, N = R.shape
print(f"{N} stocks, T = {T} days (the paper: 417 stocks, T = 230)")
rng = np.random.default_rng(54)
draws = [rng.choice(N, P, replace=False) for _ in range(DRAWS)]
Rv = R.to_numpy()


def backtest(r, n, c):
    """Out-of-sample daily returns of each strategy on one portfolio, windows ending at t = n..T-1."""
    rets = {"traditional": [], "Frahm-Memmel": [], "bona fide": [], "equal weights": []}
    for t in range(n, T):
        S = sample_cov(r[t - n:t])
        Sinv = np.linalg.inv(S) if c < 1 else pinv(S)
        nxt = r[t]
        rets["traditional"].append(gmv(Sinv) @ nxt)
        rets["bona fide"].append(bona_fide(S, n, Sinv=Sinv)[0] @ nxt)
        rets["equal weights"].append(nxt.mean())
        if c < 1:
            rets["Frahm-Memmel"].append(frahm_memmel(S, n) @ nxt)
    return rets


rows = []
for c in (CS if not os.path.exists(f"{OUT}_all.csv") else ()):
    n = int(round(P / c))
    for k, idx in enumerate(draws):
        for name, x in backtest(Rv[:, idx], n, c).items():
            if not x:
                continue
            x = np.asarray(x)
            mu = x.mean()
            var = ((x - mu) ** 2).sum() / (T - n - 1)                 # (4.1)
            rows.append({"c": c, "n": n, "draw": k, "estimator": name, "variance": var, "sharpe": mu / np.sqrt(var)})
    print("c =", c, "done", flush=True)
if rows:
    pd.DataFrame(rows).to_csv(f"{OUT}_all.csv", index=False)
res = pd.read_csv(f"{OUT}_all.csv")      # delete this file to recompute (about 5 minutes)

# Summary: medians and how often the bona fide estimator wins, portfolio by portfolio
summ = []
for c, g in res.groupby("c"):
    piv_v = g.pivot(index="draw", columns="estimator", values="variance")
    piv_s = g.pivot(index="draw", columns="estimator", values="sharpe")
    for name in piv_v.columns:
        summ.append({"c": c, "estimator": name, "median variance (x1e-5)": 1e5 * piv_v[name].median(),
                     "median Sharpe": piv_s[name].median(),
                     "bona fide has lower variance (share)": (piv_v["bona fide"] < piv_v[name]).mean() if name != "bona fide" else np.nan,
                     "bona fide has higher Sharpe (share)": (piv_s["bona fide"] > piv_s[name]).mean() if name != "bona fide" else np.nan})
summ = pd.DataFrame(summ)
pd.set_option("display.width", 200)
print(summ.round(3).to_string(index=False))
summ.to_csv(f"{OUT}.csv", index=False)

# The shrinkage intensity the bona fide estimator actually uses (first 100 portfolios, every fifth window)
alpha = []
for c in CS:
    n = int(round(P / c))
    a = []
    for idx in draws[:100]:
        r = Rv[:, idx]
        for t in range(n, T, 5):
            S = sample_cov(r[t - n:t])
            a.append(bona_fide(S, n, Sinv=np.linalg.inv(S) if c < 1 else pinv(S))[1])
    alpha.append({"c": c, "mean alpha": np.mean(a), "median alpha": np.median(a), "share negative": np.mean(np.array(a) < 0)})
alpha = pd.DataFrame(alpha)
print(alpha.round(3).to_string(index=False))
alpha.to_csv(f"{OUT}_alpha.csv", index=False)

ORDER = ("traditional", "Frahm-Memmel", "equal weights", "bona fide")
os.makedirs("results/slides", exist_ok=True)


def ecdf(a, x, name, lw=2.6):
    x = np.sort(x)
    a.plot(x, np.arange(1, len(x) + 1) / len(x), color=COLOURS[name], ls=STYLES[name], lw=lw, label=name)


# Overview: the paper's Figures 15 and 16, with the equal-weight portfolio added
fig, ax = plt.subplots(2, 4, figsize=(20, 9))
xmax = {0.5: 2.5e-4, 0.9: 1.5e-3, 1.5: 2e-3, 2.0: 2e-3}
for i, c in enumerate(CS):
    g = res[res.c == c]
    for name in ORDER:
        h = g[g.estimator == name]
        if len(h):
            ecdf(ax[0, i], h.variance.to_numpy(), name); ecdf(ax[1, i], h.sharpe.to_numpy(), name)
    ax[0, i].set_xlim(0, xmax[c]); ax[0, i].set_title(f"Out-of-sample variance, c = {c}")
    ax[1, i].set_xlim(-0.2, 0.25); ax[1, i].set_title(f"Out-of-sample Sharpe ratio, c = {c}")
    ax[0, i].set_xlabel("daily variance"); ax[1, i].set_xlabel("daily Sharpe ratio")
    ax[0, i].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
ax[0, 0].set_ylabel("share of portfolios"); ax[1, 0].set_ylabel("share of portfolios"); ax[0, 0].legend(loc="lower right")
plt.tight_layout(); save(fig, f"{OUT}.png"); plt.close(fig)

# Slide: c = 0.9, the paper's ranking reproduces, and the equal-weight portfolio
g = res[res.c == 0.9]
fig, a = plt.subplots(figsize=SLIDE)
for name in ORDER:
    ecdf(a, 1e4 * g[g.estimator == name].variance.to_numpy(), name, lw=3)
a.set_xscale("log"); a.set_xlim(0.35, 12); a.set_ylim(0, 1.02)
a.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
a.set_xlabel(r"out-of-sample daily variance, $\times 10^{-4}$ (log scale)"); a.set_ylabel("share of portfolios")
a.text(2.45, 0.30, "Frahm-Memmel", color=DARK, fontsize=16)
a.text(6.2, 0.55, "traditional", color=WARM, fontsize=16)
note(a, 1.0, 0.13, "shrinkage beats both rivals, as in the paper")
arrow(a, (0.98, 0.18), (0.66, 0.38), -0.3)
note(a, 0.37, 0.88, "but plain 1/N\nis lower still")
arrow(a, (0.5, 0.77), (0.55, 0.6), 0.2)
save(fig, "results/slides/s3_variance_c09.png"); plt.close(fig)

# Slide: how often the paper's estimator wins, portfolio by portfolio
fig, ax2 = plt.subplots(1, 2, figsize=SLIDE, sharey=True)
w = 0.26
for a, col, title in ((ax2[0], "bona fide has lower variance (share)", "lower out-of-sample variance"),
                      (ax2[1], "bona fide has higher Sharpe (share)", "higher out-of-sample Sharpe ratio")):
    for k, rival in enumerate(("traditional", "Frahm-Memmel", "equal weights")):
        sub = summ[summ.estimator == rival].set_index("c")[col].reindex(CS)
        x = np.arange(len(CS)) + (k - 1) * w
        a.bar(x, 100 * sub.to_numpy(), w * 0.92, color=COLOURS[rival], label=f"vs {rival}" if rival != "equal weights" else "vs 1/N")
    a.axhline(50, color=MUTED, lw=1.2, ls=":")
    a.set_xticks(np.arange(len(CS)), [f"c = {c}" for c in CS]); a.set_title(f"Shrinkage wins: {title}", fontsize=15)
    a.grid(axis="x", visible=False)
ax2[0].set_ylabel("% of the 1000 portfolios"); ax2[0].set_ylim(0, 128); ax2[0].set_yticks([0, 25, 50, 75, 100])
fig.legend(*ax2[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=13)
note(ax2[1], 1.5, 116, "against 1/N, shrinkage loses 2 times out of 3", ha="center", size=22)
save(fig, "results/slides/s3_win_rates.png"); plt.close(fig)
