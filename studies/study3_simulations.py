"""Study 3. The paper's simulation study (Figures 7-14).

Normal or Student-t(5) returns, bounded or unbounded (one-factor) spectrum; c in {0.1, 0.5, 0.9, 1.8}.
Average loss as p grows (300 samples per point) and its distribution at p = 306 (500 samples).
    python -m studies.study3_simulations run normal bounded      one setting (~3 min); also t5, unbounded
    python -m studies.study3_simulations plot
"""
import os
import sys
import numpy as np
import pandas as pd
from gmv.data import results_dir
from gmv.designs import make_sigma, simulate
from gmv.estimators import pinv, sample_cov, gmv, bona_fide, oracle, s_star, frahm_memmel
from gmv.theory import relative_loss, gmv_variance

OUT = results_dir(3)
CS = (0.1, 0.5, 0.9, 1.8)
PS = (9, 18, 27, 36, 54, 72, 99, 144, 207, 288)
P_LOCAL = 306
REPS_GLOBAL, REPS_LOCAL = 300, 500
SETTINGS = [("normal", "bounded"), ("t5", "bounded"), ("normal", "unbounded"), ("t5", "unbounded")]
NAMES = ("traditional", "Frahm-Memmel", "oracle", "bona fide")


def one_rep(Sigma, root, inv_root, v, n, c, dist, rng):
    Y, X = simulate(root, n, rng, dist)
    S = sample_cov(Y)
    Sinv = np.linalg.inv(S) if c < 1 else pinv(S)
    return [relative_loss(gmv(Sinv), Sigma, v),
            relative_loss(frahm_memmel(S, n), Sigma, v) if c < 1 else np.nan,
            relative_loss(oracle(Sinv if c < 1 else s_star(X, inv_root), Sigma)[0], Sigma, v),
            relative_loss(bona_fide(S, n, Sinv=Sinv)[0], Sigma, v)]


def run(dist, spec, only=None):
    """All c, or with `only` a single c that replaces its part of a saved run."""
    rng = np.random.default_rng(sum(map(ord, dist + spec)) + (0 if only is None else 7))
    path = f"{OUT}/{dist}_{spec}.npz"
    if only is not None and os.path.exists(path):
        saved = np.load(path)
        glob, local = saved["glob"].copy(), saved["local"].copy()
    else:
        glob = np.full((len(CS), len(PS), REPS_GLOBAL, 4), np.nan)
        local = np.full((len(CS), REPS_LOCAL, 4), np.nan)
    for j, p in enumerate(PS + (P_LOCAL,)):
        Sigma, root, inv_root = make_sigma(p, spec, rng)
        v = gmv_variance(Sigma)
        for i, c in enumerate(CS):
            if only is not None and c != only:
                continue
            n = int(round(p / c))
            reps = REPS_LOCAL if p == P_LOCAL else REPS_GLOBAL
            res = np.array([one_rep(Sigma, root, inv_root, v, n, c, dist, rng) for _ in range(reps)])
            if p == P_LOCAL:
                local[i] = res
            else:
                glob[i, j] = res
        print(dist, spec, "p =", p, flush=True)
    np.savez(path, glob=glob, local=local)


def plot():
    from gmv.style import plt, INK, MUTED, DARK, COLOURS, STYLES, SLIDE, note, arrow, save
    label = {"normal": "normal", "t5": "Student-t"}
    rows, D = [], {}
    fig, ax = plt.subplots(4, 4, figsize=(20, 17))
    for r, (dist, spec) in enumerate(SETTINGS):
        d = D[(dist, spec)] = np.load(f"{OUT}/{dist}_{spec}.npz")
        for i, c in enumerate(CS):
            a = ax[r, i]
            for k, name in enumerate(NAMES):
                if c > 1 and name == "Frahm-Memmel":
                    continue
                m = np.nanmean(d["glob"][i, :, :, k], axis=1)
                a.plot(PS, m, marker="o", color=COLOURS[name], ls=STYLES[name], label=name)
                loc = d["local"][i, :, k]
                rows.append({"returns": dist, "spectrum": spec, "c": c, "estimator": name,
                             "mean loss, p=9": m[0], "mean loss, p=288": m[-1], "mean loss, p=306": np.mean(loc),
                             "median, p=306": np.median(loc), "5th pct, p=306": np.percentile(loc, 5),
                             "95th pct, p=306": np.percentile(loc, 95)})
            a.set_ylim(0, 10 if c == 0.9 else None)
            a.set_title(f"{label[dist]}, {spec}, c = {c}", fontsize=14)
            if r == 3:
                a.set_xlabel("p")
            if i == 0:
                a.set_ylabel("average relative loss")
    ax[0, 0].legend(fontsize=11)
    plt.tight_layout()
    save(fig, f"{OUT}/global.png")

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    d = D[("normal", "bounded")]
    for i, c in enumerate(CS):
        a = ax[i]
        for k, name in enumerate(NAMES):
            if c > 1 and name == "Frahm-Memmel":
                continue
            x = np.sort(d["local"][i, :, k])
            a.plot(x, np.arange(1, len(x) + 1) / len(x), color=COLOURS[name], ls=STYLES[name], label=name)
        a.set_xlim(0, {0.1: 0.3, 0.5: 2, 0.9: 14, 1.8: 8}[c])
        a.set(title=f"c = {c}, p = {P_LOCAL}", xlabel="relative loss")
    ax[0].set_ylabel("share of repetitions")
    ax[0].legend(fontsize=11)
    plt.tight_layout()
    save(fig, f"{OUT}/local.png")
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/summary.csv", index=False)
    print(t.round(3).to_string(index=False))

    # medians: at p = 9 (n = 10) a few samples with losses in the thousands dominate the mean
    m = {name: np.nanmedian(d["glob"][2, :, :, k], axis=1) for k, name in enumerate(NAMES)}
    fig, a = plt.subplots(figsize=SLIDE)
    for name in NAMES:
        a.plot(PS, m[name], marker="o", color=COLOURS[name], ls=STYLES[name], lw=3, ms=8)
    a.set(ylim=(0, 11), xlim=(0, 300), xlabel="number of stocks p  (n = p / 0.9)",
          ylabel=f"median relative loss, {REPS_GLOBAL} runs")
    a.text(232, m["traditional"][-1] + 0.45, "traditional", color=MUTED, fontsize=16)
    a.text(214, m["Frahm-Memmel"][-1] + 0.45, "Frahm-Memmel", color=DARK, fontsize=16)
    note(a, 95, 3.4, f"shrinkage: {m['bona fide'][-1]:.2f} at p = 288, the oracle {m['oracle'][-1]:.2f}")
    arrow(a, (100, 2.85), (145, 0.5), 0.3)
    note(a, 60, 6.2, "traditional: about 9, as the theorem says")
    arrow(a, (150, 6.75), (175, 8.6), -0.3)
    save(fig, f"{OUT}/c09.png")

    fig, a = plt.subplots(figsize=SLIDE)
    w = 0.2
    for k, name in enumerate(NAMES):
        vals = [np.nanmean(D[s]["glob"][2, -1, :, k]) for s in SETTINGS]
        x = np.arange(4) + (k - 1.5) * w
        a.bar(x, vals, w * 0.92, color=COLOURS[name], label=name)
        for xi, y in zip(x, vals):
            a.text(xi, y * 1.12, f"{y:.2f}" if y < 1 else f"{y:.1f}", ha="center", fontsize=12, color=INK)
    a.set_yscale("log")
    a.set_ylim(0.1, 90)
    a.grid(axis="x", visible=False)
    a.set_xticks(np.arange(4), [f"{label[d]} returns,\n{s} spectrum" for d, s in SETTINGS])
    a.set_ylabel("average relative loss, c = 0.9, p = 288", fontsize=13)
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=13)
    note(a, 1.5, 38, "the same ranking with heavy tails and a factor structure", ha="center")
    save(fig, f"{OUT}/settings.png")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else None)
    else:
        plot()
