"""Study 2. The unproven step: S⁺ in place of the oracle inverse S* for c > 1 (the paper's Figures 5 and 6).

Bounded design (eigenvalues 1/9 at 2, 4/9 at 5, 4/9 at 10), normal returns.
Figure 5: p = 207, c from 0.1 to 10, 40 samples per c. Figure 6: mean estimated α with S⁺ and with S*, 200 samples.
    python -m studies.study2_pseudoinverse
"""
import numpy as np
import pandas as pd
from gmv.data import results_dir
from gmv.designs import make_sigma, simulate
from gmv.estimators import sample_cov, gmv, inverse, bona_fide, oracle, s_star
from gmv.style import plt, INK, MUTED, SURFACE, RED, WARM, DARK, SLIDE, note, arrow, save
from gmv.theory import relative_loss, gmv_variance, target_loss

OUT = results_dir(2)
P, REPS = 207, 40
PS6, REPS6 = (18, 27, 36, 54, 72, 99, 144, 207, 288), 200


def figure5(rng):
    Sigma, root, inv_root = make_sigma(P, "bounded", rng)
    v = gmv_variance(Sigma)
    cs = np.r_[np.linspace(0.1, 0.9, 9), 0.95, 1.05, 1.1, np.linspace(1.25, 3, 8), np.linspace(3.5, 10, 8)]
    rows = []
    for c in cs:
        n = max(int(round(P / c)), 3)
        acc = {k: [] for k in ("trad_S*", "trad_S+", "oracle", "bona fide", "alpha_oracle", "alpha_bf")}
        for _ in range(REPS):
            Y, X = simulate(root, n, rng)
            S = sample_cov(Y)
            Splus = inverse(S)
            Sstar = Splus if c < 1 else s_star(X, inv_root)
            w_or, a_or = oracle(Sstar, Sigma)
            w_bf, a_bf = bona_fide(S, n, Sinv=Splus)
            acc["trad_S*"].append(relative_loss(gmv(Sstar), Sigma, v))
            acc["trad_S+"].append(relative_loss(gmv(Splus), Sigma, v))
            acc["oracle"].append(relative_loss(w_or, Sigma, v))
            acc["bona fide"].append(relative_loss(w_bf, Sigma, v))
            acc["alpha_oracle"].append(a_or)
            acc["alpha_bf"].append(a_bf)
        rows.append({"c": round(c, 3), "n": n, **{k: (np.median(x) if k.startswith("trad") else np.mean(x)) for k, x in acc.items()}})
    print(f"Figure 5: p = {P}, R_b = {target_loss(Sigma):.3f}")
    return pd.DataFrame(rows).set_index("c")


def figure6(rng):
    rows = []
    for c in (1.8, 2.25, 3.0, 5.0):
        for p in PS6:
            Sigma, root, inv_root = make_sigma(p, "bounded", rng)
            n = max(int(round(p / c)), 3)
            a_plus, a_star, a_true = [], [], []
            for _ in range(REPS6):
                Y, X = simulate(root, n, rng)
                S = sample_cov(Y)
                a_plus.append(bona_fide(S, n)[1])
                a_star.append(bona_fide(S, n, Sinv=s_star(X, inv_root))[1])
                a_true.append(oracle(s_star(X, inv_root), Sigma)[1])
            rows.append({"c": c, "p": p, "alpha_S+": np.mean(a_plus), "alpha_S*": np.mean(a_star), "alpha_oracle": np.mean(a_true)})
    return pd.DataFrame(rows)


def plot(f5, f6):
    fig, ax = plt.subplots(1, 2, figsize=(17, 6))
    a = ax[0]
    a.plot(f5.index, f5["trad_S*"], color=WARM, label="traditional, oracle inverse S* (median)")
    a.plot(f5.index, f5["trad_S+"], color=WARM, ls="--", label="traditional, Moore-Penrose $S^+$ (median)")
    a.plot(f5.index, f5["oracle"], color=INK, label="oracle shrinkage")
    a.plot(f5.index, f5["bona fide"], color=RED, ls="--", label="bona fide shrinkage ($S^+$)")
    a.set(ylim=(0, 16), xlabel="c = p / n", ylabel="relative loss", title=f"Figure 5: oracle and bona fide, p = {P}")
    a.legend(loc="upper right")
    a = ax[1]
    for colour, (c, g) in zip((INK, RED, WARM, DARK), f6.groupby("c")):
        a.plot(g.p, g["alpha_S*"], color=colour, marker="o", label=f"c = {c}: S*")
        a.plot(g.p, g["alpha_S+"], color=colour, marker="o", ls="--", mfc=SURFACE, label=f"c = {c}: $S^+$")
    a.axhline(0, color=MUTED, lw=1)
    a.set(xlabel="p", ylabel=r"mean estimated intensity $\alpha$", title="Figure 6: intensity with S* (solid) and $S^+$ (dashed)")
    a.legend(ncol=2, fontsize=11)
    plt.tight_layout()
    save(fig, f"{OUT}/overview.png")

    fig, a = plt.subplots(figsize=SLIDE)
    a.plot(f5.index, f5["oracle"], color=INK, lw=3)
    a.plot(f5.index, f5["bona fide"], color=RED, lw=3, ls="--")
    a.axvline(1, color=MUTED, lw=1, ls=":")
    a.set(ylim=(0, 0.4), xlim=(0, 10), xlabel="c = p / n", ylabel="average relative loss")
    a.text(6.5, 0.218, "oracle (knows the true $\\Sigma$)", color=INK, fontsize=16)
    a.text(6.5, 0.338, "bona fide, with $S^+$", color=RED, fontsize=16)
    hi = f5.index > 1.4
    gap = (f5.loc[hi, "bona fide"] / f5.loc[hi, "oracle"] - 1).mean()
    note(a, 2.0, 0.1, f"c > 1: about {100 * gap:.0f}% more loss than the oracle,\nthe price of the unproven $S^+$ step")
    arrow(a, (3.2, 0.14), (3.6, 0.28), -0.3)
    note(a, 0.25, 0.36, "c < 1: proven, and\nit matches the oracle")
    arrow(a, (0.55, 0.31), (0.6, 0.235), 0.3)
    save(fig, f"{OUT}/pseudoinverse_cost.png")


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    f5, f6 = figure5(rng), figure6(rng)
    pd.concat({"figure5": f5.reset_index(), "figure6": f6}).to_csv(f"{OUT}/figures_5_6.csv")
    print(f5.round(3).to_string(), f6.round(3).to_string(index=False), sep="\n\n")
    plot(f5, f6)
