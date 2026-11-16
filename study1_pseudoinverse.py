"""
Study 1 — The unproven step: S⁺ in place of S* for c > 1 (the paper's Figures 5 and 6).
For c > 1 the theory is proven for the oracle inverse S* (it needs the true Σ); the usable estimator plugs in
the Moore-Penrose inverse S⁺. Figure 5 compares oracle and bona fide versions over c in (0, 10] for p = 207;
Figure 6 compares the estimated intensity with S⁺ and with S* as p grows, for c in {1.8, 2.25, 3, 5}.
Design as in the paper: eigenvalues 1/9 = 2, 4/9 = 5, 4/9 = 10, Haar eigenvectors, normal returns.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from common import (INK, MUTED, SURFACE, RED, WARM, DARK, COLOURS, SLIDE, plot_style, note, arrow, save, make_sigma,
                    simulate, sample_cov, gmv, inverse, bona_fide, oracle, s_star, relative_loss, gmv_variance, target_loss)

plot_style()
OUT = "results/study1"
rng = np.random.default_rng(1)

# --- Figure 5: p = 207 across c ------------------------------------------------------------------
P, REPS = 207, 40
Sigma, root, inv_root = make_sigma(P, "bounded", rng)
v_gmv = gmv_variance(Sigma)
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
        acc["trad_S*"].append(relative_loss(gmv(Sstar), Sigma, v_gmv))
        acc["trad_S+"].append(relative_loss(gmv(Splus), Sigma, v_gmv))
        acc["oracle"].append(relative_loss(w_or, Sigma, v_gmv))
        acc["bona fide"].append(relative_loss(w_bf, Sigma, v_gmv))
        acc["alpha_oracle"].append(a_or); acc["alpha_bf"].append(a_bf)
    rows.append({"c": round(c, 3), "n": n, **{k: (np.median(v) if k.startswith("trad") else np.mean(v)) for k, v in acc.items()}})
f5 = pd.DataFrame(rows).set_index("c")
print(f"Figure 5 (p = {P}, R_b = {target_loss(Sigma):.3f}, {REPS} repetitions per c)")
print(f5.round(3).to_string())

# --- Figure 6: intensity with S⁺ vs S* as p grows ------------------------------------------------
PS, REPS6 = (18, 27, 36, 54, 72, 99, 144, 207, 288), 200
rows6 = []
for c in (1.8, 2.25, 3.0, 5.0):
    for p in PS:
        Sg, rt, irt = make_sigma(p, "bounded", rng)
        n = max(int(round(p / c)), 3)
        a_plus, a_star, a_true = [], [], []
        for _ in range(REPS6):
            Y, X = simulate(rt, n, rng)
            S = sample_cov(Y)
            a_plus.append(bona_fide(S, n)[1])
            a_star.append(bona_fide(S, n, Sinv=s_star(X, irt))[1])
            a_true.append(oracle(s_star(X, irt), Sg)[1])
        rows6.append({"c": c, "p": p, "alpha_S+": np.mean(a_plus), "alpha_S*": np.mean(a_star), "alpha_oracle": np.mean(a_true)})
f6 = pd.DataFrame(rows6)
print("\nFigure 6: mean estimated intensity")
print(f6.round(3).to_string(index=False))
pd.concat({"figure5": f5.reset_index(), "figure6": f6}).to_csv(f"{OUT}.csv")

os.makedirs("results/slides", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
a = ax[0]
a.plot(f5.index, f5["trad_S*"], color=WARM, label="traditional, oracle inverse S* (median)")
a.plot(f5.index, f5["trad_S+"], color=WARM, ls="--", label="traditional, Moore-Penrose $S^+$ (median)")
a.plot(f5.index, f5["oracle"], color=INK, label="oracle shrinkage")
a.plot(f5.index, f5["bona fide"], color=RED, ls="--", label="bona fide shrinkage ($S^+$)")
a.set_ylim(0, 16); a.set_xlabel("c = p / n"); a.set_ylabel("relative loss")
a.set_title(f"Figure 5: oracle and bona fide, p = {P}"); a.legend(loc="upper right")
a = ax[1]
for colour, (c, g) in zip((INK, RED, WARM, DARK), f6.groupby("c")):
    a.plot(g.p, g["alpha_S*"], color=colour, marker="o", label=f"c = {c}: S*")
    a.plot(g.p, g["alpha_S+"], color=colour, marker="o", ls="--", mfc=SURFACE, label=f"c = {c}: $S^+$")
a.axhline(0, color=MUTED, lw=1)
a.set_xlabel("p"); a.set_ylabel(r"mean estimated intensity $\alpha$")
a.set_title("Figure 6: intensity with S* (solid) and $S^+$ (dashed)")
a.legend(ncol=2, fontsize=11)
plt.tight_layout(); save(fig, f"{OUT}.png"); plt.close(fig)

# Slide: the cost of the unproven step
fig, a = plt.subplots(figsize=SLIDE)
a.plot(f5.index, f5["oracle"], color=INK, lw=3)
a.plot(f5.index, f5["bona fide"], color=RED, lw=3, ls="--")
a.axvline(1, color=MUTED, lw=1, ls=":")
a.set_ylim(0, 0.4); a.set_xlim(0, 10); a.set_xlabel("c = p / n"); a.set_ylabel("average relative loss")
a.text(6.5, 0.218, "oracle (knows the true Σ)".replace("Σ", "$\\Sigma$"), color=INK, fontsize=16)
a.text(6.5, 0.338, "bona fide, with $S^+$", color=RED, fontsize=16)
gap = (f5.loc[f5.index > 1.4, "bona fide"] / f5.loc[f5.index > 1.4, "oracle"] - 1).mean()
note(a, 2.0, 0.1, f"c > 1: about {100 * gap:.0f}% more loss than the oracle,\nthe price of the unproven $S^+$ step")
arrow(a, (3.2, 0.14), (3.6, 0.28), -0.3)
note(a, 0.25, 0.36, "c < 1: proven, and\nit matches the oracle")
arrow(a, (0.55, 0.31), (0.6, 0.235), 0.3)
save(fig, "results/slides/s1_pseudoinverse_cost.png"); plt.close(fig)
