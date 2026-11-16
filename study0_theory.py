"""
Study 0 — Do the paper's theorems hold, and do we reproduce Figures 1-4?
The paper's design: p = 200, eigenvalues 20% = 3, 40% = 1, 40% = 0.5, Haar eigenvectors, equal-weight target.
For each c we compare the asymptotic formulas with a finite-sample simulation (p = 200, n = p/c),
which also checks our code: the estimators must land on the curves.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from common import (INK, MUTED, SURFACE, WARM, RED, COLOURS, SLIDE, plot_style, note, arrow, save, make_sigma,
                    simulate, sample_cov, gmv, bona_fide, oracle, s_star, relative_loss, gmv_variance, target_loss,
                    theory_loss_traditional, theory_alpha, theory_loss_shrinkage)

plot_style()
OUT = "results/study0"
os.makedirs("results/slides", exist_ok=True)
rng = np.random.default_rng(2015)
P, REPS = 200, 100
Sigma, root, inv_root = make_sigma(P, "fig1", rng)
v_gmv, Rb = gmv_variance(Sigma), target_loss(Sigma)
print(f"Relative loss of the equal-weight target in this design: R_b = {Rb:.3f}")

rows = []
for c in (0.1, 0.3, 0.5, 0.7, 0.9, 1.5, 2.0, 3.0, 5.0, 8.0):
    n = round(P / c)
    out = {k: [] for k in ("trad", "oracle", "bonafide", "alpha_oracle", "alpha_bf")}
    for _ in range(REPS):
        Y, X = simulate(root, n, rng)
        S = sample_cov(Y)
        Sinv = np.linalg.inv(S) if c < 1 else s_star(X, inv_root)    # the oracle inverse of Section 2.2 for c > 1
        w_trad = gmv(Sinv)
        w_or, a_or = oracle(Sinv, Sigma)
        w_bf, a_bf = bona_fide(S, n)
        out["trad"].append(relative_loss(w_trad, Sigma, v_gmv))
        out["oracle"].append(relative_loss(w_or, Sigma, v_gmv))
        out["bonafide"].append(relative_loss(w_bf, Sigma, v_gmv))
        out["alpha_oracle"].append(a_or); out["alpha_bf"].append(a_bf)
    m = {k: np.mean(v) for k, v in out.items()}
    rows.append({"c": c, "n": n,
                 "loss_trad_theory": float(theory_loss_traditional(c)), "loss_trad_sim": m["trad"],
                 "loss_shrink_theory": float(theory_loss_shrinkage(c, Rb)), "loss_oracle_sim": m["oracle"],
                 "loss_bonafide_sim": m["bonafide"],
                 "alpha_theory": float(theory_alpha(c, Rb)), "alpha_oracle_sim": m["alpha_oracle"],
                 "alpha_bonafide_sim": m["alpha_bf"],
                 "fig3_curve": c ** 2 / (c - 1) if c > 1 else np.nan})
t = pd.DataFrame(rows).set_index("c")
pd.set_option("display.width", 200)
print(t.round(3).to_string())
t.to_csv(f"{OUT}.csv")

# ---------------------------------------------------------------------------------------------------
# Figures. Overview: the paper's Figures 1-4 as four panels, theory lines and simulated points.
# ---------------------------------------------------------------------------------------------------
cl, cr = np.linspace(0.005, 0.97, 300), np.linspace(1.03, 10, 400)
L, H = t[t.index < 1], t[t.index > 1]
dot = dict(s=80, edgecolor=SURFACE, linewidth=2, zorder=4)
fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
a = ax[0, 0]
a.plot(cl, theory_loss_traditional(cl), color=COLOURS["traditional"], label="traditional: theory c/(1−c)")
a.plot(cl, theory_loss_shrinkage(cl, Rb), color=RED, label="optimal shrinkage: theory")
a.scatter(L.index, L.loss_trad_sim, color=INK, label="our simulation, p = 200", **dot)
a.scatter(L.index, L.loss_bonafide_sim, color=INK, **dot)
a.set_ylim(0, 10); a.set_xlabel("c = p / n"); a.set_ylabel("relative loss")
a.set_title("Figure 1: relative loss, c < 1"); a.legend(loc="upper left")
a = ax[0, 1]
a.plot(cl, theory_alpha(cl, Rb), color=RED, label="theory $\\alpha^*$")
a.scatter(L.index, L.alpha_oracle_sim, color=INK, label="oracle, simulated", **dot)
a.scatter(L.index, L.alpha_bonafide_sim, color=WARM, marker="D", label="bona fide, simulated", **dot)
a.set_xlabel("c = p / n"); a.set_ylabel("shrinkage intensity $\\alpha$"); a.set_title("Figure 2: $\\alpha^*$, c < 1"); a.legend()
a = ax[1, 0]
a.plot(cr, theory_loss_traditional(cr), color=COLOURS["traditional"], label="traditional: theory (c²−c+1)/(c−1)")
a.plot(cr, cr ** 2 / (cr - 1), color=MUTED, ls=":", label="c²/(c−1): the curve in the paper's Figure 3")
a.plot(cr, theory_loss_shrinkage(cr, Rb), color=RED, label="optimal shrinkage: theory")
a.scatter(H.index, H.loss_trad_sim, color=INK, label="our simulation, p = 200", **dot)
a.scatter(H.index, H.loss_oracle_sim, color=INK, **dot)
a.set_ylim(0, 14); a.set_xlabel("c = p / n"); a.set_ylabel("relative loss"); a.set_title("Figure 3: relative loss, c > 1")
a.legend(loc="upper right")
a = ax[1, 1]
a.plot(cr, theory_alpha(cr, Rb), color=RED, label="theory $\\alpha^+$")
a.scatter(H.index, H.alpha_oracle_sim, color=INK, label="oracle (S*), simulated", **dot)
a.scatter(H.index, H.alpha_bonafide_sim, color=WARM, marker="D", label="bona fide ($S^+$), simulated", **dot)
a.set_xlabel("c = p / n"); a.set_ylabel("shrinkage intensity $\\alpha$"); a.set_title("Figure 4: $\\alpha^+$, c > 1"); a.legend()
plt.tight_layout(); save(fig, f"{OUT}.png"); plt.close(fig)

# Slide 1: the theory holds (c < 1)
fig, a = plt.subplots(figsize=SLIDE)
a.plot(cl, theory_loss_traditional(cl), color=COLOURS["traditional"], lw=3)
a.plot(cl, theory_loss_shrinkage(cl, Rb), color=RED, lw=3)
a.scatter(L.index, L.loss_trad_sim, color=INK, **dot); a.scatter(L.index, L.loss_bonafide_sim, color=INK, **dot)
a.set_ylim(0, 10.5); a.set_xlim(0, 1); a.set_xlabel("c = p / n"); a.set_ylabel("relative loss")
a.text(0.5, 2.1, "traditional", color=MUTED, fontsize=17); a.text(0.62, 0.75, "shrinkage", color=RED, fontsize=17)
note(a, 0.12, 6.0, "lines: the paper's theorem\ndots: our simulation, p = 200")
arrow(a, (0.3, 5.0), (0.5, 1.2), -0.25)
note(a, 0.62, 9.2, f"c = 0.9: {L.loc[0.9, 'loss_trad_sim']:.1f} simulated, 9 in theory")
save(fig, "results/slides/s0_theory_holds.png"); plt.close(fig)

# Slide 2: the error in Figure 3
fig, a = plt.subplots(figsize=SLIDE)
a.plot(cr, cr ** 2 / (cr - 1), color=WARM, ls="--", lw=2.6)
a.plot(cr, theory_loss_traditional(cr), color=INK, lw=3)
a.scatter(H.index, H.loss_trad_sim, color=RED, **dot)
a.scatter([2, 2], [3, 4], color=[INK, WARM], s=90, edgecolor=SURFACE, linewidth=2, zorder=5)
a.set_ylim(0, 12); a.set_xlim(1, 10); a.set_xlabel("c = p / n"); a.set_ylabel("relative loss, traditional estimator")
note(a, 3.3, 9.6, "the paper's Figure 3: c²/(c−1), minimum 4")
arrow(a, (3.4, 9.0), (2.15, 4.15), 0.25)
note(a, 4.0, 1.5, "their own formula (c²−c+1)/(c−1): minimum 3")
arrow(a, (4.1, 2.0), (2.15, 2.9), -0.25)
note(a, 7.0, 5.3, "red dots: our simulation")
save(fig, "results/slides/s0_figure3_error.png"); plt.close(fig)

# Slide 3: the intensity for c > 1 (S⁺ versus the theory)
fig, a = plt.subplots(figsize=SLIDE)
a.plot(cr, theory_alpha(cr, Rb), color=INK, lw=3)
a.scatter(H.index, H.alpha_oracle_sim, color=INK, **dot)
a.scatter(H.index, H.alpha_bonafide_sim, color=RED, marker="D", **dot)
a.set_xlim(1, 10); a.set_ylim(0, 0.17); a.set_xlabel("c = p / n"); a.set_ylabel("shrinkage intensity $\\alpha$")
note(a, 3.6, 0.155, "theory, and the oracle that uses the true $\\Sigma$")
arrow(a, (3.7, 0.148), (3.05, 0.125), 0.3)
note(a, 5.9, 0.012, "bona fide with $S^+$: too little weight on the data")
arrow(a, (5.95, 0.018), (5.1, 0.027), -0.3)
save(fig, "results/slides/s0_alpha_c_above_1.png"); plt.close(fig)
