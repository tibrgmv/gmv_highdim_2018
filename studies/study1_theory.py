"""Study 1. Do the paper's theorems hold, and does our code reproduce Figures 1-4?

Design of Figures 1-4: p = 200, eigenvalues 20% at 3, 40% at 1, 40% at 0.5, target 1/N; 100 samples per c.
    python -m studies.study1_theory
"""
import numpy as np
import pandas as pd
from gmv import theory
from gmv.data import results_dir
from gmv.designs import make_sigma, simulate
from gmv.estimators import sample_cov, gmv, bona_fide, oracle, s_star
from gmv.style import plt, INK, MUTED, SURFACE, WARM, RED, COLOURS, SLIDE, note, arrow, save

OUT = results_dir(1)
P, REPS = 200, 100


def simulate_all():
    rng = np.random.default_rng(2015)
    Sigma, root, inv_root = make_sigma(P, "fig1", rng)
    v, Rb = theory.gmv_variance(Sigma), theory.target_loss(Sigma)
    print(f"R_b = {Rb:.3f}")
    rows = []
    for c in (0.1, 0.3, 0.5, 0.7, 0.9, 1.5, 2.0, 3.0, 5.0, 8.0):
        n = round(P / c)
        out = {k: [] for k in ("trad", "oracle", "bonafide", "alpha_oracle", "alpha_bf")}
        for _ in range(REPS):
            Y, X = simulate(root, n, rng)
            S = sample_cov(Y)
            Sinv = np.linalg.inv(S) if c < 1 else s_star(X, inv_root)
            w_or, a_or = oracle(Sinv, Sigma)
            w_bf, a_bf = bona_fide(S, n)
            out["trad"].append(theory.relative_loss(gmv(Sinv), Sigma, v))
            out["oracle"].append(theory.relative_loss(w_or, Sigma, v))
            out["bonafide"].append(theory.relative_loss(w_bf, Sigma, v))
            out["alpha_oracle"].append(a_or)
            out["alpha_bf"].append(a_bf)
        m = {k: np.mean(x) for k, x in out.items()}
        rows.append({"c": c, "n": n,
                     "loss_trad_theory": float(theory.loss_traditional(c)), "loss_trad_sim": m["trad"],
                     "loss_shrink_theory": float(theory.loss_shrinkage(c, Rb)), "loss_oracle_sim": m["oracle"],
                     "loss_bonafide_sim": m["bonafide"],
                     "alpha_theory": float(theory.alpha(c, Rb)), "alpha_oracle_sim": m["alpha_oracle"],
                     "alpha_bonafide_sim": m["alpha_bf"],
                     "fig3_curve": c ** 2 / (c - 1) if c > 1 else np.nan})
    t = pd.DataFrame(rows).set_index("c")
    t.to_csv(f"{OUT}/theory.csv")
    print(t.round(3).to_string())
    return t, Rb


def plot(t, Rb):
    cl, cr = np.linspace(0.005, 0.97, 300), np.linspace(1.03, 10, 400)
    L, H = t[t.index < 1], t[t.index > 1]
    dot = dict(s=80, edgecolor=SURFACE, linewidth=2, zorder=4)

    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
    a = ax[0, 0]
    a.plot(cl, theory.loss_traditional(cl), color=COLOURS["traditional"], label="traditional: theory c/(1−c)")
    a.plot(cl, theory.loss_shrinkage(cl, Rb), color=RED, label="optimal shrinkage: theory")
    a.scatter(L.index, L.loss_trad_sim, color=INK, label="our simulation, p = 200", **dot)
    a.scatter(L.index, L.loss_bonafide_sim, color=INK, **dot)
    a.set(ylim=(0, 10), xlabel="c = p / n", ylabel="relative loss", title="Figure 1: relative loss, c < 1")
    a.legend(loc="upper left")
    a = ax[0, 1]
    a.plot(cl, theory.alpha(cl, Rb), color=RED, label="theory $\\alpha^*$")
    a.scatter(L.index, L.alpha_oracle_sim, color=INK, label="oracle, simulated", **dot)
    a.scatter(L.index, L.alpha_bonafide_sim, color=WARM, marker="D", label="bona fide, simulated", **dot)
    a.set(xlabel="c = p / n", ylabel="shrinkage intensity $\\alpha$", title="Figure 2: $\\alpha^*$, c < 1")
    a.legend()
    a = ax[1, 0]
    a.plot(cr, theory.loss_traditional(cr), color=COLOURS["traditional"], label="traditional: theory (c²−c+1)/(c−1)")
    a.plot(cr, cr ** 2 / (cr - 1), color=MUTED, ls=":", label="c²/(c−1): the curve in the paper's Figure 3")
    a.plot(cr, theory.loss_shrinkage(cr, Rb), color=RED, label="optimal shrinkage: theory")
    a.scatter(H.index, H.loss_trad_sim, color=INK, label="our simulation, p = 200", **dot)
    a.scatter(H.index, H.loss_oracle_sim, color=INK, **dot)
    a.set(ylim=(0, 14), xlabel="c = p / n", ylabel="relative loss", title="Figure 3: relative loss, c > 1")
    a.legend(loc="upper right")
    a = ax[1, 1]
    a.plot(cr, theory.alpha(cr, Rb), color=RED, label="theory $\\alpha^+$")
    a.scatter(H.index, H.alpha_oracle_sim, color=INK, label="oracle (S*), simulated", **dot)
    a.scatter(H.index, H.alpha_bonafide_sim, color=WARM, marker="D", label="bona fide ($S^+$), simulated", **dot)
    a.set(xlabel="c = p / n", ylabel="shrinkage intensity $\\alpha$", title="Figure 4: $\\alpha^+$, c > 1")
    a.legend()
    plt.tight_layout()
    save(fig, f"{OUT}/overview.png")

    fig, a = plt.subplots(figsize=SLIDE)
    a.plot(cl, theory.loss_traditional(cl), color=COLOURS["traditional"], lw=3)
    a.plot(cl, theory.loss_shrinkage(cl, Rb), color=RED, lw=3)
    a.scatter(L.index, L.loss_trad_sim, color=INK, **dot)
    a.scatter(L.index, L.loss_bonafide_sim, color=INK, **dot)
    a.set(ylim=(0, 10.5), xlim=(0, 1), xlabel="c = p / n", ylabel="relative loss")
    a.text(0.5, 2.1, "traditional", color=MUTED, fontsize=17)
    a.text(0.62, 0.75, "shrinkage", color=RED, fontsize=17)
    note(a, 0.12, 6.0, "lines: the paper's theorem\ndots: our simulation, p = 200")
    arrow(a, (0.3, 5.0), (0.5, 1.2), -0.25)
    note(a, 0.62, 9.2, f"c = 0.9: {L.loc[0.9, 'loss_trad_sim']:.1f} simulated, 9 in theory")
    save(fig, f"{OUT}/theory_holds.png")

    fig, a = plt.subplots(figsize=SLIDE)
    a.plot(cr, cr ** 2 / (cr - 1), color=WARM, ls="--", lw=2.6)
    a.plot(cr, theory.loss_traditional(cr), color=INK, lw=3)
    a.scatter(H.index, H.loss_trad_sim, color=RED, **dot)
    a.scatter([2, 2], [3, 4], color=[INK, WARM], s=90, edgecolor=SURFACE, linewidth=2, zorder=5)
    a.set(ylim=(0, 12), xlim=(1, 10), xlabel="c = p / n", ylabel="relative loss, traditional estimator")
    note(a, 3.3, 9.6, "the paper's Figure 3: c²/(c−1), minimum 4")
    arrow(a, (3.4, 9.0), (2.15, 4.15), 0.25)
    note(a, 4.0, 1.5, "their own formula (c²−c+1)/(c−1): minimum 3")
    arrow(a, (4.1, 2.0), (2.15, 2.9), -0.25)
    note(a, 7.0, 5.3, "red dots: our simulation")
    save(fig, f"{OUT}/figure3_error.png")

    fig, a = plt.subplots(figsize=SLIDE)
    a.plot(cr, theory.alpha(cr, Rb), color=INK, lw=3)
    a.scatter(H.index, H.alpha_oracle_sim, color=INK, **dot)
    a.scatter(H.index, H.alpha_bonafide_sim, color=RED, marker="D", **dot)
    a.set(xlim=(1, 10), ylim=(0, 0.17), xlabel="c = p / n", ylabel="shrinkage intensity $\\alpha$")
    note(a, 3.6, 0.155, "theory, and the oracle that uses the true $\\Sigma$")
    arrow(a, (3.7, 0.148), (3.05, 0.125), 0.3)
    note(a, 5.9, 0.012, "bona fide with $S^+$: too little weight on the data")
    arrow(a, (5.95, 0.018), (5.1, 0.027), -0.3)
    save(fig, f"{OUT}/alpha_above_1.png")


if __name__ == "__main__":
    plot(*simulate_all())
