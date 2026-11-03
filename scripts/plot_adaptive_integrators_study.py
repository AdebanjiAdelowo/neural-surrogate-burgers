"""Figures for report/adaptive_tensorial_pod_integrators_study.md from report/adaptive_integrators/*.json.

Figures 8, 9, 10 and 13 (step-size histories, trajectories, front profiles, matrix-function costs) need data
not stored in the JSON files; they are recomputed deterministically on validation / stress-validation cases
(never on test or holdout cases).

Run:  python scripts/plot_adaptive_integrators_study.py
"""
import json
import os
import sys
import timeit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.adaptive_integrators_study import (N_ROM, OUT_DIR, R_LIST, REF_MULT, RTOLS, TIMES,  # noqa: E402
                                                case_sets, load_data, reference_rom, rom_models, run_config)
from src.convergence import grid, sine_ic, sine_solution  # noqa: E402

FIG_DIR = os.path.join(OUT_DIR, "figures")
COL = {"B1": "#8a8984", "B2": "#eb6834", "B3": "#2a78d6", "B3fd": "#184f95", "E1": "#eda100", "E2": "#e87ba4",
       "X1": "#1baf7a", "X2": "#008300"}
MK = {"B1": "s", "B2": "v", "B3": "o", "B3fd": "x", "E1": "D", "E2": "P", "X1": "^", "X2": "*"}
LAB = {"B1": "B1 reduced-CFL RK4", "B2": "B2 reduced IF-RK4", "B3": "B3 Radau (analytic J)", "B3fd": "B3fd Radau (FD J)",
       "E1": "E1 RK45", "E2": "E2 DOP853", "X1": "X1 adaptive Lawson IF-DP5(4)", "X2": "X2 ETD-RK4 (Krogstad), step doubling"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 1.5, "lines.markersize": 5,
                     "legend.frameon": False, "savefig.dpi": 160})


def load(name):
    with open(os.path.join(OUT_DIR, f"{name}.json")) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name))
    plt.close(fig)
    print("saved", os.path.join(FIG_DIR, name))


def method_of(cfg):
    return cfg.split("_")[0]


def fig01_03_stiffness(st):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, r in enumerate(R_LIST):
        ev = -np.array(st["by_r"][str(r)]["diffusion"]["eigenvalues"])
        ax.semilogy(np.arange(1, ev.size + 1), np.sort(ev), marker="o", ms=3, label=f"r = {r}",
                    color=["#2a78d6", "#eb6834", "#1baf7a"][i])
    ax.set_xlabel("index")
    ax.set_ylabel("-eigenvalue of L (decay rate per unit nu)")
    ax.legend()
    save(fig, "fig01_diffusion_eigenvalues.png")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, r in zip(axes, R_LIST):
        cs = st["by_r"][str(r)]["cases"]
        for lab, c in cs.items():
            col = "#eb6834" if c["set"] == "stress" else "#2a78d6"
            ax.plot(TIMES[: len(c["per_snapshot"]["spectral_abscissa"])], c["per_snapshot"]["spectral_abscissa"],
                    color=col, lw=0.6, alpha=0.6)
            ax.plot(TIMES[: len(c["per_snapshot"]["numerical_abscissa"])], c["per_snapshot"]["numerical_abscissa"],
                    color=col, lw=0.6, alpha=0.6, ls="--")
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_title(f"r = {r} (solid: spectral, dashed: numerical abscissa)", fontsize=8)
        ax.set_xlabel("t (reference trajectory)")
    axes[0].set_ylabel("abscissa of J(a(t))")
    axes[0].plot([], [], color="#2a78d6", label="validation")
    axes[0].plot([], [], color="#eb6834", label="stress-validation")
    axes[0].legend(fontsize=7)
    save(fig, "fig02_jacobian_abscissae.png")

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    keys = [("diffusion", "rho", "rho(L)"), ("summary", "spectral_radius_max", "max rho(J), nu included"),
            ("summary", "transient_amplification_max", "max transient amplification"),
            ("summary", "frozen_rk4_dt_min", "min frozen-Jacobian RK4 step")]
    for j, (grp, k, lab) in enumerate(keys):
        ax.semilogy(R_LIST, [st["by_r"][str(r)][grp][k] for r in R_LIST], marker="osDv"[j], label=lab)
    ax.set_xticks(R_LIST)
    ax.set_xlabel("POD rank r")
    ax.legend(fontsize=7)
    save(fig, "fig03_stiffness_vs_rank.png")


def fig04_05_06_07_tolerance(ex):
    for fname, key, ylab in (("fig04_error_vs_tolerance.png", "time_err_max", "max time error (val + stress)"),
                             ("fig05_runtime_vs_tolerance.png", "ms_median", "median single-run time (ms)")):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
        for ax, r in zip(axes, R_LIST):
            s = ex["by_r"][str(r)]["summary"]
            for m in ("B3", "B3fd", "E1", "E2", "X1", "X2"):
                ax.loglog(RTOLS, [s[f"{m}_rtol{rt:g}"]["both"][key] for rt in RTOLS], color=COL[m], marker=MK[m],
                          label=LAB[m])
            if key == "time_err_max":
                ax.loglog(RTOLS, RTOLS, color="#0b0b0a", lw=0.8, ls=":", label="error = rtol")
            ax.set_title(f"r = {r}", fontsize=9)
            ax.set_xlabel("rtol")
        axes[0].set_ylabel(ylab)
        axes[0].legend(fontsize=6.5)
        save(fig, fname)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, r in zip(axes, R_LIST):
        s = ex["by_r"][str(r)]["summary"]
        for m in ("B1", "B2", "B3", "B3fd", "E1", "E2", "X1", "X2"):
            pts = sorted((v["both"]["ms_median"], v["both"]["time_err_max"]) for n, v in s.items() if method_of(n) == m)
            ax.loglog(*zip(*pts), color=COL[m], marker=MK[m], label=LAB[m])
        ax.set_title(f"r = {r} (validation + stress)", fontsize=9)
        ax.set_xlabel("median single-run time (ms)")
    axes[0].set_ylabel("max time-integration error")
    axes[0].legend(fontsize=6)
    save(fig, "fig06_pareto_validation.png")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, r in zip(axes, R_LIST):
        rows = ex["by_r"][str(r)]["rows"]
        x = np.arange(len(RTOLS))
        w = 0.13
        for j, m in enumerate(("E1", "E2", "X1", "X2")):
            acc = [np.mean([row["cfg"][f"{m}_rtol{rt:g}"].get("accepted", np.nan) or np.nan for row in rows]) for rt in RTOLS]
            rej = [np.mean([row["cfg"][f"{m}_rtol{rt:g}"].get("rejected", np.nan) or 0 for row in rows]) for rt in RTOLS]
            ax.bar(x + (j - 1.5) * w, rej, w, color=COL[m], label=f"{LAB[m]} rejected" if r == 8 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{rt:g}" for rt in RTOLS])
        ax.set_title(f"r = {r}", fontsize=9)
        ax.set_xlabel("rtol")
    axes[0].set_ylabel("rejected steps per trajectory (mean)")
    axes[0].legend(fontsize=6.5)
    save(fig, "fig07_rejected_steps.png")


def fig08_09_10_recomputed():
    data = load_data()
    models = rom_models(data=data)
    lab, A, nu = [c for c in case_sets(("stress",), data)["stress"] if c[0] == "stress_A2.0_nu0.01"][0]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, r in zip(axes, R_LIST):
        Phi, m = models[r]
        u0 = sine_ic(A, N_ROM)
        a0 = Phi.T @ u0
        for meth in ("E1", "E2", "B3", "X1", "X2"):
            _, st = run_config((meth, 1e-6), r, models, u0, a0, A, nu, record_steps=True)
            if st.get("steps"):
                t, h = zip(*st["steps"])
                ax.semilogy(t, h, color=COL[meth], lw=1, label=LAB[meth])
        ax.set_title(f"r = {r}, A = 2, nu = 0.01, rtol 1e-6", fontsize=8)
        ax.set_xlabel("t")
    axes[0].set_ylabel("accepted step size")
    axes[0].legend(fontsize=6)
    save(fig, "fig08_step_size_evolution.png")

    Phi, m = models[16]
    u0 = sine_ic(A, N_ROM)
    a0 = Phi.T @ u0
    ref = reference_rom(m, a0, A, nu, REF_MULT)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for meth, cfg in (("B2", ("B2", 1)), ("B3", ("B3", 1e-4)), ("X1", ("X1", 1e-4)), ("X2", ("X2", 1e-4))):
        a, _ = run_config(cfg, 16, models, u0, a0, A, nu)
        err_t = np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
        axes[0].semilogy(TIMES[1:], err_t[1:], color=COL[meth], marker=MK[meth], ms=3, label=f"{LAB[meth]} {cfg[1]:g}")
        if meth in ("B2", "X1"):
            x = grid(N_ROM)
            win = np.abs(x - np.pi) < 0.25
            axes[1].plot(x[win], (a[-1] @ Phi.T)[win] - (ref[-1] @ Phi.T)[win], color=COL[meth], label=LAB[meth])
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("relative coefficient error vs reference ROM")
    axes[0].legend(fontsize=6.5)
    axes[1].set_xlabel("x (near the front)")
    axes[1].set_ylabel("u(x, 1): method minus reference ROM")
    axes[1].set_title("r = 16, stress case A = 2, nu = 0.01", fontsize=8)
    axes[1].legend(fontsize=7)
    save(fig, "fig09_trajectory_errors.png")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = grid(N_ROM)
    win = np.abs(x - np.pi) < 0.3
    ex = sine_solution(A, nu)(x, 1.0)
    ax.plot(x[win], ex[win], color="#0b0b0a", lw=2.5, alpha=0.35, label="exact (Cole-Hopf)")
    for i, r in enumerate(R_LIST):
        Phi, m = models[r]
        a, _ = run_config(("B3", 1e-6), r, models, sine_ic(A, N_ROM), Phi.T @ sine_ic(A, N_ROM), A, nu)
        ax.plot(x[win], (a[-1] @ Phi.T)[win], lw=1.1, color=["#2a78d6", "#eb6834", "#1baf7a"][i], label=f"ROM r = {r} (Radau 1e-6)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, 1)")
    ax.set_title("stress case A = 2, nu = 0.01: the POD error dominates near the front", fontsize=8)
    ax.legend(fontsize=7)
    save(fig, "fig10_steep_front_profiles.png")


def fig11_energy(fin):
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for r in R_LIST:
        s = fin["by_r"][str(r)]["summary"]["holdout"]
        names = [n for n in s if not n.startswith("_")]
        ax.scatter([r] * len(names), [max(s[n]["energy_increase_max"], 1e-17) for n in names], s=18,
                   c=[COL[method_of(n)] for n in names])
    ax.set_yscale("log")
    ax.set_xticks(R_LIST)
    ax.set_xlabel("POD rank r")
    ax.set_ylabel("max energy increase / E(0) (holdout; 1e-17 = none)")
    save(fig, "fig11_energy_increase.png")


def fig12_rank_runtime(fin):
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for m in ("B1", "B2", "B3", "B3fd", "E1", "E2", "X1", "X2"):
        xs, ys = [], []
        for r in R_LIST:
            s = fin["by_r"][str(r)]["summary"]["holdout"]
            cand = [n for n in s if not n.startswith("_") and method_of(n) == m]
            if cand:
                xs.append(r)
                ys.append(min(s[n]["e2e_ms_median"] for n in cand))
        if xs:
            ax.semilogy(xs, ys, color=COL[m], marker=MK[m], label=LAB[m])
    ax.set_xticks(R_LIST)
    ax.set_xlabel("POD rank r")
    ax.set_ylabel("fastest frozen setting, median ms (holdout)")
    ax.legend(fontsize=6.5)
    save(fig, "fig12_runtime_vs_rank.png")


def fig13_matrix_functions():
    from src.exponential_integrators import EigenBasis, phi, phi_matrix
    data = load_data()
    models = rom_models(data=data)
    rows = []
    for r in R_LIST:
        Phi, m = models[r]
        t_basis = min(timeit.repeat(lambda: EigenBasis(m.Q, m.L), number=3, repeat=3)) / 3
        D = 0.02 * np.linalg.eigvalsh(m.L) * 1e-3
        t_phi = min(timeit.repeat(lambda: [phi(k, D) for k in (1, 2, 3)], number=200, repeat=3)) / 200
        t_blk = min(timeit.repeat(lambda: phi_matrix(0.02 * 1e-3 * m.L, 3), number=5, repeat=3)) / 5
        rows.append((t_basis, t_phi, t_blk))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for j, lab in enumerate(("one-time eigen-basis (eigh + tensor transform)", "phi_1..3 per step size (diagonal)",
                             "phi_1..3 via block-matrix expm (general)")):
        ax.semilogy(R_LIST, [1e3 * row[j] for row in rows], marker="osD"[j], label=lab)
    ax.set_xticks(R_LIST)
    ax.set_xlabel("POD rank r")
    ax.set_ylabel("time (ms)")
    ax.legend(fontsize=7)
    save(fig, "fig13_matrix_function_costs.png")
    return rows


def fig14_generalization(ex, fin):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
    for ax, r in zip(axes, R_LIST):
        sv = ex["by_r"][str(r)]["summary"]
        for sname, mk in (("test", "o"), ("holdout", "^")):
            s = fin["by_r"][str(r)]["summary"][sname]
            for n in s:
                if n.startswith("_"):
                    continue
                ax.scatter(sv[n]["both"]["time_err_max"], s[n]["time_err_max"], s=20, marker=mk, color=COL[method_of(n)],
                           edgecolors="white", linewidths=0.4)
        lim = np.array([1e-14, 1e-3])
        ax.loglog(lim, lim, color="#0b0b0a", lw=0.8, ls="--")
        ax.loglog(lim, 2 * lim, color="#8a8984", lw=0.8, ls=":")
        ax.set_title(f"r = {r} (o test, ^ holdout)", fontsize=8)
        ax.set_xlabel("max time error, validation + stress")
    axes[0].set_ylabel("max time error, held-out set")
    save(fig, "fig14_generalization.png")


if __name__ == "__main__":
    st, ex = load("stiffness_run1"), load("explore_run1")
    fig01_03_stiffness(st)
    fig04_05_06_07_tolerance(ex)
    fig08_09_10_recomputed()
    costs = fig13_matrix_functions()
    print("matrix-function costs (s): basis, phi diagonal, phi block", costs)
    if os.path.exists(os.path.join(OUT_DIR, "final_run1.json")):
        fin = load("final_run1")
        fig11_energy(fin)
        fig12_rank_runtime(fin)
        fig14_generalization(ex, fin)
