"""Figures for report/burgers_spatial_convergence_study.md from report/spatial_convergence/*_run*.json.

Figures 10 and 14 (energy evolution, steep-front profiles) need full fields, which the JSON files do not
store; they are recomputed with the same deterministic solvers.

Run:  python scripts/plot_spatial_convergence_study.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.spatial_convergence_study import OUT_DIR, TIMES, cases  # noqa: E402
from src.convergence import grid, sine_ic, sine_solution, solve_burgers_f64  # noqa: E402

FIG_DIR = os.path.join(OUT_DIR, "figures")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8a8984", "#52514e"]
MARKERS = ["o", "v", "^", "D", "P", "*", "s", "x"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "lines.markersize": 5, "legend.frameon": False, "savefig.dpi": 160})
SHOW = ["corner_lowA_highnu", "centre", "corner_highA_highnu", "test0", "test5", "corner_highA_lownu"]
LABEL = {"corner_lowA_highnu": "A = 0.5, nu = 0.1", "corner_highA_highnu": "A = 2, nu = 0.1",
         "corner_lowA_lownu": "A = 0.5, nu = 0.01", "corner_highA_lownu": "A = 2, nu = 0.01",
         "centre": "A = 1.25, nu = 0.055", "test5": "test 5: A = 1.89, nu = 0.016",
         "test33": "test 33: A = 1.50, nu = 0.012", "test0": "test 0: A = 1.92, nu = 0.038"}


def load(name, tag="run1"):
    with open(os.path.join(OUT_DIR, f"{name}_{tag}.json")) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name))
    plt.close(fig)
    print("saved", os.path.join(FIG_DIR, name))


def style(i):
    return dict(color=PALETTE[i % len(PALETTE)], marker=MARKERS[i % len(MARKERS)])


def floor_line(ax, y, text):
    ax.axhline(y, color="#8a8984", lw=0.9, ls=":")
    ax.text(ax.get_xlim()[0], y * 1.6, text, fontsize=7, color="#52514e")


def fig01_spatial(sp):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for i, name in enumerate(SHOW):
        rows = sp["representative"][name]["by_N"]
        Ns = sorted(int(n) for n in rows)
        axes[0].semilogy(Ns, [rows[str(n)]["vs_exact"]["space_time"] for n in Ns], label=LABEL[name], **style(i))
        Nc = [n for n in Ns if "consecutive_E_N_2N" in rows[str(n)]]
        axes[1].semilogy(Nc, [rows[str(n)]["consecutive_E_N_2N"]["space_time"] for n in Nc], **style(i))
    for ax, t in zip(axes, ("error against the exact (Cole-Hopf) solution", "consecutive difference E(N, 2N)")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N")
        ax.set_title(t, fontsize=9)
    axes[0].set_ylabel("relative L2 error (space-time)")
    axes[0].legend(fontsize=7)
    save(fig, "fig01_spatial_convergence.png")


def fig02_temporal(tp):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, N in zip(axes, ("256", "1024", "2048")):
        for i, name in enumerate(tp["cases"]):
            rec = tp["cases"][name]["by_N"][N]
            dts = [rec["dt_base"] / int(m) for m in rec["errors"]]
            ax.loglog(dts, list(rec["errors"].values()), label=LABEL.get(name, name), **style(i))
        x = np.array([2e-4, 2e-3]) * (256 / int(N))
        ax.loglog(x, 3e-9 * (x / x[1]) ** 4, color="#0b0b0a", lw=1, ls="--", label="slope 4")
        ax.axhspan(1e-14, tp["floor"], color="#e4e3df", alpha=0.6, lw=0)
        ax.set_title(f"N = {N}", fontsize=9)
        ax.set_xlabel("time step")
    axes[0].set_ylabel("relative L2 error vs 64x steps (same N)")
    axes[0].legend(fontsize=6.5)
    save(fig, "fig02_temporal_convergence.png")


def fig03_spectra(dg):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for i, name in enumerate(SHOW):
        c = np.array(dg["cases"][name]["exact_spectrum_steepest"])
        k = np.arange(c.size)
        axes[0].semilogy(k[1:2049:2], np.maximum(c[1:2049:2], 1e-18), lw=1.3, color=PALETTE[i], label=LABEL[name])
    for N in (128, 512, 2048):
        axes[0].axvline(np.ceil(N / 3) - 1, color="#8a8984", lw=0.9, ls=":")
        axes[0].text(np.ceil(N / 3) - 1, 1e-15, f" N={N}", fontsize=7, color="#52514e")
    axes[0].set_xscale("log")
    axes[0].set_ylim(1e-17, 2)
    axes[0].set_xlabel("wavenumber k (odd modes)")
    axes[0].set_ylabel("|c_k| of the exact solution")
    axes[0].set_title("exact spectrum at the time of steepest gradient", fontsize=9)
    axes[0].legend(fontsize=7)
    name = "test5"
    for j, N in enumerate((128, 256, 512, 1024, 2048)):
        c = np.array(dg["cases"][name]["by_N"][str(N)]["fom_spectrum_steepest"])
        axes[1].semilogy(np.arange(1, c.size, 2), np.maximum(c[1::2], 1e-18), lw=1.2, color=PALETTE[j], label=f"FOM N={N}")
    ce = np.array(dg["cases"][name]["exact_spectrum_steepest"])
    axes[1].semilogy(np.arange(1, 1400, 2), np.maximum(ce[1:1400:2], 1e-18), color="#0b0b0a", lw=1, ls="--", label="exact")
    axes[1].set_xscale("log")
    axes[1].set_ylim(1e-17, 2)
    axes[1].set_xlabel("wavenumber k (odd modes)")
    axes[1].set_title(f"{LABEL[name]}: FOM spectra vs exact", fontsize=9)
    axes[1].legend(fontsize=7)
    save(fig, "fig03_fourier_spectra.png")


def fig04_reference(rf):
    names = list(rf["cases"])
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    x = np.arange(len(names))
    q = [rf["cases"][n]["quadrature_refinement"] for n in names]
    f1 = [rf["cases"][n]["fom"]["N4096_x1"]["space_time"] for n in names]
    f8 = [rf["cases"][n]["fom"].get("N8192_x2", {}).get("space_time", np.nan) for n in names]
    ax.semilogy(x, q, "o", ms=3.5, color=PALETTE[0], label="Cole-Hopf: quadrature step halved")
    ax.semilogy(x, f1, "v", ms=3.5, color=PALETTE[1], label="FOM N = 4096, baseline steps")
    ax.semilogy(x, f8, "^", ms=4.5, color=PALETTE[2], label="FOM N = 8192, 2x steps")
    ax.set_xticks(x[::3])
    ax.set_xticklabels([n.replace("corner_", "").replace("extrap_", "x:")[:12] for n in names[::3]], rotation=60, fontsize=6)
    ax.set_ylabel("relative L2 difference to Cole-Hopf")
    ax.legend(fontsize=7)
    save(fig, "fig04_reference_verification.png")


def fig05_06_param(sp):
    t = sp["test40"]
    for fname, key, xl in (("fig05_error_vs_viscosity.png", "nu", "viscosity nu"),
                           ("fig06_error_vs_amplitude.png", "A", "amplitude A")):
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for j, N in enumerate(("128", "256", "512", "1024", "2048")):
            ax.scatter([t[i][key] for i in t], [t[i]["by_N"][N]["space_time"] for i in t], s=16,
                       color=PALETTE[j], marker=MARKERS[j], label=f"N = {N}", edgecolors="white", linewidths=0.4)
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("FOM error vs exact (space-time)")
        ax.legend(fontsize=7)
        save(fig, fname)


def fig07_08_09_rom(rm):
    Ns = [int(n) for n in rm["by_N"]]
    rs = rm["r_list"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for j, N in enumerate(Ns):
        s = rm["by_N"][str(N)]["summary"]["test"]
        ax.semilogy(rs, [s["by_r"][str(r)]["rom_vs_exact"]["mean"] for r in rs], label=f"N = {N}", **style(j))
        ax.axhline(s["fom_vs_exact_mean"], color=PALETTE[j], lw=0.8, ls=":")
    ax.set_xlabel("POD rank r")
    ax.set_ylabel("tensorial POD error vs exact (test mean)")
    ax.set_title("dotted: the FOM's own error at the same N", fontsize=8)
    ax.legend(fontsize=7)
    save(fig, "fig07_rom_error_vs_rank.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for i, r in enumerate(rs):
        g = lambda key: [rm["by_N"][str(N)]["summary"]["test"]["by_r"][str(r)][key]["mean"] for N in Ns]
        axes[0].loglog(Ns, g("rom_vs_exact"), label=f"r = {r}", **style(i))
        axes[1].loglog(Ns, g("rom_vs_fom"), **style(i))
    axes[0].loglog(Ns, [rm["by_N"][str(N)]["summary"]["test"]["fom_vs_exact_mean"] for N in Ns], color="#0b0b0a",
                   lw=1.2, ls="--", label="FOM vs exact")
    for ax, t in zip(axes, ("ROM vs exact solution", "ROM vs FOM at the same N")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N")
        ax.set_title(t, fontsize=9)
    axes[0].set_ylabel("relative L2 error (test mean)")
    axes[0].legend(fontsize=7)
    save(fig, "fig08_rom_error_vs_resolution.png")

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for i, r in enumerate(rs):
        rows = [x for x in rm["by_N"]["2048"]["rows"] if x["set"] == "test"]
        ax.scatter([x["by_r"][str(r)]["proj_exact"] for x in rows], [x["by_r"][str(r)]["rom_vs_exact"] for x in rows],
                   s=14, color=PALETTE[i], marker=MARKERS[i], label=f"r = {r}", edgecolors="white", linewidths=0.4)
    lim = np.array([1e-12, 1e-1])
    ax.loglog(lim, lim, color="#0b0b0a", lw=0.9, ls="--", label="trajectory = projection")
    ax.set_xlabel("POD projection error of the exact solution")
    ax.set_ylabel("ROM trajectory error vs exact")
    ax.set_title("N = 2048, 40 test cases", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "fig09_projection_vs_trajectory_error.png")


def fig10_energy_and_14_profiles():
    cs = cases()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, name in zip(axes, ("test5", "corner_highA_lownu")):
        A, nu = cs[name]
        ex = sine_solution(A, nu).trajectory(4096, TIMES)
        E = lambda u: 0.5 * np.sum(u**2, axis=1) * 2 * np.pi / u.shape[1]
        ax.plot(TIMES, E(ex), color="#0b0b0a", lw=2.4, alpha=0.35, label="exact")
        for j, N in enumerate((64, 128, 512, 2048)):
            ax.plot(TIMES, E(solve_burgers_f64(sine_ic(A, N), nu)[1]), color=PALETTE[j], marker=MARKERS[j],
                    markersize=3, label=f"FOM N = {N}")
        ax.set_title(LABEL[name], fontsize=9)
        ax.set_xlabel("t")
    axes[0].set_ylabel("energy 0.5 * int u^2 dx")
    axes[0].legend(fontsize=7)
    save(fig, "fig10_energy_evolution.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, name in zip(axes, ("test5", "corner_highA_lownu")):
        A, nu = cs[name]
        xf = grid(4096)
        ax.plot(xf, sine_solution(A, nu)(xf, 1.0), color="#0b0b0a", lw=2.6, alpha=0.35, label="exact")
        for j, N in enumerate((128, 256, 512, 2048)):
            ax.plot(grid(N), solve_burgers_f64(sine_ic(A, N), nu)[1][-1], color=PALETTE[j], lw=1.1,
                    marker=MARKERS[j], markersize=2.5, label=f"FOM N = {N}")
        ax.set_xlim(np.pi - 0.45, np.pi + 0.45)
        ax.set_title(f"{LABEL[name]}, t = 1 (near the front)", fontsize=9)
        ax.set_xlabel("x")
    axes[0].set_ylabel("u(x, 1)")
    axes[0].legend(fontsize=7)
    save(fig, "fig14_steep_front_profiles.png")


def fig13_runtime(rm):
    Ns = [int(n) for n in rm["by_N"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    s = lambda N: rm["by_N"][str(N)]["summary"]["test"]
    ax.loglog(Ns, [s(N)["fom_ms_mean"] for N in Ns], color="#52514e", marker="s", label="FOM (float64)")
    for i, r in enumerate((8, 16, 32)):
        ax.loglog(Ns, [s(N)["by_r"][str(r)]["solve_ms_mean"] for N in Ns], label=f"tensorial POD r = {r} (solver only)",
                  **style(i))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("time per trajectory (ms, test mean, single run)")
    ax.legend(fontsize=7)
    save(fig, "fig13_runtime_scaling.png")


def timestep_figures(ts):
    keys = list(ts["by_N_r"])
    pols = ts["policies"]
    lab = {"P0_reduced": "P0 current", "P1_reduced_cfl": "P1 reduced CFL", "P2_if_rk4": "P2 reduced IF-RK4",
           "P3_rk45": "P3 RK45", "P4_radau": "P4 Radau"}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w = 0.15
    x = np.arange(len(keys))
    for j, p in enumerate(pols):
        ax.bar(x + (j - 2) * w, [ts["by_N_r"][k]["splits"]["test"]["summary"][p]["rhs_evals_mean"] for k in keys], w,
               color=PALETTE[j], label=lab[p])
    ax.plot(x, [4 * ts["by_N_r"][k]["splits"]["test"]["summary"]["fom_steps_mean"] for k in keys], "k_", ms=18, mew=1.5,
            label="FOM (4 x steps)")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", ", ") for k in keys], fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("RHS evaluations per trajectory (test mean)")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "fig11_reduced_step_counts.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, k in zip(axes, ("N128_r16", "N2048_r16")):
        wp = ts["by_N_r"][k]["work_precision_val"]
        for j, pfx in enumerate(("P0_reduced", "P1_reduced_cfl", "P2_if_rk4", "P3_rk45", "DOP853")):
            pts = sorted((v["rhs_evals_mean"], v["time_err_max"]) for n, v in wp.items() if n.startswith(pfx))
            if pts:
                ax.loglog(*zip(*pts), label=lab.get(pfx, pfx), **style(j))
        ax.set_title(k.replace("_", ", "), fontsize=9)
        ax.set_xlabel("RHS evaluations per trajectory (val mean)")
    axes[0].set_ylabel("max time-integration error (val)")
    axes[0].legend(fontsize=7)
    save(fig, "fig12_accuracy_vs_cost.png")


if __name__ == "__main__":
    sp, tp, rf, dg, rm = (load("spatial"), load("temporal", "run2"), load("reference"), load("diagnostics"), load("rom"))
    fig01_spatial(sp)
    fig02_temporal(tp)
    fig03_spectra(dg)
    fig04_reference(rf)
    fig05_06_param(sp)
    fig07_08_09_rom(rm)
    fig13_runtime(rm)
    fig10_energy_and_14_profiles()
    if os.path.exists(os.path.join(OUT_DIR, "timestep_run1.json")):
        timestep_figures(load("timestep"))
