"""Figures for the hyper-reduction audit, drawn only from report/audit/*.json (plus, for the
trajectory figure, a re-run of the stored basis on two test cases).

Run:  python scripts/plot_hyperreduction_audit.py [--sweep TAG] [--scaling TAG] [--reference TAG]
Output: report/audit/fig_*.png
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hyperreduction_benchmark import aligned_dt, aligned_reference, native_reduced_steps, wavenumbers
from src.deim import deim_rom_predict
from src.metrics import relative_l2
from src.pod_rom import rom_predict
from src.solver import solve_burgers

D = "report/audit"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8983"  # slots 1-3 + neutral

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.dpi": 130, "savefig.dpi": 130})


def load(name, tag):
    with open(os.path.join(D, f"{name}_{tag}.json")) as f:
        return json.load(f)


def fig_pareto(sw):
    rows = sw["rows"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fom = sw["fom"]["online_ms_mean_per_trajectory"]
    ax.axvline(fom, color=GRAY, lw=1.2, ls="--")
    ax.text(fom * 1.03, 3e3, "FOM (Nx=128)", color=MUTED, fontsize=8, va="top")
    for kind, col, mk in (("rom", BLUE, "s"), ("deim", ORANGE, "o")):
        sel = [r for r in rows if r["kind"] == kind]
        x = [r["online_ms_mean_per_trajectory"] for r in sel]
        y = [max(100 * r["test_err"]["mean"], 1e-4) for r in sel]
        ax.scatter(x, y, s=34, marker=mk, color=col, edgecolor="white", linewidth=0.8, zorder=3,
                   label="POD-Galerkin ROM (r = 4 ... 16)" if kind == "rom"
                   else "POD-DEIM/local-FD (r = 4...16, m = 6...32)")
    front = [r for r in rows if r["pareto_on_validation_error_vs_time"]]
    ax.scatter([r["online_ms_mean_per_trajectory"] for r in front],
               [max(100 * r["test_err"]["mean"], 1e-4) for r in front], s=120, facecolor="none",
               edgecolor=INK, linewidth=1.0, zorder=4, label="non-dominated on validation error")
    for r in rows:
        if (r["kind"], r["r"], r["m"]) in (("rom", 8, None), ("deim", 8, 16), ("deim", 8, 12),
                                            ("rom", 4, None)):
            lab = {("rom", 8, None): "ROM r=8", ("rom", 4, None): "ROM r=4",
                   ("deim", 8, 16): "DEIM r=8, m=16\n(historical config)",
                   ("deim", 8, 12): "r=8, m=12: unstable\non 9 of 40 cases"}[(r["kind"], r["r"], r["m"])]
            ax.annotate(lab, (r["online_ms_mean_per_trajectory"], max(100 * r["test_err"]["mean"], 1e-4)),
                        xytext=(8, -3 if r["m"] != 12 else -22), textcoords="offset points",
                        fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(2.5, 30)
    ax.set_xticks([3, 5, 10, 20])
    ax.set_xticklabels(["3", "5", "10", "20"])
    ax.set_xlabel("mean online time per trajectory [ms] (solver + reconstruction, native step rule)")
    ax.set_ylabel("mean relative $L^2$ error, test set [%]")
    ax.set_title("Speed-accuracy trade-off, 40 test cases, time-aligned error protocol",
                 loc="left", fontsize=10)
    hist = next(r for r in rows if (r["kind"], r["r"], r["m"]) == ("deim", 8, 16))
    ax.scatter([hist["online_ms_mean_per_trajectory"]], [100 * hist["test_err"]["mean"]], s=90,
               marker="*", color=ORANGE, edgecolor=INK, linewidth=0.8, zorder=5,
               label="historical config (r=8, m=16)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_pareto_error_vs_runtime.png"))
    plt.close(fig)


def fig_scaling(sc):
    per = sc["per_rhs"]
    tr = sc["trajectory"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax = axs[0]
    Nx = [p["Nx"] for p in per]
    for key, lab, col in (("rom_rhs_us", "plain ROM right-hand side", BLUE),
                          ("fom_nonlinear_us", "FOM nonlinear term", GRAY),
                          ("deim_rhs_us", "DEIM/local-FD right-hand side", ORANGE)):
        ax.plot(Nx, [p[key] for p in per], "-o", color=col, lw=1.6, ms=4.5, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("grid points Nx")
    ax.set_ylabel("cost of one RHS evaluation [$\\mu$s]")
    ax.set_title("(a) per-stage cost", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axs[1]
    Nt = [t["Nx"] for t in tr]
    ax.axhline(1.0, color=GRAY, lw=1.0, ls="--")
    ax.plot(Nt, [t["speedup_vs_fom"]["rom_solver_only"] for t in tr], "-s", color=BLUE, lw=1.6,
            ms=4.5, label="plain ROM")
    ax.plot(Nt, [t["speedup_vs_fom"]["deim_solver_only"] for t in tr], "-o", color=ORANGE, lw=1.6,
            ms=4.5, label="DEIM/local-FD")
    for t in tr:
        ax.annotate(f"{t['n_steps']['deim']} vs {t['n_steps']['fom']}\nsteps", (t["Nx"], t["speedup_vs_fom"]["deim_solver_only"]),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("grid points Nx")
    ax.set_ylabel("solver-only speed-up over FOM (native step rule)")
    ax.set_title("(b) end-to-end, one steep case (labels: RK4 steps)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_ylim(0.07, 6)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_scaling_with_grid_size.png"))
    plt.close(fig)


def fig_modes_points(sw):
    rows = {r["name"]: r for r in sw["rows"]}
    ab = sw["ablation_test_err"]
    rs, ms = sw["r_list"], sw["m_list"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax = axs[0]
    fl = lambda v: max(100 * v, 1e-4)
    ax.plot(rs, [fl(rows[f"rom_r{r}"]["test_err"]["mean"]) for r in rs], "-s", color=BLUE, lw=1.6,
            ms=4.5, label="ROM, time-aligned")
    ax.plot(rs, [fl(rows[f"rom_r{r}"]["test_err_native_protocol"]["mean"]) for r in rs], "--s",
            color=BLUE, lw=1.0, ms=3.5, alpha=0.6, label="ROM, historical index-wise protocol")
    ax.plot(rs, [fl(ab[f"fd_fullgrid_r{r}"]["mean"]) for r in rs], "-^", color=AQUA, lw=1.6, ms=4.5,
            label="full-grid local-FD RHS, no DEIM")
    ax.plot(rs, [fl(rows[f"deim_r{r}_m16"]["test_err"]["mean"]) for r in rs], "-o", color=ORANGE,
            lw=1.6, ms=4.5, label="DEIM/local-FD, m=16")
    ax.set_yscale("log")
    ax.set_xlabel("POD modes r")
    ax.set_ylabel("mean relative $L^2$ error, test [%]")
    ax.set_title("(a) error vs POD modes", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax = axs[1]
    ax.axhline(fl(rows["rom_r8"]["test_err"]["mean"]), color=BLUE, lw=1.0, ls="--")
    ax.text(ms[-1], fl(rows["rom_r8"]["test_err"]["mean"]) * 1.15, "plain ROM r=8", color=MUTED,
            fontsize=7.5, ha="right")
    ax.plot(ms, [fl(ab[f"exactF_deim_r8_m{m}"]["mean"]) for m in ms], "-^", color=AQUA, lw=1.6,
            ms=4.5, label="DEIM interpolation only (exact F, full grid)")
    ax.plot(ms, [fl(rows[f"deim_r8_m{m}"]["test_err"]["mean"]) for m in ms], "-o", color=ORANGE,
            lw=1.6, ms=4.5, label="DEIM/local-FD, mean")
    ax.plot(ms, [fl(rows[f"deim_r8_m{m}"]["test_err"]["median"]) for m in ms], ":o", color=ORANGE,
            lw=1.2, ms=3.5, label="DEIM/local-FD, median")
    ax.set_yscale("log")
    ax.set_xlabel("DEIM points m (r = 8)")
    ax.set_title("(b) error vs DEIM points", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_error_vs_modes_and_points.png"))
    plt.close(fig)


def fig_trajectories(ref):
    st = np.load("experiments/deim_basis.npz")
    Phi, pts, M = st["Phi"], st["points"], st["M"]
    te = np.load("data/test.npz")
    k, dealias = wavenumbers(128)
    errs = np.array([r["aligned.deim_vs_fom128"] for r in ref["rows"]])
    order = np.argsort(errs)
    cases = [("median-error case", int(order[len(order) // 2])),
             ("worst case (steep, low viscosity)", int(order[-1]))]
    x128 = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    x1024 = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=False)
    for ax, (title, i) in zip(axs, cases):
        A, nu = float(te["A"][i]), float(te["nu"][i])
        u0 = te["u"][i, 0]
        dt = aligned_dt(native_reduced_steps(np.abs(u0).max(), 128, nu))
        fom = aligned_reference(A, nu, 128)
        fine = aligned_reference(A, nu, 1024)
        rom = rom_predict(u0, Phi, k, nu, dealias, T=1.0, n_save=20, dt=dt)[1]
        with np.errstate(all="ignore"):
            deim = deim_rom_predict(u0, Phi, M, pts, 128, nu=nu, T=1.0, n_save=20, dt=dt)[1]
        ax.plot(x1024, fine[-1], color=GRAY, lw=1.0, label="FOM Nx=1024")
        ax.plot(x128, fom[-1], color=INK, lw=1.4, label="FOM Nx=128 (reference)")
        ax.plot(x128, rom[-1], color=BLUE, lw=1.4, ls="--",
                label=f"ROM r=8 ({100 * relative_l2(rom, fom):.2f}%)")
        ax.plot(x128, deim[-1], color=ORANGE, lw=1.4, ls="-.",
                label=f"DEIM m=16 ({100 * relative_l2(deim, fom):.1f}%)")
        ax.set_title(f"{title}: A={A:.2f}, $\\nu$={nu:.3f}, t=T", loc="left", fontsize=9.5)
        ax.set_xlabel("x")
        ax.legend(frameon=False, fontsize=7.5)
    axs[0].set_ylabel("u(x, T)")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_representative_trajectories.png"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="run2")
    ap.add_argument("--scaling", default="run2")
    ap.add_argument("--reference", default="run1")
    a = ap.parse_args()
    sw, sc, ref = load("sweep", a.sweep), load("scaling", a.scaling), load("reference", a.reference)
    fig_pareto(sw)
    fig_scaling(sc)
    fig_modes_points(sw)
    fig_trajectories(ref)
    print("figures written to", D)
