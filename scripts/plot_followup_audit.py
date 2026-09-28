"""Figures for the follow-up audit (reads report/audit/followup_*.json, fno_ablation_*.json, reference_run1).

Run: python scripts/plot_followup_audit.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "report/audit"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a8983"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.dpi": 130, "savefig.dpi": 130})


def J(name):
    with open(os.path.join(D, name)) as f:
        return json.load(f)


def fig_speedup_vs_nx(sc):
    rows = [r for r in sc["rows"] if "case" in r]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.7))
    for ax, case, title in ((axs[0], 0, "(a) steep, low-viscosity case"),
                            (axs[1], 13, "(b) smooth, high-viscosity case")):
        rr = sorted([r for r in rows if r["case"] == case], key=lambda r: r["Nx"])
        for key, col, ls, lab in (("deim_R", ORANGE, "-", "DEIM, revised step rule"),
                                  ("deim_A", ORANGE, "--", "DEIM, historical step rule"),
                                  ("rom_R", BLUE, "-", "plain ROM, revised"),
                                  ("rom_A", BLUE, "--", "plain ROM, historical")):
            pts = [(r["Nx"], r["speedup_vs_fom_solver_only"][key]) for r in rr if key in r["speedup_vs_fom_solver_only"]]
            ax.plot(*zip(*pts), ls, color=col, marker="o", ms=4, lw=1.6, label=lab)
        ax.axhline(1, color=GRAY, lw=1, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("grid points Nx")
        ax.set_title(title, loc="left", fontsize=10)
    axs[0].set_ylabel("solver-only speed-up over FOM")
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    ax = axs[2]
    rr = sorted([r for r in rows if r["case"] == 0], key=lambda r: r["Nx"])
    for key, col, ls, lab in (("fom_aligned", GRAY, "-", "FOM"), ("deim_A", ORANGE, "--", "reduced, historical rule"),
                              ("deim_R", ORANGE, "-", "reduced, revised rule")):
        ax.plot([r["Nx"] for r in rr], [r["n_steps"][key] for r in rr], ls, color=col, marker="o", ms=4, lw=1.6, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("grid points Nx")
    ax.set_ylabel("RK4 steps")
    ax.set_title("(c) steps, steep case", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_followup_speedup_vs_nx.png"))
    plt.close(fig)


def fig_multiplier(mu):
    rows = mu["rows"]
    cases = sorted({r["case"] for r in rows}, key=lambda c: next(r["A"] / r["nu"] for r in rows if r["case"] == c))
    cmap = plt.cm.Blues(np.linspace(0.35, 0.95, len(cases)))
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.7), sharey=True)
    for ax, model, title in zip(axs, ("rom_r8", "deim_r8_m16", "deim_r8_m12"),
                                ("(a) plain ROM, r=8", "(b) DEIM/local-FD, r=8, m=16", "(c) DEIM/local-FD, r=8, m=12")):
        for c, col in zip(cases, cmap):
            rr = sorted([r for r in rows if r["case"] == c and r["model"] == model], key=lambda r: r["multiplier"])
            ok = [r for r in rr if r["stable"]]
            bad = [r for r in rr if not r["stable"]]
            ax.plot([r["multiplier"] for r in ok], [100 * r["rel_l2"] for r in ok], "-o", color=col, ms=4, lw=1.4,
                    label=f"A/$\\nu$={rr[0]['A'] / rr[0]['nu']:.0f}")
            ax.scatter([r["multiplier"] for r in bad], [5e3] * len(bad), marker="x", color=col, s=28, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("time step as a multiple of the historical rule")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(3e-2, 1e4)
    axs[0].set_ylabel("relative $L^2$ error [%]  (x = unstable)")
    axs[0].legend(frameon=False, fontsize=7.5, loc="center left", title="steepness rank", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_followup_error_vs_timestep_multiplier.png"))
    plt.close(fig)


def fig_deim_diagnosis(st):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax = axs[0]
    pc = st["nx128_test40_policyR"]["per_case"]
    for key, col, lab in (("rom", BLUE, "plain ROM"), ("exactF_deim", AQUA, "DEIM interpolation only (exact F)"),
                          ("fd2_fullgrid_no_deim", YELLOW, "local-FD F, full grid, no DEIM"),
                          ("deim_order2", ORANGE, "DEIM/local-FD, order 2"), ("deim_order4", MAGENTA, "DEIM/local-FD, order 4")):
        v = np.sort(np.array(pc[key]) * 100)
        ax.plot(np.arange(1, 41) / 40, v, "-", color=col, lw=1.6, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("fraction of the 40 test cases (sorted per method)")
    ax.set_ylabel("relative $L^2$ error [%]")
    ax.set_title("(a) error distribution, Nx = 128, r=8, m=16", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax = axs[1]
    nx = [128] + [h["Nx"] for h in st["higher_resolution_first10"]]
    e2 = [st["nx128_test40_policyR"]["errors"]["deim_order2"]["mean"]] + [h["order2"]["mean"] for h in st["higher_resolution_first10"]]
    e4 = [st["nx128_test40_policyR"]["errors"]["deim_order4"]["mean"]] + [h["order4"]["mean"] for h in st["higher_resolution_first10"]]
    ax.plot(nx, np.array(e2) * 100, "-o", color=ORANGE, lw=1.6, ms=4.5, label="order 2 (3-point)")
    ax.plot(nx, np.array(e4) * 100, "-o", color=MAGENTA, lw=1.6, ms=4.5, label="order 4 (5-point)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("grid points Nx  (Nx = 128: 40 cases; 256, 512: first 10 cases)")
    ax.set_ylabel("mean relative $L^2$ error [%]")
    ax.set_title("(b) stencil order vs resolution", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_followup_deim_accuracy_diagnosis.png"))
    plt.close(fig)


def fig_fno(fn, floor_key="stored_targets_current_solver_native_times"):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
    names = [("historical", "historical\n$\\nu$-blind\n(3 seeds)", GRAY), ("blind_rerun", "$\\nu$-blind\nretrained\n(5 seeds)", GRAY),
             ("nu_aware", "$\\nu$-aware\n(5 seeds)", AQUA)]
    for ax, key, title in ((axs[0], "in_dist", "(a) in-distribution (40 held-out cases)"),
                           (axs[1], "out_of_family", "(b) out-of-family two-mode initial conditions (20)")):
        for i, (v, lab, col) in enumerate(names):
            seeds = np.array(fn["variants"][v]["aggregate"][key]["seed_means"]) * 100
            ax.bar(i, seeds.mean(), color=col, alpha=0.85, width=0.55)
            ax.scatter(np.full(len(seeds), i) + np.linspace(-0.12, 0.12, len(seeds)), seeds, color=INK, s=12, zorder=3)
            ax.text(i, seeds.mean() * 1.03, f"{seeds.mean():.2f}%", ha="center", fontsize=8)
        fl = fn["nu_blind_floor"][floor_key][key]["mean"] * 100
        ax.axhline(fl, color=ORANGE, lw=1.4, ls="--")
        ax.text(2.45, fl * 1.05, f"$\\nu$-blind floor {fl:.2f}%", ha="right", fontsize=8, color=MUTED)
        ax.set_xticks(range(3))
        ax.set_xticklabels([n[1] for n in names], fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    axs[0].set_ylabel("mean relative $L^2$ error [%]  (dots: seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_followup_fno_nu_ablation.png"))
    plt.close(fig)


def fig_alignment(ref):
    rows = ref["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for ax, key, col, title in ((axs[0], "rom_vs_fom128", BLUE, "(a) plain ROM r=8"),
                                (axs[1], "deim_vs_fom128", ORANGE, "(b) DEIM/local-FD r=8, m=16")):
        x = np.array([r["native." + key] for r in rows]) * 100
        y = np.array([r["aligned." + key] for r in rows]) * 100
        ax.scatter(x, y, s=16, color=col, edgecolor="white", linewidth=0.5, zorder=3)
        lo, hi = min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], color=GRAY, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("historical index-wise error [%]")
        ax.set_title(title, loc="left", fontsize=10)
        ax.text(0.05, 0.92, f"mean {x.mean():.3f}% -> {y.mean():.3f}%", transform=ax.transAxes, fontsize=8, color=MUTED)
    axs[0].set_ylabel("time-aligned error [%]")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_followup_historical_vs_aligned_error.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_speedup_vs_nx(J("followup_scaling2_run1.json"))
    fig_multiplier(J("followup_multiplier_run1.json"))
    fig_deim_diagnosis(J("followup_stencil_run2.json"))
    fig_fno(J("fno_ablation_run1.json"))
    fig_alignment(J("reference_run1.json"))
    print("figures written")
