"""Figures for report/tensorial_pod_spectral_deim_study.md, from report/tensorial_deim/*_run1.json.

Figures 8 and 9 (energy evolution, predicted vs reference) need full trajectories, which the JSON
files do not store; they are recomputed here with the same models and protocol (deterministic).

Run:  python scripts/plot_tensorial_deim_study.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.tensorial_deim_study import (COMPARISON_DEIM, NX, OUT_DIR, case_set, deim_bases,  # noqa: E402
                                          load_data, method_registry, stored_basis, training_sigma,
                                          wavenumbers)

FIG_DIR = os.path.join(OUT_DIR, "figures")
# categorical slots in fixed order (dataviz reference palette, light mode); markers as secondary encoding
STYLE = {
    "fom": dict(color="#52514e", marker="s", label="Full-order solver"),
    "rom": dict(color="#8a8984", marker="x", label="POD-Galerkin (original)"),
    "tensorial": dict(color="#2a78d6", marker="o", label="Tensorial POD-Galerkin"),
    "hybrid_localfd_m16": dict(color="#eb6834", marker="v", label="Hybrid DEIM/local-FD"),
    "spec_deim_m16": dict(color="#1baf7a", marker="^", label="Spectral DEIM"),
    "spec_qdeim_m16": dict(color="#eda100", marker="D", label="Spectral Q-DEIM"),
    "spec_oversampled_m16_s32": dict(color="#e87ba4", marker="P", label="Spectral oversampled (s = 2m)"),
    "wec_hybrid_localfd_m16": dict(color="#008300", marker="*", label="Weighted EC hybrid"),
}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False,
                     "savefig.dpi": 160, "figure.dpi": 100})


def load(name):
    with open(os.path.join(OUT_DIR, f"{name}_run1.json")) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name))
    plt.close(fig)
    print("saved", os.path.join(FIG_DIR, name))


def pct_axis(ax, axis="y"):
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fig_rank(sens, tb):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    rows = sens["rows"]
    rs = sorted({r["r"] for r in rows})
    series = [("tensorial", lambda r: "tensorial", "tensorial"),
              ("spec_qdeim_m16", lambda r: f"spec_qdeim_m{r}_s{r}", "Spectral Q-DEIM, m = r"),
              ("spec_oversampled_m16_s32", lambda r: f"spec_qdeim_m{2*r}_s{4*r}", "Spectral Q-DEIM, m = 2r, s = 2m"),
              ("hybrid_localfd_m16", lambda r: f"hybrid_localfd_m{2*r}", "Hybrid DEIM/local-FD, m = 2r")]
    for key, name, label in series:
        y = [next(x["summary"]["mean"] for x in rows if x["r"] == r and x["method"] == name(r)) for r in rs]
        st = dict(STYLE[key])
        st["label"] = label if key != "tensorial" else "Tensorial = original POD-Galerkin"
        ax.plot(rs, y, **st)
    ax.set_xlabel("POD rank r")
    ax.set_ylabel("Mean relative L2 error (40 test cases)")
    pct_axis(ax)
    ax.set_xticks(rs)
    ax.legend(fontsize=8)
    save(fig, "fig01_error_vs_rank.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    rr = sorted(int(r) for r in tb["rank_sweep"])
    ax.plot(rr, [tb["rank_sweep"][str(r)]["rom_end_to_end_ms_mean_10cases"] for r in rr], **STYLE["rom"])
    ax.plot(rr, [tb["rank_sweep"][str(r)]["tensorial_end_to_end_ms_mean_10cases"] for r in rr], **STYLE["tensorial"])
    ax.axhline(tb["timing"]["workload_mean_ms"]["fom"], color=STYLE["fom"]["color"], ls="--", lw=1.2,
               label="Full-order solver (40-case mean)")
    ax.set_xlabel("POD rank r")
    ax.set_ylabel("End-to-end time per trajectory (ms)")
    ax.set_xticks(rr)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    save(fig, "fig02_runtime_vs_rank.png")


def fig_viscosity(cmp_):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for key in ("hybrid_localfd_m16", "wec_hybrid_localfd_m16", "spec_qdeim_m16", "tensorial"):
        rows = cmp_["accuracy"]["test"][key]["rows"]
        ax.scatter([r["nu"] for r in rows], [r["rel_l2"] for r in rows], s=22, color=STYLE[key]["color"],
                   marker=STYLE[key]["marker"], label=STYLE[key]["label"], edgecolors="white", linewidths=0.6)
    ax.set_xlabel("Viscosity nu")
    ax.set_ylabel("Relative L2 error per test case")
    pct_axis(ax)
    ax.legend(fontsize=8)
    save(fig, "fig03_error_vs_viscosity.png")


def fig_deim_dimension(sw):
    cf = sw["configs"]
    ms = sorted({c["m"] for c in cf.values()})
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for target, sel, key in (("nonlinear", "deim", "spec_deim_m16"), ("nonlinear", "qdeim", "spec_qdeim_m16"),
                             ("full", "deim", None), ("full", "qdeim", None)):
        y = [cf[f"{target}|m={m}|s={m}|{sel}"]["test"]["summary"]["mean"] for m in ms]
        st = dict(STYLE[key]) if key else dict(color=STYLE["spec_deim_m16" if sel == "deim" else "spec_qdeim_m16"]["color"],
                                                marker=STYLE["spec_deim_m16" if sel == "deim" else "spec_qdeim_m16"]["marker"])
        st["label"] = f"Spectral {'DEIM' if sel == 'deim' else 'Q-DEIM'}, DEIM on {'N only' if target == 'nonlinear' else 'full RHS'}"
        if target == "full":
            st["ls"] = "--"
        ax.plot(ms, y, **st)
    ax.plot(ms, [sw["hybrid_localfd"][str(m)]["test"]["summary"]["mean"] for m in ms], **STYLE["hybrid_localfd_m16"])
    ax.axhline(sw["tensorial_test_mean"], color=STYLE["tensorial"]["color"], lw=1.2, ls=":",
               label="Tensorial POD-Galerkin (r = 8)")
    ax.set_xlabel("DEIM basis dimension m (s = m points)")
    ax.set_ylabel("Mean relative L2 error (40 test cases)")
    ax.set_xticks(ms)
    pct_axis(ax)
    ax.legend(fontsize=7.5)
    save(fig, "fig04_error_vs_deim_dimension.png")


def fig_oversampling(sw):
    cf = sw["configs"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, sel, title in ((axes[0], "qdeim+greedy", "Q-DEIM + greedy sigma_min"),
                           (axes[1], "qdeim_extended", "Q-DEIM on extended basis")):
        for m, key in zip((8, 12, 16, 24), ("spec_deim_m16", "spec_qdeim_m16", "spec_oversampled_m16_s32", "tensorial")):
            ss = [m, int(np.ceil(1.5 * m)), 2 * m]
            y = [cf[f"nonlinear|m={m}|s={s}|{'qdeim' if s == m else sel}"]["test"]["summary"]["mean"] for s in ss]
            ax.plot([s / m for s in ss], y, color=STYLE[key]["color"], marker=STYLE[key]["marker"], label=f"m = {m}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Oversampling ratio s / m")
        ax.set_xticks([1, 1.5, 2])
    pct_axis(axes[0])
    axes[0].set_ylabel("Mean relative L2 error (test)")
    axes[1].legend(fontsize=8, title="DEIM on N only", title_fontsize=8)
    save(fig, "fig05_error_vs_oversampling.png")


def fig_conditioning(sw):
    cf = sw["configs"]
    ms = (8, 12, 16, 24)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for label, fn, key in (("Greedy DEIM (s = m)", lambda m: f"nonlinear|m={m}|s={m}|deim", "spec_deim_m16"),
                           ("Q-DEIM (s = m)", lambda m: f"nonlinear|m={m}|s={m}|qdeim", "spec_qdeim_m16"),
                           ("Q-DEIM + greedy (s = 2m)", lambda m: f"nonlinear|m={m}|s={2*m}|qdeim+greedy",
                            "spec_oversampled_m16_s32"),
                           ("Q-DEIM extended (s = 2m)", lambda m: f"nonlinear|m={m}|s={2*m}|qdeim_extended", "tensorial")):
        ax.plot(ms, [cf[fn(m)]["conditioning"]["cond"] for m in ms], color=STYLE[key]["color"],
                marker=STYLE[key]["marker"], label=label)
    ax.plot(ms, [sw["hybrid_localfd"][str(m)]["conditioning"]["cond"] for m in ms], color=STYLE["hybrid_localfd_m16"]["color"],
            marker="v", ls="--", label="Hybrid points (full-RHS basis)")
    ax.set_xlabel("DEIM basis dimension m")
    ax.set_ylabel("cond(P^T Psi)")
    ax.set_xticks(ms)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    save(fig, "fig06_interpolation_conditioning.png")


def fig_jacobian(stab):
    names = [("tensorial", "tensorial"), ("spec_full_same_pts", "spec_deim_m16"),
             ("hybrid_localfd", "hybrid_localfd_m16"), ("WEC_hybrid_localfd", "wec_hybrid_localfd_m16")]
    labels = {"spec_full_same_pts": "Spectral DEIM, same basis and points as hybrid"}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    t = np.linspace(0, 1, 20)
    for ax, case in zip(axes, ("5", "34")):
        c = stab["cases"][case]
        for name, key in names:
            re = c["models"][name]["jacobian_on_projected_reference"]["max_re_eig_per_snapshot"]
            ax.plot(t, re, color=STYLE[key]["color"], marker=STYLE[key]["marker"], markersize=3.5,
                    label=labels.get(name, STYLE[key]["label"]))
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_title(f"Test case {case}: A = {c['A']:.2f}, nu = {c['nu']:.3f} ({c['kind']})", fontsize=9)
        ax.set_xlabel("t (projected reference state)")
    axes[0].set_ylabel("max Re eig of reduced Jacobian")
    axes[1].legend(fontsize=7.5)
    save(fig, "fig07_jacobian_eigenvalues.png")


def trajectories(case_ids, keys):
    data = load_data()
    k, dealias = wavenumbers(NX)
    st = stored_basis()
    Phi = st["Phi"]
    reg = method_registry(Phi, data, k, dealias, hybrid_op=(st["M"], st["points"]), db=deim_bases(data=data),
                          sigma=training_sigma(Phi, data["train"][2]),
                          deim_cfgs={n: COMPARISON_DEIM[n] for n in keys if n in COMPARISON_DEIM})
    cases = case_set("test", data)
    out = {}
    for i in case_ids:
        lab, u0, nu, ref = cases[i]
        with np.errstate(all="ignore"):
            out[i] = {"ref": ref.astype(np.float64), "nu": nu,
                      **{n: reg[n]["predict"](u0, nu).astype(np.float64) for n in keys}}
    return out


def fig_energy_and_profiles(stab):
    keys = ["tensorial", "spec_qdeim_m16", "hybrid_localfd_m16", "wec_hybrid_localfd_m16"]
    tr = trajectories([5, 34], keys)
    dx = 2 * np.pi / NX
    t = np.linspace(0, 1, 20)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, i in zip(axes, (5, 34)):
        E = lambda u: 0.5 * dx * np.sum(u**2, axis=1)
        ax.plot(t, E(tr[i]["ref"]), color="#0b0b0a", lw=2.4, alpha=0.35, label="Reference (FOM)")
        for n in keys:
            ax.plot(t, E(tr[i][n]), color=STYLE[n]["color"], marker=STYLE[n]["marker"], markersize=3.5,
                    label=STYLE[n]["label"])
        c = stab["cases"][str(i)]
        ax.set_title(f"Test case {i}: A = {c['A']:.2f}, nu = {c['nu']:.3f}", fontsize=9)
        ax.set_xlabel("t")
    axes[0].set_ylabel("Energy 0.5 * sum(u^2) dx")
    axes[1].legend(fontsize=7.5)
    save(fig, "fig08_energy_evolution.png")

    x = np.linspace(0, 2 * np.pi, NX, endpoint=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax, j in zip(axes, (10, 19)):
        ax.plot(x, tr[5]["ref"][j], color="#0b0b0a", lw=2.6, alpha=0.35, label="Reference (FOM)")
        for n in keys:
            ax.plot(x, tr[5][n][j], color=STYLE[n]["color"], lw=1.3, label=STYLE[n]["label"])
        c = stab["cases"]["5"]
        ax.set_title(f"Test case 5 (A = {c['A']:.2f}, nu = {c['nu']:.3f}), t = {t[j]:.2f}", fontsize=9)
        ax.set_xlabel("x")
    axes[0].set_ylabel("u(x, t)")
    axes[0].legend(fontsize=7.5, loc="lower left")
    save(fig, "fig09_predicted_vs_reference.png")


def fig_distribution(cmp_):
    keys = ["tensorial", "spec_deim_m16", "spec_qdeim_m16", "spec_oversampled_m16_s32", "wec_hybrid_localfd_m16",
            "hybrid_localfd_m16"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    rng = np.random.default_rng(0)
    for j, n in enumerate(keys):
        e = np.array([r["rel_l2"] for r in cmp_["accuracy"]["test"][n]["rows"]])
        ax.scatter(e, j + rng.uniform(-0.18, 0.18, e.size), s=14, color=STYLE[n]["color"], marker=STYLE[n]["marker"],
                   edgecolors="white", linewidths=0.4)
        ax.plot([np.median(e)] * 2, [j - 0.3, j + 0.3], color="#0b0b0a", lw=1.6)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([STYLE[n]["label"] for n in keys])
    ax.invert_yaxis()
    pct_axis(ax, "x")
    ax.set_xlabel("Relative L2 error per test case (bar = median)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig10_error_distribution.png")


def fig_scaling(sc):
    rows = sc["rows"]
    nx = [r["Nx"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for n in ("fom", "rom|e2e", "tensorial|e2e", "hybrid_localfd_m16|e2e", "spec_qdeim_m16|e2e",
              "spec_oversampled_m16_s32|e2e"):
        key = n.split("|")[0]
        ax.plot(nx, [r["mean_of_case_medians_ms"][n] for r in rows], **STYLE[key])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(nx)
    ax.set_xticklabels([str(v) for v in nx])
    ax.set_xlabel("Grid size Nx")
    ax.set_ylabel(f"End-to-end time (ms, mean over {len(rows[0]['cases'])} cases)")
    ax.legend(fontsize=7.5)
    save(fig, "fig11_runtime_scaling_nx.png")


def fig_tradeoff(cmp_):
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    tm = cmp_["timing"]["workload_mean_ms"]
    for n in ("rom", "tensorial", "hybrid_localfd_m16", "spec_deim_m16", "spec_qdeim_m16",
              "spec_oversampled_m16_s32", "wec_hybrid_localfd_m16"):
        s_ = cmp_["accuracy"]["test"][n]["summary"]
        ax.scatter(tm[n + "|e2e"], s_["mean"], s=60, color=STYLE[n]["color"], marker=STYLE[n]["marker"],
                   label=STYLE[n]["label"], edgecolors="white" if n != "rom" else None, linewidths=0.8, zorder=3)
    ax.axvline(tm["fom"], color=STYLE["fom"]["color"], ls="--", lw=1.2, label="Full-order solver time")
    ax.set_xlabel("End-to-end time per trajectory (ms, 40-case mean)")
    ax.set_ylabel("Mean relative L2 error (test)")
    pct_axis(ax)
    ax.set_xlim(left=0)
    ax.legend(fontsize=7.5, loc="center right")
    save(fig, "fig12_accuracy_vs_runtime.png")


if __name__ == "__main__":
    sens, tb, cmp_, sw, stab, sc = (load(n) for n in ("sensitivity", "tensorial_bench", "comparison",
                                                      "deim_sweep", "stability", "scaling"))
    fig_rank(sens, tb)
    fig_viscosity(cmp_)
    fig_deim_dimension(sw)
    fig_oversampling(sw)
    fig_conditioning(sw)
    fig_jacobian(stab)
    fig_energy_and_profiles(stab)
    fig_distribution(cmp_)
    fig_scaling(sc)
    fig_tradeoff(cmp_)
