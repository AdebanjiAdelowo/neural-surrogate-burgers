"""Follow-up audit: reduced-model time-step/stability, DEIM error diagnosis, aligned snapshot data.

Stages (JSON in report/audit/followup_<stage>[_tag].json; historical results are never touched):
  stability    the historical step rule written out, reduced diffusion operators, frozen-Jacobian
               RK4 stability limits along trajectories, and the regression case behind the rule.
  multiplier   predetermined multipliers of the historical dt on 5 a-priori cases (A/nu ranks).
  policy       all 40 val + 40 test cases: A historical rule, B FOM step count, R revised rule.
  scaling2     Nx = 128..2048 under the historical and revised policies (per-stage vs end-to-end).
  stencil      attribution re-check, stencil order 2 vs 4, unstable-case / conditioning diagnosis.
  aligned_data corrected (aligned) datasets, rebuild + re-evaluation, legacy-data resampling check.

Every accuracy number uses the TIME-ALIGNED protocol (aligned FOM reference, aligned reduced models),
see scripts/hyperreduction_benchmark.py and src/snapshot_times.py.
Run: python scripts/hyperreduction_followup.py <stage> [--tag TAG]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hyperreduction_benchmark import (aligned_dt as _aligned_dt, L, M_POINTS, N_SAVE, NX, R_MODES, T_FINAL, _fom_ensemble,
                                               build_bases, deim_operator, environment_info, err_stats,
                                               finish_env, interleaved_timing, load_data,
                                               rhs_components, save, stats, wavenumbers)
from src.deim import _stencil_support, deim_rom_predict, deim_rom_reduced_trajectory
from src.metrics import relative_l2
from src.pod_rom import (_burgers_rhs_physical, build_pod_basis, rom_predict,
                          rom_reduced_trajectory)
from src.snapshot_times import intended_times, resample_in_time, saved_times
from src.solver import solve_burgers
from src.timestep import (RK4_IMAG_LIMIT, RK4_REAL_LIMIT, diffusion_dt, fom_advective_dt,
                           historical_dt, historical_rule_fractions_of_rk4_limit,
                           numerical_jacobian, reduced_diffusion_matrix_deim,
                           reduced_diffusion_matrix_rom, revised_dt, rk4_max_stable_dt,
                           spectral_radius)

BLOWUP = 10.0  # a run is "unstable" if non-finite or max|u| exceeds BLOWUP * max|u0| (Burgers obeys
#               the maximum principle, so the true solution never exceeds max|u0|)


# ------------------------------------------------------------------------------------- helpers
def make_ctx(Phi, M, pts, Nx=NX):
    k, dealias = wavenumbers(Nx)
    return {"Phi": Phi, "M": M, "pts": pts, "Nx": Nx, "k": k, "dealias": dealias,
            "K": {"rom": reduced_diffusion_matrix_rom(Phi, k),
                  "deim": reduced_diffusion_matrix_deim(Phi, M, pts, Nx)}}


def stored_ctx():
    st = np.load("experiments/deim_basis.npz")
    return make_ctx(st["Phi"], st["M"], st["points"])


def rom_rhs_fn(C, nu):
    return lambda a: C["Phi"].T @ _burgers_rhs_physical(C["Phi"] @ a, C["k"], nu, C["dealias"])


def deim_rhs_fn(C, nu):
    support, neigh = _stencil_support(C["pts"], C["Nx"], 1)
    idx, Ps, dx, M = np.searchsorted(support, neigh), C["Phi"][support, :], L / C["Nx"], C["M"]

    def f(a):
        us = Ps @ a
        ul, uc, ur = us[idx[:, 0]], us[idx[:, 1]], us[idx[:, 2]]
        return M @ (-uc * (ur - ul) / (2 * dx) + nu * (ur - 2 * uc + ul) / dx**2)
    return f


def run_red(kind, u0, nu, dt, C, stencil_order=2, min_steps=None, aligned=True):
    a0, u0max = C["Phi"].T @ u0, np.abs(u0).max()
    with np.errstate(all="ignore"):
        if kind == "rom":
            _, a, n = rom_reduced_trajectory(a0, u0max, C["Phi"], C["k"], nu, C["dealias"],
                                             T_FINAL, N_SAVE, dt, aligned, min_steps)
        else:
            _, a, n = deim_rom_reduced_trajectory(a0, u0max, C["Phi"], C["M"], C["pts"], C["Nx"],
                                                  nu, T_FINAL, N_SAVE, dt, stencil_order, aligned,
                                                  min_steps)
    return a, n


def assess(a, C, ref, u0max):
    u = (a @ C["Phi"].T).astype(np.float32)
    finite = bool(np.all(np.isfinite(u)))
    with np.errstate(all="ignore"):
        mx = float(np.max(np.abs(u))) if finite else float("inf")
        bad = np.where(~np.isfinite(u).all(axis=1) | (np.abs(np.nan_to_num(u, nan=1e30, posinf=1e30)).max(axis=1)
                                                       > BLOWUP * u0max))[0]
    err = relative_l2(u, ref) if finite else float("inf")
    return {"finite": finite, "stable": bool(finite and mx <= BLOWUP * u0max), "max_abs_over_u0max":
            mx / float(u0max), "rel_l2": err, "first_bad_snapshot": int(bad[0]) if bad.size else None}


def dt_policy(policy, kind, u0max, nu, Nx, C):
    if policy == "A":
        return historical_dt(u0max, Nx, nu)
    if policy == "B":
        return fom_advective_dt(u0max, Nx)
    return revised_dt(u0max, Nx, nu, C["K"][kind])


def snap_dt(dt):
    """Align a step size for the full-grid DIAGNOSTIC variants (they use scripts.hyperreduction_benchmark.
    _reduced_rk4, which does not align by itself): same n_steps the aligned reduced models would use."""
    return _aligned_dt(max(4 * N_SAVE, int(np.ceil(T_FINAL / dt))))


def aligned_ref(A, nu, Nx=NX):
    return solve_burgers(A=A, nu=nu, Nx=Nx, T=T_FINAL, n_save=N_SAVE, align_snapshots=True)[2]


def apriori_cases(tA, tnu, ranks=(0, 10, 20, 30, 39)):
    """Cases chosen by the RANK of A/nu (a known problem quantity, a steepness proxy), never by any
    observed error: rank 0 = smoothest ... rank 39 = steepest."""
    order = np.argsort(tA / tnu)
    return [int(order[r]) for r in ranks]


# ================================================================================== stability
def stage_stability(tag):
    env = environment_info()
    C = stored_ctx()
    data = load_data()
    tA, tnu, tU = data["test"]
    out = {"rk4_real_limit": RK4_REAL_LIMIT, "rk4_imag_limit": RK4_IMAG_LIMIT,
           "historical_rule": "dt = min(0.25 dx/(max|u0|+1e-8), 0.4 dx^2/(2 nu+1e-12)); n_steps = "
                              "max(80, ceil(T/dt))  (src/pod_rom.py, src/deim.py; added in 8fdbca0)",
           "historical_diffusion_constant_as_fraction_of_rk4_limit":
               historical_rule_fractions_of_rk4_limit()}
    dx = L / NX
    out["fullgrid_lambda_max"] = {"spectral_(pi/dx)^2": (np.pi / dx) ** 2, "fd3_4/dx^2": 4 / dx**2}
    out["reduced_diffusion"] = {}
    for kind in ("rom", "deim"):
        K = C["K"][kind]
        ev = np.linalg.eigvals(K)
        out["reduced_diffusion"][kind] = {
            "rho": spectral_radius(K), "eig_min_real": float(ev.real.min()),
            "eig_max_real": float(ev.real.max()), "max_abs_imag": float(np.abs(ev.imag).max()),
            "eigenvalues": sorted(ev.real.tolist()),
            "implied_dt_limit_at_safety_0.5": {f"nu={nu}": diffusion_dt(nu, K) for nu in (0.01, 0.1)},
            "historical_dt_diffusive": {f"nu={nu}": 0.4 * dx**2 / (2 * nu) for nu in (0.01, 0.1)}}
    out["reduced_diffusion"]["ratio_rho_fullgrid_over_reduced"] = {
        "rom": float((np.pi / dx) ** 2 / spectral_radius(C["K"]["rom"])),
        "deim": float((4 / dx**2) / spectral_radius(C["K"]["deim"]))}
    print("rho(K) rom %.1f deim %.1f | full-grid spectral %.0f fd3 %.0f" % (
        spectral_radius(C["K"]["rom"]), spectral_radius(C["K"]["deim"]), (np.pi / dx) ** 2, 4 / dx**2))

    cases = []
    for i in apriori_cases(tA, tnu):
        A, nu = float(tA[i]), float(tnu[i])
        ref = aligned_ref(A, nu)
        u0max = np.abs(ref[0]).max()
        row = {"i": i, "A": A, "nu": nu, "dt_historical": float(historical_dt(u0max, NX, nu)),
               "dt_fom": float(fom_advective_dt(u0max, NX)),
               "dt_revised": {kd: float(revised_dt(u0max, NX, nu, C["K"][kd])) for kd in ("rom", "deim")},
               "states": []}
        for j in (0, 5, 10, 15, 19):
            a = C["Phi"].T @ ref[j]
            st = {"snapshot": j}
            for kd, f in (("rom", rom_rhs_fn(C, nu)), ("deim", deim_rhs_fn(C, nu))):
                ev = np.linalg.eigvals(numerical_jacobian(f, a))
                st[kd] = {"max_re_eig": float(ev.real.max()), "spectral_radius": float(np.abs(ev).max()),
                          "dt_star_frozen_rk4": rk4_max_stable_dt(ev)}
            row["states"].append(st)
        for kd in ("rom", "deim"):
            row[f"min_dt_star_{kd}"] = min(s[kd]["dt_star_frozen_rk4"] for s in row["states"])
            row[f"dt_star_over_dt_historical_{kd}"] = row[f"min_dt_star_{kd}"] / row["dt_historical"]
            row[f"dt_star_over_dt_fom_{kd}"] = row[f"min_dt_star_{kd}"] / row["dt_fom"]
        cases.append(row)
        print(f"case {i} A={A:.2f} nu={nu:.3f}: dt_hist {row['dt_historical']:.4f} dt_fom {row['dt_fom']:.4f} "
              f"dt* rom {row['min_dt_star_rom']:.3f} deim {row['min_dt_star_deim']:.3f}")
    out["cases"] = cases

    # the regression behind the rule (tests/test_pod_rom.py::test_rom_stable_at_higher_viscosity_..)
    rng = np.random.default_rng(0)
    ens = [solve_burgers(A=rng.uniform(0.5, 2.0), nu=rng.uniform(0.01, 0.1), Nx=NX, T=1.0,
                         n_save=10)[2] for _ in range(10)]
    Phi10, _ = build_pod_basis(np.concatenate(ens, axis=0), r=8)
    C10 = make_ctx(Phi10, np.zeros((8, 1)), np.array([0]))
    gt = solve_burgers(A=1.0, nu=0.15, Nx=NX, T=1.0, n_save=10)[2]
    u0 = gt[0]
    res = {"rho_K_rom_10traj_basis": spectral_radius(C10["K"]["rom"]),
           "rho_K_rom_stored_basis": spectral_radius(C["K"]["rom"])}
    for label, dt in (("historical", historical_dt(np.abs(u0).max(), NX, 0.15)),
                      ("fom_advective_only", fom_advective_dt(np.abs(u0).max(), NX))):
        for bname, CC in (("10traj_basis", C10), ("stored_basis", C)):
            _, a_ = None, run_red("rom", u0, 0.15, dt, CC, aligned=False)[0]
            u = a_ @ CC["Phi"].T
            res[f"{label}__{bname}"] = {"finite": bool(np.all(np.isfinite(u))),
                                        "max_abs": float(np.nanmax(np.abs(u))) if np.all(np.isfinite(u)) else None}
    res["dt_revised_10traj_basis"] = float(revised_dt(np.abs(u0).max(), NX, 0.15, C10["K"]["rom"]))
    out["regression_case_nu0.15_A1"] = res
    print(res)
    out["env"] = finish_env(env)
    save("followup_stability", tag, out)


# ================================================================================== multiplier
def stage_multiplier(tag, mults=(1, 2, 4, 8, 16, 32)):
    env = environment_info()
    data = load_data()
    trA, trnu, trU = data["train"]
    tA, tnu, tU = data["test"]
    Phi_full, _, deim, k, dealias = build_bases(trU, trnu, NX, r_max=R_MODES, m_list=(12, 16))
    Phi = Phi_full[:, :R_MODES]
    ctxs = {"rom_r8": (make_ctx(Phi, np.zeros((8, 1)), np.array([0])), "rom")}
    for m in (12, 16):
        pts = deim[m]["points"]
        ctxs[f"deim_r8_m{m}"] = (make_ctx(Phi, deim_operator(Phi, deim[m]["Psi"], pts), pts), "deim")
    rows = []
    for i in apriori_cases(tA, tnu):
        A, nu = float(tA[i]), float(tnu[i])
        ref = aligned_ref(A, nu)
        u0, u0max = ref[0], np.abs(ref[0]).max()
        dt_h = historical_dt(u0max, NX, nu)
        for name, (C, kind) in ctxs.items():
            for mu in mults:
                dt = mu * dt_h
                a, n = run_red(kind, u0, nu, dt, C, min_steps=N_SAVE - 1)
                res = assess(a, C, ref, u0max)
                t = interleaved_timing({"x": lambda: run_red(kind, u0, nu, dt, C, min_steps=N_SAVE - 1)},
                                       reps=5, warmup=1)["x"]
                res.update({"case": i, "A": A, "nu": nu, "model": name, "multiplier": mu, "n_steps": n,
                           "dt_over_dt_fom": float(dt / fom_advective_dt(u0max, NX)),
                           "solver_only_ms": 1e3 * float(np.median(t))})
                rows.append(res)
        print(f"case {i} (A={A:.2f}, nu={nu:.3f}) done")
    out = {"multipliers": mults, "case_selection": "test-set ranks 0,10,20,30,39 of A/nu, fixed a priori",
           "stability_criterion": f"finite and max|u| <= {BLOWUP} max|u0|",
           "step_rule": "n_steps = aligned(ceil(T/(mult*dt_hist))) with a floor of n_save-1 = 19",
           "rows": rows, "env": finish_env(env)}
    for r in rows:
        if r["model"] != "deim_r8_m12":
            print(f"{r['model']:12s} case {r['case']:2d} x{r['multiplier']:2d} steps {r['n_steps']:4d} "
                  f"{'ok ' if r['stable'] else 'BAD'} err {100*r['rel_l2']:9.4f}% max|u|/u0 {r['max_abs_over_u0max']:.2f}")
    save("followup_multiplier", tag, out)


# ===================================================================================== policy
def stage_policy(tag, reps=7):
    env = environment_info()
    data = load_data()
    C = stored_ctx()
    out = {"policies": {"A": "historical rule", "B": "FOM step count (advective bound only)",
                        "R": "revised: min(B, 0.5*RK4-real-limit/(nu*rho(K_reduced)))"},
           "config": {"r": R_MODES, "m": M_POINTS}, "splits": {}}
    for split in ("val", "test"):
        sA, snu, sU = data[split]
        rows = []
        for i in range(len(sA)):
            A, nu, u0 = float(sA[i]), float(snu[i]), sU[i, 0]
            ref = aligned_ref(A, nu)
            u0max = np.abs(u0).max()
            row = {"i": i, "A": A, "nu": nu}
            calls = {}
            if split == "test":
                calls["fom"] = lambda: solve_burgers(A=A, nu=nu, Nx=NX, T=T_FINAL, n_save=N_SAVE,
                                                     align_snapshots=True)
            for kind in ("rom", "deim"):
                for pol in "ABR":
                    dt = dt_policy(pol, kind, u0max, nu, NX, C)
                    a, n = run_red(kind, u0, nu, dt, C)
                    res = assess(a, C, ref, u0max)
                    res.update({"n_steps": n, "dt": float(dt)})
                    row[f"{kind}_{pol}"] = res
                    if split == "test":
                        calls[f"{kind}_{pol}"] = (lambda kind=kind, dt=dt: run_red(kind, u0, nu, dt, C))
            if split == "test":
                t = interleaved_timing(calls, reps=reps, warmup=1)
                for n_, v in t.items():
                    (row if n_ == "fom" else row[n_]).__setitem__("median_ms" if n_ != "fom" else "fom_ms",
                                                                  1e3 * float(np.median(v)))
            rows.append(row)
        summ = {}
        for kind in ("rom", "deim"):
            for pol in "ABR":
                key = f"{kind}_{pol}"
                errs = np.array([r[key]["rel_l2"] for r in rows])
                s = {"n_stable": int(sum(r[key]["stable"] for r in rows)), "n": len(rows),
                     "err": err_stats(errs), "n_steps": stats([r[key]["n_steps"] for r in rows]),
                     "max_abs_over_u0max_worst": float(max(r[key]["max_abs_over_u0max"] for r in rows)),
                     "failures": [{"i": r["i"], "A": r["A"], "nu": r["nu"], "err": r[key]["rel_l2"],
                                   "max_abs_over_u0max": r[key]["max_abs_over_u0max"]}
                                  for r in rows if not r[key]["stable"]]}
                if split == "test":
                    fom = np.array([r["fom_ms"] for r in rows])
                    mine = np.array([r[key]["median_ms"] for r in rows])
                    s["workload_mean_ms"] = float(mine.mean())
                    s["speedup_vs_fom_workload"] = float(fom.mean() / mine.mean())
                    s["speedup_vs_fom_per_case"] = stats(fom / mine)
                    s["fraction_faster_than_fom"] = float(np.mean(fom / mine > 1))
                summ[key] = s
        if split == "test":
            summ["fom_workload_mean_ms"] = float(np.mean([r["fom_ms"] for r in rows]))
        # R vs B: how often does the reduced-diffusion bound bind?
        summ["R_differs_from_B_cases"] = {kd: int(sum(abs(r[f"{kd}_R"]["dt"] - r[f"{kd}_B"]["dt"]) > 1e-12
                                                     for r in rows)) for kd in ("rom", "deim")}
        out["splits"][split] = {"summary": summ, "rows": rows}
        print(f"--- {split}")
        for key in [k_ for k_ in summ if k_[:3] in ("rom", "dei")]:
            s = summ[key]
            print(f"{key:7s} stable {s['n_stable']}/{s['n']}  err mean {100*(s['err']['mean'] or np.nan):.4f}% "
                  f"med {100*(s['err']['median'] or np.nan):.4f}% max {100*(s['err']['max'] or np.nan):.3f}%  "
                  f"steps {s['n_steps']['mean']:.0f}"
                  + (f"  x{s['speedup_vs_fom_workload']:.2f} vs FOM" if split == "test" else ""))
    out["env"] = finish_env(env)
    save("followup_policy", tag, out)


# =================================================================================== scaling2
def stage_scaling2(tag):
    env = environment_info()
    data = load_data()
    tA, tnu, _ = data["test"]
    cases_idx = (0, 13)  # steep/low-nu and high-nu/low-A test cases, fixed a priori
    out = {"cases": {str(i): {"A": float(tA[i]), "nu": float(tnu[i])} for i in cases_idx}, "rows": []}
    for Nx, n_train, rom_A, n_acc, reps in ((128, 200, True, 10, 9), (256, 100, True, 10, 7),
                                            (512, 100, True, 10, 5), (1024, 100, True, 10, 3),
                                            (2048, 40, False, 3, 2)):
        print(f"Nx={Nx}: ensemble ...")
        A_, nu_, U_ = _fom_ensemble(Nx, n_train)
        Phi_full, _, deim, k, dealias = build_bases(U_, nu_, Nx, r_max=R_MODES, m_list=(M_POINTS,))
        Phi = Phi_full[:, :R_MODES]
        pts = deim[M_POINTS]["points"]
        C = make_ctx(Phi, deim_operator(Phi, deim[M_POINTS]["Psi"], pts), pts, Nx)
        a_mid = Phi.T @ solve_burgers(A=float(tA[0]), nu=float(tnu[0]), Nx=Nx, T=T_FINAL,
                                      n_save=N_SAVE)[2][10]
        comp = rhs_components(Phi, pts, C["M"], Nx, float(tnu[0]), a_mid, k, dealias)
        stage = {"rom_rhs_us": 1e6 * sum(comp[c] for c in comp if c.startswith("rom.")),
                 "deim_rhs_us": 1e6 * sum(comp[c] for c in comp if c.startswith("deim.")),
                 "fom_nonlinear_us": 1e6 * comp["fom.nonlinear (1 ifft + 1 fft)"]}
        for i in cases_idx:
            A, nu = float(tA[i]), float(tnu[i])
            u0 = (A * np.sin(np.linspace(0, L, Nx, endpoint=False))).astype(np.float32)
            u0max = np.abs(u0).max()
            calls = {"fom": lambda: solve_burgers(A=A, nu=nu, Nx=Nx, T=T_FINAL, n_save=N_SAVE,
                                                  align_snapshots=True)}
            steps = {}
            for kind in ("rom", "deim"):
                for pol in "AR":
                    if kind == "rom" and pol == "A" and not rom_A:
                        continue
                    dt = dt_policy(pol, kind, u0max, nu, Nx, C)
                    steps[f"{kind}_{pol}"] = run_red(kind, u0, nu, dt, C)[1]
                    calls[f"{kind}_{pol}__solver"] = (lambda kind=kind, dt=dt: run_red(kind, u0, nu, dt, C))
                    if kind == "rom":
                        calls[f"{kind}_{pol}__full"] = (lambda dt=dt: rom_predict(
                            u0, Phi, k, nu, dealias, T=T_FINAL, n_save=N_SAVE, dt=dt, align_snapshots=True))
                    else:
                        calls[f"{kind}_{pol}__full"] = (lambda dt=dt: deim_rom_predict(
                            u0, Phi, C["M"], pts, Nx, nu=nu, T=T_FINAL, n_save=N_SAVE, dt=dt,
                            align_snapshots=True))
            t = interleaved_timing(calls, reps=reps, warmup=1)
            med = {n: 1e3 * float(np.median(v)) for n, v in t.items()}
            from scripts.hyperreduction_benchmark import fom_steps
            row = {"Nx": Nx, "case": i, "A": A, "nu": nu, "n_steps": {"fom_aligned": None, **steps},
                   "median_ms": med, "per_stage_us": stage,
                   "speedup_vs_fom_solver_only": {n[:-8]: med["fom"] / med[n] for n in med if n.endswith("__solver")},
                   "speedup_vs_fom_reconstruction_inclusive": {n[:-6]: med["fom"] / med[n] for n in med
                                                              if n.endswith("__full")},
                   "us_per_step_solver_only": {n[:-8]: 1e3 * med[n] / steps[n[:-8]] for n in med if n.endswith("__solver")}}
            row["n_steps"]["fom_aligned"] = int(-(-fom_steps(A, nu, Nx, T_FINAL, N_SAVE) // 19) * 19)
            out["rows"].append(row)
            print(f"  case {i}: steps {row['n_steps']}  x solver-only "
                  f"{ {k_: round(v, 2) for k_, v in row['speedup_vs_fom_solver_only'].items()} }")
        # accuracy on the first n_acc test parameter pairs
        acc = {f"{k_}_{p_}": [] for k_ in ("rom", "deim") for p_ in "AR"}
        for i in range(n_acc):
            A, nu = float(tA[i]), float(tnu[i])
            ref = aligned_ref(A, nu, Nx)
            u0max = np.abs(ref[0]).max()
            for kind in ("rom", "deim"):
                for pol in "AR":
                    if kind == "rom" and pol == "A" and not rom_A:
                        continue
                    a, _ = run_red(kind, ref[0], nu, dt_policy(pol, kind, u0max, nu, Nx, C), C)
                    acc[f"{kind}_{pol}"].append(assess(a, C, ref, u0max)["rel_l2"])
        out["rows"].append({"Nx": Nx, "accuracy_first_test_params": n_acc,
                            "accuracy": {k_: err_stats(v) for k_, v in acc.items() if v}})
        print("  accuracy:", {k_: f"{100*np.mean(v):.3f}%" for k_, v in acc.items() if v})
    out["env"] = finish_env(env)
    save("followup_scaling2", tag, out)


# ==================================================================================== stencil
def stage_stencil(tag):
    env = environment_info()
    data = load_data()
    trA, trnu, trU = data["train"]
    tA, tnu, tU = data["test"]
    C = stored_ctx()
    from scripts.hyperreduction_benchmark import _reduced_rk4, predict_fd_fullgrid
    out = {}
    refs = [aligned_ref(float(a), float(v)) for a, v in zip(tA, tnu)]

    # ---- (1) attribution re-check + stencil orders, Nx=128, all 40 test cases, policy R
    K4 = reduced_diffusion_matrix_deim(C["Phi"], C["M"], C["pts"], NX, stencil_order=4)
    variants = {"rom": [], "exactF_deim": [], "fd2_fullgrid_no_deim": [], "deim_order2": [],
                "deim_order4": []}
    steps = {"deim_order2": [], "deim_order4": []}
    for i in range(len(tA)):
        A, nu, u0 = float(tA[i]), float(tnu[i]), tU[i, 0]
        u0max, ref = np.abs(u0).max(), refs[i]
        dt2 = dt_policy("R", "deim", u0max, nu, NX, C)
        dt4 = revised_dt(u0max, NX, nu, K4)
        dtr = dt_policy("R", "rom", u0max, nu, NX, C)
        with np.errstate(all="ignore"):
            variants["rom"].append(assess(run_red("rom", u0, nu, dtr, C)[0], C, ref, u0max)["rel_l2"])
            a2, n2 = run_red("deim", u0, nu, dt2, C)
            a4, n4 = run_red("deim", u0, nu, dt4, C, stencil_order=4)
            variants["deim_order2"].append(assess(a2, C, ref, u0max)["rel_l2"])
            variants["deim_order4"].append(assess(a4, C, ref, u0max)["rel_l2"])
            steps["deim_order2"].append(n2)
            steps["deim_order4"].append(n4)
            rhs_ex = lambda a: C["M"] @ _burgers_rhs_physical(C["Phi"] @ a, C["k"], nu, C["dealias"])[C["pts"]]
            ae, _ = _reduced_rk4(rhs_ex, C["Phi"].T @ u0, u0max, NX, nu, T_FINAL, N_SAVE, snap_dt(dt2))
            variants["exactF_deim"].append(assess(ae, C, ref, u0max)["rel_l2"])
            uf = predict_fd_fullgrid(u0, C["Phi"], nu, T_FINAL, N_SAVE, snap_dt(dt2))
            variants["fd2_fullgrid_no_deim"].append(relative_l2(uf, ref))
    ss = {}
    for k_, v in variants.items():
        e = err_stats(v)
        e["n_above_100pct"] = int(np.sum(np.array(v) > 1.0))
        ss[k_] = e
        print(f"{k_:22s} mean {100*(e['mean'] or np.nan):8.4f}% median {100*(e['median'] or np.nan):.4f}% "
              f"max {100*(e['max'] or np.nan):.3f}%  >10%: {e['n_above_10pct']}")
    out["nx128_test40_policyR"] = {"errors": ss, "per_case": {k_: v for k_, v in variants.items()},
                                   "n_steps_mean": {k_: float(np.mean(v)) for k_, v in steps.items()}}
    # runtime: order 2 vs 4, solver-only, interleaved, per-stage = time / (4 steps)
    tim = {"o2": [], "o4": []}
    per_stage = {"o2": [], "o4": []}
    for i in range(len(tA)):
        nu, u0 = float(tnu[i]), tU[i, 0]
        u0max = np.abs(u0).max()
        dt2, dt4 = dt_policy("R", "deim", u0max, nu, NX, C), revised_dt(u0max, NX, nu, K4)
        t = interleaved_timing({"o2": lambda: run_red("deim", u0, nu, dt2, C),
                                "o4": lambda: run_red("deim", u0, nu, dt4, C, stencil_order=4)},
                               reps=7, warmup=1)
        for key, n in (("o2", steps["deim_order2"][i]), ("o4", steps["deim_order4"][i])):
            tim[key].append(1e3 * float(np.median(t[key])))
            per_stage[key].append(1e6 * float(np.median(t[key])) / (4 * n))
    out["nx128_runtime"] = {k_: {"trajectory_ms_mean": float(np.mean(v)),
                                 "per_stage_us_mean": float(np.mean(per_stage[k_]))} for k_, v in tim.items()}
    print(out["nx128_runtime"])

    # ---- (2) order 2 vs 4 at higher resolution (first 10 test parameter pairs)
    hi = []
    for Nx, n_train in ((256, 100), (512, 100)):
        A_, nu_, U_ = _fom_ensemble(Nx, n_train)
        Pf, _, dm, k, dealias = build_bases(U_, nu_, Nx, r_max=R_MODES, m_list=(M_POINTS,))
        Phi = Pf[:, :R_MODES]
        pts = dm[M_POINTS]["points"]
        Cn = make_ctx(Phi, deim_operator(Phi, dm[M_POINTS]["Psi"], pts), pts, Nx)
        K4n = reduced_diffusion_matrix_deim(Phi, Cn["M"], pts, Nx, stencil_order=4)
        e2, e4 = [], []
        for i in range(10):
            A, nu = float(tA[i]), float(tnu[i])
            ref = aligned_ref(A, nu, Nx)
            u0max = np.abs(ref[0]).max()
            e2.append(assess(run_red("deim", ref[0], nu, dt_policy("R", "deim", u0max, nu, Nx, Cn), Cn)[0],
                             Cn, ref, u0max)["rel_l2"])
            e4.append(assess(run_red("deim", ref[0], nu, revised_dt(u0max, Nx, nu, K4n), Cn, stencil_order=4)[0],
                             Cn, ref, u0max)["rel_l2"])
        hi.append({"Nx": Nx, "order2": err_stats(e2), "order4": err_stats(e4)})
        print(f"Nx={Nx}: order2 mean {100*np.mean(e2):.3f}% max {100*np.max(e2):.2f}% | order4 mean "
              f"{100*np.mean(e4):.3f}% max {100*np.max(e4):.2f}%")
    out["higher_resolution_first10"] = hi

    # ---- (3) DEIM diagnosis for r=8 and several m: conditioning, placement, timestep, Jacobian
    ms = (6, 8, 10, 12, 16, 24, 32)
    Pf, _, dm, k, dealias = build_bases(trU, trnu, NX, r_max=R_MODES, m_list=ms)
    Phi = Pf[:, :R_MODES]
    rows = []
    for m in ms:
        Psi, pts = dm[m]["Psi"], dm[m]["points"]
        M = deim_operator(Phi, Psi, pts)
        Cm = make_ctx(Phi, M, pts)
        srt = np.sort(pts)
        row = {"m": m, "cond_PtPsi": float(np.linalg.cond(Psi[pts, :])), "norm2_M": float(np.linalg.norm(M, 2)),
               "points": srt.tolist(), "min_gap": int(np.min(np.diff(np.r_[srt, srt[0] + NX]))),
               "n_points_in_left_half_[0,63]": int(np.sum(pts < 64)),
               "nearest_point_to_shock_index_64": int(np.min(np.abs(pts - 64))),
               "rho_K_deim": spectral_radius(Cm["K"]["deim"])}
        e_R, e_R4, e_ex, bad = [], [], [], []
        for i in range(len(tA)):
            nu, u0 = float(tnu[i]), tU[i, 0]
            u0max, ref = np.abs(u0).max(), refs[i]
            dtR = dt_policy("R", "deim", u0max, nu, NX, Cm)
            r1 = assess(run_red("deim", u0, nu, dtR, Cm)[0], Cm, ref, u0max)
            r4 = assess(run_red("deim", u0, nu, dtR / 4, Cm)[0], Cm, ref, u0max)
            e_R.append(r1["rel_l2"])
            e_R4.append(r4["rel_l2"])
            rhs_ex = lambda a: M @ _burgers_rhs_physical(Phi @ a, k, nu, dealias)[pts]
            ae, _ = _reduced_rk4(rhs_ex, Phi.T @ u0, u0max, NX, nu, T_FINAL, N_SAVE, snap_dt(dtR))
            e_ex.append(assess(ae, Cm, ref, u0max)["rel_l2"])
            if r1["rel_l2"] > 0.1:
                bad.append(i)
        row["err_policyR"] = err_stats(e_R)
        row["err_policyR_over_4_(4x steps)"] = err_stats(e_R4)
        row["err_exactF_deim"] = err_stats(e_ex)
        row["cases_above_10pct"] = bad
        # frozen-Jacobian growth along the trajectory for the (up to 5) worst cases
        growth = []
        for i in sorted(bad, key=lambda j: -e_R[j])[:5]:
            nu = float(tnu[i])
            mre = []
            for j in (0, 5, 10, 15, 19):
                ev = np.linalg.eigvals(numerical_jacobian(deim_rhs_fn(Cm, nu), Phi.T @ refs[i][j]))
                mre.append(float(ev.real.max()))
            growth.append({"case": i, "A": float(tA[i]), "nu": nu, "max_re_eig_at_snapshots": mre})
        row["worst_cases_frozen_jacobian"] = growth
        rows.append(row)
        print(f"m={m:2d} cond {row['cond_PtPsi']:.1f} |M| {row['norm2_M']:.2f} err_R mean "
              f"{100*row['err_policyR']['mean']:.3f}% max {100*row['err_policyR']['max']:.1f}% | dt/4 mean "
              f"{100*row['err_policyR_over_4_(4x steps)']['mean']:.3f}% | exactF mean {100*row['err_exactF_deim']['mean']:.3f}% "
              f"| n>10%: {len(bad)}")
    out["diagnosis_r8"] = rows
    out["env"] = finish_env(env)
    save("followup_stencil", tag, out)


# ============================================================================== aligned_data
def stage_aligned_data(tag):
    """Generate corrected (time-aligned) datasets next to the historical ones (never overwriting them),
    rebuild POD/DEIM from the aligned training set, re-evaluate, and check whether the legacy data can
    instead be time-corrected by resampling."""
    from scripts.generate_dataset import A_RANGE, NU_RANGE, SPLIT_SEED_OFFSETS, SPLITS
    from src.general_ic_solver import solve_burgers_general_ic
    env = environment_info()
    hist = load_data()
    aligned, A64 = {}, {}
    for split, n in SPLITS.items():
        rng = np.random.default_rng(SPLIT_SEED_OFFSETS[split])
        A = rng.uniform(*A_RANGE, n)
        nu = rng.uniform(*NU_RANGE, n)
        A64[split] = A
        U = np.stack([solve_burgers(A=float(a), nu=float(v), Nx=NX, T=T_FINAL, n_save=N_SAVE,
                                    align_snapshots=True)[2] for a, v in zip(A, nu)])
        aligned[split] = (A.astype(np.float32), nu.astype(np.float32), U)
        np.savez(f"data/aligned_{split}.npz", A=aligned[split][0], nu=aligned[split][1], u=U)
    ood = np.load("data/ood_family_test.npz")
    x = np.linspace(0, L, NX, endpoint=False)
    Uo = np.stack([solve_burgers_general_ic(ood["A1"][i] * np.sin(x) + ood["A2"][i] * np.sin(2 * x),
                                            float(ood["nu"][i]), T=T_FINAL, n_save=N_SAVE,
                                            align_snapshots=True)[2] for i in range(len(ood["nu"]))])
    np.savez("data/aligned_ood_family_test.npz", A1=ood["A1"], A2=ood["A2"], nu=ood["nu"], u=Uo)
    out = {"files_written": ["data/aligned_train.npz", "data/aligned_val.npz", "data/aligned_test.npz",
                             "data/aligned_ood_family_test.npz"], "historical_files_untouched": True}
    d = {s_: relative_l2(aligned[s_][2], hist[s_][2]) for s_ in ("train", "val", "test")}
    out["aligned_vs_historical_index_wise_rel_l2"] = d
    out["test_examples_bitwise_identical_to_historical"] = int(sum(
        np.array_equal(aligned["test"][2][i], hist["test"][2][i]) for i in range(40)))

    # ---- legacy stored data, time-corrected by resampling instead of regeneration
    rows = []
    for i in range(40):
        A = A64["test"][i]
        dt_old = 0.25 * (L / NX) / (abs(A) + 1e-8)
        n_old = max(1, int(np.ceil(T_FINAL / dt_old)))  # the pre-fix solver had no 80-step floor
        t_old = saved_times(n_old, N_SAVE, T_FINAL)
        res = resample_in_time(t_old, hist["test"][2][i], intended_times(N_SAVE, T_FINAL)).astype(np.float32)
        rows.append({"i": i, "n_steps_legacy": n_old, "raw_index_wise": relative_l2(hist["test"][2][i], aligned["test"][2][i]),
                     "resampled": relative_l2(res, aligned["test"][2][i])})
    out["legacy_resampling"] = {"raw_index_wise": err_stats([r["raw_index_wise"] for r in rows]),
                                "resampled_to_uniform_times": err_stats([r["resampled"] for r in rows])}

    # ---- rebuild everything on the aligned data
    Phi_full, _, dm, k, dealias = build_bases(aligned["train"][2], aligned["train"][1], NX, r_max=R_MODES,
                                              m_list=(M_POINTS,))
    Phi = Phi_full[:, :R_MODES]
    pts = dm[M_POINTS]["points"]
    Ca = make_ctx(Phi, deim_operator(Phi, dm[M_POINTS]["Psi"], pts), pts)
    Cs = stored_ctx()
    sub = np.linalg.svd(Phi.T @ Cs["Phi"], compute_uv=False)
    out["aligned_basis_vs_stored_basis"] = {"principal_cosines_min": float(sub.min()),
                                            "deim_points_identical": bool(np.array_equal(np.sort(pts), np.sort(Cs["pts"]))),
                                            "n_points_common": int(len(set(pts.tolist()) & set(Cs["pts"].tolist())))}
    res = {}
    for label, C_ in (("aligned_basis", Ca), ("historical_basis", Cs)):
        for kind in ("rom", "deim"):
            for pol in ("A", "R"):
                e = []
                for i in range(40):
                    nu, u0 = float(aligned["test"][1][i]), aligned["test"][2][i, 0]
                    u0max = np.abs(u0).max()
                    a, _ = run_red(kind, u0, nu, dt_policy(pol, kind, u0max, nu, NX, C_), C_)
                    e.append(assess(a, C_, aligned["test"][2][i], u0max)["rel_l2"])
                res[f"{label}__{kind}_{pol}"] = err_stats(e)
    out["aligned_test_accuracy"] = res
    # out-of-family
    ro = {}
    for label, C_ in (("aligned_basis", Ca), ("historical_basis", Cs)):
        for kind in ("rom", "deim"):
            e = []
            for i in range(len(Uo)):
                nu, u0 = float(ood["nu"][i]), Uo[i, 0]
                u0max = np.abs(u0).max()
                a, _ = run_red(kind, u0, nu, dt_policy("R", kind, u0max, nu, NX, C_), C_)
                e.append(assess(a, C_, Uo[i], u0max)["rel_l2"])
            ro[f"{label}__{kind}_R"] = err_stats(e)
    out["aligned_ood_accuracy"] = ro
    for k_, v in {**res, **ro}.items():
        print(f"{k_:28s} mean {100*v['mean']:.4f}%  median {100*v['median']:.4f}%  max {100*v['max']:.3f}%")
    print("legacy raw", 100 * out["legacy_resampling"]["raw_index_wise"]["mean"], "% -> resampled",
          100 * out["legacy_resampling"]["resampled_to_uniform_times"]["mean"], "%")
    out["env"] = finish_env(env)
    save("followup_aligned_data", tag, out)


# ============================================================================== production
def stage_production(tag):
    """Regression verification of the PRODUCTION path: `rom_predict` / `deim_rom_predict` with their
    default (reduced-operator) time-step policy and `align_snapshots=True`, no helper choosing dt.
    Compared with `timestep_policy="historical"` under the same aligned evaluation."""
    env = environment_info()
    data = load_data()
    C = stored_ctx()
    k, dealias = C["k"], C["dealias"]

    def predict(kind, u0, nu, Cx, policy=None, Nx=NX):
        kw = {} if policy is None else {"timestep_policy": policy}
        with np.errstate(all="ignore"):
            if kind == "rom":
                return rom_predict(u0, Cx["Phi"], Cx["k"], nu, Cx["dealias"], T=T_FINAL, n_save=N_SAVE,
                                   align_snapshots=True, **kw)[1]
            return deim_rom_predict(u0, Cx["Phi"], Cx["M"], Cx["pts"], Nx, nu=nu, T=T_FINAL, n_save=N_SAVE,
                                    align_snapshots=True, **kw)[1]

    def steps(kind, u0, nu, Cx, policy=None, Nx=NX):
        kw = {} if policy is None else {"timestep_policy": policy}
        a0, u0max = Cx["Phi"].T @ u0, np.abs(u0).max()
        with np.errstate(all="ignore"):
            if kind == "rom":
                return rom_reduced_trajectory(a0, u0max, Cx["Phi"], Cx["k"], nu, Cx["dealias"], T_FINAL,
                                              N_SAVE, align_snapshots=True, **kw)[2]
            return deim_rom_reduced_trajectory(a0, u0max, Cx["Phi"], Cx["M"], Cx["pts"], Nx, nu, T_FINAL,
                                               N_SAVE, align_snapshots=True, **kw)[2]

    out = {"production_call": "rom_predict/deim_rom_predict(..., align_snapshots=True) with default "
                              "timestep_policy (= 'reduced'); historical = timestep_policy='historical'",
           "splits": {}}
    for split in ("val", "test"):
        sA, snu, sU = data[split]
        rows = []
        for i in range(len(sA)):
            A, nu, u0 = float(sA[i]), float(snu[i]), sU[i, 0]
            ref, u0max = aligned_ref(A, nu), np.abs(u0).max()
            row = {"i": i, "A": A, "nu": nu}
            for kind in ("rom", "deim"):
                for label, pol in (("default", None), ("historical", "historical")):
                    u = predict(kind, u0, nu, C, pol)
                    res = assess((u.astype(np.float64) @ C["Phi"]) if False else C["Phi"].T @ u.T.astype(np.float64)
                                 if False else np.linalg.lstsq(C["Phi"], u.T.astype(np.float64), rcond=None)[0].T
                                 if False else (u @ C["Phi"]), C, ref, u0max)
                    res["n_steps"] = steps(kind, u0, nu, C, pol)
                    res["rel_l2_direct"] = relative_l2(u, ref)
                    row[f"{kind}_{label}"] = res
            rows.append(row)
        summ = {}
        for kind in ("rom", "deim"):
            d = np.array([r[f"{kind}_default"]["rel_l2_direct"] for r in rows])
            h = np.array([r[f"{kind}_historical"]["rel_l2_direct"] for r in rows])
            fin = lambda key: all(np.isfinite(r[f"{kind}_{key}"]["rel_l2_direct"]) and
                                  r[f"{kind}_{key}"]["max_abs_over_u0max"] < BLOWUP for r in rows)
            summ[kind] = {"n": len(rows), "default_stable_all": bool(fin("default")),
                          "historical_stable_all": bool(fin("historical")),
                          "default_err": err_stats(d), "historical_err": err_stats(h),
                          "max_abs_diff_of_case_errors": float(np.max(np.abs(d - h))),
                          "max_rel_diff_of_case_errors": float(np.max(np.abs(d - h) / np.maximum(h, 1e-12))),
                          "steps_default": stats([r[f"{kind}_default"]["n_steps"] for r in rows]),
                          "steps_historical": stats([r[f"{kind}_historical"]["n_steps"] for r in rows])}
        out["splits"][split] = {"summary": summ, "rows": rows}
        for kind in ("rom", "deim"):
            s_ = summ[kind]
            print(f"{split} {kind:4s} stable(default) {s_['default_stable_all']} | err default {100*s_['default_err']['mean']:.4f}% "
                  f"(med {100*s_['default_err']['median']:.4f}%, max {100*s_['default_err']['max']:.3f}%) vs historical "
                  f"{100*s_['historical_err']['mean']:.4f}% | max case diff {s_['max_abs_diff_of_case_errors']:.2e} | steps "
                  f"{s_['steps_default']['mean']:.0f} vs {s_['steps_historical']['mean']:.0f}")

    # ---- nu = 0.15 regression (outside the training range)
    rng = np.random.default_rng(0)
    ens = [solve_burgers(A=rng.uniform(0.5, 2.0), nu=rng.uniform(0.01, 0.1), Nx=NX, T=1.0, n_save=10)[2]
           for _ in range(10)]
    Phi10, _ = build_pod_basis(np.concatenate(ens, axis=0), r=8)
    C10 = make_ctx(Phi10, np.zeros((8, 1)), np.array([0]))
    reg = {}
    for A, nu in ((1.0, 0.15),):
        ref = aligned_ref(A, nu)
        u0, u0max = ref[0], np.abs(ref[0]).max()
        for bname, Cx in (("stored_basis_r8", C), ("small_10traj_basis_r8", C10)):
            with np.errstate(all="ignore"):
                fom_step = rom_predict(u0, Cx["Phi"], Cx["k"], nu, Cx["dealias"], T=T_FINAL, n_save=N_SAVE,
                                       dt=fom_advective_dt(u0max, NX))[1]
            e = {"FOM_step_count_unaligned": {"finite": bool(np.all(np.isfinite(fom_step)))}}
            for label, pol in (("default_reduced", None), ("historical", "historical")):
                u = predict("rom", u0, nu, Cx, pol)
                e[label] = {"finite": bool(np.all(np.isfinite(u))), "rel_l2": relative_l2(u, ref),
                            "n_steps": steps("rom", u0, nu, Cx, pol)}
            reg[f"A={A}_nu={nu}__{bname}"] = e
        u = predict("deim", u0, nu, C)
        reg[f"A={A}_nu={nu}__deim_r8_m16"] = {"default_reduced": {"finite": bool(np.all(np.isfinite(u))),
                                                                   "rel_l2": relative_l2(u, ref)}}
    out["nu_0.15_regression"] = reg
    print(json.dumps(reg, indent=1, default=float))

    # ---- scaling through the production path
    tA, tnu, _ = data["test"]
    out["scaling"] = []
    for Nx, n_train, hist_rom, hist_deim, n_acc, reps in ((128, 200, True, True, 10, 5), (512, 100, True, True, 10, 3),
                                                          (1024, 100, False, True, 10, 3), (2048, 40, False, True, 3, 2)):
        A_, nu_, U_ = _fom_ensemble(Nx, n_train)
        Pf, _, dm, k_, de_ = build_bases(U_, nu_, Nx, r_max=R_MODES, m_list=(M_POINTS,))
        Phi = Pf[:, :R_MODES]
        pts = dm[M_POINTS]["points"]
        Cx = make_ctx(Phi, deim_operator(Phi, dm[M_POINTS]["Psi"], pts), pts, Nx)
        for i in (0, 13):
            A, nu = float(tA[i]), float(tnu[i])
            u0 = (A * np.sin(np.linspace(0, L, Nx, endpoint=False))).astype(np.float32)
            calls = {"fom": lambda: solve_burgers(A=A, nu=nu, Nx=Nx, T=T_FINAL, n_save=N_SAVE, align_snapshots=True),
                     "rom_default": lambda: predict("rom", u0, nu, Cx, None, Nx),
                     "deim_default": lambda: predict("deim", u0, nu, Cx, None, Nx)}
            st = {"fom": int(-(-max(int(np.ceil(T_FINAL / fom_advective_dt(np.abs(u0).max(), Nx))), 80) // 19) * 19),
                  "rom_default": steps("rom", u0, nu, Cx, None, Nx), "deim_default": steps("deim", u0, nu, Cx, None, Nx)}
            if hist_rom:
                calls["rom_historical"] = lambda: predict("rom", u0, nu, Cx, "historical", Nx)
                st["rom_historical"] = steps("rom", u0, nu, Cx, "historical", Nx)
            if hist_deim:
                calls["deim_historical"] = lambda: predict("deim", u0, nu, Cx, "historical", Nx)
                st["deim_historical"] = steps("deim", u0, nu, Cx, "historical", Nx)
            t = interleaved_timing(calls, reps=reps, warmup=1)
            med = {n: 1e3 * float(np.median(v)) for n, v in t.items()}
            out["scaling"].append({"Nx": Nx, "case": i, "A": A, "nu": nu, "n_steps": st, "median_ms": med,
                                   "speedup_vs_fom": {n: med["fom"] / med[n] for n in med if n != "fom"}})
            print(f"Nx={Nx} case {i}: steps {st} speed-up { {n: round(v, 2) for n, v in out['scaling'][-1]['speedup_vs_fom'].items()} }")
        acc = {"rom_default": [], "deim_default": []}
        for i in range(n_acc):
            A, nu = float(tA[i]), float(tnu[i])
            ref = aligned_ref(A, nu, Nx)
            for kind in ("rom", "deim"):
                acc[f"{kind}_default"].append(relative_l2(predict(kind, ref[0], nu, Cx, None, Nx), ref))
        out["scaling"].append({"Nx": Nx, "accuracy_first_test_params": n_acc,
                               "accuracy": {k_: err_stats(v) for k_, v in acc.items()}})
        print("  accuracy", {k_: f"{100*np.mean(v):.3f}%" for k_, v in acc.items()})
    out["env"] = finish_env(env)
    save("followup_production_validation", tag, out)


STAGES = {"stability": stage_stability, "multiplier": stage_multiplier, "policy": stage_policy,
          "scaling2": stage_scaling2, "stencil": stage_stencil,
          "aligned_data": stage_aligned_data, "production": stage_production}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    STAGES[args.stage](args.tag)
