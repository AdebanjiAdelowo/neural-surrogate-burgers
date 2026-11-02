"""Spatial / temporal convergence, reference verification and ROM accuracy against an exact solution
(report/burgers_spatial_convergence_study.md; pre-registered plan in report/spatial_convergence/PLAN.md).

Every stage writes report/spatial_convergence/<stage>[_<tag>].json; nothing else is modified.

Stages:
  spatial      FOM error vs N for the representative cases (common fine step) and the 40 test cases
               (baseline step rule), against the Cole-Hopf solution; consecutive differences E_{N,2N}
  temporal     FOM time-integration error at fixed N (step refinement vs 64x steps), observed orders
  reference    verification and uncertainty of the Cole-Hopf reference (quadrature, sampling, FOM)
  diagnostics  Fourier spectra, front resolution, mass, energy budget, maximum principle, gradients
  rom          tensorial POD vs N and r: errors vs FOM and exact, projection errors, offline cost
  timestep     reduced time-step policies: Jacobian analysis, pre-registered acceptance, work-precision

Run:  python scripts/spatial_convergence_study.py <stage> [--tag TAG]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_dataset import N_SAVE, T_FINAL
from scripts.hyperreduction_benchmark import environment_info, finish_env, load_data
from src.convergence import (default_step_count, rel_l2, rel_l2_per_snapshot, sine_ic, sine_solution,
                             solve_burgers_f64)

OUT_DIR = "report/spatial_convergence"
TIMES = np.linspace(0.0, T_FINAL, N_SAVE)  # exactly the aligned snapshot times j/19
N_REF = 8192                                # grid on which the exact solution is sampled

# pre-registered representative cases (report/spatial_convergence/PLAN.md, Section 3)
CASES = {
    "corner_lowA_highnu": (0.5, 0.1), "corner_highA_highnu": (2.0, 0.1),
    "corner_lowA_lownu": (0.5, 0.01), "corner_highA_lownu": (2.0, 0.01), "centre": (1.25, 0.055),
    "test5": None, "test33": None, "test0": None,
    "extrap_A2.5_nu0.05": (2.5, 0.05), "extrap_A1.0_nu0.15": (1.0, 0.15), "extrap_A0.2_nu0.05": (0.2, 0.05),
}


def save(name, tag, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}{'_' + tag if tag else ''}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"saved {path}")


def cases(data=None):
    """{name: (A, nu)} with the test-case entries filled from data/test.npz."""
    data = data or load_data()
    tA, tnu, _ = data["test"]
    out = {}
    for name, p in CASES.items():
        if p is None:
            i = int(name[4:])
            p = (float(tA[i]), float(tnu[i]))
        out[name] = p
    return out


def exact(A, nu, N=N_REF):
    """Cole-Hopf solution at the 20 snapshot times on N points (float64)."""
    return sine_solution(A, nu).trajectory(N, TIMES)


def err_record(u, ref):
    per = rel_l2_per_snapshot(u, ref)
    return {"space_time": rel_l2(u, ref), "final": float(per[-1]), "per_snapshot": per,
            "max_snapshot": float(np.nanmax(per[1:]))}


# =================================================================================== stage: spatial
def stage_spatial(tag, n_list=(64, 128, 256, 512, 1024, 2048, 4096, 8192),
                  test_n_list=(64, 128, 256, 512, 1024, 2048)):
    env = environment_info()
    data = load_data()
    out = {"N_ref_exact": N_REF, "times": TIMES, "representative": {}, "test40": {}}
    # (a) representative cases: common step count = the baseline count of the N = 4096 grid
    for name, (A, nu) in cases(data).items():
        ref = exact(A, nu)
        n_steps = default_step_count(A, 4096)
        sols, rows = {}, {}
        for N in n_list:
            t0 = time.perf_counter()
            sols[N] = solve_burgers_f64(sine_ic(A, N), nu, n_steps=n_steps)[1]
            rows[N] = {"vs_exact": err_record(sols[N], ref), "fom_s": time.perf_counter() - t0}
        for N in n_list[:-1]:
            rows[N]["consecutive_E_N_2N"] = err_record(sols[N], sols[2 * N])
        out["representative"][name] = {"A": A, "nu": nu, "n_steps_common": n_steps, "by_N": rows}
        print(f"{name:22s} A={A:.2f} nu={nu:.4f} | " + " ".join(
            f"{N}:{rows[N]['vs_exact']['space_time']:.1e}" for N in n_list))
    # (b) the 40 test cases with the baseline step rule (the FOM as used for the ROM studies)
    tA, tnu, _ = data["test"]
    for i in range(len(tA)):
        A, nu = float(tA[i]), float(tnu[i])
        ref = exact(A, nu)
        rows = {}
        for N in test_n_list:
            u = solve_burgers_f64(sine_ic(A, N), nu)[1]
            rows[N] = {"space_time": rel_l2(u, ref), "final": float(rel_l2_per_snapshot(u, ref)[-1]),
                       "n_steps": default_step_count(A, N)}
        out["test40"][str(i)] = {"A": A, "nu": nu, "by_N": rows}
    summ = {}
    for N in test_n_list:
        e = np.array([out["test40"][str(i)]["by_N"][N]["space_time"] for i in range(len(tA))])
        summ[N] = {"mean": float(e.mean()), "median": float(np.median(e)), "max": float(e.max()),
                   "n_above_1pct": int((e > 1e-2).sum()), "n_above_1e-3": int((e > 1e-3).sum()),
                   "n_above_1e-6": int((e > 1e-6).sum())}
        print(f"test40 N={N:5d}: mean {e.mean():.2e} median {np.median(e):.2e} max {e.max():.2e} "
              f">1%: {summ[N]['n_above_1pct']}  >1e-3: {summ[N]['n_above_1e-3']}")
    out["test40_summary"] = summ
    out["env"] = finish_env(env)
    save("spatial", tag, out)


# ================================================================================== stage: temporal
FLOOR = 1e-11  # 10x the round-off plateau observed at N >= 1024 (about 1e-12, temporal_run1.json): no order below it


def stage_temporal(tag, names=("corner_lowA_highnu", "centre", "corner_highA_highnu", "corner_lowA_lownu",
                               "corner_highA_lownu", "test5", "test33"),
                   n_list=(256, 1024, 2048), mults=(1, 2, 4, 8), ref_mult=64):
    """Time-integration error of the FOM at fixed N: n_steps = m x (baseline count) against
    ref_mult x (baseline count) at the SAME N, so the spatial error cancels exactly. Observed order
    p = log2(e_m / e_2m), reported only when both errors exceed FLOOR."""
    env = environment_info()
    cs = cases()
    out = {"mults": mults, "ref_mult": ref_mult, "floor": FLOOR, "cases": {}}
    for name in names:
        A, nu = cs[name]
        rec = {"A": A, "nu": nu, "by_N": {}}
        for N in n_list:
            n0 = default_step_count(A, N)
            ref = solve_burgers_f64(sine_ic(A, N), nu, n_steps=n0 * ref_mult)[1]
            errs = {m: rel_l2(solve_burgers_f64(sine_ic(A, N), nu, n_steps=n0 * m)[1], ref) for m in mults}
            orders = {f"{m}->{2 * m}": (float(np.log2(errs[m] / errs[2 * m]))
                                         if errs[m] > FLOOR and errs[2 * m] > FLOOR else None)
                      for m in mults[:-1]}
            rec["by_N"][N] = {"base_steps": n0, "dt_base": T_FINAL / n0, "errors": errs, "orders": orders}
            print(f"{name:20s} N={N:5d} base {n0:5d} steps | errs " + " ".join(f"{errs[m]:.1e}" for m in mults)
                  + " | orders " + " ".join("-" if o is None else f"{o:.2f}" for o in orders.values()))
        out["cases"][name] = rec
    out["env"] = finish_env(env)
    save("temporal", tag, out)


# ================================================================================= stage: reference
def stage_reference(tag, fom_n=(4096, 8192)):
    """Verification of the exact (Cole-Hopf) reference and its uncertainty estimate, per case:
    (1) quadrature refinement (step / 2), (2) sampling: exact-solution energy beyond |k| = N_REF/4 on the
    N_REF grid, (3) the FOM at N = 4096 and 8192 with the baseline step count and twice it."""
    env = environment_info()
    data = load_data()
    tA, tnu, _ = data["test"]
    todo = dict(cases(data))
    todo.update({f"test{i}": (float(tA[i]), float(tnu[i])) for i in range(len(tA))})
    out = {"N_ref_exact": N_REF, "cases": {}}
    for name, (A, nu) in todo.items():
        ref = exact(A, nu)
        refined = sine_solution(A, nu, step_fraction=1 / 80).trajectory(N_REF, TIMES)
        c = np.abs(np.fft.fft(ref[1:], axis=-1)) ** 2
        k = np.abs(np.fft.fftfreq(N_REF, 1 / N_REF))
        tail = float(np.max(np.sum(c[:, k > N_REF // 4], axis=1) / np.sum(c, axis=1)))
        rec = {"A": A, "nu": nu, "quadrature_refinement": rel_l2(ref, refined),
               "exact_tail_energy_beyond_N_ref_over_4": tail, "fom": {}}
        rep = name in CASES
        for N in (fom_n if rep else fom_n[:1]):
            n0 = default_step_count(A, N)
            for m in ((1, 2) if rep else (1,)):
                u = solve_burgers_f64(sine_ic(A, N), nu, n_steps=n0 * m)[1]
                rec["fom"][f"N{N}_x{m}"] = err_record(u, ref)
        fom_disc = min(v["space_time"] for v in rec["fom"].values())
        rec["uncertainty_space_time"] = max(rec["quadrature_refinement"], fom_disc)
        rec["uncertainty_max_snapshot"] = max(rec["quadrature_refinement"],
                                              min(v["max_snapshot"] for v in rec["fom"].values()))
        out["cases"][name] = rec
        print(f"{name:22s} quad {rec['quadrature_refinement']:.1e} tail {tail:.1e} | " +
              " ".join(f"{k_}:{v['space_time']:.1e}" for k_, v in rec["fom"].items()) +
              f" | U {rec['uncertainty_space_time']:.1e}")
    u_all = [v["uncertainty_space_time"] for v in out["cases"].values()]
    out["max_uncertainty_space_time"] = float(max(u_all))
    out["max_uncertainty_max_snapshot"] = float(max(v["uncertainty_max_snapshot"] for v in out["cases"].values()))
    print(f"max reference uncertainty: space-time {max(u_all):.1e}, worst snapshot {out['max_uncertainty_max_snapshot']:.1e}")
    out["env"] = finish_env(env)
    save("reference", tag, out)


# =============================================================================== stage: diagnostics
def _dx_spectral(u, N):
    k = np.fft.fftfreq(N, 1 / N)
    return np.real(np.fft.ifft(1j * k * np.fft.fft(u, axis=-1), axis=-1))


def stage_diagnostics(tag, n_list=(64, 128, 256, 512, 1024, 2048, 4096)):
    """Fourier spectra and physical diagnostics of the FOM (baseline step rule) and of the exact
    solution, for the representative cases."""
    from src.pod_rom import _burgers_rhs_physical
    from scripts.hyperreduction_benchmark import wavenumbers
    env = environment_info()
    out = {"cases": {}}
    kk = np.fft.fftfreq(N_REF, 1 / N_REF)
    for name, (A, nu) in cases().items():
        ref = exact(A, nu)
        g_ref = _dx_spectral(ref, N_REF)
        j_star = int(np.argmax(np.abs(g_ref).max(axis=1)))       # snapshot of steepest gradient
        c_ref = np.abs(np.fft.fft(ref, axis=-1)) / N_REF
        dx_ref = 2 * np.pi / N_REF
        E_ref = 0.5 * dx_ref * np.sum(ref**2, axis=1)
        width = 2 * nu / max(np.abs(ref[j_star]).max(), 1e-12)  # tanh front width 2 nu / U
        rec = {"A": A, "nu": nu, "steepest_snapshot": j_star, "t_steepest": float(TIMES[j_star]),
               "exact_max_abs_ux": float(np.abs(g_ref).max()), "front_width_2nu_over_U": width,
               "exact_spectrum_steepest": [float(c_ref[j_star, k]) for k in range(0, N_REF // 2)],
               "exact_spectrum_final": [float(c_ref[-1, k]) for k in range(0, N_REF // 2)],
               "by_N": {}}
        for N in n_list:
            u = solve_burgers_f64(sine_ic(A, N), nu)[1]
            k, dealias = wavenumbers(N)
            kc = int(np.ceil(N / 3)) - 1
            c = np.abs(np.fft.fft(u, axis=-1)) / N
            dx = 2 * np.pi / N
            E = 0.5 * dx * np.sum(u**2, axis=1)
            mass = dx * np.sum(u, axis=1)
            # instantaneous discrete energy budget: convective (aliasing) vs viscous term
            p_nl = np.array([dx * s @ _burgers_rhs_physical(s, k, 0.0, dealias) for s in u[1:]])
            p_v = np.array([nu * dx * s @ np.real(np.fft.ifft(-(k**2) * np.fft.fft(s))) for s in u[1:]])
            g = _dx_spectral(u, N)
            tail_exact = float(np.sum(c_ref[j_star, np.abs(kk) > kc] ** 2) / np.sum(c_ref[j_star] ** 2))
            rec["by_N"][N] = {
                "resolved_kmax": kc, "points_per_front_width": width / dx,
                "exact_energy_fraction_beyond_kmax_at_steepest": tail_exact,
                "fom_coeff_at_kmax_over_max_steepest": float(c[j_star, kc] / c[j_star].max()),
                "fom_spectrum_steepest": [float(c[j_star, q]) for q in range(0, N // 2)],
                "mass_drift_max": float(np.abs(mass - mass[0]).max()),
                "energy_rel_err_max": float(np.abs(E - E_ref).max() / E_ref[0]),
                "energy_increase_max": float(max(0.0, np.diff(E).max()) / E[0]),
                "aliasing_over_dissipation_max": float(np.max(np.abs(p_nl) / np.abs(p_v))),
                "max_abs_ux_ratio": float(np.abs(g).max() / np.abs(g_ref).max()),
                "overshoot_over_max_u0": float(np.abs(u).max() / np.abs(u[0]).max() - 1.0),
                "final_rel_err": float(rel_l2_per_snapshot(u, ref)[-1])}
        out["cases"][name] = rec
        print(f"{name:22s} t*={TIMES[j_star]:.2f} width {width:.4f} | " + " ".join(
            f"{N}:ppw {rec['by_N'][N]['points_per_front_width']:.1f} tail {rec['by_N'][N]['exact_energy_fraction_beyond_kmax_at_steepest']:.0e} "
            f"os {rec['by_N'][N]['overshoot_over_max_u0']:+.0e}" for N in (128, 512, 2048)))
    out["env"] = finish_env(env)
    save("diagnostics", tag, out)


# ======================================================================================= stage: rom
R_LIST = (4, 8, 12, 16, 24, 32)


def training_ensemble(N, data=None):
    """FOM (float64, baseline step rule, aligned) at N for the 200 historical training parameters.
    Cached in data/convergence_train_N<N>.npz (gitignored, regenerated deterministically if absent)."""
    data = data or load_data()
    trA, trnu, _ = data["train"]
    path = f"data/convergence_train_N{N}.npz"
    if os.path.exists(path):
        st = np.load(path)
        if np.array_equal(st["A"], trA) and np.array_equal(st["nu"], trnu):
            return st["U"], trnu
    U = np.stack([solve_burgers_f64(sine_ic(float(a), N), float(v))[1] for a, v in zip(trA, trnu)])
    np.savez(path, A=trA, nu=trnu, U=U)
    return U, trnu


def eval_sets(data):
    """[(set, label, u0 builder N -> (N,), nu, exact solution object)] for test, extrapolation and the
    two-mode family (same cases as the earlier studies)."""
    from src.convergence import grid, two_mode_solution
    tA, tnu, _ = data["test"]
    out = [("test", f"test{i}", (lambda N, A=float(tA[i]): sine_ic(A, N)), float(tnu[i]),
            sine_solution(float(tA[i]), float(tnu[i]))) for i in range(len(tA))]
    for A, nu in ((2.5, 0.05), (1.0, 0.15), (0.2, 0.05)):
        out.append(("extrap", f"A={A},nu={nu}", (lambda N, A=A: sine_ic(A, N)), nu, sine_solution(A, nu)))
    o = np.load("data/ood_family_test.npz")
    for i in range(len(o["nu"])):
        A1, A2, nu = float(o["A1"][i]), float(o["A2"][i]), float(o["nu"][i])
        out.append(("ood", f"ood{i}", (lambda N, A1=A1, A2=A2: A1 * np.sin(grid(N)) + A2 * np.sin(2 * grid(N))),
                    nu, two_mode_solution(A1, A2, nu)))
    return out


def project(u, Phi):
    return (u @ Phi) @ Phi.T


def stage_rom(tag, n_list=(128, 256, 512, 1024, 2048), r_list=R_LIST):
    """Tensorial POD at several resolutions and ranks. For every case: errors of the ROM against the FOM
    at the same N and against the exact solution, the FOM's own error against the exact solution, and
    the POD projection errors of the FOM trajectory and of the exact solution (restricted to N)."""
    from scripts.hyperreduction_benchmark import wavenumbers
    from src.convergence import resample
    from src.pod_rom import build_pod_basis
    from src.tensorial_pod import build_tensorial_rom, tensorial_reduced_trajectory
    env = environment_info()
    data = load_data()
    sets = eval_sets(data)
    exact_cache = {lab: ch.trajectory(N_REF, TIMES) for _, lab, _, _, ch in sets}
    out = {"n_list": n_list, "r_list": r_list, "by_N": {}}
    for N in n_list:
        t0 = time.perf_counter()
        U, trnu = training_ensemble(N, data)
        t_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        Phi_all, S = build_pod_basis(U.reshape(-1, N), max(r_list))
        t_pod = time.perf_counter() - t0
        k, dealias = wavenumbers(N)
        models, t_tensor = {}, {}
        for r in r_list:
            t0 = time.perf_counter()
            models[r] = build_tensorial_rom(np.ascontiguousarray(Phi_all[:, :r]), k, dealias)
            t_tensor[r] = time.perf_counter() - t0
        rows = []
        for set_name, lab, ic, nu, _ in sets:
            u0 = ic(N)
            t0 = time.perf_counter()
            fom = solve_burgers_f64(u0, nu)[1]
            t_fom = time.perf_counter() - t0
            ex = exact_cache[lab]
            ex_N = resample(ex, N)
            row = {"set": set_name, "case": lab, "nu": nu, "fom_vs_exact": rel_l2(fom, ex),
                   "fom_s": t_fom, "by_r": {}}
            for r in r_list:
                Phi = np.ascontiguousarray(Phi_all[:, :r])
                t0 = time.perf_counter()
                with np.errstate(all="ignore"):
                    _, a, n_steps = tensorial_reduced_trajectory(Phi.T @ u0, np.abs(u0).max(), models[r], nu,
                                                                 T_FINAL, N_SAVE, align_snapshots=True)
                t_solve = time.perf_counter() - t0
                rom = a @ Phi.T
                finite = bool(np.all(np.isfinite(rom)))
                row["by_r"][r] = {
                    "stable": finite and float(np.abs(rom).max()) <= 10 * float(np.abs(u0).max()),
                    "rom_vs_fom": rel_l2(rom, fom) if finite else float("inf"),
                    "rom_vs_exact": rel_l2(rom, ex) if finite else float("inf"),
                    "proj_fom": rel_l2(project(fom, Phi), fom),
                    "proj_exact": rel_l2(project(ex_N, Phi), ex),
                    "n_steps": int(n_steps), "solve_s": t_solve}
            rows.append(row)
        summ = {}
        for set_name in ("test", "extrap", "ood"):
            sel = [x for x in rows if x["set"] == set_name]
            s_ = {"fom_vs_exact_mean": float(np.mean([x["fom_vs_exact"] for x in sel])),
                  "fom_vs_exact_max": float(np.max([x["fom_vs_exact"] for x in sel])), "by_r": {}}
            for r in r_list:
                g = lambda key: np.array([x["by_r"][r][key] for x in sel], dtype=float)
                s_["by_r"][r] = {key: {"mean": float(np.mean(g(key))), "median": float(np.median(g(key))),
                                       "max": float(np.max(g(key)))}
                                 for key in ("rom_vs_fom", "rom_vs_exact", "proj_fom", "proj_exact")}
                s_["by_r"][r]["n_unstable"] = int(sum(not x["by_r"][r]["stable"] for x in sel))
                s_["by_r"][r]["steps_mean"] = float(np.mean(g("n_steps")))
                s_["by_r"][r]["solve_ms_mean"] = 1e3 * float(np.mean(g("solve_s")))
            s_["fom_ms_mean"] = 1e3 * float(np.mean([x["fom_s"] for x in sel]))
            summ[set_name] = s_
        out["by_N"][N] = {"offline_s": {"training_fom": t_train, "pod_svd": t_pod, "tensor": t_tensor},
                          "singular_values": S[:64], "summary": summ, "rows": rows}
        st = summ["test"]
        print(f"N={N}: FOM vs exact mean {st['fom_vs_exact_mean']:.1e} | " + " ".join(
            f"r{r}: rom/fom {st['by_r'][r]['rom_vs_fom']['mean']:.1e} rom/ex {st['by_r'][r]['rom_vs_exact']['mean']:.1e}"
            for r in (8, 16, 32)) + f" | offline FOM {t_train:.0f}s SVD {t_pod:.1f}s")
    out["env"] = finish_env(env)
    save("rom", tag, out)


# ================================================================================== stage: timestep
POLICIES = ("P0_reduced", "P1_reduced_cfl", "P2_if_rk4", "P3_rk45", "P4_radau")
ADAPTIVE_TOL = {"rtol": 1e-6, "atol": 1e-9}  # fixed before the runs (plan, Section 3)


def run_policy(policy, model, Phi, u0, nu, mult=1, tol=None):
    """(a_save (20, r), info) of one reduced-integration policy; `mult` multiplies the step SIZE of the
    fixed-step policies (work-precision sweep), `tol` overrides the adaptive tolerances."""
    from src.reduced_timestep import (aligned_steps, if_rk4_dt, if_rk4_trajectory, reduced_cfl_dt,
                                      rk4_trajectory, scipy_trajectory)
    from src.tensorial_pod import reduced_step_count
    a0, mx, rho = Phi.T @ u0, float(np.abs(u0).max()), model.diffusion_radius
    with np.errstate(all="ignore"):
        if policy == "P0_reduced":
            n0 = reduced_step_count(mx, model.Nx, nu, T_FINAL, N_SAVE, align_snapshots=True,
                                    diffusion_radius=rho)[0]
            n = aligned_steps(T_FINAL / n0 * mult, T_FINAL, N_SAVE, min_steps=1)
            return rk4_trajectory(model, a0, nu, T_FINAL, N_SAVE, n)[1], {"n_steps": n, "rhs_evals": 4 * n}
        if policy == "P1_reduced_cfl":
            n = aligned_steps(reduced_cfl_dt(mx, nu, rho) * mult, T_FINAL, N_SAVE, min_steps=1)
            return rk4_trajectory(model, a0, nu, T_FINAL, N_SAVE, n)[1], {"n_steps": n, "rhs_evals": 4 * n}
        if policy == "P2_if_rk4":
            n = aligned_steps(if_rk4_dt(mx, rho) * mult, T_FINAL, N_SAVE, min_steps=1)
            return if_rk4_trajectory(model, a0, nu, T_FINAL, N_SAVE, n)[1], {"n_steps": n, "rhs_evals": 4 * n}
        method = {"P3_rk45": "RK45", "P4_radau": "Radau", "DOP853": "DOP853"}[policy]
        t = tol or ADAPTIVE_TOL
        a, info = scipy_trajectory(model, a0, nu, TIMES, method, t["rtol"], t["atol"])
        info["rhs_evals"] = info["nfev"]
        return a, info


def stage_timestep(tag, n_list=(128, 512, 2048), r_list=(8, 16), reps=5):
    from scripts.hyperreduction_benchmark import interleaved_timing, wavenumbers
    from src.pod_rom import build_pod_basis
    from src.reduced_timestep import advective_rate_bound, count_steps, rk4_trajectory
    from src.tensorial_pod import build_tensorial_rom, quadratic_jacobian, reduced_step_count
    from src.timestep import fom_advective_dt, rk4_max_stable_dt
    env = environment_info()
    data = load_data()
    out = {"policies": POLICIES, "adaptive_tol": ADAPTIVE_TOL, "by_N_r": {}}
    for N in n_list:
        U, _ = training_ensemble(N, data)
        Phi_all, _ = build_pod_basis(U.reshape(-1, N), max(r_list))
        k, dealias = wavenumbers(N)
        for r in r_list:
            Phi = np.ascontiguousarray(Phi_all[:, :r])
            model = build_tensorial_rom(Phi, k, dealias)
            rho = model.diffusion_radius
            key = f"N{N}_r{r}"
            rec = {"N": N, "r": r, "rho_L": rho, "sqrt_rho_L": float(np.sqrt(rho)), "splits": {}}
            for split in ("val", "test"):
                sA, snu, _ = data[split]
                rows = []
                for i in range(len(sA)):
                    A, nu = float(sA[i]), float(snu[i])
                    u0 = sine_ic(A, N)
                    fom = solve_burgers_f64(u0, nu)[1]
                    n0 = reduced_step_count(A, N, nu, T_FINAL, N_SAVE, align_snapshots=True, diffusion_radius=rho)[0]
                    conv = rk4_trajectory(model, Phi.T @ u0, nu, T_FINAL, N_SAVE, 16 * n0)[1]  # time-converged ROM
                    row = {"i": i, "A": A, "nu": nu, "fom_steps": default_step_count(A, N),
                           "fom_dt": fom_advective_dt(A, N), "policies": {}}
                    if split == "val":  # Part A: Jacobian analysis along the projected FOM trajectory
                        ratios, dts, re_max = [], [], []
                        for s_ in fom[1:]:
                            a = Phi.T @ s_
                            Jn = quadratic_jacobian(model.Q, 0 * model.L, a)
                            ratios.append(np.linalg.norm(Jn, 2) / advective_rate_bound(np.abs(Phi @ a).max(), rho))
                            ev = np.linalg.eigvals(Jn + nu * model.L)
                            re_max.append(float(ev.real.max()))
                            dts.append(rk4_max_stable_dt(ev, dt_max=1.0))
                        row["jacobian"] = {"norm_over_bound_max": float(max(ratios)), "max_re_eig": float(max(re_max)),
                                           "frozen_rk4_dt_min": float(min(dts))}
                    for pol in POLICIES:
                        a, info = run_policy(pol, model, Phi, u0, nu)
                        rom = a @ Phi.T
                        fin = bool(np.all(np.isfinite(rom)))
                        row["policies"][pol] = {
                            **{k_: v for k_, v in info.items() if k_ != "message"},
                            "stable": fin and float(np.abs(rom).max()) <= 10 * float(np.abs(u0).max()),
                            "time_err": float(np.linalg.norm(a - conv) / np.linalg.norm(conv)) if fin else float("inf"),
                            "rom_vs_fom": rel_l2(rom, fom) if fin else float("inf")}
                    for meth in ("RK45",):
                        row["policies"]["P3_rk45"]["steps"] = count_steps(model, Phi.T @ u0, nu, T_FINAL, meth,
                                                                          **ADAPTIVE_TOL)
                    if split == "test" or i % 4 == 0:  # runtime: all test cases, every 4th val case
                        t = interleaved_timing({p: (lambda p=p: run_policy(p, model, Phi, u0, nu)) for p in POLICIES},
                                               reps=reps, warmup=1)
                        for p in POLICIES:
                            row["policies"][p]["solve_ms"] = 1e3 * float(np.median(t[p]))
                    rows.append(row)
                # pre-registered acceptance: stable everywhere and e_fom <= 1.05 e_fom(P0) + 1e-6 per case
                acc = {}
                for pol in POLICIES:
                    ok = [x["policies"][pol]["stable"] and x["policies"][pol]["rom_vs_fom"] <=
                          1.05 * x["policies"]["P0_reduced"]["rom_vs_fom"] + 1e-6 for x in rows]
                    g = lambda key: [x["policies"][pol][key] for x in rows if key in x["policies"][pol]]
                    acc[pol] = {"accepted": bool(all(ok)), "n_fail": int(len(ok) - sum(ok)),
                                "steps_mean": float(np.mean(g("n_steps"))) if g("n_steps") else None,
                                "rhs_evals_mean": float(np.mean(g("rhs_evals"))),
                                "time_err_max": float(np.max(g("time_err"))),
                                "rom_vs_fom_mean": float(np.mean(g("rom_vs_fom"))),
                                "solve_ms_mean": float(np.mean(g("solve_ms"))) if g("solve_ms") else None}
                acc["fom_steps_mean"] = float(np.mean([x["fom_steps"] for x in rows]))
                rec["splits"][split] = {"summary": acc, "rows": rows}
                print(f"{key} {split}: fom steps {acc['fom_steps_mean']:.0f} | " + " | ".join(
                    f"{p.split('_')[0]} {'OK' if acc[p]['accepted'] else 'FAIL'} ev {acc[p]['rhs_evals_mean']:.0f} "
                    f"terr {acc[p]['time_err_max']:.0e} ms {acc[p]['solve_ms_mean'] or float('nan'):.2f}" for p in POLICIES))
            vr = rec["splits"]["val"]["rows"]
            rec["jacobian_summary_val"] = {k_: (float(max(x["jacobian"][k_] for x in vr)) if k_ != "frozen_rk4_dt_min"
                                                else float(min(x["jacobian"][k_] for x in vr)))
                                           for k_ in ("norm_over_bound_max", "max_re_eig", "frozen_rk4_dt_min")}
            # Part C: work-precision on the validation cases (time error vs RHS evaluations / runtime)
            wp = {}
            for pol, mults in (("P0_reduced", (1, 2, 4, 8)), ("P1_reduced_cfl", (0.5, 1, 2, 4)),
                               ("P2_if_rk4", (0.5, 1, 2, 4))):
                for m in mults:
                    errs, evals = [], []
                    for x in vr[::4]:
                        u0 = sine_ic(x["A"], N)
                        n0 = reduced_step_count(x["A"], N, x["nu"], T_FINAL, N_SAVE, align_snapshots=True,
                                                diffusion_radius=rho)[0]
                        conv = rk4_trajectory(model, Phi.T @ u0, x["nu"], T_FINAL, N_SAVE, 16 * n0)[1]
                        a, info = run_policy(pol, model, Phi, u0, x["nu"], mult=m)
                        errs.append(float(np.linalg.norm(a - conv) / np.linalg.norm(conv)) if np.all(np.isfinite(a)) else float("inf"))
                        evals.append(info["rhs_evals"])
                    wp[f"{pol}_x{m}"] = {"time_err_max": float(max(errs)), "rhs_evals_mean": float(np.mean(evals))}
            for pol in ("P3_rk45", "DOP853"):
                for rt in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
                    errs, evals = [], []
                    for x in vr[::4]:
                        u0 = sine_ic(x["A"], N)
                        n0 = reduced_step_count(x["A"], N, x["nu"], T_FINAL, N_SAVE, align_snapshots=True,
                                                diffusion_radius=rho)[0]
                        conv = rk4_trajectory(model, Phi.T @ u0, x["nu"], T_FINAL, N_SAVE, 16 * n0)[1]
                        a, info = run_policy(pol, model, Phi, u0, x["nu"], tol={"rtol": rt, "atol": rt * 1e-3})
                        errs.append(float(np.linalg.norm(a - conv) / np.linalg.norm(conv)))
                        evals.append(info["rhs_evals"])
                    wp[f"{pol}_rtol{rt:g}"] = {"time_err_max": float(max(errs)), "rhs_evals_mean": float(np.mean(evals))}
            rec["work_precision_val"] = wp
            out["by_N_r"][key] = rec
            print(f"{key}: rho(L) {rho:.0f} | Jacobian norm/bound max {rec['jacobian_summary_val']['norm_over_bound_max']:.2f}"
                  f" max Re eig {rec['jacobian_summary_val']['max_re_eig']:+.2f} frozen RK4 dt min "
                  f"{rec['jacobian_summary_val']['frozen_rk4_dt_min']:.2e}")
    out["env"] = finish_env(env)
    save("timestep", tag, out)


STAGES = {"spatial": stage_spatial, "temporal": stage_temporal, "reference": stage_reference,
          "diagnostics": stage_diagnostics, "rom": stage_rom, "timestep": stage_timestep}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    STAGES[args.stage](args.tag)
