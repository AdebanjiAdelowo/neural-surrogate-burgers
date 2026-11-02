"""Tensorial POD-Galerkin and spectral-consistent DEIM study (report/tensorial_pod_spectral_deim_study.md).

Every stage writes report/tensorial_deim/<stage>[_<tag>].json and touches nothing else; the historical
audit files under report/audit/ and report/research/ are read, never rewritten.

Protocol (the time-aligned protocol of report/audit/followup_production_validation_run1.json):
  * stored basis experiments/deim_basis.npz (r = 8 POD modes and the m = 16 DEIM/local-FD operator,
    built from the 200 training trajectories); other ranks / DEIM sizes are rebuilt from data/train.npz
    with the same recipe (scripts/hyperreduction_benchmark.build_bases);
  * initial conditions u0 = data/<split>.npz[u][i, 0] (float32, as stored);
  * reference = src.solver.solve_burgers(..., align_snapshots=True), saved at exactly j/19;
  * reduced models use align_snapshots=True and the default "reduced" step-size policy;
  * error = src.metrics.relative_l2 over all 20 snapshots x Nx points.
Historical index-wise numbers (the README Results table) are a different protocol and are only
reproduced in `baseline`, never mixed with the aligned ones.

Stages:
  baseline     environment, verified-baseline reproduction (aligned + historical protocols)
  equivalence  tensorial vs original POD-Galerkin: RHS and trajectories, several ranks

Run:  python scripts/tensorial_deim_study.py <stage> [--tag TAG]
"""
import argparse
import json
import os
import sys
import time
import timeit

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_dataset import N_SAVE, NX, T_FINAL
from scripts.hyperreduction_benchmark import (build_bases, environment_info, err_stats, finish_env,
                                               interleaved_timing, load_data, stats, wavenumbers)
from src.deim import deim_rom_predict
from src.metrics import relative_l2
from src.pod_rom import _burgers_rhs_physical, rom_predict, rom_reduced_trajectory
from src.solver import solve_burgers
from src.deim import build_deim_projector, collect_nonlinear_snapshots
from src.spectral_deim import (build_basis, build_spectral_deim, deim_rhs_fn,
                               interpolation_conditioning, interpolation_pinv, nonlinear_snapshots,
                               sampled_diffusion, sampled_quadratic_tensor, select_points,
                               spectral_deim_predict, spectral_deim_rhs)
from src.tensorial_pod import (build_tensorial_rom, quadratic_rhs, tensorial_predict,
                               tensorial_reduced_trajectory, tensorial_rhs)

OUT_DIR = "report/tensorial_deim"
L = 2 * np.pi


# ------------------------------------------------------------------------------------- helpers
def save(name, tag, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}{'_' + tag if tag else ''}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist")
                  else str(o))
    print(f"saved {path}")


def full_err_stats(e):
    """err_stats plus the quantiles and thresholds this study reports."""
    out = err_stats(e)
    e = np.asarray(e, dtype=float)
    good = e[np.isfinite(e)]
    out.update({"p95": float(np.percentile(good, 95)) if good.size else None,
                "n_above_5pct": int((good > 0.05).sum())})
    return out


BLOWUP = 10.0  # unstable: non-finite, or max|u| > BLOWUP * max|u0| (as scripts/hyperreduction_followup.py)


def physical_diagnostics(u, ref, Nx=NX):
    """Mass, energy and gradient diagnostics of a predicted (n_save, Nx) trajectory against its
    reference. Energy E = 0.5 sum u^2 dx; the true solution's E is non-increasing, so any increase
    between consecutive saved snapshots is spurious (reported relative to E(0))."""
    u = np.asarray(u, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    dx = L / Nx
    finite = bool(np.all(np.isfinite(u)))
    if not finite:
        return {"finite": False, "stable": False, "rel_l2": float("inf")}
    k, _ = wavenumbers(Nx)
    grad = lambda v: np.real(np.fft.ifft(1j * k * np.fft.fft(v, axis=1), axis=1))
    E, Er = 0.5 * dx * np.sum(u**2, axis=1), 0.5 * dx * np.sum(ref**2, axis=1)
    mass, mass_r = dx * u.sum(axis=1), dx * ref.sum(axis=1)
    gmax, gmax_r = np.abs(grad(u)).max(axis=1), np.abs(grad(ref)).max(axis=1)
    mx = float(np.abs(u).max())
    return {"finite": True, "stable": bool(mx <= BLOWUP * np.abs(ref[0]).max()),
            "rel_l2": relative_l2(u.astype(np.float32), ref.astype(np.float32)),
            "max_abs_over_u0max": mx / float(np.abs(ref[0]).max()),
            "mass_drift_max": float(np.abs(mass - mass[0]).max()),
            "mass_err_vs_ref_max": float(np.abs(mass - mass_r).max()),
            "energy_rel_err_max": float(np.abs(E - Er).max() / Er[0]),
            "energy_spurious_growth": float(max(0.0, np.max(np.diff(E))) / E[0]),
            "energy_final_ratio": float(E[-1] / Er[-1]),
            "max_grad_ratio_final": float(gmax[-1] / gmax_r[-1]),
            "max_grad_ref_final": float(gmax_r[-1])}


def summarize(rows, key="rel_l2"):
    e = np.array([r_[key] for r_ in rows], dtype=float)
    s_ = full_err_stats(e)
    s_["n_unstable"] = int(sum(not r_["stable"] for r_ in rows))
    s_["energy_spurious_growth_max"] = float(max(r_.get("energy_spurious_growth", np.inf) for r_ in rows))
    s_["n_with_spurious_energy_growth_gt_1e-6"] = int(sum(r_.get("energy_spurious_growth", np.inf) > 1e-6
                                                          for r_ in rows))
    return s_


def stored_basis():
    st = np.load("experiments/deim_basis.npz")
    return {k_: st[k_] for k_ in st.files}


def aligned_ref(A, nu, Nx=NX):
    return solve_burgers(A=A, nu=nu, Nx=Nx, T=T_FINAL, n_save=N_SAVE, align_snapshots=True)[2]


def aligned_refs(split, data, Nx=NX):
    sA, snu, _ = data[split]
    return [aligned_ref(float(sA[i]), float(snu[i]), Nx) for i in range(len(sA))]


EXTRAP_CASES = ((2.5, 0.05), (1.0, 0.15), (0.2, 0.05))  # scripts/evaluate_comparison.py


def case_set(name, data):
    """[(label, u0 float32, nu, aligned reference float32 (n_save, Nx))] for one evaluation set.

    test / val     the stored splits (u0 = stored IC);
    extrap         the three historical parameter-extrapolation cases (u0 = the aligned FOM's IC);
    ood            the unseen two-mode IC family A1 sin x + A2 sin 2x of data/ood_family_test.npz,
                   reference from src.general_ic_solver with align_snapshots=True."""
    if name in ("test", "val"):
        sA, snu, sU = data[name]
        return [(f"{name}{i}", sU[i, 0], float(snu[i]), aligned_ref(float(sA[i]), float(snu[i])))
                for i in range(len(sA))]
    if name == "extrap":
        out = []
        for A, nu in EXTRAP_CASES:
            ref = aligned_ref(A, nu)
            out.append((f"A={A},nu={nu}", ref[0], nu, ref))
        return out
    if name == "ood":
        from src.general_ic_solver import solve_burgers_general_ic
        o = np.load("data/ood_family_test.npz")
        out = []
        for i in range(len(o["nu"])):
            u0, nu = o["u"][i, 0], float(o["nu"][i])
            ref = solve_burgers_general_ic(u0.astype(np.float64), nu, T=T_FINAL, n_save=N_SAVE,
                                           align_snapshots=True)[2]
            out.append((f"ood{i}", u0, nu, ref))
        return out
    raise ValueError(name)


def case_params(name, data):
    """(A, nu) per case for in-family sets (None for the two-mode family)."""
    if name in ("test", "val"):
        return [(float(a), float(n)) for a, n in zip(data[name][0], data[name][1])]
    if name == "extrap":
        return list(EXTRAP_CASES)
    return None


def pod_bases(r_list, Nx=NX, data=None):
    """{r: Phi_r} rebuilt from data/train.npz (r = 8 is replaced by the stored basis, identical to
    5e-14 up to sign, so that every r = 8 number uses the exact stored operator)."""
    data = data or load_data()
    _, tnu, tU = data["train"]
    Pf, S, _, _, _ = build_bases(tU, tnu, Nx, r_max=max(r_list), m_list=())
    out = {r: np.ascontiguousarray(Pf[:, :r]) for r in r_list}
    if 8 in out:
        out[8] = stored_basis()["Phi"]
    return out, S


# ================================================================================ stage: baseline
def stage_baseline(tag):
    """Reproduce the verified baseline numbers in this environment before anything new is run."""
    env = environment_info()
    data = load_data()
    st = stored_basis()
    Phi, M, pts = st["Phi"], st["M"], st["points"]
    k, dealias = wavenumbers(NX)
    out = {"protocol_aligned": "u0 = stored float32 IC; reference solve_burgers(align_snapshots=True); "
                               "models align_snapshots=True, default timestep_policy='reduced'",
           "protocol_historical": "scripts/evaluate_comparison.py: reference = stored data/test.npz "
                                  "snapshots, models with timestep_policy='historical', no alignment",
           "recorded": {}, "reproduced": {}}
    rec = json.load(open("report/audit/followup_production_validation_run1.json"))
    for split in ("val", "test"):
        for kind in ("rom", "deim"):
            out["recorded"][f"aligned_{split}_{kind}"] = rec["splits"][split]["summary"][kind]["default_err"]
    for split in ("val", "test"):
        sA, snu, sU = data[split]
        refs = aligned_refs(split, data)
        e = {"rom": [], "deim_localfd": []}
        for i in range(len(sA)):
            u0, nu = sU[i, 0], float(snu[i])
            with np.errstate(all="ignore"):
                e["rom"].append(relative_l2(rom_predict(u0, Phi, k, nu, dealias, T=T_FINAL, n_save=N_SAVE,
                                                        align_snapshots=True)[1], refs[i]))
                e["deim_localfd"].append(relative_l2(deim_rom_predict(u0, Phi, M, pts, NX, nu=nu, T=T_FINAL,
                                                                      n_save=N_SAVE, align_snapshots=True)[1],
                                                     refs[i]))
        for kname, v in e.items():
            out["reproduced"][f"aligned_{split}_{kname}"] = full_err_stats(v)
            out["reproduced"][f"aligned_{split}_{kname}_per_case"] = v
    # historical index-wise protocol (README Results table: POD-ROM 0.152% +- 0.077%)
    sA, snu, sU = data["test"]
    hist = [relative_l2(rom_predict(sU[i, 0], Phi, k, float(snu[i]), dealias, T=T_FINAL, n_save=N_SAVE,
                                    timestep_policy="historical")[1], sU[i]) for i in range(len(sA))]
    out["reproduced"]["historical_test_rom"] = full_err_stats(hist)
    for key in ("aligned_val_rom", "aligned_val_deim_localfd", "aligned_test_rom", "aligned_test_deim_localfd",
                "historical_test_rom"):
        s_ = out["reproduced"][key]
        print(f"{key:28s} mean {100*s_['mean']:.4f}%  median {100*s_['median']:.4f}%  max {100*s_['max']:.3f}%")
    out["env"] = finish_env(env)
    save("baseline", tag, out)


# ============================================================================= stage: equivalence
def stage_equivalence(tag, r_list=(4, 6, 8, 12, 16)):
    """RHS equivalence on random and on data-derived reduced states, and trajectory equivalence on
    all 40 test cases, tensorial vs src.pod_rom, for several ranks."""
    env = environment_info()
    data = load_data()
    k, dealias = wavenumbers(NX)
    bases, _ = pod_bases(r_list, data=data)
    sA, snu, sU = data["test"]
    out = {"ranks": {}}
    for r in r_list:
        Phi = bases[r]
        model = build_tensorial_rom(Phi, k, dealias)
        rng = np.random.default_rng(1000 + r)
        rhs_abs, rhs_rel = [], []
        # (a) random states over a wide amplitude range and viscosities incl. outside training
        states = [(rng.standard_normal(r) * rng.uniform(0.1, 10.0), rng.uniform(1e-3, 0.3))
                  for _ in range(200)]
        # (b) projections of reference snapshots (the states the ROM actually visits)
        states += [((Phi.T @ sU[i, j]).astype(np.float64), float(snu[i]))
                   for i in range(len(sA)) for j in (0, 5, 10, 19)]
        for a, nu in states:
            fo = Phi.T @ _burgers_rhs_physical(Phi @ a, k, nu, dealias)
            ft = tensorial_rhs(model, a, nu)
            rhs_abs.append(float(np.linalg.norm(fo - ft)))
            rhs_rel.append(float(np.linalg.norm(fo - ft) / np.linalg.norm(fo)))
        traj, traj_f32, ctrl, steps_equal = [], [], [], True
        eps = np.finfo(float).eps
        for i in range(len(sA)):
            u0, nu = sU[i, 0], float(snu[i])
            a0_32 = Phi.T @ u0
            a0 = a0_32.astype(np.float64)
            mx = np.abs(u0).max()
            _, ao, no = rom_reduced_trajectory(a0, mx, Phi, k, nu, dealias, T_FINAL, N_SAVE, align_snapshots=True)
            _, at, nt = tensorial_reduced_trajectory(a0, mx, model, nu, T_FINAL, N_SAVE, align_snapshots=True)
            steps_equal &= (no == nt)
            s_ = np.abs(ao).max()
            traj.append(float(np.abs(ao - at).max() / s_))
            # original with the stored float32 a0 (the production path), see tests/test_tensorial_pod.py
            _, ao32, _ = rom_reduced_trajectory(a0_32, mx, Phi, k, nu, dealias, T_FINAL, N_SAVE,
                                                align_snapshots=True)
            traj_f32.append(float(np.abs(ao32 - at).max() / s_))
            # control: the original against itself with a0 perturbed at machine epsilon
            d = np.random.default_rng(i).standard_normal(r)
            d *= eps * np.linalg.norm(a0) / np.linalg.norm(d)
            _, ap, _ = rom_reduced_trajectory(a0 + d, mx, Phi, k, nu, dealias, T_FINAL, N_SAVE,
                                              align_snapshots=True)
            ctrl.append(float(np.abs(ao - ap).max() / s_))
        Q = model.Q
        out["ranks"][str(r)] = {
            "n_rhs_states": len(states), "rhs_abs_diff_max": max(rhs_abs), "rhs_rel_diff_max": max(rhs_rel),
            "rhs_rel_diff_median": float(np.median(rhs_rel)),
            "Q_jk_asymmetry_max": float(np.abs(Q - Q.transpose(0, 2, 1)).max()),
            "traj_rel_diff_max_float64_a0": max(traj), "traj_rel_diff_median_float64_a0": float(np.median(traj)),
            "traj_rel_diff_max_original_float32_a0": max(traj_f32),
            "control_eps_perturbation_rel_diff_max": max(ctrl), "step_counts_identical": bool(steps_equal)}
        o = out["ranks"][str(r)]
        print(f"r={r:2d} rhs rel max {o['rhs_rel_diff_max']:.1e} | traj (f64 a0) {o['traj_rel_diff_max_float64_a0']:.1e}"
              f" | control {o['control_eps_perturbation_rel_diff_max']:.1e} | vs f32-a0 original "
              f"{o['traj_rel_diff_max_original_float32_a0']:.1e} | steps equal {steps_equal}")
    out["env"] = finish_env(env)
    save("equivalence", tag, out)


# ========================================================================= stage: tensorial_bench
def stage_tensorial_bench(tag, reps=7, r_list=(4, 6, 8, 12, 16, 24)):
    """Tensorial POD vs the original POD-Galerkin ROM and the FOM: accuracy on every evaluation set,
    interleaved runtime on the 40 test cases (solver-only and end-to-end), offline cost, and the
    rank dependence of error and runtime."""
    env = environment_info()
    data = load_data()
    k, dealias = wavenumbers(NX)
    st = stored_basis()
    Phi = st["Phi"]
    t0 = time.perf_counter()
    model = build_tensorial_rom(Phi, k, dealias)
    offline_tensor_s = time.perf_counter() - t0
    out = {"r": 8, "offline_tensor_build_s": offline_tensor_s,
           "tensor_bytes": int(model.Q.nbytes + model.L.nbytes), "accuracy": {}, "timing": {}}

    def rom(u0, nu):
        return rom_predict(u0, Phi, k, nu, dealias, T=T_FINAL, n_save=N_SAVE, align_snapshots=True)[1]

    def ten(u0, nu, mdl=model, P=Phi):
        return tensorial_predict(u0, P, mdl, nu, T=T_FINAL, n_save=N_SAVE, align_snapshots=True)[1]

    # ---- accuracy (identical protocol for both; FOM is the reference)
    for set_name in ("test", "val", "extrap", "ood"):
        rows = {"rom": [], "tensorial": []}
        diff = []
        for label, u0, nu, ref in case_set(set_name, data):
            with np.errstate(all="ignore"):
                uo, ut = rom(u0, nu), ten(u0, nu)
            rows["rom"].append({"case": label, "nu": nu, **physical_diagnostics(uo, ref)})
            rows["tensorial"].append({"case": label, "nu": nu, **physical_diagnostics(ut, ref)})
            diff.append(float(np.abs(uo.astype(np.float64) - ut).max() / np.abs(uo).max()))
        out["accuracy"][set_name] = {m_: {"summary": summarize(v), "rows": v} for m_, v in rows.items()}
        out["accuracy"][set_name]["max_rel_field_diff_rom_vs_tensorial"] = max(diff)
        print(f"{set_name:6s} ROM {100*out['accuracy'][set_name]['rom']['summary']['mean']:.4f}%  "
              f"tensorial {100*out['accuracy'][set_name]['tensorial']['summary']['mean']:.4f}%  "
              f"max field diff {max(diff):.1e}")

    # ---- runtime: all 40 test cases, interleaved, same process
    sA, snu, sU = data["test"]
    per_case = []
    for i in range(len(sA)):
        A, nu, u0 = float(sA[i]), float(snu[i]), sU[i, 0]
        a0, mx = Phi.T @ u0, np.abs(u0).max()
        calls = {
            "fom": lambda: solve_burgers(A=A, nu=nu, Nx=NX, T=T_FINAL, n_save=N_SAVE, align_snapshots=True),
            "rom_solver_only": lambda: rom_reduced_trajectory(a0, mx, Phi, k, nu, dealias, T_FINAL, N_SAVE,
                                                              align_snapshots=True),
            "rom_end_to_end": lambda: rom(u0, nu),
            "tensorial_solver_only": lambda: tensorial_reduced_trajectory(a0, mx, model, nu, T_FINAL, N_SAVE,
                                                                          align_snapshots=True),
            "tensorial_end_to_end": lambda: ten(u0, nu)}
        t = interleaved_timing(calls, reps=reps, warmup=2)
        steps = {"rom": rom_reduced_trajectory(a0, mx, Phi, k, nu, dealias, T_FINAL, N_SAVE,
                                               align_snapshots=True)[2],
                 "tensorial": tensorial_reduced_trajectory(a0, mx, model, nu, T_FINAL, N_SAVE,
                                                           align_snapshots=True)[2]}
        per_case.append({"i": i, "A": A, "nu": nu, "median_s": {n: float(np.median(v)) for n, v in t.items()},
                         "iqr_s": {n: float(np.subtract(*np.percentile(v, [75, 25]))) for n, v in t.items()},
                         "n_steps": steps})
    names = list(per_case[0]["median_s"])
    work = {n: float(np.sum([c["median_s"][n] for c in per_case])) for n in names}
    out["timing"] = {
        "reps": reps, "per_case": per_case,
        "workload_total_s": work,
        "workload_mean_ms": {n: 1e3 * work[n] / len(per_case) for n in names},
        "speedup_vs_fom_workload": {n: work["fom"] / work[n] for n in names if n != "fom"},
        "speedup_vs_rom_end_to_end_workload": {n: work["rom_end_to_end"] / work[n] for n in names},
        "per_case_speedup_vs_fom": {n: stats([c["median_s"]["fom"] / c["median_s"][n] for c in per_case])
                                    for n in names if n != "fom"}}
    print("workload mean ms", {n: round(v, 3) for n, v in out["timing"]["workload_mean_ms"].items()})
    print("speed-up vs FOM", {n: round(v, 2) for n, v in out["timing"]["speedup_vs_fom_workload"].items()})

    # ---- online RHS micro-benchmark (one evaluation, r = 8)
    a = Phi.T @ sU[0, 5]
    nu = float(snu[0])
    rhs_t = quadratic_rhs(model.Q_flat, nu * model.L)
    out["rhs_eval_us"] = {
        "original_full_grid_fft": _med_us(lambda: Phi.T @ _burgers_rhs_physical(Phi @ a, k, nu, dealias)),
        "tensorial": _med_us(lambda: rhs_t(a))}
    print("RHS us", out["rhs_eval_us"])

    # ---- rank dependence (test split, aligned): error and end-to-end runtime
    bases, _ = pod_bases(r_list, data=data)
    cases = case_set("test", data)
    out["rank_sweep"] = {}
    for r in r_list:
        P = bases[r]
        t0 = time.perf_counter()
        mdl = build_tensorial_rom(P, k, dealias)
        build_s = time.perf_counter() - t0
        errs_t, errs_o, tt, to = [], [], [], []
        for label, u0, nu, ref in cases:
            with np.errstate(all="ignore"):
                errs_t.append(relative_l2(ten(u0, nu, mdl, P), ref))
                errs_o.append(relative_l2(rom_predict(u0, P, k, nu, dealias, T=T_FINAL, n_save=N_SAVE,
                                                      align_snapshots=True)[1], ref))
        for label, u0, nu, ref in cases[::4]:  # runtime on every 4th case (10 cases), interleaved
            tm = interleaved_timing({"t": lambda: ten(u0, nu, mdl, P),
                                     "o": lambda: rom_predict(u0, P, k, nu, dealias, T=T_FINAL,
                                                              n_save=N_SAVE, align_snapshots=True)},
                                    reps=5, warmup=1)
            tt.append(np.median(tm["t"]))
            to.append(np.median(tm["o"]))
        a = P.T @ sU[0, 5]
        rhs_r = quadratic_rhs(mdl.Q_flat, float(snu[0]) * mdl.L)
        out["rank_sweep"][str(r)] = {
            "tensorial_err": full_err_stats(errs_t), "rom_err": full_err_stats(errs_o),
            "tensorial_end_to_end_ms_mean_10cases": 1e3 * float(np.mean(tt)),
            "rom_end_to_end_ms_mean_10cases": 1e3 * float(np.mean(to)),
            "tensorial_rhs_us": _med_us(lambda: rhs_r(a)),
            "offline_tensor_build_s": build_s, "tensor_bytes": int(mdl.Q.nbytes)}
        o = out["rank_sweep"][str(r)]
        print(f"r={r:2d} err tens {100*o['tensorial_err']['mean']:.4f}% rom {100*o['rom_err']['mean']:.4f}% | "
              f"ms tens {o['tensorial_end_to_end_ms_mean_10cases']:.2f} rom {o['rom_end_to_end_ms_mean_10cases']:.2f}"
              f" | rhs {o['tensorial_rhs_us']:.2f} us")
    out["env"] = finish_env(env)
    save("tensorial_bench", tag, out)


# ===================================================================== stage: spectral_consistency
def deim_bases(Nx=NX, data=None, m_max=48):
    """Nonlinear-only (N) and full-RHS (f) snapshot bases from the training split, m_max columns."""
    data = data or load_data()
    _, tnu, tU = data["train"]
    k, dealias = wavenumbers(Nx)
    ens = tU.reshape(-1, Nx)
    nu_snap = np.repeat(tnu, tU.shape[1])
    Psi_n, S_n = build_basis(nonlinear_snapshots(ens, k, dealias), m_max)
    Psi_f, S_f = build_basis(collect_nonlinear_snapshots(ens, k, nu_snap, dealias), m_max)
    return {"nonlinear": (Psi_n, S_n), "full": (Psi_f, S_f)}


def stage_spectral_consistency(tag):
    """(1) the sampled tensor reproduces P^T N(Phi a) and P^T f(Phi a) of the full-grid spectral
    operator; (2) cross-check: target "full" with the STORED basis, points and step size reproduces
    the historical full-grid exact-F DEIM diagnostic (report/audit/followup_stencil_run2.json)."""
    env = environment_info()
    data = load_data()
    k, dealias = wavenumbers(NX)
    bases, _ = pod_bases((4, 8, 12, 16), data=data)
    db = deim_bases(data=data)
    sA, snu, sU = data["test"]
    out = {"sampled_operator": [], "historical_crosscheck": {}}
    rng = np.random.default_rng(7)
    for r, Phi in bases.items():
        states = [(Phi.T @ sU[i, j]).astype(np.float64) for i in range(len(sA)) for j in range(0, 20, 3)]
        states += [rng.standard_normal(r) * rng.uniform(0.1, 10) for _ in range(100)]
        for m in (8, 16, 24):
            Psi = db["nonlinear"][0][:, :m]
            for method, s in (("deim", m), ("qdeim", m), ("qdeim+greedy", 2 * m)):
                pts = select_points(Psi, s=s, method=method)
                C, Ds = sampled_quadratic_tensor(Phi, pts, k, dealias), sampled_diffusion(Phi, pts, k)
                en, ef = [], []
                for a in states:
                    nu = float(rng.uniform(0.005, 0.2))
                    u = Phi @ a
                    ref_n = _burgers_rhs_physical(u, k, 0.0, dealias)[pts]
                    ref_f = _burgers_rhs_physical(u, k, nu, dealias)[pts]
                    cn = (C.reshape(-1, r) @ a).reshape(len(pts), r) @ a
                    en.append(np.abs(cn - ref_n).max() / np.abs(ref_n).max())
                    ef.append(np.abs(cn + nu * Ds @ a - ref_f).max() / np.abs(ref_f).max())
                out["sampled_operator"].append({"r": r, "m": m, "selection": method, "s": s,
                                                "n_states": len(states),
                                                "nonlinear_rel_err_max": float(max(en)),
                                                "full_rhs_rel_err_max": float(max(ef))})
    worst_n = max(o["nonlinear_rel_err_max"] for o in out["sampled_operator"])
    worst_f = max(o["full_rhs_rel_err_max"] for o in out["sampled_operator"])
    print(f"sampled operator: max rel err nonlinear {worst_n:.2e}, full RHS {worst_f:.2e} "
          f"over {len(out['sampled_operator'])} (r, m, selection) configurations")

    # ---- historical cross-check (same Psi, points, dt as the exact-F diagnostic)
    st = stored_basis()
    Phi, Psi, pts = st["Phi"], st["Psi"], st["points"]
    model = build_spectral_deim(Phi, Psi, pts, k, dealias, target="full")
    from scripts.hyperreduction_followup import dt_policy, snap_dt, stored_ctx
    Cx = stored_ctx()
    hist = json.load(open("report/audit/followup_stencil_run2.json"))["nx128_test40_policyR"]["per_case"]["exactF_deim"]
    mine = []
    for i in range(len(sA)):
        u0, nu = sU[i, 0], float(snu[i])
        dt = snap_dt(dt_policy("R", "deim", np.abs(u0).max(), nu, NX, Cx))
        with np.errstate(all="ignore"):
            u = spectral_deim_predict(u0, Phi, model, nu, T=T_FINAL, n_save=N_SAVE, dt=dt,
                                      align_snapshots=True)[1]
        mine.append(relative_l2(u, aligned_ref(float(sA[i]), nu)))
    d = np.abs(np.array(mine) - np.array(hist))
    out["historical_crosscheck"] = {
        "description": "spectral DEIM target='full', stored Psi/points (m=16), dt of the historical run, "
                       "vs report/audit/followup_stencil_run2.json exactF_deim (full-grid FFT then sampled)",
        "historical_mean": float(np.mean(hist)), "spectral_deim_full_mean": float(np.mean(mine)),
        "max_abs_diff_per_case": float(d.max()), "max_rel_diff_per_case": float((d / np.array(hist)).max()),
        "per_case_spectral_deim_full": mine}
    print(f"cross-check: historical exact-F mean {100*np.mean(hist):.5f}% vs sampled-tensor "
          f"{100*np.mean(mine):.5f}%, max per-case abs diff {d.max():.1e}")
    out["env"] = finish_env(env)
    save("spectral_consistency", tag, out)


# =============================================================================== stage: deim_sweep
M_LIST = (8, 12, 16, 24)


def deim_configs(m_list=M_LIST, targets=("nonlinear", "full")):
    """Every (target, m, s, selection) of the sweep: s = m with DEIM and Q-DEIM; s = ceil(1.5 m) and
    2 m with the two oversampling rules."""
    out = []
    for target in targets:
        for m in m_list:
            for sel in ("deim", "qdeim"):
                out.append((target, m, m, sel))
            for s in (int(np.ceil(1.5 * m)), 2 * m):
                for sel in ("qdeim+greedy", "qdeim_extended"):
                    out.append((target, m, s, sel))
    return out


def config_name(target, m, s, sel):
    return f"{target}|m={m}|s={s}|{sel}"


def build_config(Phi, db, target, m, s, sel, k, dealias):
    Psi_all = db[target][0]
    Psi = Psi_all[:, :m]
    pts = select_points(Psi, s=s, method=sel, Psi_extended=Psi_all)
    return build_spectral_deim(Phi, Psi, pts, k, dealias, target=target), Psi


def hybrid_models(Phi, data, m_list=M_LIST):
    """{m: (M, points, Psi)} of src/deim.py's DEIM/local-FD model (full-RHS basis, greedy DEIM
    points) by the historical recipe (scripts/hyperreduction_benchmark.build_bases); m = 16 is the
    stored operator, so the canonical configuration is bit-identical to the historical one."""
    _, tnu, tU = data["train"]
    _, _, dm, _, _ = build_bases(tU, tnu, NX, r_max=8, m_list=m_list)
    out = {m: (build_deim_projector(Phi, dm[m]["Psi"], dm[m]["points"]), dm[m]["points"], dm[m]["Psi"])
           for m in m_list}
    if 16 in out:
        st = stored_basis()
        out[16] = (st["M"], st["points"], st["Psi"])
    return out


def run_spectral(model, Phi, u0, nu):
    with np.errstate(all="ignore"):
        return spectral_deim_predict(u0, Phi, model, nu, T=T_FINAL, n_save=N_SAVE, align_snapshots=True)[1]


def run_hybrid(M, pts, Phi, u0, nu, stencil_order=2):
    with np.errstate(all="ignore"):
        return deim_rom_predict(u0, Phi, M, pts, NX, nu=nu, T=T_FINAL, n_save=N_SAVE,
                                stencil_order=stencil_order, align_snapshots=True)[1]


def stage_deim_sweep(tag):
    """All spectral-DEIM configurations at r = 8 (stored POD basis): conditioning, approximation
    errors at the states the ROM visits, trajectory accuracy on val (selection) and test (report),
    stability, energy, and RHS / end-to-end cost; plus the hybrid DEIM/local-FD at each m."""
    env = environment_info()
    data = load_data()
    k, dealias = wavenumbers(NX)
    Phi = stored_basis()["Phi"]
    tens = build_tensorial_rom(Phi, k, dealias)
    db = deim_bases(data=data)
    sets = {s_: case_set(s_, data) for s_ in ("val", "test")}
    # states visited: projections of the aligned reference snapshots of the test cases
    states = [((Phi.T @ ref[j]).astype(np.float64), nu, ref[j].astype(np.float64))
              for _, _, nu, ref in sets["test"] for j in range(N_SAVE)]
    out = {"r": 8, "configs": {}, "hybrid_localfd": {}}
    for cfg in deim_configs():
        target, m, s, sel = cfg
        t0 = time.perf_counter()
        model, Psi = build_config(Phi, db, target, m, s, sel, k, dealias)
        offline_s = time.perf_counter() - t0
        interp = Psi @ interpolation_pinv(Psi, model.points)  # (Nx, s) full-field reconstruction
        nl_err, rhs_err = [], []
        for a, nu, uref in states:
            # (i) DEIM interpolation error of the TRUE nonlinear quantity on the reference field
            q = (_burgers_rhs_physical(uref, k, 0.0, dealias) if target == "nonlinear"
                 else _burgers_rhs_physical(uref, k, nu, dealias))
            nl_err.append(np.linalg.norm(interp @ q[model.points] - q) / np.linalg.norm(q))
            # (ii) reduced RHS error against the exact Galerkin RHS at the same reduced state
            ft = tensorial_rhs(tens, a, nu)
            rhs_err.append(np.linalg.norm(spectral_deim_rhs(model, a, nu) - ft) / np.linalg.norm(ft))
        entry = {"target": target, "m": m, "s": s, "selection": sel, "points": model.points,
                 "conditioning": model.conditioning, "offline_s": offline_s,
                 "diffusion_radius": model.diffusion_radius,
                 "field_interp_rel_err": stats(nl_err), "reduced_rhs_rel_err": stats(rhs_err),
                 "bytes_online": int(model.M.nbytes + model.C.nbytes + model.Lin.nbytes)}
        for split, cases in sets.items():
            rows = [{"case": lab, "nu": nu, **physical_diagnostics(run_spectral(model, Phi, u0, nu), ref)}
                    for lab, u0, nu, ref in cases]
            entry[split] = {"summary": summarize(rows), "rows": rows}
        rhs = deim_rhs_fn(model, float(sets["test"][0][2]))
        entry["rhs_us"] = _med_us(lambda: rhs(states[5][0]))
        tm = [np.median(interleaved_timing({"x": lambda: run_spectral(model, Phi, u0, nu)}, reps=5,
                                           warmup=1)["x"]) for _, u0, nu, _ in sets["test"][::4]]
        entry["end_to_end_ms_mean_10cases"] = 1e3 * float(np.mean(tm))
        out["configs"][config_name(*cfg)] = entry
        sv, stt = entry["val"]["summary"], entry["test"]["summary"]
        print(f"{config_name(*cfg):34s} cond {model.conditioning['cond']:7.2f} | rhs err med "
              f"{entry['reduced_rhs_rel_err']['median']:.1e} | val {100*sv['mean']:.4f}% test {100*stt['mean']:.4f}%"
              f" (max {100*stt['max']:.3f}%, unstable {stt['n_unstable']}) | {entry['rhs_us']:.2f} us")
    # ---- hybrid DEIM/local-FD at the same m (s = m, greedy DEIM, full-RHS basis)
    for m, (M, pts, Psi_h) in hybrid_models(Phi, data).items():
        entry = {"m": m, "points": pts, "conditioning": interpolation_conditioning(Psi_h, pts)}
        for split, cases in sets.items():
            rows = [{"case": lab, "nu": nu, **physical_diagnostics(run_hybrid(M, pts, Phi, u0, nu), ref)}
                    for lab, u0, nu, ref in cases]
            entry[split] = {"summary": summarize(rows), "rows": rows}
        out["hybrid_localfd"][str(m)] = entry
        stt = entry["test"]["summary"]
        print(f"hybrid local-FD m={m:2d} val {100*entry['val']['summary']['mean']:.4f}% test {100*stt['mean']:.4f}%"
              f" (max {100*stt['max']:.3f}%, unstable {stt['n_unstable']}, >10% {stt['n_above_10pct']})")
    # ---- selection on val only (lowest mean val error among configurations stable on all val cases)
    ok = {n: e for n, e in out["configs"].items() if e["val"]["summary"]["n_unstable"] == 0}
    for target in ("nonlinear", "full"):
        cand = {n: e for n, e in ok.items() if e["target"] == target}
        best = min(cand, key=lambda n: cand[n]["val"]["summary"]["mean"])
        out[f"best_on_val_{target}"] = best
        print(f"best on val ({target}): {best} -> test {100*cand[best]['test']['summary']['mean']:.4f}%")
    out["tensorial_test_mean"] = summarize([{"rel_l2": relative_l2(
        tensorial_predict(u0, Phi, tens, nu, T=T_FINAL, n_save=N_SAVE, align_snapshots=True)[1], ref),
        "stable": True} for _, u0, nu, ref in sets["test"]])["mean"]
    out["env"] = finish_env(env)
    save("deim_sweep", tag, out)


def _med_us(fn, number=2000, repeat=9):
    return 1e6 * float(np.median(timeit.repeat(fn, number=number, repeat=repeat))) / number


STAGES = {"baseline": stage_baseline, "equivalence": stage_equivalence,
          "tensorial_bench": stage_tensorial_bench, "spectral_consistency": stage_spectral_consistency,
          "deim_sweep": stage_deim_sweep}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    STAGES[args.stage](args.tag)
