"""Adaptive, error-controlled time integration for tensorial POD-Galerkin models
(report/adaptive_tensorial_pod_integrators_study.md; plan in report/adaptive_integrators/PLAN.md).

Every stage writes report/adaptive_integrators/<stage>[_<tag>].json; nothing else is modified. The
earlier studies' modules and results (src/tensorial_pod.py, src/reduced_timestep.py,
scripts/spatial_convergence_study.py, report/spatial_convergence/, report/tensorial_deim/) are only read.

Conventions (verified in Stage 1): da/dt = nu L a + Q(a, a), L = Phi^T D2 Phi symmetric negative definite
(nu is NOT inside L), Q symmetric in its last two indices; POD bases at N = 2048 from the 200 historical
training trajectories (cached ensemble, scripts/spatial_convergence_study.training_ensemble).

Stages:
  if_failure   why the reduced integrating-factor RK4 (P2 of the previous study) failed its accuracy
               criterion at r = 16 on test cases 5 and 33

Run:  python scripts/adaptive_integrators_study.py <stage> [--tag TAG]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_dataset import N_SAVE, T_FINAL
from scripts.hyperreduction_benchmark import environment_info, finish_env, load_data, wavenumbers
from scripts.spatial_convergence_study import TIMES, run_policy, training_ensemble
from src.convergence import sine_ic, solve_burgers_f64
from src.pod_rom import build_pod_basis
from src.reduced_timestep import aligned_steps, if_rk4_dt, if_rk4_trajectory, rk4_trajectory
from src.tensorial_pod import build_tensorial_rom, reduced_step_count

OUT_DIR = "report/adaptive_integrators"
N_ROM = 2048
R_LIST = (8, 16, 32)


def save(name, tag, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}{'_' + tag if tag else ''}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"saved {path}")


_MODELS = {}


def rom_models(r_list=R_LIST, data=None):
    """{r: (Phi, TensorialROM)} at N = 2048 from the cached training ensemble (built once per process)."""
    key = tuple(r_list)
    if key not in _MODELS:
        U, _ = training_ensemble(N_ROM, data or load_data())
        Phi_all, _ = build_pod_basis(U.reshape(-1, N_ROM), max(r_list))
        k, dealias = wavenumbers(N_ROM)
        _MODELS[key] = {r: (np.ascontiguousarray(Phi_all[:, :r]),
                            build_tensorial_rom(np.ascontiguousarray(Phi_all[:, :r]), k, dealias)) for r in r_list}
    return _MODELS[key]


def p0_steps(model, A, nu):
    """Step count of the previous studies' default ("reduced") policy for this case."""
    return reduced_step_count(A, model.Nx, nu, T_FINAL, N_SAVE, align_snapshots=True,
                              diffusion_radius=model.diffusion_radius)[0]


def reference_rom(model, a0, A, nu, mult=16):
    """High-accuracy solution of the SAME reduced ODE: classical RK4 with mult x the default step count
    (independent of every candidate method; its own accuracy is checked by doubling mult)."""
    return rk4_trajectory(model, a0, nu, T_FINAL, N_SAVE, mult * p0_steps(model, A, nu))[1]


def rel(a, b):
    """Relative space-time L2 difference of two coefficient trajectories (= relative discrete L2 error of
    the reconstructed fields, since Phi is orthonormal)."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# =============================================================================== stage: if_failure
def stage_if_failure(tag):
    env = environment_info()
    data = load_data()
    tA, tnu, _ = data["test"]
    vA, vnu, _ = data["val"]
    models = rom_models(data=data)
    v_order = np.argsort(vA / vnu)
    cases = {"test5": (float(tA[5]), float(tnu[5])), "test33": (float(tA[33]), float(tnu[33])),
             "test0": (float(tA[0]), float(tnu[0])),
             "val_steepest": (float(vA[v_order[-1]]), float(vnu[v_order[-1]])),
             "val_smoothest": (float(vA[v_order[0]]), float(vnu[v_order[0]]))}
    out = {"cases": cases, "by_r": {}}
    for r in (8, 16, 32):
        Phi, m = models[r]
        lam, V = np.linalg.eigh(0.5 * (m.L + m.L.T))
        out["by_r"][r] = {"rho_L": m.diffusion_radius, "cases": {}}
        for name, (A, nu) in cases.items():
            u0 = sine_ic(A, N_ROM)
            a0 = Phi.T @ u0
            ref = reference_rom(m, a0, A, nu, 16)
            ref32 = reference_rom(m, a0, A, nu, 32)
            n_if = aligned_steps(if_rk4_dt(A, m.diffusion_radius), T_FINAL, N_SAVE, min_steps=1)
            rec = {"A": A, "nu": nu, "A_over_nu": A / nu, "reference_self_check_16_vs_32": rel(ref, ref32),
                   "if_steps": n_if, "z_stiff_nu_rho_dt": nu * m.diffusion_radius / n_if, "refinement": {}}
            # (1)-(2) instability vs truncation error: refine the IF-RK4 step at fixed everything else
            errs = {}
            for q in (1, 2, 4, 8):
                t_s, a = if_rk4_trajectory(m, a0, nu, T_FINAL, N_SAVE, n_if * q)
                errs[q] = rel(a, ref)
                fin = bool(np.all(np.isfinite(a)))
                E = 0.5 * np.sum(a**2, axis=1)
                rec["refinement"][q] = {"time_err": errs[q], "finite": fin,
                                        "energy_increase_max": float(max(0.0, np.diff(E).max()) / E[0]),
                                        "max_abs_coeff_over_ref": float(np.abs(a).max() / np.abs(ref).max()),
                                        "snapshot_times_exact": bool(np.allclose(t_s, TIMES, atol=1e-13, rtol=0))}
            rec["observed_orders"] = [float(np.log2(errs[q] / errs[2 * q])) for q in (1, 2, 4)]
            # (3) implementation: IF-RK4 must converge to the independent RK4 reference as the step shrinks
            #     (refinement above); additionally record how badly the GLOBAL integrating-factor transform
            #     v = exp(-nu L t) a is scaled: exp(nu rho t) overflows double precision once nu rho t > 709,
            #     so any integrating-factor method must reset its frame every step.
            rec["global_transform_log10_scale_at_T"] = float(nu * m.diffusion_radius * T_FINAL / np.log(10))
            rec["global_transform_overflows_double"] = bool(nu * m.diffusion_radius * T_FINAL > 709.0)
            rec["if_rk4_finest_vs_reference"] = errs[8]
            # (5) floating point: the same IF-RK4 run in a perturbed operation order (a0 + eps)
            eps = np.finfo(float).eps * np.linalg.norm(a0) * np.random.default_rng(0).standard_normal(r) / np.sqrt(r)
            rec["float_perturbation_effect"] = rel(if_rk4_trajectory(m, a0 + eps, nu, T_FINAL, N_SAVE, n_if)[1],
                                                   if_rk4_trajectory(m, a0, nu, T_FINAL, N_SAVE, n_if)[1])
            # where the error lives: components in L's eigenbasis, grouped by stiffness
            _, a1 = if_rk4_trajectory(m, a0, nu, T_FINAL, N_SAVE, n_if)
            e_eig = (a1 - ref) @ V                        # (20, r) error in eigen-coordinates
            stiff = nu * (-lam) * (T_FINAL / n_if)        # |z| per eigenmode at the IF step
            share = np.sum(e_eig**2, axis=0) / np.sum(e_eig**2)
            rec["error_share_modes_with_z_gt_1"] = float(share[stiff > 1].sum())
            rec["error_share_modes_with_z_le_1"] = float(share[stiff <= 1].sum())
            # comparison: the previous study's criterion quantities for this case
            fom = solve_burgers_f64(u0, nu)[1]
            from src.convergence import rel_l2
            p0, _ = run_policy("P0_reduced", m, Phi, u0, nu)
            p2, _ = run_policy("P2_if_rk4", m, Phi, u0, nu)
            rec["previous_criterion"] = {"P2_rom_vs_fom": rel_l2(p2 @ Phi.T, fom), "P0_rom_vs_fom": rel_l2(p0 @ Phi.T, fom),
                                         "P0_time_err": rel(p0, ref), "P2_time_err": rel(p2, ref)}
            pc = rec["previous_criterion"]
            pc["passes"] = bool(pc["P2_rom_vs_fom"] <= 1.05 * pc["P0_rom_vs_fom"] + 1e-6)
            out["by_r"][r]["cases"][name] = rec
            print(f"r={r:2d} {name:13s} A/nu {A / nu:6.1f} z {rec['z_stiff_nu_rho_dt']:7.1f} | IF time err "
                  + " ".join(f"{errs[q]:.1e}" for q in (1, 2, 4, 8)) + " | orders "
                  + " ".join(f"{o:.2f}" for o in rec["observed_orders"])
                  + f" | stiff-mode share {rec['error_share_modes_with_z_gt_1']:.2f} | ref16/32 {rec['reference_self_check_16_vs_32']:.0e}"
                  + f" | P2 {pc['P2_rom_vs_fom']:.2e} vs P0 {pc['P0_rom_vs_fom']:.2e} {'pass' if pc['passes'] else 'FAIL'}")
    out["env"] = finish_env(env)
    save("if_failure", tag, out)


# ===================================================================================== shared setup
RTOLS = (1e-4, 1e-6, 1e-8, 1e-10)
DIVISORS = (1, 2, 4, 8)          # fixed-step baselines: step divided by these factors
TARGETS = (1e-4, 1e-6, 1e-8)
REF_MULT, REF_CHECK_MULT = 4, 8  # reference ROM: RK4 with 4x (checked against 8x) the default step count
STRESS_A = (1.5, 1.75, 2.0)
STRESS_NU = (0.010, 0.0125, 0.015, 0.0175, 0.020)


def configs():
    """Every configuration of the study (PLAN.md Section 5): (method id, setting)."""
    out = [("B1", d) for d in DIVISORS] + [("B2", d) for d in DIVISORS]
    out += [(m, rt) for m in ("B3", "B3fd", "E1", "E2", "X1", "X2") for rt in RTOLS]
    return out


def cfg_name(cfg):
    m, s = cfg
    return f"{m}_div{s}" if m in ("B1", "B2") else f"{m}_rtol{s:g}"


def case_sets(names, data=None):
    """{set: [(label, A, nu)]} for the sets of PLAN.md Section 4. The fresh holdout is drawn only when
    requested (Stage 7) from the committed seeds."""
    data = data or load_data()
    out = {}
    for n in names:
        if n == "val":
            A, nu, _ = data["val"]
            out[n] = [(f"val{i}", float(A[i]), float(nu[i])) for i in range(len(A))]
        elif n == "stress":
            out[n] = [(f"stress_A{A}_nu{nu}", A, nu) for A in STRESS_A for nu in STRESS_NU]
        elif n == "test":
            A, nu, _ = data["test"]
            out[n] = [(f"test{i}", float(A[i]), float(nu[i])) for i in range(len(A))]
        elif n == "holdout":
            g = np.random.default_rng(300000)
            A, nu = g.uniform(0.5, 2.0, 40), g.uniform(0.01, 0.1, 40)
            g2 = np.random.default_rng(300001)
            As, nus = g2.uniform(1.5, 2.0, 10), g2.uniform(0.010, 0.020, 10)
            out[n] = ([(f"hold{i}", float(A[i]), float(nu[i])) for i in range(40)]
                      + [(f"hold_steep{i}", float(As[i]), float(nus[i])) for i in range(10)])
        else:
            raise ValueError(n)
    return out


_BASES = {}


def eigen_basis(r, models):
    from src.exponential_integrators import EigenBasis
    if r not in _BASES:
        _BASES[r] = EigenBasis(models[r][1].Q, models[r][1].L)
    return _BASES[r]


def run_config(cfg, r, models, u0, a0, A, nu, record_steps=False):
    """(a_save (20, r), stats) of one configuration."""
    from src.adaptive_integrators import adaptive_solve, default_atol, scipy_solve
    Phi, m = models[r]
    meth, s = cfg
    with np.errstate(all="ignore"):
        if meth in ("B1", "B2"):
            a, info = run_policy({"B1": "P1_reduced_cfl", "B2": "P2_if_rk4"}[meth], m, Phi, u0, nu, mult=1.0 / s)
            return a, {"rhs_evals": info["rhs_evals"], "accepted": info["n_steps"], "rejected": 0}
        atol = default_atol(a0, s)
        if meth in ("X1", "X2"):
            a, st = adaptive_solve({"X1": "lawson_dp54", "X2": "etdrk4_doubling"}[meth], m.Q, m.L, nu, a0, TIMES,
                                   s, atol, record_steps=record_steps, basis=eigen_basis(r, models))
            st["rhs_evals"] = st["n_evals"]
            return a, st
        method = {"B3": "Radau", "B3fd": "Radau", "E1": "RK45", "E2": "DOP853"}[meth]
        a, st = scipy_solve(m.Q, m.L, nu, a0, TIMES, method, s, atol, jacobian="fd" if meth == "B3fd" else "analytic",
                            count_steps=record_steps)
        st["rhs_evals"] = st["nfev"]
        return a, st


def exact_traj(A, nu):
    from scripts.spatial_convergence_study import exact
    return exact(A, nu)


# ================================================================================= stage: stiffness
def stage_stiffness(tag):
    """Reduced diffusion spectra and Jacobian diagnostics along the reference trajectories of the
    validation and stress-validation cases (test and holdout are not used)."""
    from src.reduced_stiffness import diffusion_spectrum, jacobian_diagnostics
    from src.timestep import rk4_max_stable_dt
    from src.tensorial_pod import quadratic_jacobian
    env = environment_info()
    data = load_data()
    models = rom_models(data=data)
    sets = case_sets(("val", "stress"), data)
    out = {"by_r": {}}
    for r in R_LIST:
        Phi, m = models[r]
        ds = diffusion_spectrum(m.L)
        rec = {"diffusion": ds, "nu_rho_range": [0.01 * ds["rho"], 0.1 * ds["rho"]], "cases": {}}
        for sname, cases in sets.items():
            for lab, A, nu in cases:
                a0 = Phi.T @ sine_ic(A, N_ROM)
                ref = reference_rom(m, a0, A, nu, REF_MULT)
                diag = [jacobian_diagnostics(m.Q, m.L, nu, a, transient=(j in (0, 10, 19))) for j, a in enumerate(ref)]
                frozen_dt = [rk4_max_stable_dt(np.linalg.eigvals(quadratic_jacobian(m.Q, nu * m.L, a)), dt_max=1.0)
                             for a in ref[1:]]
                rec["cases"][lab] = {
                    "set": sname, "A": A, "nu": nu,
                    "spectral_abscissa_max": max(d["spectral_abscissa"] for d in diag),
                    "numerical_abscissa_max": max(d["numerical_abscissa"] for d in diag),
                    "spectral_radius_max": max(d["spectral_radius"] for d in diag),
                    "max_abs_imag_max": max(d["max_abs_imag"] for d in diag),
                    "henrici_max": max(d["henrici_departure"] for d in diag),
                    "nonlinear_over_diffusion_norm_max": max(d["nonlinear_part_norm"] / d["diffusion_part_norm"] for d in diag),
                    "transient_amplification_max": max(d["transient_amplification"] for d in diag if "transient_amplification" in d),
                    "frozen_rk4_dt_min": float(min(frozen_dt)),
                    "per_snapshot": {k_: [d[k_] for d in diag] for k_ in ("spectral_abscissa", "numerical_abscissa",
                                                                         "spectral_radius", "henrici_departure")}}
        cs = rec["cases"].values()
        rec["summary"] = {k_: (float(max(c[k_] for c in cs)) if k_ != "frozen_rk4_dt_min" else float(min(c[k_] for c in cs)))
                          for k_ in ("spectral_abscissa_max", "numerical_abscissa_max", "spectral_radius_max", "henrici_max",
                                     "nonlinear_over_diffusion_norm_max", "transient_amplification_max", "frozen_rk4_dt_min")}
        out["by_r"][r] = rec
        print(f"r={r}: L eig [{ds['eigenvalues'][0]:.0f}, {ds['eigenvalues'][-1]:.2f}] rho {ds['rho']:.0f} sym {ds['symmetry_defect']:.0e} "
              f"| J: alpha max {rec['summary']['spectral_abscissa_max']:+.2f} omega max {rec['summary']['numerical_abscissa_max']:+.2f} "
              f"rho max {rec['summary']['spectral_radius_max']:.0f} henrici max {rec['summary']['henrici_max']:.2f} "
              f"transient max {rec['summary']['transient_amplification_max']:.3f} frozen RK4 dt min {rec['summary']['frozen_rk4_dt_min']:.1e}")
    out["env"] = finish_env(env)
    save("stiffness", tag, out)


# =================================================================================== stage: explore
def stage_explore(tag):
    """All configurations on the validation and stress-validation sets only: time-integration error
    against the reference ROM, solver statistics, single-run times, step histories (one case)."""
    env = environment_info()
    data = load_data()
    models = rom_models(data=data)
    sets = case_sets(("val", "stress"), data)
    out = {"configs": [cfg_name(c) for c in configs()], "rtols": RTOLS, "divisors": DIVISORS, "by_r": {}}
    steep_lab = "stress_A2.0_nu0.01"
    for r in R_LIST:
        Phi, m = models[r]
        rows = []
        for sname, cases in sets.items():
            for lab, A, nu in cases:
                u0 = sine_ic(A, N_ROM)
                a0 = Phi.T @ u0
                ref = reference_rom(m, a0, A, nu, REF_MULT)
                row = {"set": sname, "case": lab, "A": A, "nu": nu, "cfg": {}}
                if sname == "stress":
                    row["reference_self_check"] = rel(ref, reference_rom(m, a0, A, nu, REF_CHECK_MULT))
                for cfg in configs():
                    rec_steps = lab == steep_lab and cfg[1] in (1e-6, 1)
                    t0 = time.perf_counter()
                    a, st = run_config(cfg, r, models, u0, a0, A, nu, record_steps=rec_steps)
                    dt = time.perf_counter() - t0
                    fin = bool(np.all(np.isfinite(a)))
                    row["cfg"][cfg_name(cfg)] = {"time_err": rel(a, ref) if fin else float("inf"), "finite": fin,
                                                 "ms": 1e3 * dt, **{k_: v for k_, v in st.items() if k_ not in ("message",)}}
                rows.append(row)
            print(f"r={r} {sname} done")
        summ = {}
        for c in out["configs"]:
            for sname in ("val", "stress", "both"):
                sel = [x for x in rows if sname == "both" or x["set"] == sname]
                e = [x["cfg"][c]["time_err"] for x in sel]
                summ.setdefault(c, {})[sname] = {"time_err_max": float(np.max(e)), "time_err_median": float(np.median(e)),
                                                 "rhs_evals_mean": float(np.mean([x["cfg"][c]["rhs_evals"] for x in sel])),
                                                 "ms_median": float(np.median([x["cfg"][c]["ms"] for x in sel])),
                                                 "n_nonfinite": int(sum(not x["cfg"][c]["finite"] for x in sel))}
        out["by_r"][r] = {"rows": rows, "summary": summ,
                          "reference_self_check_max": float(max(x["reference_self_check"] for x in rows if "reference_self_check" in x))}
        print(f"r={r}: reference self-check (4x vs 8x) max {out['by_r'][r]['reference_self_check_max']:.1e}")
        for c in out["configs"]:
            s_ = summ[c]["both"]
            print(f"   {c:16s} max time err {s_['time_err_max']:.1e} (val {summ[c]['val']['time_err_max']:.1e}, stress "
                  f"{summ[c]['stress']['time_err_max']:.1e}) evals {s_['rhs_evals_mean']:8.0f} ms {s_['ms_median']:7.2f}")
    out["env"] = finish_env(env)
    save("explore", tag, out)


# ==================================================================================== stage: select
METHODS = ("B1", "B2", "B3", "B3fd", "E1", "E2", "X1", "X2")


def select_settings(explore_summary_by_r, margin=2.0):
    """PLAN.md Section 7: for each method, rank and target, the loosest setting whose maximum time error
    over validation + stress-validation is <= target / margin; None if no setting qualifies."""
    sel = {}
    for r, summ in explore_summary_by_r.items():
        for meth in METHODS:
            settings = list(DIVISORS) if meth in ("B1", "B2") else sorted(RTOLS, reverse=True)  # loosest first
            for tau in TARGETS:
                choice = None
                for s in settings:
                    name = cfg_name((meth, s))
                    if summ[name]["both"]["n_nonfinite"] == 0 and summ[name]["both"]["time_err_max"] <= tau / margin:
                        choice = s
                        break
                sel[f"r{r}|{meth}|{tau:g}"] = choice
    return sel


def stage_select(tag):
    ex = json.load(open(os.path.join(OUT_DIR, "explore_run1.json")))
    summ = {int(r): v["summary"] for r, v in ex["by_r"].items()}
    sel = select_settings(summ)
    out = {"rule": "loosest setting with max time error over validation + stress-validation <= target / 2",
           "source": "explore_run1.json (validation and stress-validation only)", "selection": sel}
    for key, s in sel.items():
        r, meth, tau = key.split("|")
        detail = summ[int(r[1:])][cfg_name((meth, s))]["both"] if s is not None else None
        print(f"{key:18s} -> {('not eligible' if s is None else cfg_name((meth, s))):18s}"
              + (f" val+stress max {detail['time_err_max']:.1e}, evals {detail['rhs_evals_mean']:.0f}, ms {detail['ms_median']:.2f}"
                 if detail else ""))
    with open(os.path.join(OUT_DIR, "selection_frozen.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved", os.path.join(OUT_DIR, "selection_frozen.json"))


# ===================================================================================== stage: final
def stage_final(tag, reps=5):
    """Frozen configurations on the original test set and the fresh holdout (reported separately):
    time-integration error, reliability per target, total error vs Cole-Hopf, physical diagnostics and
    round-robin timing (one-time preparation, per-trajectory setup, integration, reconstruction)."""
    from scripts.hyperreduction_benchmark import interleaved_timing
    from src.convergence import grid, rel_l2, sine_solution
    from src.exponential_integrators import EigenBasis, EigenSystem
    env = environment_info()
    data = load_data()
    models = rom_models(data=data)
    frozen = json.load(open(os.path.join(OUT_DIR, "selection_frozen.json")))["selection"]
    sets = case_sets(("test", "holdout"), data)
    out = {"frozen": frozen, "reps": reps, "by_r": {}, "holdout_cases": sets["holdout"]}
    xg = grid(N_ROM)
    front = np.abs(xg - np.pi) < 0.3
    for r in R_LIST:
        Phi, m = models[r]
        t0 = time.perf_counter()
        _BASES.pop(r, None)
        eigen_basis(r, models)
        prep_s = time.perf_counter() - t0
        cfgs = sorted({(meth, s) for key, s in frozen.items() if s is not None
                       for rr, meth, _ in [key.split("|")] if int(rr[1:]) == r}, key=cfg_name)
        names = [cfg_name(c) for c in cfgs]
        rows = []
        for sname, cases in sets.items():
            for lab, A, nu in cases:
                u0 = sine_ic(A, N_ROM)
                a0 = Phi.T @ u0
                ref = reference_rom(m, a0, A, nu, REF_MULT)
                ex = exact_traj(A, nu)
                ch = sine_solution(A, nu)
                ex_front = ch(xg, 1.0)[front]
                row = {"set": sname, "case": lab, "A": A, "nu": nu, "ref_rom_vs_exact": rel_l2(ref @ Phi.T, ex),
                       "cfg": {}}
                calls = {}
                for c, n in zip(cfgs, names):
                    a, st = run_config(c, r, models, u0, a0, A, nu)
                    fin = bool(np.all(np.isfinite(a)))
                    u = a @ Phi.T
                    E = 0.5 * np.sum(a**2, axis=1)
                    row["cfg"][n] = {
                        "time_err": rel(a, ref) if fin else float("inf"), "finite": fin,
                        "err_vs_exact": rel_l2(u, ex) if fin else float("inf"),
                        "front_err_final": float(np.linalg.norm(u[-1][front] - ex_front) / np.linalg.norm(ex_front)) if fin else float("inf"),
                        "energy_increase_max": float(max(0.0, np.diff(E).max()) / E[0]) if fin else float("inf"),
                        "mean_abs_max": float(np.abs(u.mean(axis=1)).max()) if fin else float("inf"),
                        "max_abs_over_u0": float(np.abs(u).max() / np.abs(u0).max()) if fin else float("inf"),
                        **{k_: v for k_, v in st.items() if k_ in ("rhs_evals", "accepted", "rejected", "njev", "nlu")}}
                    calls[n + "|integrate"] = (lambda c=c: run_config(c, r, models, u0, a0, A, nu))
                calls["reconstruct"] = (lambda a=ref: a @ Phi.T)
                calls["project"] = (lambda: Phi.T @ u0)
                calls["setup_eigensystem"] = (lambda: EigenSystem(m.Q, m.L, nu, basis=_BASES[r]))
                t = interleaved_timing(calls, reps=reps, warmup=1)
                row["median_ms"] = {k_: 1e3 * float(np.median(v)) for k_, v in t.items()}
                row["iqr_ms"] = {k_: 1e3 * float(np.subtract(*np.percentile(v, [75, 25]))) for k_, v in t.items()}
                rows.append(row)
            print(f"r={r} {sname} done", flush=True)
        summ = {}
        for sname in ("test", "holdout"):
            sel_rows = [x for x in rows if x["set"] == sname]
            for n in names:
                e = np.array([x["cfg"][n]["time_err"] for x in sel_rows])
                tm = np.array([x["median_ms"][n + "|integrate"] + x["median_ms"]["reconstruct"] + x["median_ms"]["project"]
                               for x in sel_rows])
                summ.setdefault(sname, {})[n] = {
                    "time_err_max": float(e.max()), "time_err_median": float(np.median(e)),
                    "reliability": {f"{tau:g}": float(np.mean(e <= tau)) for tau in TARGETS},
                    "err_vs_exact_mean": float(np.mean([x["cfg"][n]["err_vs_exact"] for x in sel_rows])),
                    "err_vs_exact_max": float(np.max([x["cfg"][n]["err_vs_exact"] for x in sel_rows])),
                    "front_err_max": float(np.max([x["cfg"][n]["front_err_final"] for x in sel_rows])),
                    "energy_increase_max": float(np.max([x["cfg"][n]["energy_increase_max"] for x in sel_rows])),
                    "mean_abs_max": float(np.max([x["cfg"][n]["mean_abs_max"] for x in sel_rows])),
                    "max_abs_over_u0_max": float(np.max([x["cfg"][n]["max_abs_over_u0"] for x in sel_rows])),
                    "rhs_evals_mean": float(np.mean([x["cfg"][n]["rhs_evals"] for x in sel_rows])),
                    "rejected_mean": (float(np.mean([x["cfg"][n]["rejected"] for x in sel_rows]))
                                      if all(x["cfg"][n].get("rejected") is not None for x in sel_rows) else None),
                    "e2e_ms_median": float(np.median(tm)), "e2e_ms_p10": float(np.percentile(tm, 10)),
                    "e2e_ms_p90": float(np.percentile(tm, 90)),
                    "integrate_ms_median": float(np.median([x["median_ms"][n + "|integrate"] for x in sel_rows])),
                    "within_case_rel_iqr_median": float(np.median([x["iqr_ms"][n + "|integrate"] / x["median_ms"][n + "|integrate"]
                                                                   for x in sel_rows]))}
            summ[sname]["_reference"] = {
                "ref_rom_vs_exact_mean": float(np.mean([x["ref_rom_vs_exact"] for x in sel_rows])),
                "ref_rom_vs_exact_max": float(np.max([x["ref_rom_vs_exact"] for x in sel_rows])),
                "reconstruct_us_median": 1e3 * float(np.median([x["median_ms"]["reconstruct"] for x in sel_rows])),
                "project_us_median": 1e3 * float(np.median([x["median_ms"]["project"] for x in sel_rows])),
                "setup_eigensystem_us_median": 1e3 * float(np.median([x["median_ms"]["setup_eigensystem"] for x in sel_rows]))}
        # fastest frozen configuration meeting each target on each set
        fastest = {}
        for sname in ("test", "holdout"):
            for tau in TARGETS:
                ok = [(summ[sname][n]["e2e_ms_median"], n) for n in names if summ[sname][n]["time_err_max"] <= tau]
                fastest[f"{sname}|{tau:g}"] = sorted(ok)[:4]
        out["by_r"][r] = {"one_time_prep_s": prep_s, "configs": names, "summary": summ, "fastest": fastest, "rows": rows}
        print(f"r={r}: one-time eigen prep {1e3 * prep_s:.1f} ms")
        for k_, v in fastest.items():
            print(f"   fastest meeting {k_}: " + ", ".join(f"{n} {t_:.2f} ms" for t_, n in v))
    out["env"] = finish_env(env)
    save("final", tag, out)


STAGES = {"if_failure": stage_if_failure, "stiffness": stage_stiffness, "explore": stage_explore,
          "select": stage_select, "final": stage_final}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    STAGES[args.stage](args.tag)
