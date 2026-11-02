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


STAGES = {"if_failure": stage_if_failure}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    STAGES[args.stage](args.tag)
