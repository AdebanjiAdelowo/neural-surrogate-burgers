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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_dataset import N_SAVE, NX, T_FINAL
from scripts.hyperreduction_benchmark import (build_bases, environment_info, err_stats, finish_env,
                                               load_data, wavenumbers)
from src.deim import deim_rom_predict
from src.metrics import relative_l2
from src.pod_rom import _burgers_rhs_physical, rom_predict, rom_reduced_trajectory
from src.solver import solve_burgers
from src.tensorial_pod import (build_tensorial_rom, tensorial_reduced_trajectory, tensorial_rhs)

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


def stored_basis():
    st = np.load("experiments/deim_basis.npz")
    return {k_: st[k_] for k_ in st.files}


def aligned_ref(A, nu, Nx=NX):
    return solve_burgers(A=A, nu=nu, Nx=Nx, T=T_FINAL, n_save=N_SAVE, align_snapshots=True)[2]


def aligned_refs(split, data, Nx=NX):
    sA, snu, _ = data[split]
    return [aligned_ref(float(sA[i]), float(snu[i]), Nx) for i in range(len(sA))]


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


STAGES = {"baseline": stage_baseline, "equivalence": stage_equivalence}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    STAGES[args.stage](args.tag)
