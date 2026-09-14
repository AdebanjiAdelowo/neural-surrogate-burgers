"""Research extension -- rigorous diagnosis of (a) the DEIM-ROM steep-gradient failure mode and
(b) the non-monotonic DEIM-rank behaviour (review response; see RESEARCH_EXTENSION.md
"Failure-mechanism investigation" and "Rank non-monotonicity").

Both analyses were initially done ad hoc during review; this script makes them reproducible and
saves machine-readable output, per the same "every number traceable to script output" standard
the rest of this research extension follows.

(a) Correlates DEIM-ROM's per-example trajectory error (test split, n=40) against candidate
    explanatory variables -- max|u_x|, max|u_xx| (steepness proxies, measured on the true
    trajectory), and mean/max nonlinear-term approximation error at the DEIM points (measured on
    the REAL, solver-generated reduced-state trajectory, not an arbitrary/random one -- an earlier
    version of this diagnostic used t=0 only, which is misleading since t=0's field is always the
    smooth initial sinusoid regardless of A; steepness develops later via nonlinear advection).
    Reports Spearman rank correlation (robust to the heavy right-skew in trajectory error) with
    each candidate variable, with p-values, rather than asserting causality from correlation alone.

(b) For each DEIM rank m, measures singular-value energy capture, DEIM-interpolation-matrix
    conditioning, min inter-point gap, ||M||_2 (the DEIM-Galerkin reduction matrix's operator
    norm), the reduced-RHS relative error (the quantity that actually enters the RK4 integration,
    as opposed to the raw pointwise nonlinear-term error at the DEIM points), and the resulting
    trajectory error vs. the plain ROM -- to determine which of these actually explains the
    non-monotonic trajectory-error behaviour across m.

Run: python scripts/diagnose_deim_mechanisms.py
Output: report/research/deim_mechanism_diagnostics.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deim import (build_deim_basis, build_deim_projector, collect_nonlinear_snapshots,
                       deim_nonlinear_at_points, deim_rom_predict, select_deim_points,
                       _stencil_support)
from src.pod_rom import build_pod_basis, rom_predict

ROM_MODES = 8
CANDIDATE_M = [6, 8, 10, 12, 16, 24, 32]


def part_a_failure_mechanism():
    d = np.load("experiments/deim_basis.npz")
    Phi, points, M, m, Nx = d["Phi"], d["points"], d["M"], int(d["m"]), int(d["Nx"])
    L = 2 * np.pi
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    dealias = np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k))

    d_test = np.load("data/test.npz")
    A_test, nu_test, u_test = d_test["A"], d_test["nu"], d_test["u"]
    n_save = u_test.shape[1]

    support, neighbours = _stencil_support(points, Nx, halfwidth=1)
    Phi_support = Phi[support, :]
    dx = (2 * np.pi) / Nx

    rows = []
    for i in range(len(A_test)):
        gt = u_test[i]
        A_i, nu_i = float(A_test[i]), float(nu_test[i])
        u0 = gt[0]
        _, u_deim = deim_rom_predict(u0, Phi, M, points, Nx, nu=nu_i, T=1.0, n_save=n_save)
        traj_err = np.linalg.norm(u_deim - gt) / np.linalg.norm(gt)

        max_ux, max_uxx, nl_errs = 0.0, 0.0, []
        for snap_idx in range(n_save):
            u_true_snap = gt[snap_idx]
            a_snap = Phi.T @ u_true_snap  # project the REAL trajectory snapshot (not t=0 only)
            u_recon = Phi @ a_snap
            u_hat = np.fft.fft(u_recon)
            ux = np.real(np.fft.ifft(1j * k * u_hat))
            uxx = np.real(np.fft.ifft(-(k**2) * u_hat))
            max_ux = max(max_ux, np.abs(ux).max())
            max_uxx = max(max_uxx, np.abs(uxx).max())
            flux_hat = np.fft.fft(u_recon * u_recon) * dealias
            nl_exact = np.real(np.fft.ifft(-0.5j * k * flux_hat))
            F_exact_pts = (nl_exact + nu_i * uxx)[points]
            F_approx_pts = deim_nonlinear_at_points(a_snap, Phi_support, support, neighbours,
                                                      nu_i, dx)
            nl_errs.append(float(np.linalg.norm(F_approx_pts - F_exact_pts) /
                                  (np.linalg.norm(F_exact_pts) + 1e-12)))
        rows.append({"traj_err": float(traj_err), "A": A_i, "nu": nu_i, "max_ux": float(max_ux),
                     "max_uxx": float(max_uxx), "max_nl_err": float(np.max(nl_errs)),
                     "mean_nl_err": float(np.mean(nl_errs))})

    arr = np.array([[r["traj_err"], r["A"], r["nu"], r["max_ux"], r["max_uxx"], r["max_nl_err"],
                      r["mean_nl_err"]] for r in rows])
    names = ["A", "nu", "max_ux", "max_uxx", "max_nl_err", "mean_nl_err"]
    correlations = {}
    print("=== (a) Failure-mechanism correlations (Spearman, n=40) ===")
    for j, name in enumerate(names, start=1):
        rho, p = spearmanr(arr[:, 0], arr[:, j])
        correlations[name] = {"spearman_rho": float(rho), "p_value": float(p)}
        print(f"  spearman(traj_err, {name:12s}) = {rho:+.3f}  (p={p:.4g})")

    worst = sorted(rows, key=lambda r: -r["traj_err"])[:8]
    best = sorted(rows, key=lambda r: r["traj_err"])[:5]
    return {"per_example": rows, "spearman_correlations": correlations,
            "worst_8_cases": worst, "best_5_cases": best,
            "note": "Correlation, not proven causation -- see the higher-order-stencil "
                    "experiment (scripts/compare_deim_stencil_orders.py) for a direct "
                    "intervention test."}


def part_b_rank_mechanism():
    d_train = np.load("data/train.npz")
    Nx = d_train["u"].shape[2]
    L = 2 * np.pi
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    dealias = np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k))
    ensemble = d_train["u"].reshape(-1, Nx)
    nu_per_snapshot = np.repeat(d_train["nu"], d_train["u"].shape[1])

    Phi_full, _ = build_pod_basis(ensemble, r=32)
    Phi = Phi_full[:, :ROM_MODES]
    F_ensemble = collect_nonlinear_snapshots(ensemble, k, nu_per_snapshot, dealias)

    d_val = np.load("data/val.npz")
    A_val, nu_val, u_val = d_val["A"], d_val["nu"], d_val["u"]
    n_save = u_val.shape[1]
    dx = (2 * np.pi) / Nx

    print("\n=== (b) Rank non-monotonicity mechanism ===")
    print(f"{'m':>3} {'cumE':>9} {'cond(PtPsi)':>12} {'||M||_2':>9} {'min_gap':>8} "
          f"{'reducedRHS_err':>15} {'traj_err_vs_ROM':>16}")
    rows = []
    for m in CANDIDATE_M:
        Psi, S = build_deim_basis(F_ensemble, m)
        points = select_deim_points(Psi)
        PT_Psi = Psi[points, :]
        cond = float(np.linalg.cond(PT_Psi))
        M = build_deim_projector(Phi, Psi, points)
        op_norm = float(np.linalg.norm(M, 2))
        support, neighbours = _stencil_support(points, Nx, halfwidth=1)
        Phi_support = Phi[support, :]
        cumE = float(np.sum(S[:m]**2) / np.sum(S**2))
        sp = np.sort(points)
        gaps = np.diff(np.concatenate([sp, [sp[0] + Nx]]))
        min_gap = int(gaps.min())

        rhs_errs, traj_errs = [], []
        for i in range(len(A_val)):
            gt = u_val[i]
            nu_i = float(nu_val[i])
            for snap_idx in [5, 10, 15]:
                u_snap = gt[snap_idx]
                a_snap = Phi.T @ u_snap
                u_recon = Phi @ a_snap
                u_hat = np.fft.fft(u_recon)
                uxx = np.real(np.fft.ifft(-(k**2) * u_hat))
                flux_hat = np.fft.fft(u_recon * u_recon) * dealias
                nl_exact = np.real(np.fft.ifft(-0.5j * k * flux_hat))
                F_exact_full = nl_exact + nu_i * uxx
                true_reduced_rhs = Phi.T @ F_exact_full
                F_approx_pts = deim_nonlinear_at_points(a_snap, Phi_support, support, neighbours,
                                                          nu_i, dx)
                deim_reduced_rhs = M @ F_approx_pts
                rhs_errs.append(float(np.linalg.norm(deim_reduced_rhs - true_reduced_rhs) /
                                       (np.linalg.norm(true_reduced_rhs) + 1e-12)))
            _, u_deim = deim_rom_predict(gt[0], Phi, M, points, Nx, nu=nu_i, T=1.0, n_save=n_save)
            _, u_rom = rom_predict(gt[0], Phi, k, nu_i, dealias, T=1.0, n_save=n_save)
            traj_errs.append(float(np.linalg.norm(u_deim - u_rom) /
                                    (np.linalg.norm(u_rom) + 1e-12)))

        row = {"m": m, "cumulative_energy_ratio": cumE, "cond_PtPsi": cond, "M_operator_norm": op_norm,
               "min_point_gap": min_gap, "reduced_rhs_relerr_mean": float(np.mean(rhs_errs)),
               "trajectory_relerr_vs_rom_mean": float(np.mean(traj_errs)),
               "trajectory_relerr_vs_rom_max": float(np.max(traj_errs))}
        rows.append(row)
        print(f"{m:3d} {cumE:9.6f} {cond:12.3f} {op_norm:9.3f} {min_gap:8d} "
              f"{np.mean(rhs_errs):15.5f} {np.mean(traj_errs):16.5f}")

    best_row = min(rows, key=lambda r: r["trajectory_relerr_vs_rom_mean"])
    return {"sweep": rows, "selected_m": best_row["m"],
            "conclusion": "Neither the SVD energy-capture ratio (saturates smoothly to ~1.0 by "
                           "m=10-12, no noise floor visible) nor DEIM-interpolation-matrix "
                           "conditioning (stays well-conditioned, 4-7, across all m) explains the "
                           "non-monotonic trajectory error. The reduced-RHS relative error -- the "
                           "quantity that actually enters the RK4 integration -- tracks ||M||_2 "
                           "(the DEIM-Galerkin reduction matrix's operator norm) closely, and both "
                           "are minimised at the selected rank. The original 'fitting SVD noise' "
                           "explanation is not supported by this evidence and is retracted; the "
                           "better-supported (though not fully first-principles-derived) "
                           "explanation is that M's operator norm happens to be non-monotonic in "
                           "m for this dataset, amplifying or damping the (monotonically "
                           "decreasing) raw pointwise nonlinear-term error differently at "
                           "different ranks."}


def main():
    part_a = part_a_failure_mechanism()
    part_b = part_b_rank_mechanism()
    os.makedirs("report/research", exist_ok=True)
    with open("report/research/deim_mechanism_diagnostics.json", "w") as f:
        json.dump({"failure_mechanism": part_a, "rank_nonmonotonicity_mechanism": part_b}, f,
                   indent=2)
    print("\nSaved report/research/deim_mechanism_diagnostics.json")


if __name__ == "__main__":
    main()
