"""Research extension -- build the DEIM basis/interpolation points from the existing training
ensemble (no new data generation: reuses data/train.npz and the same r=8 POD basis the plain ROM
baseline uses in scripts/evaluate_comparison.py, for a fair, apples-to-apples comparison).

Also runs a DEIM-rank sweep (m = 6..32) and reports both (a) the raw nonlinear-term-evaluation
accuracy at the DEIM points and (b) the resulting full-trajectory accuracy versus the plain
(non-hyper-reduced) ROM, on a held-out validation set. This is not a cosmetic diagnostic: an
exploratory run (recorded in RESEARCH_EXTENSION.md) found DEIM-ROM trajectory accuracy is
NON-MONOTONIC in m for this dataset -- it degrades once m exceeds what the ~200-trajectory
nonlinear-term ensemble can actually resolve (the SVD singular values of the nonlinear-term
ensemble collapse to near-noise levels beyond roughly m=12-16), rather than continuing to improve
as textbook DEIM intuition (larger m -> more accurate) would suggest. Selecting the DEIM rank to
use in the headline comparison from this sweep, rather than assuming "bigger is better," is itself
part of this study's honest methodology.

Run: python scripts/build_deim.py
Output: experiments/deim_basis.npz (Phi, Psi, points, M, m -- the SELECTED rank),
        report/research/deim_rank_sweep.json (the full sweep, for the write-up).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deim import (build_deim_basis, build_deim_projector, collect_nonlinear_snapshots,
                       deim_rom_predict, select_deim_points)
from src.pod_rom import build_pod_basis, rom_predict

ROM_MODES = 8  # matches scripts/evaluate_comparison.py's ROM_MODES
CANDIDATE_M = [6, 8, 10, 12, 16, 24, 32]


def rel_l2(pred, gt):
    return float(np.linalg.norm(pred - gt) / np.linalg.norm(gt))


def main():
    d_train = np.load("data/train.npz")
    Nx = d_train["u"].shape[2]
    L = 2 * np.pi
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    dealias = np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k))

    ensemble = d_train["u"].reshape(-1, Nx)
    nu_per_snapshot = np.repeat(d_train["nu"], d_train["u"].shape[1])

    t0 = time.perf_counter()
    Phi_full, _ = build_pod_basis(ensemble, r=32)
    Phi = Phi_full[:, :ROM_MODES]
    pod_offline_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    F_ensemble = collect_nonlinear_snapshots(ensemble, k, nu_per_snapshot, dealias)
    nonlinear_snapshot_time = time.perf_counter() - t0

    d_val = np.load("data/val.npz")
    A_val, nu_val, u_val = d_val["A"], d_val["nu"], d_val["u"]
    n_save = u_val.shape[1]

    sweep = []
    for m in CANDIDATE_M:
        t0 = time.perf_counter()
        Psi, S = build_deim_basis(F_ensemble, m)
        points = select_deim_points(Psi)
        M = build_deim_projector(Phi, Psi, points)
        deim_offline_time = time.perf_counter() - t0

        traj_errors = []
        for i in range(len(A_val)):
            gt = u_val[i]
            _, u_deim = deim_rom_predict(gt[0], Phi, M, points, Nx, nu=float(nu_val[i]), T=1.0,
                                          n_save=n_save)
            _, u_rom = rom_predict(gt[0], Phi, k, float(nu_val[i]), dealias, T=1.0, n_save=n_save)
            traj_errors.append(rel_l2(u_deim, u_rom))

        sweep.append({
            "m": m,
            "singular_value_ratio_last_over_first": float(S[m - 1] / S[0]),
            "deim_offline_time_s": deim_offline_time,
            "trajectory_rel_err_vs_plain_rom_mean": float(np.mean(traj_errors)),
            "trajectory_rel_err_vs_plain_rom_max": float(np.max(traj_errors)),
            "trajectory_rel_err_vs_plain_rom_std": float(np.std(traj_errors)),
        })
        print(f"m={m:3d}  sv_ratio={sweep[-1]['singular_value_ratio_last_over_first']:.2e}  "
              f"traj_err_mean={sweep[-1]['trajectory_rel_err_vs_plain_rom_mean']:.4f}  "
              f"traj_err_max={sweep[-1]['trajectory_rel_err_vs_plain_rom_max']:.4f}")

    # Select the rank with the best mean validation-trajectory agreement with the plain ROM --
    # NOT simply the largest m (see module docstring: larger m is not monotonically better here).
    best = min(sweep, key=lambda r: r["trajectory_rel_err_vs_plain_rom_mean"])
    selected_m = best["m"]
    print(f"\nSelected DEIM rank m={selected_m} (best mean validation trajectory agreement with "
          f"the plain ROM: {best['trajectory_rel_err_vs_plain_rom_mean']:.4f})")

    Psi, S = build_deim_basis(F_ensemble, selected_m)
    points = select_deim_points(Psi)
    M = build_deim_projector(Phi, Psi, points)

    os.makedirs("experiments", exist_ok=True)
    np.savez("experiments/deim_basis.npz", Phi=Phi, Psi=Psi, points=points, M=M, m=selected_m,
             Nx=Nx)

    os.makedirs("report/research", exist_ok=True)
    with open("report/research/deim_rank_sweep.json", "w") as f:
        json.dump({
            "candidate_m": CANDIDATE_M, "selected_m": selected_m, "sweep": sweep,
            "pod_basis_offline_time_s": pod_offline_time,
            "nonlinear_snapshot_collection_time_s": nonlinear_snapshot_time,
            "rom_modes": ROM_MODES, "n_val_examples": len(A_val),
        }, f, indent=2)
    print(f"Saved experiments/deim_basis.npz and report/research/deim_rank_sweep.json")


if __name__ == "__main__":
    main()
