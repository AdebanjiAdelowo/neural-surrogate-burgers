"""Research extension -- head-to-head comparison of the 2nd-order (baseline) vs. 4th-order local
finite-difference stencil used for the DEIM nonlinear-term evaluation at the interpolation points.

Tests the specific question raised during review: can the steep-gradient accuracy loss identified
for the 2nd-order stencil (see RESEARCH_EXTENSION.md "Failure-mechanism investigation") be reduced
by a higher-order local stencil, without destroying DEIM's online speed advantage?

Same POD rank (r=8), DEIM rank (m=16, from experiments/deim_basis.npz), datasets, and test regimes
as scripts/evaluate_research_extension.py -- only the stencil order varies.

Run: python scripts/compare_deim_stencil_orders.py
Output: report/research/stencil_order_comparison.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deim import deim_rom_predict
from src.general_ic_solver import solve_burgers_general_ic
from src.solver import solve_burgers

# Historical protocol: this script reproduces published numbers, so it pins the historical time-step
# rule (the predictors' default is now the reduced-operator policy; see src/timestep.py).
from functools import partial  # noqa: E402

deim_rom_predict = partial(deim_rom_predict, timestep_policy="historical")

N_TIMING_REPEATS = 20


def rel_l2(pred, gt):
    return float(np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-12))


def main():
    d = np.load("experiments/deim_basis.npz")
    Phi, points, M, m, Nx = d["Phi"], d["points"], d["M"], int(d["m"]), int(d["Nx"])

    results = {"deim_m": m, "orders_compared": [2, 4]}

    # ---- (1) in-distribution test set ----
    d_test = np.load("data/test.npz")
    A_test, nu_test, u_test = d_test["A"], d_test["nu"], d_test["u"]
    n_save = u_test.shape[1]

    for order in (2, 4):
        errs = []
        for i in range(len(A_test)):
            gt = u_test[i]
            nu_i = float(nu_test[i])
            _, u_deim = deim_rom_predict(gt[0], Phi, M, points, Nx, nu=nu_i, T=1.0, n_save=n_save,
                                          stencil_order=order)
            errs.append(rel_l2(u_deim, gt))
        results.setdefault("in_distribution", {})[f"order_{order}"] = {
            "mean": float(np.mean(errs)), "std": float(np.std(errs)), "max": float(np.max(errs)),
            "n": len(errs),
        }
        print(f"[in-distribution] order={order}: mean={np.mean(errs):.4f} "
              f"max={np.max(errs):.4f}")

    # ---- (2) parameter extrapolation (existing 3 cases) ----
    extrap_cases = [
        {"A": 2.5, "nu": 0.05, "label": "A above training range (2.5 > 2.0)"},
        {"A": 1.0, "nu": 0.15, "label": "nu above training range (0.15 > 0.10)"},
        {"A": 0.2, "nu": 0.05, "label": "A below training range (0.2 < 0.5)"},
    ]
    results["parameter_extrapolation"] = []
    for case in extrap_cases:
        A_e, nu_e = case["A"], case["nu"]
        _, _, gt_e = solve_burgers(A=A_e, nu=nu_e, Nx=Nx, T=1.0, n_save=n_save)
        row = {"label": case["label"]}
        for order in (2, 4):
            _, u_deim_e = deim_rom_predict(gt_e[0], Phi, M, points, Nx, nu=nu_e, T=1.0,
                                            n_save=n_save, stencil_order=order)
            row[f"order_{order}_err"] = rel_l2(u_deim_e, gt_e)
        results["parameter_extrapolation"].append(row)
        print(f"[extrap] {case['label']}: order2={row['order_2_err']:.4f} "
              f"order4={row['order_4_err']:.4f}")

    # ---- (3) out-of-family IC ----
    d_ood = np.load("data/ood_family_test.npz")
    nu_ood, u_ood = d_ood["nu"], d_ood["u"]
    for order in (2, 4):
        errs = []
        for i in range(len(nu_ood)):
            gt = u_ood[i]
            nu_i = float(nu_ood[i])
            _, u_deim = deim_rom_predict(gt[0], Phi, M, points, Nx, nu=nu_i, T=1.0, n_save=n_save,
                                          stencil_order=order)
            errs.append(rel_l2(u_deim, gt))
        results.setdefault("out_of_family_ic", {})[f"order_{order}"] = {
            "mean": float(np.mean(errs)), "std": float(np.std(errs)), "max": float(np.max(errs)),
            "n": len(errs),
        }
        print(f"[out-of-family IC] order={order}: mean={np.mean(errs):.4f} max={np.max(errs):.4f}")

    # ---- (4) the specific worst-case steep-gradient examples identified in review ----
    worst_cases = [
        {"A": 1.889, "nu": 0.0158, "label": "worst in-distribution case (A=1.889, nu=0.0158)"},
        {"A": 1.503, "nu": 0.0122, "label": "2nd-worst (A=1.503, nu=0.0122)"},
    ]
    results["worst_case_steep_gradient"] = []
    for case in worst_cases:
        A_w, nu_w = case["A"], case["nu"]
        _, _, gt_w = solve_burgers(A=A_w, nu=nu_w, Nx=Nx, T=1.0, n_save=n_save)
        row = {"label": case["label"]}
        for order in (2, 4):
            _, u_deim_w = deim_rom_predict(gt_w[0], Phi, M, points, Nx, nu=nu_w, T=1.0,
                                            n_save=n_save, stencil_order=order)
            row[f"order_{order}_err"] = rel_l2(u_deim_w, gt_w)
        results["worst_case_steep_gradient"].append(row)
        print(f"[worst-case] {case['label']}: order2={row['order_2_err']:.4f} "
              f"order4={row['order_4_err']:.4f}")

    # ---- (5) online timing (does the wider 4th-order stencil erode the speedup?) ----
    A0, nu0 = float(A_test[0]), float(nu_test[0])
    u0_0 = u_test[0, 0]

    def time_it(fn, n=N_TIMING_REPEATS):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n

    for order in (2, 4):
        t_ms = time_it(lambda: deim_rom_predict(u0_0, Phi, M, points, Nx, nu=nu0, T=1.0,
                                                 n_save=n_save, stencil_order=order)) * 1000
        results.setdefault("online_cost_ms", {})[f"order_{order}"] = t_ms
        print(f"[timing] order={order}: {t_ms:.3f} ms")

    os.makedirs("report/research", exist_ok=True)
    with open("report/research/stencil_order_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved report/research/stencil_order_comparison.json")


if __name__ == "__main__":
    main()
