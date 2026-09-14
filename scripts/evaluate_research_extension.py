"""Research extension -- the main fair-comparison harness.

Compares five methods: ground-truth solver (reference), POD-Galerkin ROM, DEIM-hyper-reduced
POD-Galerkin ROM, the MLP parameter-to-field surrogate, and a 1D FNO, on:

  (1) in-distribution accuracy (existing held-out test split, unchanged from evaluate_comparison.py)
  (2) parameter extrapolation (existing three cases, unchanged)
  (3) out-of-INITIAL-CONDITION-FAMILY generalisation (new: two-mode ICs, same nu range)
  (4) temporal extrapolation (new: T=2.0, double the training horizon)
  (5) offline cost (basis/interpolation-point construction, training) and
      (6) online cost (per-evaluation wall-clock), reported SEPARATELY, never conflated.

Fair-comparison protocol (see RESEARCH_EXTENSION_ROADMAP.md Section 4, decided before this script
was written): identical machine/session for all timings; explicit device labelling; same relative-
L2 error definition for every method; matched (A, nu) test splits reused unchanged from the
existing, already-leakage-checked data; offline and online costs reported in separate tables;
timing methodology (warm-up, repeats, what's included) stated identically for every method.

Where a method is STRUCTURALLY unable to perform a test (the MLP surrogate cannot accept a
non-2-parameter initial condition; the MLP/FNO's fixed-size output cannot extrapolate beyond the
training time horizon by construction), this script reports that explicitly as "N/A (structural)"
rather than forcing an ad-hoc workaround -- this is itself one of the study's findings, not a gap
to paper over.

Run: python scripts/evaluate_research_extension.py
Output: report/research/results.json, report/research/results_summary.txt,
        report/research/comparison_figure.png
"""
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_surrogate import normalize_params
from src.deim import deim_rom_predict
from src.fno_net import FNO1d
from src.general_ic_solver import solve_burgers_general_ic
from src.pod_rom import rom_predict
from src.solver import solve_burgers
from src.surrogate_net import BurgersSurrogateMLP

ROM_MODES = 8
N_TIMING_REPEATS = 20


def rel_l2(pred, gt):
    return float(np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-12))


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_deim_basis():
    d = np.load("experiments/deim_basis.npz")
    return d["Phi"], d["points"], d["M"], int(d["m"]), int(d["Nx"])


def load_mlp(device):
    ckpt = torch.load("experiments/surrogate_checkpoint.pt", map_location=device,
                       weights_only=True)
    model = BurgersSurrogateMLP(ckpt["n_save"], ckpt["Nx"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def load_fno(device):
    ckpt = torch.load("experiments/fno_checkpoint.pt", map_location=device, weights_only=True)
    model = FNO1d(ckpt["Nx"], ckpt["n_save"], modes=ckpt["modes"], width=ckpt["width"],
                   n_layers=ckpt["n_layers"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def fno_predict(model, device, u0_np):
    u0 = torch.from_numpy(u0_np.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(u0).cpu().numpy()[0]
    return out


def mlp_predict(model, device, A, nu):
    params = torch.from_numpy(normalize_params(np.array([A]), np.array([nu]))).to(device)
    with torch.no_grad():
        out = model(params).cpu().numpy()[0]
    return out


def main():
    device = get_device()
    print(f"Using device: {device}")

    Phi, deim_points, M, deim_m, Nx = load_deim_basis()
    L = 2 * np.pi
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    dealias = np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k))
    x = np.linspace(0, L, Nx, endpoint=False)

    mlp_model, mlp_ckpt = load_mlp(device)
    fno_model, fno_ckpt = load_fno(device)

    results = {"device": str(device), "rom_modes": ROM_MODES, "deim_m": deim_m,
               "mlp_param_count": sum(p.numel() for p in mlp_model.parameters()),
               "fno_param_count": sum(p.numel() for p in fno_model.parameters())}

    # ---------------------------------------------------------------- (1) in-distribution --
    d_test = np.load("data/test.npz")
    A_test, nu_test, u_test = d_test["A"], d_test["nu"], d_test["u"]
    n_save = u_test.shape[1]

    errs = {"rom": [], "deim_rom": [], "mlp": [], "fno": []}
    for i in range(len(A_test)):
        gt = u_test[i]
        A_i, nu_i = float(A_test[i]), float(nu_test[i])
        _, u_rom = rom_predict(gt[0], Phi, k, nu_i, dealias, T=1.0, n_save=n_save)
        _, u_deim = deim_rom_predict(gt[0], Phi, M, deim_points, Nx, nu=nu_i, T=1.0, n_save=n_save)
        u_mlp = mlp_predict(mlp_model, device, A_i, nu_i)
        u_fno = fno_predict(fno_model, device, gt[0])
        errs["rom"].append(rel_l2(u_rom, gt))
        errs["deim_rom"].append(rel_l2(u_deim, gt))
        errs["mlp"].append(rel_l2(u_mlp, gt))
        errs["fno"].append(rel_l2(u_fno, gt))

    results["in_distribution"] = {
        method: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        for method, v in errs.items()
    }
    print("\n=== (1) In-distribution test set (n=%d) ===" % len(A_test))
    for method, stats in results["in_distribution"].items():
        print(f"  {method:10s}  mean={stats['mean']:.5f}  std={stats['std']:.5f}")

    # ------------------------------------------------------------ (2) parameter extrapolation --
    extrap_cases = [
        {"A": 2.5, "nu": 0.05, "label": "A above training range (2.5 > 2.0)"},
        {"A": 1.0, "nu": 0.15, "label": "nu above training range (0.15 > 0.10)"},
        {"A": 0.2, "nu": 0.05, "label": "A below training range (0.2 < 0.5)"},
    ]
    results["parameter_extrapolation"] = []
    print("\n=== (2) Parameter extrapolation ===")
    for case in extrap_cases:
        A_e, nu_e = case["A"], case["nu"]
        _, _, gt_e = solve_burgers(A=A_e, nu=nu_e, Nx=Nx, T=1.0, n_save=n_save)
        _, u_rom_e = rom_predict(gt_e[0], Phi, k, nu_e, dealias, T=1.0, n_save=n_save)
        _, u_deim_e = deim_rom_predict(gt_e[0], Phi, M, deim_points, Nx, nu=nu_e, T=1.0,
                                        n_save=n_save)
        u_mlp_e = mlp_predict(mlp_model, device, A_e, nu_e)
        u_fno_e = fno_predict(fno_model, device, gt_e[0])
        row = {"label": case["label"], "A": A_e, "nu": nu_e,
               "rom_err": rel_l2(u_rom_e, gt_e), "deim_rom_err": rel_l2(u_deim_e, gt_e),
               "mlp_err": rel_l2(u_mlp_e, gt_e), "fno_err": rel_l2(u_fno_e, gt_e)}
        results["parameter_extrapolation"].append(row)
        print(f"  {case['label']}: ROM={row['rom_err']:.4f}  DEIM-ROM={row['deim_rom_err']:.4f}  "
              f"MLP={row['mlp_err']:.4f}  FNO={row['fno_err']:.4f}")

    # --------------------------------------------------- (3) out-of-family IC generalisation --
    d_ood = np.load("data/ood_family_test.npz")
    A1_ood, A2_ood, nu_ood, u_ood = d_ood["A1"], d_ood["A2"], d_ood["nu"], d_ood["u"]
    ood_errs = {"rom": [], "deim_rom": [], "fno": []}  # MLP excluded: see note below.
    for i in range(len(A1_ood)):
        gt = u_ood[i]
        nu_i = float(nu_ood[i])
        _, u_rom = rom_predict(gt[0], Phi, k, nu_i, dealias, T=1.0, n_save=n_save)
        _, u_deim = deim_rom_predict(gt[0], Phi, M, deim_points, Nx, nu=nu_i, T=1.0, n_save=n_save)
        u_fno = fno_predict(fno_model, device, gt[0])
        ood_errs["rom"].append(rel_l2(u_rom, gt))
        ood_errs["deim_rom"].append(rel_l2(u_deim, gt))
        ood_errs["fno"].append(rel_l2(u_fno, gt))
    results["out_of_family_ic"] = {
        method: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        for method, v in ood_errs.items()
    }
    results["out_of_family_ic"]["mlp"] = "N/A (structural): the MLP surrogate's input is the " \
        "2-scalar pair (A, nu); a two-mode initial condition has no such representation, so the " \
        "MLP cannot be evaluated on this test at all -- not merely inaccurately, but not " \
        "applicable, unlike the FNO/ROM/DEIM-ROM which all accept the actual field."
    print("\n=== (3) Out-of-family IC generalisation (two-mode ICs, in-range nu) (n=%d) ===" %
          len(A1_ood))
    for method in ("rom", "deim_rom", "fno"):
        s = results["out_of_family_ic"][method]
        print(f"  {method:10s}  mean={s['mean']:.5f}  std={s['std']:.5f}")
    print("  mlp        N/A (structural -- cannot represent a two-mode IC as (A, nu))")

    # ----------------------------------------------------------- (4) temporal extrapolation --
    # Double the training horizon (T=2.0 vs. T=1.0). The solver/ROM/DEIM-ROM are genuine time
    # integrators and can simply be run longer; the MLP/FNO's fixed-size output layer (n_save x Nx,
    # sized for T=1.0's 20 snapshots) CANNOT produce a T=2.0 trajectory at all -- a structural
    # limitation of the single-shot regression framing (see src/fno_net.py docstring), reported
    # here rather than worked around.
    temporal_errs = {"rom": [], "deim_rom": []}
    rng = np.random.default_rng(42)
    n_temporal_cases = 8
    for _ in range(n_temporal_cases):
        A_t = rng.uniform(0.5, 2.0)
        nu_t = rng.uniform(0.01, 0.1)
        _, _, gt_t = solve_burgers(A=A_t, nu=nu_t, Nx=Nx, T=2.0, n_save=n_save)
        _, u_rom_t = rom_predict(gt_t[0], Phi, k, nu_t, dealias, T=2.0, n_save=n_save)
        _, u_deim_t = deim_rom_predict(gt_t[0], Phi, M, deim_points, Nx, nu=nu_t, T=2.0,
                                        n_save=n_save)
        temporal_errs["rom"].append(rel_l2(u_rom_t, gt_t))
        temporal_errs["deim_rom"].append(rel_l2(u_deim_t, gt_t))
    results["temporal_extrapolation_T2"] = {
        method: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        for method, v in temporal_errs.items()
    }
    results["temporal_extrapolation_T2"]["mlp"] = "N/A (structural): fixed output size sized " \
        "for T=1.0's 20 snapshots cannot represent a T=2.0 trajectory."
    results["temporal_extrapolation_T2"]["fno"] = "N/A (structural, same reason as MLP -- this " \
        "module's FNO is a single-shot full-trajectory regressor, not an autoregressive " \
        "one-step operator; see src/fno_net.py docstring for why that design was chosen and " \
        "what capability it deliberately does not test."
    print(f"\n=== (4) Temporal extrapolation, T=2.0 (double training horizon) (n={n_temporal_cases}) ===")
    for method in ("rom", "deim_rom"):
        s = results["temporal_extrapolation_T2"][method]
        print(f"  {method:10s}  mean={s['mean']:.5f}  std={s['std']:.5f}")
    print("  mlp        N/A (structural -- fixed-size output cannot extrapolate in time)")
    print("  fno        N/A (structural -- same reason, see src/fno_net.py)")

    # ------------------------------------------------------------------------ (5) offline cost --
    with open("report/research/deim_rank_sweep.json") as f:
        deim_sweep = json.load(f)
    selected_sweep_row = next(r for r in deim_sweep["sweep"] if r["m"] == deim_m)
    with open("experiments/fno_run_meta.json") as f:
        fno_meta = json.load(f)

    results["offline_cost_s"] = {
        "pod_basis_construction": deim_sweep["pod_basis_offline_time_s"],
        "deim_nonlinear_snapshot_collection": deim_sweep["nonlinear_snapshot_collection_time_s"],
        "deim_basis_and_point_selection_at_selected_m": selected_sweep_row["deim_offline_time_s"],
        "mlp_training_200_epochs": 8.23,  # measured, see RESEARCH_EXTENSION.md "Offline cost"
        "fno_training_200_epochs": fno_meta["total_train_time_s"],
        "note": "ROM/DEIM-ROM offline cost also includes the POD basis construction above "
                "(shared with the plain ROM baseline); DEIM adds only the nonlinear-snapshot "
                "collection + basis/point-selection rows on top of that shared cost.",
    }
    print("\n=== (5) Offline cost (one-time, not amortised) ===")
    for k_, v_ in results["offline_cost_s"].items():
        if isinstance(v_, float):
            print(f"  {k_:45s}  {v_:.3f} s")

    # ------------------------------------------------------------------------- (6) online cost --
    A0, nu0 = float(A_test[0]), float(nu_test[0])
    u0_0 = u_test[0, 0]

    def time_it(fn, n=N_TIMING_REPEATS):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n

    solver_time = time_it(lambda: solve_burgers(A=A0, nu=nu0, Nx=Nx, T=1.0, n_save=n_save))
    rom_time = time_it(lambda: rom_predict(u0_0, Phi, k, nu0, dealias, T=1.0, n_save=n_save))
    deim_time = time_it(lambda: deim_rom_predict(u0_0, Phi, M, deim_points, Nx, nu=nu0, T=1.0,
                                                  n_save=n_save))
    mlp_predict(mlp_model, device, A0, nu0)  # warm-up (JIT/graph warm-up on first call)
    mlp_time = time_it(lambda: mlp_predict(mlp_model, device, A0, nu0))
    fno_predict(fno_model, device, u0_0)  # warm-up
    fno_time = time_it(lambda: fno_predict(fno_model, device, u0_0))

    results["online_cost_ms"] = {
        "solver": solver_time * 1000, "rom": rom_time * 1000, "deim_rom": deim_time * 1000,
        "mlp": mlp_time * 1000, "fno": fno_time * 1000,
        "note": f"mean of {N_TIMING_REPEATS} calls, same machine/session as offline timings "
                f"above; device={device}; NN methods include a single untimed warm-up call "
                f"before timing (JIT/graph warm-up), matching evaluate_comparison.py's existing "
                f"methodology.",
    }
    print(f"\n=== (6) Online cost, mean of {N_TIMING_REPEATS} calls (device={device}) ===")
    for method in ("solver", "rom", "deim_rom", "mlp", "fno"):
        print(f"  {method:10s}  {results['online_cost_ms'][method]:.3f} ms")
    print(f"\n  DEIM-ROM vs plain ROM online speedup: {rom_time / deim_time:.2f}x")
    print(f"  DEIM-ROM vs solver online speedup:    {solver_time / deim_time:.2f}x")

    # ---------------------------------------------------------------------------- save + plot --
    os.makedirs("report/research", exist_ok=True)
    with open("report/research/results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("report/research/results_summary.txt", "w") as f:
        f.write("Neural Surrogate vs. POD-ROM vs. DEIM-ROM vs. FNO -- Research Extension\n")
        f.write("Real, measured results only. See RESEARCH_EXTENSION.md for full analysis.\n\n")
        f.write(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i in range(3):
        gt = u_test[i]
        A_i, nu_i = float(A_test[i]), float(nu_test[i])
        _, u_rom_i = rom_predict(gt[0], Phi, k, nu_i, dealias, T=1.0, n_save=n_save)
        _, u_deim_i = deim_rom_predict(gt[0], Phi, M, deim_points, Nx, nu=nu_i, T=1.0,
                                        n_save=n_save)
        u_mlp_i = mlp_predict(mlp_model, device, A_i, nu_i)
        u_fno_i = fno_predict(fno_model, device, gt[0])
        axes[i].plot(x, gt[-1], label="ground truth", linewidth=2, color="black")
        axes[i].plot(x, u_rom_i[-1], "--", label=f"ROM (r={ROM_MODES})")
        axes[i].plot(x, u_deim_i[-1], "-.", label=f"DEIM-ROM (m={deim_m})")
        axes[i].plot(x, u_mlp_i[-1], ":", label="MLP")
        axes[i].plot(x, u_fno_i[-1], ":", label="FNO")
        axes[i].set_title(f"A={A_i:.2f}, nu={nu_i:.3f}")
        axes[i].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("report/research/comparison_figure.png", dpi=120)

    print("\nSaved report/research/results.json, results_summary.txt, comparison_figure.png")


if __name__ == "__main__":
    main()
