"""Research extension -- rigorous timing verification (review response, see RESEARCH_EXTENSION.md
"Timing verification").

Fixes and checks applied here that scripts/evaluate_research_extension.py's original timing did
NOT do:
  1. torch.mps.synchronize() before stopping the timer for MLP/FNO -- MPS dispatches operations
     asynchronously, so a CPU-side timer stopped without synchronizing can under-report GPU time.
  2. Multiple independent timing TRIALS (not just one mean-of-N-calls), reporting median and
     inter-trial variability, not a single point estimate.
  3. A component-level breakdown of DEIM-ROM's online cost (via cProfile), to identify where time
     is actually spent, not just the total.

Run: python scripts/verify_timing.py [--device D] [--checkpoint-dir DIR] [--out PATH]
Output: report/research/timing_verification.json (the historical Apple MPS result; a CUDA run must
        write elsewhere via --out, see REMOTE_GPU.md)
"""
import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_surrogate import normalize_params
from src.deim import deim_rom_predict
from src.device import add_device_argument, device_info, select_device, synchronize
from src.fno_net import FNO1d
from src.pod_rom import rom_predict
from src.solver import solve_burgers
from src.surrogate_net import BurgersSurrogateMLP

# Historical protocol: this script reproduces published numbers, so it pins the historical time-step
# rule (the predictors' default is now the reduced-operator policy; see src/timestep.py).
from functools import partial  # noqa: E402

deim_rom_predict = partial(deim_rom_predict, timestep_policy="historical")
rom_predict = partial(rom_predict, timestep_policy="historical")

N_CALLS_PER_TRIAL = 20
N_TRIALS = 7
HISTORICAL_OUT = "report/research/timing_verification.json"


def time_trials(fn, n_calls=N_CALLS_PER_TRIAL, n_trials=N_TRIALS, sync_fn=None):
    """Returns per-trial mean-of-n_calls timings (ms), across n_trials independent trials."""
    trial_means = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            fn()
        if sync_fn is not None:
            sync_fn()
        trial_means.append((time.perf_counter() - t0) / n_calls * 1000)
    return trial_means


def main(device_name="auto", checkpoint_dir="experiments", out_path=HISTORICAL_OUT):
    device = select_device(device_name)
    if device.type == "cuda" and os.path.normpath(out_path) == HISTORICAL_OUT:
        raise SystemExit(f"refusing to overwrite the historical {HISTORICAL_OUT} with a CUDA run; "
                         "pass --out (see REMOTE_GPU.md)")
    # MPS and CUDA both dispatch asynchronously; CPU needs no synchronization.
    nn_sync = (lambda: synchronize(device)) if device.type in ("mps", "cuda") else None
    print(f"Device: {device}  ({device.type.upper() + ' sync enabled' if nn_sync else 'sync not applicable (CPU)'})")

    d = np.load("experiments/deim_basis.npz")
    Phi, points, M, m, Nx = d["Phi"], d["points"], d["M"], int(d["m"]), int(d["Nx"])
    L = 2 * np.pi
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    dealias = np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k))

    d_test = np.load("data/test.npz")
    A0, nu0 = float(d_test["A"][0]), float(d_test["nu"][0])
    u0_0 = d_test["u"][0, 0]
    n_save = d_test["u"].shape[1]

    ckpt = torch.load(os.path.join(checkpoint_dir, "surrogate_checkpoint.pt"), map_location=device,
                       weights_only=True)
    mlp = BurgersSurrogateMLP(ckpt["n_save"], ckpt["Nx"]).to(device)
    mlp.load_state_dict(ckpt["model_state"])
    mlp.eval()

    fno_ckpt = torch.load(os.path.join(checkpoint_dir, "fno_checkpoint.pt"), map_location=device,
                           weights_only=True)
    fno = FNO1d(fno_ckpt["Nx"], fno_ckpt["n_save"], modes=fno_ckpt["modes"],
                width=fno_ckpt["width"], n_layers=fno_ckpt["n_layers"]).to(device)
    fno.load_state_dict(fno_ckpt["model_state"])
    fno.eval()

    results = {"device": str(device), "device_info": device_info(device), "n_calls_per_trial": N_CALLS_PER_TRIAL,
               "n_trials": N_TRIALS}

    # ---- (1) repeated-trial timing, all five methods, device-synchronized for NN methods ----
    def solver_call():
        solve_burgers(A=A0, nu=nu0, Nx=Nx, T=1.0, n_save=n_save)

    def rom_call():
        rom_predict(u0_0, Phi, k, nu0, dealias, T=1.0, n_save=n_save)

    def deim_call():
        deim_rom_predict(u0_0, Phi, M, points, Nx, nu=nu0, T=1.0, n_save=n_save)

    params0 = torch.from_numpy(normalize_params(np.array([A0]), np.array([nu0]))).to(device)

    # Matches scripts/evaluate_research_extension.py's mlp_predict/fno_predict exactly (including
    # the .cpu().numpy() transfer a real caller needs to use the result) -- an earlier version of
    # this script omitted that transfer and reported a misleadingly fast number; a follow-up check
    # (see RESEARCH_EXTENSION.md "Timing verification") found .cpu().numpy() already forces an
    # implicit MPS sync (explicit torch.mps.synchronize() changes the measured time by <15%, well
    # within run-to-run noise), so the ORIGINAL report's MLP/FNO numbers were not meaningfully
    # biased by a missing synchronization call -- that specific suspicion did not hold up.
    def mlp_call():
        with torch.no_grad():
            mlp(params0).cpu().numpy()[0]

    u0_t = torch.from_numpy(u0_0.astype(np.float32)).unsqueeze(0).to(device)

    def fno_call():
        with torch.no_grad():
            fno(u0_t).cpu().numpy()[0]

    # warm-up (JIT/graph warm-up), then synchronize before starting the real trials
    mlp_call()
    fno_call()
    if nn_sync:
        nn_sync()

    for name, fn, sync in [("solver", solver_call, None), ("rom", rom_call, None),
                            ("deim_rom", deim_call, None), ("mlp", mlp_call, nn_sync),
                            ("fno", fno_call, nn_sync)]:
        trials = time_trials(fn, sync_fn=sync)
        results.setdefault("timing_ms", {})[name] = {
            "trial_means": trials, "median": float(np.median(trials)),
            "mean_of_trials": float(np.mean(trials)), "std_of_trials": float(np.std(trials)),
            "min": float(np.min(trials)), "max": float(np.max(trials)),
        }
        print(f"{name:10s}  median={np.median(trials):.4f} ms  "
              f"std={np.std(trials):.4f}  (n_trials={N_TRIALS}, {N_CALLS_PER_TRIAL} calls/trial)")

    med = {k_: v_["median"] for k_, v_ in results["timing_ms"].items()}
    results["speedups"] = {
        "deim_rom_vs_rom": med["rom"] / med["deim_rom"],
        "deim_rom_vs_solver": med["solver"] / med["deim_rom"],
        "mlp_vs_solver": med["solver"] / med["mlp"],
        "fno_vs_solver": med["solver"] / med["fno"],
    }
    print("\nMedian-based speedups:")
    for k_, v_ in results["speedups"].items():
        print(f"  {k_}: {v_:.2f}x")

    # ---- (2) component-level breakdown of DEIM-ROM online cost (cProfile) ----
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(N_CALLS_PER_TRIAL):
        deim_rom_predict(u0_0, Phi, M, points, Nx, nu=nu0, T=1.0, n_save=n_save)
    pr.disable()
    s = io.StringIO()
    # strip_dirs() removes the full filesystem path prefix (this machine's home directory,
    # Python environment site-packages path, etc.), leaving only relative-looking basenames --
    # avoids baking machine-specific absolute paths into a committed results file.
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")
    ps.print_stats(12)
    print("\n=== DEIM-ROM component breakdown (cProfile, 20 calls) ===")
    print(s.getvalue())
    results["deim_rom_profile_text"] = s.getvalue()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_device_argument(parser)
    parser.add_argument("--checkpoint-dir", default="experiments")
    parser.add_argument("--out", default=HISTORICAL_OUT)
    args = parser.parse_args()
    main(args.device, args.checkpoint_dir, args.out)
