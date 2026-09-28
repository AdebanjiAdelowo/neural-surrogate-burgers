"""DEIM/local-FD versus full-solver runtime at Nx = 128 under the installed NumPy version.

The full solver spends most of its time in FFTs; the DEIM/local-FD reduced model uses none. Their
runtime ratio therefore depends on the FFT implementation of the installed NumPy. This script repeats
the 40-case test-set timing of `scripts/hyperreduction_followup.py policy` (time-aligned snapshots,
interleaved repetitions, median per case) through both the solver-only path and the public
`deim_rom_predict` call, and records a bare FFT timing, so results from different NumPy versions can
be compared directly. Accuracy is not measured here (it does not depend on the NumPy version; see
report/audit/followup_policy_run1.json and followup_production_validation_run1.json).

Run: python scripts/timing_numpy_sensitivity.py [--reps 7]
Writes report/audit/timing_numpy_<version>.json
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.hyperreduction_followup as F  # noqa: E402
from scripts.hyperreduction_benchmark import interleaved_timing  # noqa: E402
from src.deim import deim_rom_predict  # noqa: E402
from src.solver import solve_burgers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fft_time_us(n=F.NX, reps=20000):
    x = np.random.default_rng(0).standard_normal(n)
    np.fft.fft(x)
    t0 = time.perf_counter()
    for _ in range(reps):
        np.fft.ifft(np.fft.fft(x))
    return 1e6 * (time.perf_counter() - t0) / reps


def main(reps):
    import scipy
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    env = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "machine": platform.machine(), "os": platform.platform(), "git_head": head,
           "loadavg_start": os.getloadavg()}
    data = F.load_data()
    C = F.stored_ctx()
    tA, tnu, tU = data["test"]
    per_case = {}
    for i in range(len(tA)):
        A, nu, u0 = float(tA[i]), float(tnu[i]), tU[i, 0]
        u0max = np.abs(u0).max()
        dt_hist = F.dt_policy("A", "deim", u0max, nu, F.NX, C)
        dt_rev = F.dt_policy("R", "deim", u0max, nu, F.NX, C)
        calls = {
            "fom": lambda: solve_burgers(A=A, nu=nu, Nx=F.NX, T=F.T_FINAL, n_save=F.N_SAVE,
                                         align_snapshots=True),
            "deim_solver_only_historical_rule": lambda: F.run_red("deim", u0, nu, dt_hist, C),
            "deim_solver_only_revised_rule": lambda: F.run_red("deim", u0, nu, dt_rev, C),
            "deim_predict_default": lambda: deim_rom_predict(u0, C["Phi"], C["M"], C["pts"], F.NX, nu=nu,
                                                             T=F.T_FINAL, n_save=F.N_SAVE,
                                                             align_snapshots=True),
        }
        t = interleaved_timing(calls, reps=reps, warmup=1)
        for name, v in t.items():
            per_case.setdefault(name, []).append(float(np.median(v)))
    fom = np.array(per_case["fom"])
    summary = {"workload_mean_ms": {n: 1e3 * float(np.mean(v)) for n, v in per_case.items()}, "speedup_vs_fom": {}}
    for name, v in per_case.items():
        if name == "fom":
            continue
        ratio = fom / np.array(v)
        summary["speedup_vs_fom"][name] = {"workload": float(fom.sum() / np.sum(v)), "per_case_min": float(ratio.min()),
                                           "per_case_max": float(ratio.max()),
                                           "cases_faster_than_fom": int((ratio > 1).sum()), "n": len(ratio)}
    env["loadavg_end"] = os.getloadavg()
    out = {"description": __doc__.strip().split("\n")[0], "config": {"Nx": F.NX, "r": F.R_MODES, "m": F.M_POINTS,
                                                                      "reps": reps, "cases": len(tA)},
           "fft_plus_ifft_us_n128": fft_time_us(), "summary": summary, "per_case_median_s": per_case, "env": env}
    path = os.path.join(ROOT, "report", "audit", f"timing_numpy_{np.__version__}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"numpy {np.__version__}: FFT+IFFT {out['fft_plus_ifft_us_n128']:.2f} us")
    for n, s in summary["speedup_vs_fom"].items():
        print(f"  {n:34s} {s['workload']:.2f}x (per case {s['per_case_min']:.2f} to {s['per_case_max']:.2f}, "
              f"faster in {s['cases_faster_than_fom']}/{s['n']})")
    print("saved", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    main(ap.parse_args().reps)
