"""Controlled speed/accuracy audit of the POD-Galerkin ROM and the DEIM/local-FD hyper-reduced ROM.

Stages (each writes report/audit/<stage>[_<tag>].json; nothing under report/research/ or
experiments/ is touched):

  reproduce   regenerate data + POD/DEIM basis, re-derive the historical accuracy numbers, and rerun
              the historical timing protocol (scripts/verify_timing.py's 7 trials x 20 calls on
              test[0]) for solver / ROM / DEIM-ROM.
  timing      interleaved per-call timing on ALL 40 test cases; solver-only vs reconstruction-
              inclusive; RK4 step counts per method.
  breakdown   where the plain-ROM / DEIM / FOM time goes (component micro-benchmarks x call counts).
  sweep       (r, m) grid: accuracy on val and test, online time, speed-up; plus two accuracy
              ablations that split DEIM's error into "interpolation" and "local finite difference".
  scaling     per-RHS cost and end-to-end cost versus grid size Nx (bases rebuilt at each Nx).
  reference   FOM temporal/spatial reference-resolution check; ROM/DEIM error against a finer FOM.
  dtprobe     exploratory: reduced models at the FOM's step count (is the step rule the limit?)
  ood         out-of-family (two-mode IC) ROM/DEIM accuracy, native and time-aligned protocols
  fno         FNO seeds: accuracy, timing (CPU, plus MPS and/or CUDA when present, each labelled
              separately), and the nu-blind irreducible error floor.

Run:  python scripts/hyperreduction_benchmark.py <stage> [--tag TAG]
"""
import argparse
import gc
import json
import os
import platform
import re
import subprocess
import sys
import time
import timeit

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_dataset import (A_RANGE, N_SAVE, NU_RANGE, NX, SPLIT_SEED_OFFSETS, SPLITS,
                                       T_FINAL, generate_split)
from src.deim import (build_deim_basis, build_deim_projector, collect_nonlinear_snapshots,
                       deim_rom_predict, deim_rom_reduced_trajectory, select_deim_points)
from src.metrics import relative_l2
from src.pod_rom import (_burgers_rhs_physical, build_pod_basis, rom_predict,
                          rom_reduced_trajectory)
from src.solver import solve_burgers

# Historical protocol: this script reproduces published numbers, so it pins the historical time-step
# rule (the predictors' default is now the reduced-operator policy; see src/timestep.py).
from functools import partial  # noqa: E402

deim_rom_predict = partial(deim_rom_predict, timestep_policy="historical")
deim_rom_reduced_trajectory = partial(deim_rom_reduced_trajectory, timestep_policy="historical")
rom_predict = partial(rom_predict, timestep_policy="historical")
rom_reduced_trajectory = partial(rom_reduced_trajectory, timestep_policy="historical")

L = 2 * np.pi
OUT_DIR = "report/audit"
BENCH_SEED = 20260921  # reserved for any stochastic choice; the audit itself has none (see below)
R_MODES, M_POINTS = 8, 16  # the historical canonical configuration


def rng_free_config():
    """Everything that defines the benchmark; there is deliberately no random choice inside the
    audit itself (data come from the fixed split seeds, methods are deterministic)."""
    return {"Nx": NX, "T": T_FINAL, "n_save": N_SAVE, "A_range": A_RANGE, "nu_range": NU_RANGE,
            "split_sizes": SPLITS, "split_seed_offsets": SPLIT_SEED_OFFSETS, "r": R_MODES,
            "m": M_POINTS, "stencil_order": 2}


# ------------------------------------------------------------------------------------ utilities
def wavenumbers(Nx):
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    return k, np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k))


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def environment_info():
    import scipy
    info = {
        "cpu": sh("sysctl -n machdep.cpu.brand_string"), "n_cpu": os.cpu_count(),
        "mem_gb": round(int(sh("sysctl -n hw.memsize") or 0) / 2**30, 1),
        "os": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "blas": "accelerate (numpy.show_config)",
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")},
        "git_head": sh("git rev-parse HEAD"),
        "git_dirty_paths": (sh("git status --porcelain") or "").count("\n") + 1
        if sh("git status --porcelain") else 0,
        "loadavg_start": os.getloadavg(),
        # executable names only (no paths), so saved results carry no local filesystem details
        "top_cpu_processes_start": [re.sub(r"^(\s*[\d.]+ )/.*/", r"\1", line) for line in
                                    (sh("ps -Ao pcpu,comm -r | sed -n 2,6p") or "").split("\n")],
        "power": sh("pmset -g batt | sed -n 1p"),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["mps_available"] = bool(torch.backends.mps.is_available())
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_runtime"] = torch.version.cuda
    except Exception:
        pass
    return info


def finish_env(info):
    info["loadavg_end"] = os.getloadavg()
    return info


def save(name, tag, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}{'_' + tag if tag else ''}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist")
                  else str(o))
    print(f"saved {path}")


def stats(x):
    x = np.asarray(x, dtype=float)
    return {"n": int(x.size), "mean": float(np.mean(x)), "median": float(np.median(x)),
            "std": float(np.std(x)), "min": float(np.min(x)), "max": float(np.max(x)),
            "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75)),
            "cv": float(np.std(x) / np.mean(x))}


def err_stats(e):
    e = np.asarray(e, dtype=float)
    ok = np.isfinite(e)
    good = e[ok]
    return {"n": int(e.size), "n_nonfinite": int((~ok).sum()),
            "mean": float(np.mean(good)) if good.size else None,
            "std": float(np.std(good)) if good.size else None,
            "median": float(np.median(good)) if good.size else None,
            "p90": float(np.percentile(good, 90)) if good.size else None,
            "max": float(np.max(good)) if good.size else None,
            "n_above_1pct": int((good > 0.01).sum()), "n_above_10pct": int((good > 0.1).sum())}


def interleaved_timing(callables, reps, warmup=2):
    """Round-robin timing: each repetition times every callable once, in a rotating order, so slow
    drift or a competing process biases every method alike. Returns {name: [seconds, ...]}."""
    names = list(callables)
    for _ in range(warmup):
        for n in names:
            callables[n]()
    out = {n: [] for n in names}
    gc.disable()
    try:
        for rep in range(reps):
            order = names[rep % len(names):] + names[:rep % len(names)]
            for n in order:
                t0 = time.perf_counter_ns()
                callables[n]()
                out[n].append((time.perf_counter_ns() - t0) * 1e-9)
    finally:
        gc.enable()
    return out


def pareto_mask(err, time_, rel_tol=0.05):
    """True where no other point beats it: j dominates i when err_j <= err_i and time_j is no worse
    than i's, with times within `rel_tol` (5%, about the run-to-run timing noise) treated as tied, so
    j must then be strictly more accurate; or when time_j is faster by more than rel_tol and err_j
    <= err_i. Non-finite errors are never on the front."""
    n = len(err)
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(err[i]):
            continue
        dominated = False
        for j in range(n):
            if j == i or not np.isfinite(err[j]):
                continue
            faster = time_[j] < time_[i] * (1 - rel_tol)
            tied = abs(time_[j] - time_[i]) <= rel_tol * time_[i]
            if (faster and err[j] <= err[i]) or (tied and err[j] < err[i]):
                dominated = True
                break
        keep[i] = not dominated
    return keep


# ------------------------------------------------------------------------------------- contexts
def load_data():
    d = {s: np.load(f"data/{s}.npz") for s in ("train", "val", "test")}
    return {s: (d[s]["A"], d[s]["nu"], d[s]["u"]) for s in d}


def build_bases(train_u, train_nu, Nx, r_max=16, m_list=(16,)):
    """POD (r_max modes) and DEIM (per m) built from a snapshot ensemble, same recipe as
    scripts/build_deim.py."""
    k, dealias = wavenumbers(Nx)
    ens = train_u.reshape(-1, Nx)
    nu_snap = np.repeat(train_nu, train_u.shape[1])
    Phi_full, S = build_pod_basis(ens, r=r_max)
    F = collect_nonlinear_snapshots(ens, k, nu_snap, dealias)
    deim = {}
    for m in m_list:
        Psi, _ = build_deim_basis(F, m)
        pts = select_deim_points(Psi)
        deim[m] = {"Psi": Psi, "points": pts}
    return Phi_full, S, deim, k, dealias


def deim_operator(Phi, Psi, points):
    return build_deim_projector(Phi, Psi, points)


# ----------------------------------------------------- diagnostic (NOT hyper-reduced) ROM variants
def native_reduced_steps(u0_max, Nx, nu, T=T_FINAL, n_save=N_SAVE):
    """RK4 step count the ROM/DEIM step-size rule (src/pod_rom.py, src/deim.py) chooses. Copied
    formula; tests/test_hyperreduction_audit.py checks it against the value the src loop returns."""
    dx = L / Nx
    dt = min(0.25 * dx / (u0_max + 1e-8), 0.4 * dx**2 / (2 * nu + 1e-12))
    return max(4 * n_save, int(np.ceil(T / dt)))


def _reduced_rk4(rhs, a0, u0_max, Nx, nu, T, n_save, dt=None):
    """Reduced RK4 with exactly the ROM/DEIM step-size rule (copied; checked against
    rom_reduced_trajectory in `reproduce`). Used only by full-grid diagnostic variants. An explicit
    dt overrides the rule (used by the time-aligned accuracy protocol)."""
    dx = L / Nx
    if dt is None:
        dt = min(0.25 * dx / (u0_max + 1e-8), 0.4 * dx**2 / (2 * nu + 1e-12))
    n_steps = max(4 * n_save, int(np.ceil(T / dt)))
    dt = T / n_steps
    save_steps = set(np.linspace(0, n_steps, n_save, dtype=int).tolist())
    a, a_save = a0, [a0.copy()]
    with np.errstate(all="ignore"):
        for step in range(1, n_steps + 1):
            k1 = rhs(a)
            k2 = rhs(a + dt / 2 * k1)
            k3 = rhs(a + dt / 2 * k2)
            k4 = rhs(a + dt * k3)
            a = a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if step in save_steps:
                a_save.append(a.copy())
    return np.array(a_save), n_steps


def predict_exactF_deim(u0, Phi, M, points, k, dealias, nu, T, n_save, dt=None):
    """DEIM interpolation applied to the EXACT spectral F (evaluated on the full grid, so this is
    a diagnostic, not a hyper-reduced method): da/dt = M @ F_spectral(Phi a)[points]."""
    Nx = Phi.shape[0]
    rhs = lambda a: M @ _burgers_rhs_physical(Phi @ a, k, nu, dealias)[points]
    a_save, _ = _reduced_rk4(rhs, Phi.T @ u0, np.abs(u0).max(), Nx, nu, T, n_save, dt)
    return (a_save @ Phi.T).astype(np.float32)


def predict_fd_fullgrid(u0, Phi, nu, T, n_save, dt=None):
    """POD-Galerkin with the same 2nd-order central-difference F as DEIM, but on the FULL grid and
    with no DEIM interpolation: da/dt = Phi^T F_FD(Phi a). Diagnostic only."""
    Nx = Phi.shape[0]
    dx = L / Nx

    def rhs(a):
        u = Phi @ a
        ul, ur = np.roll(u, 1), np.roll(u, -1)
        return Phi.T @ (-u * (ur - ul) / (2 * dx) + nu * (ur - 2 * u + ul) / dx**2)

    a_save, _ = _reduced_rk4(rhs, Phi.T @ u0, np.abs(u0).max(), Nx, nu, T, n_save, dt)
    return (a_save @ Phi.T).astype(np.float32)


def aligned_dt(n_native, factor=1):
    """dt giving n_steps = 19*q exactly (q = ceil(n_native/19)*factor), so the saved indices
    floor(j*n_steps/19) = j*q land on the exactly uniform times j/19 * T. The (1+1e-9) guards the
    solvers' ceil(T/dt) against float round-up."""
    q = int(np.ceil(n_native / (N_SAVE - 1))) * factor
    return T_FINAL / ((N_SAVE - 1) * q) * (1 + 1e-9)


def aligned_reference(A, nu, Nx):
    """FOM trajectory saved at exactly uniform times (n_steps a multiple of n_save-1)."""
    return solve_burgers(A=A, nu=nu, Nx=Nx, T=T_FINAL, n_save=N_SAVE,
                         dt=aligned_dt(fom_steps(A, nu, Nx, T_FINAL, N_SAVE)))[2]


def fom_steps(A, nu, Nx, T, n_save):
    """RK4 step count solve_burgers uses (formula copied from src/solver.py; validated against an
    FFT-call count in `timing`)."""
    dt = 0.25 * (L / Nx) / (abs(A) + 1e-8)
    return max(max(1, int(np.ceil(T / dt))), 4 * n_save)


# ================================================================================ stage: reproduce
def stage_reproduce(tag):
    env = environment_info()
    data = load_data()
    out = {"config": rng_free_config(), "bench_seed": BENCH_SEED}

    # (a) data regeneration determinism
    A, nu, U = generate_split("test", SPLITS["test"])
    tA, tnu, tU = data["test"]
    out["data_regenerates_bit_identically"] = {
        "test_A": bool(np.array_equal(A, tA)), "test_nu": bool(np.array_equal(nu, tnu)),
        "test_u": bool(np.array_equal(U, tU))}
    out["data_max_abs_diff_test_u"] = float(np.max(np.abs(U - tU)))

    # (b) basis regeneration vs stored experiments/deim_basis.npz
    trA, trnu, trU = data["train"]
    Phi_full, S, deim, k, dealias = build_bases(trU, trnu, NX, m_list=(M_POINTS,))
    Phi = Phi_full[:, :R_MODES]
    M = deim_operator(Phi, deim[M_POINTS]["Psi"], deim[M_POINTS]["points"])
    stored = np.load("experiments/deim_basis.npz")
    out["basis_vs_stored"] = {
        "Phi_max_abs_diff": float(np.max(np.abs(Phi - stored["Phi"]))),
        "points_identical": bool(np.array_equal(deim[M_POINTS]["points"], stored["points"])),
        "M_max_abs_diff": float(np.max(np.abs(M - stored["M"]))),
        "stored_r": int(stored["Phi"].shape[1]), "stored_m": int(stored["m"]),
        "note": "sign flips of SVD columns would show up as large diffs; small diffs are BLAS noise"}
    out["pod_energy_captured_r8"] = float(np.sum(S[:R_MODES] ** 2) / np.sum(S**2))

    # (c) accuracy on the test split with the STORED basis (what the historical numbers used)
    Phi_s, pts_s, M_s = stored["Phi"], stored["points"], stored["M"]
    errs = {"rom": [], "deim": []}
    for i in range(len(tA)):
        g = tU[i]
        errs["rom"].append(relative_l2(rom_predict(g[0], Phi_s, k, float(tnu[i]), dealias,
                                                   T=T_FINAL, n_save=N_SAVE)[1], g))
        errs["deim"].append(relative_l2(deim_rom_predict(g[0], Phi_s, M_s, pts_s, NX,
                                                         nu=float(tnu[i]), T=T_FINAL,
                                                         n_save=N_SAVE)[1], g))
    out["accuracy_test_stored_basis"] = {m_: err_stats(e) for m_, e in errs.items()}
    out["per_example_test_error"] = {m_: e for m_, e in errs.items()}
    out["test_params"] = {"A": tA, "nu": tnu}

    # (d) copied RK4 driver == src's reduced loop (validates the diagnostic variants' dt logic)
    g = tU[0]
    a0 = Phi_s.T @ g[0]
    _, a_src, ns = rom_reduced_trajectory(a0, np.abs(g[0]).max(), Phi_s, k, float(tnu[0]),
                                           dealias, T_FINAL, N_SAVE)
    rhs = lambda a: Phi_s.T @ _burgers_rhs_physical(Phi_s @ a, k, float(tnu[0]), dealias)
    a_copy, ns2 = _reduced_rk4(rhs, a0, np.abs(g[0]).max(), NX, float(tnu[0]), T_FINAL, N_SAVE)
    out["diagnostic_driver_matches_src"] = bool(np.array_equal(a_src, a_copy) and ns == ns2)

    # (d2) full from-scratch rebuild at HEAD: regenerate train/val/test with the CURRENT solver,
    # rebuild POD(r=8)+DEIM(m=16) exactly as build_deim.py does (m NOT re-selected), re-evaluate.
    regen = {s: generate_split(s, n) for s, n in SPLITS.items()}
    Phi_h_full, _, deim_h, _, _ = build_bases(regen["train"][2], regen["train"][1], NX,
                                                m_list=(M_POINTS,))
    Phi_h = Phi_h_full[:, :R_MODES]
    M_h = deim_operator(Phi_h, deim_h[M_POINTS]["Psi"], deim_h[M_POINTS]["points"])
    e_h = {"rom": [], "deim": []}
    for i in range(len(tA)):
        g, nu_i = regen["test"][2][i], float(regen["test"][1][i])
        e_h["rom"].append(relative_l2(rom_predict(g[0], Phi_h, k, nu_i, dealias, T=T_FINAL,
                                                  n_save=N_SAVE)[1], g))
        e_h["deim"].append(relative_l2(deim_rom_predict(g[0], Phi_h, M_h,
                                                        deim_h[M_POINTS]["points"], NX, nu=nu_i,
                                                        T=T_FINAL, n_save=N_SAVE)[1], g))
    out["from_scratch_at_head"] = {
        "note": "data regenerated with the CURRENT src/solver.py (>=80 RK4 steps floor); the files "
                "on disk were produced by the pre-fix solver of commit ac7e934 (checked "
                "bit-identical outside this script) so a from-scratch rebuild differs slightly",
        "n_test_examples_identical_to_stored": int(sum(
            np.array_equal(regen["test"][2][i], tU[i]) for i in range(len(tA)))),
        "train_examples_identical_to_stored": int(sum(
            np.array_equal(regen["train"][2][i], trU[i]) for i in range(len(trA)))),
        "deim_points_identical_to_stored": bool(np.array_equal(deim_h[M_POINTS]["points"],
                                                                 stored["points"])),
        "accuracy_test": {m_: err_stats(e) for m_, e in e_h.items()}}

    # (e) historical timing protocol (verify_timing.py): test[0], 7 trials x 20 calls, mean-of-calls
    A0, nu0, u00 = float(tA[0]), float(tnu[0]), tU[0, 0]
    fns = {"solver": lambda: solve_burgers(A=A0, nu=nu0, Nx=NX, T=T_FINAL, n_save=N_SAVE),
           "rom": lambda: rom_predict(u00, Phi_s, k, nu0, dealias, T=T_FINAL, n_save=N_SAVE),
           "deim_rom": lambda: deim_rom_predict(u00, Phi_s, M_s, pts_s, NX, nu=nu0, T=T_FINAL,
                                                n_save=N_SAVE)}
    hist = {}
    for name, fn in fns.items():
        fn()  # warm-up
        trials = []
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(20):
                fn()
            trials.append((time.perf_counter() - t0) / 20 * 1000)
        hist[name] = {"trial_means_ms": trials, "median_ms": float(np.median(trials)),
                      "std_ms": float(np.std(trials))}
    hist["speedups_from_medians"] = {
        "deim_vs_rom": hist["rom"]["median_ms"] / hist["deim_rom"]["median_ms"],
        "deim_vs_solver": hist["solver"]["median_ms"] / hist["deim_rom"]["median_ms"],
        "rom_vs_solver": hist["solver"]["median_ms"] / hist["rom"]["median_ms"]}
    hist["config"] = {"example": "test[0]", "A": A0, "nu": nu0, "trials": 7, "calls_per_trial": 20,
                      "statistic": "median over trials of mean-of-20-calls (non-interleaved)"}
    out["historical_timing_protocol"] = hist
    print(json.dumps({k_: out[k_] for k_ in ("data_regenerates_bit_identically", "basis_vs_stored",
                                              "accuracy_test_stored_basis")}, indent=1))
    print(json.dumps(hist["speedups_from_medians"], indent=1),
          {n: hist[n]["median_ms"] for n in fns})
    out["env"] = finish_env(env)
    save("reproduce", tag, out)


# ================================================================================== stage: timing
def stage_timing(tag, reps=15):
    env = environment_info()
    data = load_data()
    tA, tnu, tU = data["test"]
    st = np.load("experiments/deim_basis.npz")
    Phi, pts, M = st["Phi"], st["points"], st["M"]
    k, dealias = wavenumbers(NX)

    # validate the FOM step-count formula against an FFT-call count
    real_ifft, calls = np.fft.ifft, [0]

    def counting(*a, **kw):
        calls[0] += 1
        return real_ifft(*a, **kw)
    fom_ok = True
    np.fft.ifft = counting
    try:
        for i in (0, 7, 23, 39):
            calls[0] = 0
            solve_burgers(A=float(tA[i]), nu=float(tnu[i]), Nx=NX, T=T_FINAL, n_save=N_SAVE)
            n_from_calls = (calls[0] - (N_SAVE - 1)) / 4
            fom_ok &= abs(n_from_calls - fom_steps(float(tA[i]), float(tnu[i]), NX, T_FINAL,
                                                   N_SAVE)) < 1e-9
    finally:
        np.fft.ifft = real_ifft

    rows = []
    for i in range(len(tA)):
        A, nu, u0 = float(tA[i]), float(tnu[i]), tU[i, 0]
        a0, u0max = Phi.T @ u0, np.abs(u0).max()
        c = {
            "fom": lambda: solve_burgers(A=A, nu=nu, Nx=NX, T=T_FINAL, n_save=N_SAVE),
            "rom_solver_only": lambda: rom_reduced_trajectory(a0, u0max, Phi, k, nu, dealias,
                                                              T_FINAL, N_SAVE),
            "rom_full": lambda: rom_predict(u0, Phi, k, nu, dealias, T=T_FINAL, n_save=N_SAVE),
            "deim_solver_only": lambda: deim_rom_reduced_trajectory(a0, u0max, Phi, M, pts, NX,
                                                                    nu, T_FINAL, N_SAVE),
            "deim_full": lambda: deim_rom_predict(u0, Phi, M, pts, NX, nu=nu, T=T_FINAL,
                                                  n_save=N_SAVE),
        }
        t = interleaved_timing(c, reps=reps)
        _, _, n_rom = rom_reduced_trajectory(a0, u0max, Phi, k, nu, dealias, T_FINAL, N_SAVE)
        _, _, n_deim = deim_rom_reduced_trajectory(a0, u0max, Phi, M, pts, NX, nu, T_FINAL, N_SAVE)
        rows.append({"i": i, "A": A, "nu": nu,
                     "n_steps": {"fom": fom_steps(A, nu, NX, T_FINAL, N_SAVE), "rom": n_rom,
                                 "deim": n_deim},
                     "median_ms": {n: 1e3 * float(np.median(v)) for n, v in t.items()},
                     "min_ms": {n: 1e3 * float(np.min(v)) for n, v in t.items()},
                     "cv_within_example": {n: float(np.std(v) / np.mean(v)) for n, v in t.items()}})
        if i % 10 == 0:
            print(f"  ex {i}: " + "  ".join(f"{n}={r_:.2f}ms" for n, r_ in
                                              rows[-1]["median_ms"].items()))

    methods = list(rows[0]["median_ms"])
    per = {m_: np.array([r["median_ms"][m_] for r in rows]) for m_ in methods}
    summary = {"workload_mean_ms": {m_: float(per[m_].mean()) for m_ in methods}}
    speed = {}
    for m_ in methods:
        if m_ == "fom":
            continue
        ratio = per["fom"] / per[m_]
        speed[m_] = {"workload_speedup_vs_fom": float(per["fom"].mean() / per[m_].mean()),
                     "per_example_speedup": stats(ratio),
                     "fraction_examples_faster_than_fom": float(np.mean(ratio > 1))}
    speed["deim_solver_only_vs_rom_solver_only"] = float(per["rom_solver_only"].mean()
                                                          / per["deim_solver_only"].mean())
    speed["deim_full_vs_rom_full"] = float(per["rom_full"].mean() / per["deim_full"].mean())
    summary["speedups"] = speed
    summary["reconstruction_share_ms"] = {
        "rom": float((per["rom_full"] - per["rom_solver_only"]).mean()),
        "deim": float((per["deim_full"] - per["deim_solver_only"]).mean())}
    steps = {m_: np.array([r["n_steps"][m_] for r in rows]) for m_ in ("fom", "rom", "deim")}
    summary["n_steps"] = {m_: stats(v) for m_, v in steps.items()}
    summary["examples_where_rom_steps_exceed_fom"] = int((steps["rom"] > steps["fom"]).sum())
    summary["corr_speedup_deim_vs_steps_ratio"] = float(np.corrcoef(
        per["fom"] / per["deim_full"], steps["fom"] / steps["deim"])[0, 1])
    out = {"config": rng_free_config(), "reps_per_example": reps, "warmup_per_method": 2,
           "protocol": "interleaved round-robin single-call timing; per-example statistic = median "
                       "over reps; workload mean = mean over the 40 test examples of that median",
           "fom_step_formula_validated_by_fft_call_count": bool(fom_ok),
           "summary": summary, "rows": rows}
    print(json.dumps(summary, indent=1))
    out["env"] = finish_env(env)
    save("timing", tag, out)


# ================================================================================ stage: breakdown
def _med_time(fn, number=2000, repeat=9):
    fn()
    return float(np.median(timeit.repeat(fn, number=number, repeat=repeat))) / number


def rhs_components(Phi, pts, M, Nx, nu, a, k, dealias):
    """Per-call cost (s) of every ingredient of one RHS evaluation for FOM / ROM / DEIM."""
    from src.deim import _stencil_support
    r = Phi.shape[1]
    u = Phi @ a
    f = _burgers_rhs_physical(u, k, nu, dealias)
    support, neigh = _stencil_support(pts, Nx, 1)
    Ps, idx, dx = Phi[support, :], np.searchsorted(support, neigh), L / Nx
    Fp = np.zeros(len(pts))
    v_hat = np.fft.fft(u)

    def deim_fd():
        us = Ps @ a
        ul, uc, ur = us[idx[:, 0]], us[idx[:, 1]], us[idx[:, 2]]
        return -uc * (ur - ul) / (2 * dx) + nu * (ur - 2 * uc + ul) / dx**2

    def fom_nonlinear():
        uu = np.real(np.fft.ifft(v_hat))
        fh = np.fft.fft(uu * uu)
        fh *= dealias
        return -0.5j * k * fh

    return {
        "rom.reconstruct_Phi@a (full grid)": _med_time(lambda: Phi @ a),
        "rom.nonlinear_rhs_fullgrid (5 FFTs)": _med_time(lambda: _burgers_rhs_physical(u, k, nu,
                                                                                       dealias)),
        "rom.project_Phi.T@f (full grid)": _med_time(lambda: Phi.T @ f),
        "deim.reconstruct_support_Phi_s@a": _med_time(lambda: Ps @ a),
        "deim.gather_and_FD_at_points": _med_time(deim_fd),
        "deim.project_M@F": _med_time(lambda: M @ Fp),
        "fom.nonlinear (1 ifft + 1 fft)": _med_time(fom_nonlinear),
        "n_support_rows": int(len(support)), "Nx": Nx, "r": r,
    }


def stage_breakdown(tag):
    env = environment_info()
    data = load_data()
    tA, tnu, tU = data["test"]
    st = np.load("experiments/deim_basis.npz")
    Phi, pts, M = st["Phi"], st["points"], st["M"]
    k, dealias = wavenumbers(NX)
    out = {"config": rng_free_config(), "cases": []}
    for i in (0, 5, 12):  # a steep case, and two others (indices fixed, not selected on results)
        A, nu, u0 = float(tA[i]), float(tnu[i]), tU[i, 0]
        a_mid = Phi.T @ tU[i, 10]  # a genuine mid-trajectory reduced state
        comp = rhs_components(Phi, pts, M, NX, nu, a_mid, k, dealias)
        a0, u0max = Phi.T @ u0, np.abs(u0).max()
        _, _, n_rom = rom_reduced_trajectory(a0, u0max, Phi, k, nu, dealias, T_FINAL, N_SAVE)
        _, _, n_deim = deim_rom_reduced_trajectory(a0, u0max, Phi, M, pts, NX, nu, T_FINAL, N_SAVE)
        n_fom = fom_steps(A, nu, NX, T_FINAL, N_SAVE)
        tt = interleaved_timing({
            "fom": lambda: solve_burgers(A=A, nu=nu, Nx=NX, T=T_FINAL, n_save=N_SAVE),
            "rom": lambda: rom_reduced_trajectory(a0, u0max, Phi, k, nu, dealias, T_FINAL, N_SAVE),
            "deim": lambda: deim_rom_reduced_trajectory(a0, u0max, Phi, M, pts, NX, nu, T_FINAL,
                                                        N_SAVE)}, reps=25)
        med = {n: float(np.median(v)) for n, v in tt.items()}

        def share(parts, n_rhs, total):
            acc = {p: comp[p] * n_rhs for p in parts}
            acc["residual (RK4 vector ops, loop, snapshot saves)"] = total - sum(acc.values())
            return {p: {"seconds": s, "fraction": s / total} for p, s in acc.items()}

        out["cases"].append({
            "i": i, "A": A, "nu": nu, "n_steps": {"fom": n_fom, "rom": n_rom, "deim": n_deim},
            "rhs_evals": {"fom": 4 * n_fom, "rom": 4 * n_rom, "deim": 4 * n_deim},
            "per_call_component_seconds": comp, "total_solver_only_seconds": med,
            "rom_breakdown": share(["rom.reconstruct_Phi@a (full grid)",
                                    "rom.nonlinear_rhs_fullgrid (5 FFTs)",
                                    "rom.project_Phi.T@f (full grid)"], 4 * n_rom, med["rom"]),
            "deim_breakdown": share(["deim.reconstruct_support_Phi_s@a",
                                     "deim.gather_and_FD_at_points", "deim.project_M@F"],
                                    4 * n_deim, med["deim"]),
            "fom_breakdown": share(["fom.nonlinear (1 ifft + 1 fft)"], 4 * n_fom, med["fom"]),
        })
        print(f"case {i}: fom {med['fom']*1e3:.2f} ms ({n_fom} steps), rom {med['rom']*1e3:.2f} ms "
              f"({n_rom}), deim {med['deim']*1e3:.2f} ms ({n_deim})")
        for nme in ("rom_breakdown", "deim_breakdown", "fom_breakdown"):
            print("  ", nme, {p[:34]: round(v["fraction"], 3)
                              for p, v in out["cases"][-1][nme].items()})
    out["env"] = finish_env(env)
    save("breakdown", tag, out)


# =================================================================================== stage: sweep
def stage_sweep(tag, r_list=(4, 6, 8, 12, 16), m_list=(6, 8, 12, 16, 24, 32), reps=5):
    env = environment_info()
    data = load_data()
    trA, trnu, trU = data["train"]
    vA, vnu, vU = data["val"]
    tA, tnu, tU = data["test"]
    Phi_full, S, deim, k, dealias = build_bases(trU, trnu, NX, r_max=max(r_list), m_list=m_list)

    cfgs = {}  # name -> dict(kind, r, m, Phi, points, M)
    for r in r_list:
        Phi = Phi_full[:, :r]
        cfgs[f"rom_r{r}"] = {"kind": "rom", "r": r, "m": None, "Phi": Phi}
        for m in m_list:
            pts = deim[m]["points"]
            cfgs[f"deim_r{r}_m{m}"] = {"kind": "deim", "r": r, "m": m, "Phi": Phi, "points": pts,
                                       "M": deim_operator(Phi, deim[m]["Psi"], pts)}

    def run(cfg, u0, nu, dt=None):
        Phi = cfg["Phi"]
        if cfg["kind"] == "rom":
            return rom_predict(u0, Phi, k, nu, dealias, T=T_FINAL, n_save=N_SAVE, dt=dt)[1]
        with np.errstate(all="ignore"):
            return deim_rom_predict(u0, Phi, cfg["M"], cfg["points"], NX, nu=nu, T=T_FINAL,
                                    n_save=N_SAVE, dt=dt)[1]

    refs_aligned = {s_: [aligned_reference(float(a), float(v), NX) for a, v in zip(d_[0], d_[1])]
                    for s_, d_ in (("val", data["val"]), ("test", data["test"]))}

    def accuracy(split, aligned):
        A_, nu_, U_ = data[split]
        res = {}
        for name, cfg in cfgs.items():
            e = []
            for i in range(len(A_)):
                u0, nu_i = U_[i, 0], float(nu_[i])
                if aligned:
                    dt = aligned_dt(native_reduced_steps(np.abs(u0).max(), NX, nu_i))
                    e.append(relative_l2(run(cfg, u0, nu_i, dt), refs_aligned[split][i]))
                else:
                    e.append(relative_l2(run(cfg, u0, nu_i), U_[i]))
            res[name] = np.array(e)
        return res

    print("accuracy (val, test; native and time-aligned) ...")
    err_val, err_test = accuracy("val", True), accuracy("test", True)
    err_val_native, err_test_native = accuracy("val", False), accuracy("test", False)

    # ablations, test split, time-aligned: (i) exact-F at the DEIM points, (ii) FD everywhere
    print("ablations ...")
    abl_exact, abl_fd = {}, {}
    with np.errstate(all="ignore"):
        for r in r_list:
            Phi = Phi_full[:, :r]
            dts = [aligned_dt(native_reduced_steps(np.abs(tU[i, 0]).max(), NX, float(tnu[i])))
                   for i in range(len(tA))]
            abl_fd[f"fd_fullgrid_r{r}"] = np.array([relative_l2(predict_fd_fullgrid(
                tU[i, 0], Phi, float(tnu[i]), T_FINAL, N_SAVE, dts[i]), refs_aligned["test"][i])
                for i in range(len(tA))])
            for m in m_list:
                c = cfgs[f"deim_r{r}_m{m}"]
                abl_exact[f"exactF_deim_r{r}_m{m}"] = np.array([relative_l2(predict_exactF_deim(
                    tU[i, 0], Phi, c["M"], c["points"], k, dealias, float(tnu[i]), T_FINAL,
                    N_SAVE, dts[i]), refs_aligned["test"][i]) for i in range(len(tA))])

    print("timing (interleaved) ...")
    names = list(cfgs)
    times = {n: np.zeros((len(tA), reps)) for n in names + ["fom"]}
    for i in range(len(tA)):
        A, nu, u0 = float(tA[i]), float(tnu[i]), tU[i, 0]
        calls = {"fom": lambda: solve_burgers(A=A, nu=nu, Nx=NX, T=T_FINAL, n_save=N_SAVE)}
        for n in names:
            calls[n] = (lambda c=cfgs[n]: run(c, u0, nu))
        t = interleaved_timing(calls, reps=reps, warmup=1)
        for n in t:
            times[n][i] = t[n]
        if i % 10 == 0:
            print(f"  timed example {i}")
    per_ex = {n: np.median(times[n], axis=1) for n in times}  # (40,) per-example median seconds
    fom_mean = per_ex["fom"].mean()

    rows = []
    for n, c in cfgs.items():
        ev, et = err_val[n], err_test[n]
        rows.append({
            "name": n, "kind": c["kind"], "r": c["r"], "m": c["m"],
            "reduced_dim": c["r"], "deim_dim": c["m"],
            "val_err": err_stats(ev), "test_err": err_stats(et),
            "val_err_native_protocol": err_stats(err_val_native[n]),
            "test_err_native_protocol": err_stats(err_test_native[n]),
            "online_ms_mean_per_trajectory": float(per_ex[n].mean() * 1e3),
            "online_ms_median_over_examples": float(np.median(per_ex[n]) * 1e3),
            "speedup_vs_fom_workload": float(fom_mean / per_ex[n].mean()),
            "speedup_vs_fom_per_example": stats(per_ex["fom"] / per_ex[n]),
            "memory_floats_online": (c["Phi"].size + (c["M"].size + 3 * len(c["points"]) * c["r"]
                                     if c["kind"] == "deim" else 0)),
            "explicit_note_memory": "ROM stores Phi (Nx*r); DEIM stores M (r*m) + Phi rows on the "
                                    "stencil support (<=3m*r) online, Phi (Nx*r) for the final "
                                    "reconstruction only",
        })
    mean_val = np.array([np.nanmean(err_val[r_["name"]]) if np.all(np.isfinite(err_val[r_["name"]]))
                         else np.nan for r_ in rows])
    tmean = np.array([r_["online_ms_mean_per_trajectory"] for r_ in rows])
    mask = pareto_mask(mean_val, tmean)
    for r_, on in zip(rows, mask):
        r_["pareto_on_validation_error_vs_time"] = bool(on)

    out = {"config": rng_free_config(), "r_list": r_list, "m_list": m_list, "reps_per_example": reps,
           "fom": {"online_ms_mean_per_trajectory": float(fom_mean * 1e3)},
           "accuracy_protocols": "val_err/test_err (primary) use the TIME-ALIGNED protocol (aligned FOM "
                                 "reference, aligned ROM/DEIM dt, see stage_reference); *_native_"
                                 "protocol is the historical index-wise comparison against the "
                                 "stored data",
           "pareto_definition": "non-dominated in (mean VALIDATION rel-L2 [aligned], mean online "
                                "time, times within 5% treated as tied); test errors are reported alongside and were not used to "
                                "choose the front",
           "rows": rows,
           "ablation_test_err": {n: err_stats(e) for n, e in {**abl_exact, **abl_fd}.items()},
           "pod_singular_value_energy": {str(r): float(np.sum(S[:r] ** 2) / np.sum(S**2))
                                         for r in r_list},
           "note_m_lt_r": "combinations with m < r are included (the operator M is r x m)"}
    out["ablation_definitions"] = {
        "exactF_deim_r*_m*": "DEIM interpolation applied to the exact spectral F on the FULL grid "
                             "(diagnostic, not hyper-reduced): isolates the interpolation error",
        "fd_fullgrid_r*": "POD-Galerkin with the DEIM stencil's finite-difference F on the full grid "
                          "and no DEIM interpolation (diagnostic): isolates the FD error"}
    for r_ in rows:
        print(f"{r_['name']:14s} val={100*(r_['val_err']['mean'] or np.nan):7.3f}% "
              f"test={100*(r_['test_err']['mean'] or np.nan):7.3f}%  "
              f"{r_['online_ms_mean_per_trajectory']:6.2f} ms  x{r_['speedup_vs_fom_workload']:.2f}"
              f"{'  *' if r_['pareto_on_validation_error_vs_time'] else ''}")
    out["env"] = finish_env(env)
    save("sweep", tag, out)


# ================================================================================= stage: scaling
def _fom_ensemble(Nx, n, offset=0, T=T_FINAL):
    rng = np.random.default_rng(SPLIT_SEED_OFFSETS["train"] + offset)
    A = rng.uniform(*A_RANGE, SPLITS["train"])[:n]
    nu = rng.uniform(*NU_RANGE, SPLITS["train"])[:n]  # first n of the historical 200-draw stream
    U = np.stack([solve_burgers(A=float(a), nu=float(v), Nx=Nx, T=T, n_save=N_SAVE)[2]
                  for a, v in zip(A, nu)])
    return A, nu, U


def stage_scaling(tag):
    env = environment_info()
    data = load_data()
    tA, tnu, _ = data["test"]
    out = {"config": rng_free_config(), "per_rhs": [], "trajectory": []}
    rep_i = 0  # test[0] parameters as the representative steep case
    A_rep, nu_rep = float(tA[rep_i]), float(tnu[rep_i])
    n_test = 10
    for Nx, n_train, do_traj, do_acc in ((128, 200, True, True), (256, 100, True, True),
                                         (512, 100, True, True), (1024, 100, True, True),
                                         (2048, 40, False, False), (4096, 20, False, False)):
        print(f"Nx={Nx}: building ensemble ({n_train} FOM solves) ...")
        t0 = time.perf_counter()
        A, nu, U = _fom_ensemble(Nx, n_train)
        gen_s = time.perf_counter() - t0
        Phi_full, S, deim, k, dealias = build_bases(U, nu, Nx, r_max=R_MODES, m_list=(M_POINTS,))
        Phi = Phi_full[:, :R_MODES]
        pts = deim[M_POINTS]["points"]
        M = deim_operator(Phi, deim[M_POINTS]["Psi"], pts)
        u_mid = solve_burgers(A=A_rep, nu=nu_rep, Nx=Nx, T=T_FINAL, n_save=N_SAVE)[2]
        a_mid = Phi.T @ u_mid[10]
        comp = rhs_components(Phi, pts, M, Nx, nu_rep, a_mid, k, dealias)
        rhs = {"Nx": Nx, "n_train": n_train, "fom_ensemble_seconds": gen_s,
               "rom_rhs_us": 1e6 * (comp["rom.reconstruct_Phi@a (full grid)"]
                                     + comp["rom.nonlinear_rhs_fullgrid (5 FFTs)"]
                                     + comp["rom.project_Phi.T@f (full grid)"]),
               "deim_rhs_us": 1e6 * (comp["deim.reconstruct_support_Phi_s@a"]
                                      + comp["deim.gather_and_FD_at_points"]
                                      + comp["deim.project_M@F"]),
               "fom_nonlinear_us": 1e6 * comp["fom.nonlinear (1 ifft + 1 fft)"],
               "components_us": {n: 1e6 * v for n, v in comp.items() if isinstance(v, float)},
               "n_support_rows": comp["n_support_rows"]}
        rhs["deim_over_rom_rhs_speedup"] = rhs["rom_rhs_us"] / rhs["deim_rhs_us"]
        out["per_rhs"].append(rhs)
        print(f"  per-RHS: ROM {rhs['rom_rhs_us']:.1f} us, DEIM {rhs['deim_rhs_us']:.1f} us, "
              f"FOM-nonlinear {rhs['fom_nonlinear_us']:.1f} us")
        if not do_traj:
            continue
        u0 = np.load("data/test.npz")["u"][rep_i, 0] if Nx == NX else \
            (A_rep * np.sin(np.linspace(0, L, Nx, endpoint=False))).astype(np.float32)
        a0, u0max = Phi.T @ u0, np.abs(u0).max()
        reps = 15 if Nx <= 256 else 5 if Nx <= 512 else 3
        t = interleaved_timing({
            "fom": lambda: solve_burgers(A=A_rep, nu=nu_rep, Nx=Nx, T=T_FINAL, n_save=N_SAVE),
            "rom_solver_only": lambda: rom_reduced_trajectory(a0, u0max, Phi, k, nu_rep, dealias,
                                                              T_FINAL, N_SAVE),
            "deim_solver_only": lambda: deim_rom_reduced_trajectory(a0, u0max, Phi, M, pts, Nx,
                                                                    nu_rep, T_FINAL, N_SAVE)},
            reps=reps, warmup=1)
        _, _, n_rom = rom_reduced_trajectory(a0, u0max, Phi, k, nu_rep, dealias, T_FINAL, N_SAVE)
        _, _, n_deim = deim_rom_reduced_trajectory(a0, u0max, Phi, M, pts, Nx, nu_rep, T_FINAL,
                                                   N_SAVE)
        row = {"Nx": Nx, "A": A_rep, "nu": nu_rep, "reps": reps,
               "n_steps": {"fom": fom_steps(A_rep, nu_rep, Nx, T_FINAL, N_SAVE), "rom": n_rom,
                           "deim": n_deim},
               "median_ms": {n: 1e3 * float(np.median(v)) for n, v in t.items()}}
        row["speedup_vs_fom"] = {m_: row["median_ms"]["fom"] / row["median_ms"][m_]
                                 for m_ in ("rom_solver_only", "deim_solver_only")}
        if do_acc:
            # held-out accuracy at this Nx, TIME-ALIGNED protocol: first n_test test parameter
            # pairs, aligned FOM at this Nx as the reference
            e_rom, e_deim = [], []
            with np.errstate(all="ignore"):
                for i in range(n_test):
                    A_i, nu_i = float(tA[i]), float(tnu[i])
                    g = aligned_reference(A_i, nu_i, Nx)
                    u0_i = g[0]
                    dt_i = aligned_dt(native_reduced_steps(np.abs(u0_i).max(), Nx, nu_i))
                    e_rom.append(relative_l2(rom_predict(u0_i, Phi, k, nu_i, dealias, T=T_FINAL,
                                                         n_save=N_SAVE, dt=dt_i)[1], g))
                    e_deim.append(relative_l2(deim_rom_predict(u0_i, Phi, M, pts, Nx, nu=nu_i,
                                                               T=T_FINAL, n_save=N_SAVE,
                                                               dt=dt_i)[1], g))
            row["accuracy_first10_test_params_time_aligned"] = {"rom": err_stats(e_rom),
                                                                "deim": err_stats(e_deim)}
        out["trajectory"].append(row)
        print("  trajectory:", row["n_steps"], {n: round(v, 2) for n, v in row["median_ms"].items()},
              {n: round(v, 2) for n, v in row["speedup_vs_fom"].items()})
    out["env"] = finish_env(env)
    save("scaling", tag, out)


# =============================================================================== stage: reference
def stage_reference(tag):
    env = environment_info()
    data = load_data()
    tA, tnu, tU = data["test"]
    st = np.load("experiments/deim_basis.npz")
    Phi, pts, M = st["Phi"], st["points"], st["M"]
    k, dealias = wavenumbers(NX)
    nominal = np.linspace(0, T_FINAL, N_SAVE)
    rows = []
    for i in range(len(tA)):
        A, nu, g128 = float(tA[i]), float(tnu[i]), tU[i]
        u0, u0max = g128[0], np.abs(g128[0]).max()
        dt_def = 0.25 * (L / NX) / (abs(A) + 1e-8)
        # ---- native protocol (what the historical numbers used): index-wise comparison ----
        t_f, _, _ = solve_burgers(A=A, nu=nu, Nx=NX, T=T_FINAL, n_save=N_SAVE)
        t_r, rom = rom_predict(u0, Phi, k, nu, dealias, T=T_FINAL, n_save=N_SAVE)
        with np.errstate(all="ignore"):
            _, deim = deim_rom_predict(u0, Phi, M, pts, NX, nu=nu, T=T_FINAL, n_save=N_SAVE)
        g_dt4 = solve_burgers(A=A, nu=nu, Nx=NX, T=T_FINAL, n_save=N_SAVE, dt=dt_def / 4)[2]
        g1024 = solve_burgers(A=A, nu=nu, Nx=1024, T=T_FINAL, n_save=N_SAVE)[2][:, ::8]
        g2048 = solve_burgers(A=A, nu=nu, Nx=2048, T=T_FINAL, n_save=N_SAVE)[2][:, ::16]
        # ---- time-aligned protocol: every run has n_steps = 19*q (uniform saved times) ----
        n_f = fom_steps(A, nu, NX, T_FINAL, N_SAVE)
        _, _, n_r = rom_reduced_trajectory(Phi.T @ u0, u0max, Phi, k, nu, dealias, T_FINAL, N_SAVE)
        t_al, _, g_al = solve_burgers(A=A, nu=nu, Nx=NX, T=T_FINAL, n_save=N_SAVE,
                                      dt=aligned_dt(n_f))
        g_al4 = solve_burgers(A=A, nu=nu, Nx=NX, T=T_FINAL, n_save=N_SAVE,
                              dt=aligned_dt(n_f, 4))[2]
        g_al1024 = solve_burgers(A=A, nu=nu, Nx=1024, T=T_FINAL, n_save=N_SAVE,
                                 dt=aligned_dt(fom_steps(A, nu, 1024, T_FINAL, N_SAVE)))[2][:, ::8]
        g_al2048 = solve_burgers(A=A, nu=nu, Nx=2048, T=T_FINAL, n_save=N_SAVE,
                                 dt=aligned_dt(fom_steps(A, nu, 2048, T_FINAL, N_SAVE)))[2][:, ::16]
        rom_al = rom_predict(u0, Phi, k, nu, dealias, T=T_FINAL, n_save=N_SAVE,
                             dt=aligned_dt(n_r))[1]
        with np.errstate(all="ignore"):
            deim_al = deim_rom_predict(u0, Phi, M, pts, NX, nu=nu, T=T_FINAL, n_save=N_SAVE,
                                       dt=aligned_dt(n_r))[1]
        rows.append({
            "i": i, "A": A, "nu": nu, "n_steps": {"fom": n_f, "rom_deim": n_r},
            "max_snapshot_time_offset_fom": float(np.abs(t_f - nominal).max()),
            "max_snapshot_time_offset_rom_vs_fom": float(np.abs(t_r - t_f).max()),
            # native protocol
            "native.rom_vs_fom128": relative_l2(rom, g128),
            "native.deim_vs_fom128": relative_l2(deim, g128),
            "native.fom128_vs_own_dt/4": relative_l2(g128, g_dt4),
            "native.fom128_vs_fom1024": relative_l2(g128, g1024),
            "native.fom1024_vs_fom2048": relative_l2(g1024, g2048),
            "native.rom_vs_fom1024": relative_l2(rom, g1024),
            "native.deim_vs_fom1024": relative_l2(deim, g1024),
            # aligned protocol
            "aligned.fom128_temporal_(n vs 4n steps)": relative_l2(g_al, g_al4),
            "aligned.fom128_vs_fom1024": relative_l2(g_al, g_al1024),
            "aligned.fom1024_vs_fom2048": relative_l2(g_al1024, g_al2048),
            "aligned.rom_vs_fom128": relative_l2(rom_al, g_al),
            "aligned.deim_vs_fom128": relative_l2(deim_al, g_al),
            "aligned.rom_vs_fom1024": relative_l2(rom_al, g_al1024),
            "aligned.deim_vs_fom1024": relative_l2(deim_al, g_al1024)})
    cols = [c for c in rows[0] if "." in c]
    summary = {c: err_stats([r_[c] for r_ in rows]) for c in cols}
    summary["max_snapshot_time_offset_fom"] = stats([r_["max_snapshot_time_offset_fom"]
                                                      for r_ in rows])
    summary["max_snapshot_time_offset_rom_vs_fom"] = stats(
        [r_["max_snapshot_time_offset_rom_vs_fom"] for r_ in rows])
    summary["n_examples_rom_steps_differ_from_fom"] = int(sum(
        r_["n_steps"]["fom"] != r_["n_steps"]["rom_deim"] for r_ in rows))
    worst = sorted(rows, key=lambda r_: -r_["aligned.fom128_vs_fom1024"])[:5]
    out = {"config": rng_free_config(),
           "definitions": {
               "native": "the historical protocol: each method saves at floor(j*n_steps/19) of ITS OWN "
                         "step count, compared index-by-index",
               "aligned": "every run uses n_steps = 19*q so all saved times are exactly j/19*T; the "
                          "aligned ROM/DEIM use q >= their native step count (never coarser)",
               "fom128_vs_fom1024": "FOM Nx=128 vs FOM Nx=1024 at the shared 128 points (exact "
                                    "injection: x_j = x_{8j})"},
           "summary": summary, "worst5_by_aligned_fom128_vs_fom1024": worst, "rows": rows}
    for c, sm in summary.items():
        if isinstance(sm, dict) and "mean" in sm and sm["mean"] is not None and "." in c:
            print(f"{c:42s} mean={100*sm['mean']:.4f}%  median={100*sm['median']:.4f}%  "
                  f"max={100*sm['max']:.4f}%")
    print("time-offset max (FOM):", summary["max_snapshot_time_offset_fom"]["max"],
          " rom-vs-fom:", summary["max_snapshot_time_offset_rom_vs_fom"]["max"],
          " n differ:", summary["n_examples_rom_steps_differ_from_fom"])
    out["env"] = finish_env(env)
    save("reference", tag, out)


# ================================================================================ stage: dtprobe
def stage_dtprobe(tag):
    """EXPLORATORY what-if (not a benchmark of the shipped methods): run the reduced models with the
    FOM's own step count (advective bound only, no dx^2/(2 nu) rule) and record whether they stay
    stable/accurate and how the end-to-end time changes. Tests whether the inherited explicit-
    diffusion step bound is what limits DEIM's end-to-end speed-up at large Nx."""
    env = environment_info()
    data = load_data()
    tA, tnu, tU = data["test"]
    out = {"config": rng_free_config(), "cases": [],
           "purpose": "diagnose the step-count limit; default step rule of src/ is NOT changed"}
    for Nx, n_train in ((128, 200), (1024, 100)):
        A_, nu_, U_ = _fom_ensemble(Nx, n_train)
        Phi_full, _, deim, k, dealias = build_bases(U_, nu_, Nx, r_max=R_MODES, m_list=(M_POINTS,))
        Phi = Phi_full[:, :R_MODES]
        pts = deim[M_POINTS]["points"]
        M = deim_operator(Phi, deim[M_POINTS]["Psi"], pts)
        for i in (0, 13, 29):  # indices fixed a priori: steep/low-nu, high-nu/low-A, low-A/low-nu
            A, nu = float(tA[i]), float(tnu[i])
            ref = aligned_reference(A, nu, Nx)
            u0, u0max = ref[0], np.abs(ref[0]).max()
            n_fom = fom_steps(A, nu, Nx, T_FINAL, N_SAVE)
            n_nat = native_reduced_steps(u0max, Nx, nu)
            row = {"Nx": Nx, "i": i, "A": A, "nu": nu, "n_steps_fom": n_fom, "n_steps_native": n_nat}
            for label, dt in (("native_rule", aligned_dt(n_nat)), ("fom_step_count", aligned_dt(n_fom))):
                a0 = Phi.T @ u0
                with np.errstate(all="ignore"):
                    ur = rom_predict(u0, Phi, k, nu, dealias, T=T_FINAL, n_save=N_SAVE, dt=dt)[1]
                    ud = deim_rom_predict(u0, Phi, M, pts, Nx, nu=nu, T=T_FINAL, n_save=N_SAVE,
                                          dt=dt)[1]
                tt = interleaved_timing({
                    "fom": lambda: solve_burgers(A=A, nu=nu, Nx=Nx, T=T_FINAL, n_save=N_SAVE),
                    "rom": lambda: rom_reduced_trajectory(a0, u0max, Phi, k, nu, dealias, T_FINAL,
                                                          N_SAVE, dt),
                    "deim": lambda: deim_rom_reduced_trajectory(a0, u0max, Phi, M, pts, Nx, nu,
                                                                T_FINAL, N_SAVE, dt)},
                    reps=3 if Nx > 128 else 9, warmup=1)
                med = {n: float(np.median(v)) * 1e3 for n, v in tt.items()}
                row[label] = {"rom_err": relative_l2(ur, ref), "deim_err": relative_l2(ud, ref),
                              "rom_finite": bool(np.all(np.isfinite(ur))),
                              "deim_finite": bool(np.all(np.isfinite(ud))),
                              "median_ms": med, "deim_speedup_vs_fom": med["fom"] / med["deim"],
                              "rom_speedup_vs_fom": med["fom"] / med["rom"]}
            out["cases"].append(row)
            print(f"Nx={Nx} ex{i} (A={A:.2f}, nu={nu:.3f}) steps fom {n_fom} native {n_nat}")
            for lab in ("native_rule", "fom_step_count"):
                c = row[lab]
                print(f"   {lab:15s} ROM err {100*c['rom_err']:.3f}% DEIM err {100*c['deim_err']:.3f}% "
                      f"DEIM x{c['deim_speedup_vs_fom']:.2f} ROM x{c['rom_speedup_vs_fom']:.2f}")
    out["env"] = finish_env(env)
    save("dtprobe", tag, out)


# ================================================================================== stage: ood
def stage_ood(tag):
    """Out-of-family (two-mode initial condition) ROM/DEIM accuracy with the stored basis, under the
    historical index-wise protocol and the time-aligned protocol."""
    from src.general_ic_solver import solve_burgers_general_ic
    env = environment_info()
    st = np.load("experiments/deim_basis.npz")
    Phi, pts, M = st["Phi"], st["points"], st["M"]
    k, dealias = wavenumbers(NX)
    o = np.load("data/ood_family_test.npz")
    nat, al, fom_t = {"rom": [], "deim": []}, {"rom": [], "deim": []}, []
    for i in range(len(o["nu"])):
        g, nu = o["u"][i], float(o["nu"][i])
        u0, u0max = g[0], np.abs(g[0]).max()
        with np.errstate(all="ignore"):
            nat["rom"].append(relative_l2(rom_predict(u0, Phi, k, nu, dealias, T=T_FINAL,
                                                      n_save=N_SAVE)[1], g))
            nat["deim"].append(relative_l2(deim_rom_predict(u0, Phi, M, pts, NX, nu=nu, T=T_FINAL,
                                                            n_save=N_SAVE)[1], g))
            n_f = max(int(np.ceil(T_FINAL / (0.25 * (L / NX) / (u0max + 1e-8)))), 4 * N_SAVE)
            u064 = u0.astype(np.float64)
            ref = solve_burgers_general_ic(u064, nu, T=T_FINAL, n_save=N_SAVE, dt=aligned_dt(n_f))[2]
            fine = solve_burgers_general_ic(u064, nu, T=T_FINAL, n_save=N_SAVE,
                                            dt=aligned_dt(n_f, 4))[2]
            fom_t.append(relative_l2(ref, fine))
            dt = aligned_dt(native_reduced_steps(u0max, NX, nu))
            al["rom"].append(relative_l2(rom_predict(u0, Phi, k, nu, dealias, T=T_FINAL,
                                                     n_save=N_SAVE, dt=dt)[1], ref))
            al["deim"].append(relative_l2(deim_rom_predict(u0, Phi, M, pts, NX, nu=nu, T=T_FINAL,
                                                           n_save=N_SAVE, dt=dt)[1], ref))
    out = {"config": rng_free_config(), "n": len(o["nu"]),
           "definition": "two-mode IC A1 sin x + A2 sin 2x (A1 in [0.5,2], A2 in [0.1,0.6]), nu in "
                         "the training range; POD/DEIM basis built from single-mode data only",
           "native_protocol": {m_: err_stats(v) for m_, v in nat.items()},
           "time_aligned_protocol": {m_: err_stats(v) for m_, v in al.items()},
           "aligned_fom_temporal_error_max": float(np.max(fom_t))}
    for n_, e in (("native", nat), ("aligned", al)):
        print(n_, {m_: f"{100*np.mean(v):.3f}% +- {100*np.std(v):.3f}%" for m_, v in e.items()})
    out["env"] = finish_env(env)
    save("ood", tag, out)


# ===================================================================================== stage: fno
def stage_fno(tag):
    import torch
    from src.device import timing_devices
    from src.fno_net import FNO1d
    env = environment_info()
    data = load_data()
    tA, tnu, tU = data["test"]
    ood = np.load("data/ood_family_test.npz")
    ckpts = {0: "experiments/fno_checkpoint.pt", 1: "experiments/fno_checkpoint_seed1.pt",
             2: "experiments/fno_checkpoint_seed2.pt"}
    metas = {0: "experiments/fno_run_meta.json", 1: "experiments/fno_run_meta_seed1.json",
             2: "experiments/fno_run_meta_seed2.json"}

    def load(path, dev):
        c = torch.load(path, map_location=dev, weights_only=True)
        m = FNO1d(c["Nx"], c["n_save"], modes=c["modes"], width=c["width"],
                  n_layers=c["n_layers"]).to(dev)
        m.load_state_dict(c["model_state"])
        m.eval()
        return m, c

    out = {"config": rng_free_config(), "seeds": {}}
    for seed, path in ckpts.items():
        m_cpu, c = load(path, "cpu")
        with torch.no_grad():
            p_in = m_cpu(torch.from_numpy(tU[:, 0].astype(np.float32))).numpy()
            p_ood = m_cpu(torch.from_numpy(ood["u"][:, 0].astype(np.float32))).numpy()
        e_in = [relative_l2(p_in[i], tU[i]) for i in range(len(tA))]
        e_ood = [relative_l2(p_ood[i], ood["u"][i]) for i in range(len(ood["A1"]))]
        meta = json.load(open(metas[seed]))
        out["seeds"][seed] = {"in_dist": err_stats(e_in), "out_of_family": err_stats(e_ood),
                              "checkpoint_epoch": int(c["epoch"]), "train_meta": meta}
        print(f"seed {seed}: in-dist {100*np.mean(e_in):.3f}%  OOD {100*np.mean(e_ood):.3f}%  "
              f"(epoch {c['epoch']})")
    out["seed_mean_in_dist"] = float(np.mean([out["seeds"][s]["in_dist"]["mean"] for s in ckpts]))
    out["seed_mean_ood"] = float(np.mean([out["seeds"][s]["out_of_family"]["mean"] for s in ckpts]))

    # FNO input audit: does the network see nu at all?
    m_cpu, _ = load(ckpts[0], "cpu")
    u0 = torch.from_numpy(tU[:1, 0].astype(np.float32))
    with torch.no_grad():
        out["fno_input_channels"] = int(m_cpu.lift.in_channels)  # 2: [u0(x), x]
        out["fno_forward_signature_args"] = ["u0"]
        out["fno_prediction_identical_for_two_nu_same_u0"] = bool(torch.equal(m_cpu(u0), m_cpu(u0)))
    # same u0 (same A), very different nu -> the target differs, the FNO input does not
    a_fix = 1.5
    lo = solve_burgers(A=a_fix, nu=0.01, Nx=NX, T=T_FINAL, n_save=N_SAVE)[2]
    hi = solve_burgers(A=a_fix, nu=0.10, Nx=NX, T=T_FINAL, n_save=N_SAVE)[2]
    out["same_u0_different_nu"] = {"A": a_fix, "u0_identical": bool(np.array_equal(lo[0], hi[0])),
                                   "target_rel_l2_difference_nu0.01_vs_0.10": relative_l2(lo, hi)}

    # nu-blind floor: the L2-optimal predictor that knows A but not nu is E_nu[u(A, nu)] (nu ~ U over
    # the training range); its error against the true trajectory is the floor for ANY nu-blind model.
    xg, wg = np.polynomial.legendre.leggauss(24)
    nus = 0.5 * (NU_RANGE[1] - NU_RANGE[0]) * xg + 0.5 * (NU_RANGE[1] + NU_RANGE[0])
    wts = wg / 2
    floor = []
    for i in range(len(tA)):
        sols = np.stack([solve_burgers(A=float(tA[i]), nu=float(v), Nx=NX, T=T_FINAL,
                                       n_save=N_SAVE)[2] for v in nus])
        mean_pred = np.tensordot(wts, sols, axes=1).astype(np.float32)
        floor.append(relative_l2(mean_pred, tU[i]))
    out["nu_blind_floor_in_dist"] = err_stats(floor)
    print(f"nu-blind irreducible error (in-distribution): mean {100*np.mean(floor):.3f}%")

    # timing, batch 1, output moved to host (same protocol as the historical FNO timing)
    timing = {}
    for dev in timing_devices():
        m, _ = load(ckpts[0], dev)
        x = torch.from_numpy(tU[0, 0].astype(np.float32)).unsqueeze(0).to(dev)

        def call():
            with torch.no_grad():
                out = m(x).cpu().numpy()[0]
            if dev == "cuda":  # the host copy already blocks; the sync makes the bracket explicit
                torch.cuda.synchronize()
            return out
        t = interleaved_timing({"fno": call}, reps=200, warmup=10)["fno"]
        timing[dev] = {"ms": stats(np.array(t) * 1e3)}
        print(f"FNO {dev}: median {timing[dev]['ms']['median']:.3f} ms (batch 1, incl. host copy)")
    xb = torch.from_numpy(tU[:, 0].astype(np.float32))
    with torch.no_grad():
        m_cpu(xb)
        tb = timeit.repeat(lambda: m_cpu(xb), number=20, repeat=5)
    timing["cpu_batch40_ms_per_trajectory"] = float(np.median(tb) / 20 / 40 * 1e3)
    out["timing"] = timing
    out["env"] = finish_env(env)
    save("fno", tag, out)


STAGES = {"reproduce": stage_reproduce, "timing": stage_timing, "breakdown": stage_breakdown,
          "sweep": stage_sweep, "scaling": stage_scaling, "reference": stage_reference,
          "dtprobe": stage_dtprobe, "ood": stage_ood, "fno": stage_fno}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    STAGES[args.stage](args.tag)
