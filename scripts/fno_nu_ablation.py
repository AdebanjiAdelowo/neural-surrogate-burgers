"""nu-blind vs nu-aware FNO: controlled A/B, timing, and the nu-blind irreducible-error floors.

Variants (all evaluated on CPU float32 from saved checkpoints; nothing is retrained here):
  historical   experiments/fno_checkpoint[_seed{1,2}].pt         nu-blind, the original runs (seeds 0-2)
  blind_rerun  experiments/rerun_fno_checkpoint_seed{0-4}.pt     nu-blind, retrained today, same code
  nu_aware     experiments/nu_fno_checkpoint_seed{0-4}.pt        input [u0, x, nu_norm]; nothing else changed
Train them with:
  python scripts/train_fno.py --epochs 200 --seed S --nu-aware --out-prefix nu_
  python scripts/train_fno.py --epochs 200 --seed S --out-prefix rerun_
Optional aligned-data variants (prefix al_nu_ / al_rerun_, --data-prefix aligned_) use `--aligned`.

Run: python scripts/fno_nu_ablation.py [--aligned] [--tag TAG] [--overwrite]
Output: report/audit/fno_ablation[_aligned]_<tag>.json. An existing output (including the committed
        historical run1 files, which the default tag points at) is never replaced unless --overwrite
        is given; pass a new --tag for a new run.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hyperreduction_benchmark import (N_SAVE, NU_RANGE, NX, OUT_DIR, T_FINAL, environment_info,
                                               err_stats, finish_env, interleaved_timing, save, stats)
from src.device import timing_devices
from src.fno_net import FNO1d
from src.general_ic_solver import solve_burgers_general_ic
from src.metrics import relative_l2
from src.solver import solve_burgers


def load_model(path):
    c = torch.load(path, map_location="cpu", weights_only=True)
    m = FNO1d(c["Nx"], c["n_save"], modes=c["modes"], width=c["width"], n_layers=c["n_layers"],
              use_nu=c.get("use_nu", False))
    m.load_state_dict(c["model_state"])
    m.eval()
    return m, c


def predict(m, u0, nu):
    with torch.no_grad():
        return m(torch.from_numpy(u0.astype(np.float32)), torch.from_numpy(nu.astype(np.float32))).numpy()


def evaluate(m, u, nu):
    p = predict(m, u[:, 0], nu)
    return [relative_l2(p[i], u[i]) for i in range(len(nu))]


def nu_mean_predictor(solve, nus_nodes, weights):
    sols = np.stack([solve(v) for v in nus_nodes])
    return np.tensordot(weights, sols, axes=1).astype(np.float32)


def output_path(tag, aligned):
    name = "fno_ablation" + ("_aligned" if aligned else "")
    return os.path.join(OUT_DIR, f"{name}{'_' + tag if tag else ''}.json")  # same rule as save()


def stage(tag, aligned, overwrite=False):
    path = output_path(tag, aligned)
    if os.path.exists(path) and not overwrite:
        raise SystemExit(f"{path} already exists (the default tag points at the historical result); "
                         "pass a new --tag, or --overwrite to replace it deliberately")
    env = environment_info()
    pre = "aligned_" if aligned else ""
    d_test, d_ood = np.load(f"data/{pre}test.npz"), np.load(f"data/{pre}ood_family_test.npz")
    tA, tnu, tU = d_test["A"], d_test["nu"], d_test["u"]
    oA1, oA2, onu, oU = d_ood["A1"], d_ood["A2"], d_ood["nu"], d_ood["u"]
    if aligned:
        variants = {"blind_rerun": [f"experiments/al_rerun_fno_checkpoint_seed{s}.pt" for s in range(3)],
                    "nu_aware": [f"experiments/al_nu_fno_checkpoint_seed{s}.pt" for s in range(3)]}
    else:
        variants = {"historical": ["experiments/fno_checkpoint.pt", "experiments/fno_checkpoint_seed1.pt",
                                   "experiments/fno_checkpoint_seed2.pt"],
                    "blind_rerun": [f"experiments/rerun_fno_checkpoint_seed{s}.pt" for s in range(5)],
                    "nu_aware": [f"experiments/nu_fno_checkpoint_seed{s}.pt" for s in range(5)]}
    out = {"aligned_data": aligned, "variants": {}}
    for name, paths in variants.items():
        per_seed = []
        for s, p in enumerate(paths):
            m, c = load_model(p)
            e_in, e_ood = evaluate(m, tU, tnu), evaluate(m, oU, onu)
            meta_path = p.replace("fno_checkpoint", "fno_run_meta").replace(".pt", ".json")
            meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
            rec = {"seed": s, "in_dist": err_stats(e_in), "out_of_family": err_stats(e_ood),
                   "best_epoch": int(c["epoch"]), "best_val_mse": float(c["val_loss"]),
                   "train_time_s": meta.get("total_train_time_s"), "param_count": meta.get("param_count")}
            if name == "nu_aware":  # is the nu channel actually used? permute nu and re-evaluate
                perm = np.random.default_rng(0).permutation(len(tnu))
                rec["in_dist_with_shuffled_nu"] = err_stats(evaluate(m, tU, tnu[perm]))["mean"]
                rec["ood_with_shuffled_nu"] = err_stats(evaluate(m, oU, onu[np.random.default_rng(1).permutation(len(onu))]))["mean"]
            per_seed.append(rec)
        agg = {}
        for key in ("in_dist", "out_of_family"):
            means = np.array([r[key]["mean"] for r in per_seed])
            agg[key] = {"seed_means": means.tolist(), "mean_over_seeds": float(means.mean()),
                        "std_over_seeds": float(means.std()), "median_over_seeds": float(np.median(means)),
                        "worst_case_max_over_seeds": float(max(r[key]["max"] for r in per_seed))}
        out["variants"][name] = {"per_seed": per_seed, "aggregate": agg, "n_seeds": len(paths),
                                 "train_time_s_median": float(np.median([r["train_time_s"] for r in per_seed
                                                                         if r["train_time_s"]])) if per_seed[0]["train_time_s"] else None}
        a = agg
        print(f"{name:12s} in-dist {100*a['in_dist']['mean_over_seeds']:.3f}% +- {100*a['in_dist']['std_over_seeds']:.3f}%  "
              f"OOD {100*a['out_of_family']['mean_over_seeds']:.3f}% +- {100*a['out_of_family']['std_over_seeds']:.3f}%  "
              f"({len(paths)} seeds)")

    # ---- nu-blind floors: the L2-optimal predictor that knows the initial condition but not nu is the
    # nu-average of the trajectories over the training range (nu ~ uniform); its error is a floor for ANY
    # nu-blind model. Gauss-Legendre nodes; targets are the stored (index-wise) trajectories for the
    # historical protocol and aligned ones for the aligned protocol.
    xg, wg = np.polynomial.legendre.leggauss(24)
    nodes = 0.5 * (NU_RANGE[1] - NU_RANGE[0]) * xg + 0.5 * (NU_RANGE[1] + NU_RANGE[0])
    w = wg / 2
    floors = {}
    for label, use_aligned_solver, targets in (("stored_targets_current_solver_native_times", False, None),
                                               ("aligned_targets_aligned_times", True, "aligned")):
        f_in, f_ood = [], []
        for i in range(len(tA)):
            A = float(tA[i])
            mean_pred = nu_mean_predictor(lambda v: solve_burgers(A=A, nu=float(v), Nx=NX, T=T_FINAL, n_save=N_SAVE,
                                                                   align_snapshots=use_aligned_solver)[2], nodes, w)
            tgt = tU[i] if targets is None else solve_burgers(A=A, nu=float(tnu[i]), Nx=NX, T=T_FINAL, n_save=N_SAVE,
                                                               align_snapshots=True)[2]
            f_in.append(relative_l2(mean_pred, tgt))
        for i in range(len(onu)):
            u0 = oU[i, 0].astype(np.float64)
            mean_pred = nu_mean_predictor(lambda v: solve_burgers_general_ic(u0, float(v), T=T_FINAL, n_save=N_SAVE,
                                                                             align_snapshots=use_aligned_solver)[2], nodes, w)
            tgt = oU[i] if targets is None else solve_burgers_general_ic(u0, float(onu[i]), T=T_FINAL, n_save=N_SAVE,
                                                                         align_snapshots=True)[2]
            f_ood.append(relative_l2(mean_pred, tgt))
        floors[label] = {"in_dist": err_stats(f_in), "out_of_family": err_stats(f_ood)}
        print(f"nu-blind floor [{label}]: in-dist {100*np.mean(f_in):.3f}%  OOD {100*np.mean(f_ood):.3f}%")
    out["nu_blind_floor"] = floors
    out["floor_definition"] = ("mean over nu (24 Gauss-Legendre nodes on [0.01, 0.1]) of the FOM trajectory for the "
                               "same initial condition; the L2-optimal predictor without nu. The floor for OOD is "
                               "for a nu-blind model that had learned the two-mode initial-condition map perfectly.")

    # ---- inference timing (batch 1, output copied to host), blind vs nu-aware
    timing = {}
    ref_paths = {"blind": variants.get("historical", variants["blind_rerun"])[0], "nu_aware": variants["nu_aware"][0]}
    for dev in timing_devices():
        calls = {}
        for label, path in ref_paths.items():
            m, _ = load_model(path)
            m = m.to(dev)
            u = torch.from_numpy(tU[0:1, 0].astype(np.float32)).to(dev)
            nu = torch.from_numpy(tnu[0:1].astype(np.float32)).to(dev)
            calls[label] = (lambda m=m, u=u, nu=nu: m(u, nu).detach().cpu().numpy())
        if dev == "cuda":  # the host copy already blocks; the sync makes the CUDA bracket explicit
            calls = {k: (lambda f=f: (f(), torch.cuda.synchronize())) for k, f in calls.items()}
        t = interleaved_timing(calls, reps=300, warmup=20)
        timing[dev] = {k: stats(np.array(v) * 1e3) for k, v in t.items()}
        print(dev, {k: f"{v['median']:.3f} ms" for k, v in timing[dev].items()})
    out["inference_ms_batch1"] = timing
    out["env"] = finish_env(env)
    save("fno_ablation" + ("_aligned" if aligned else ""), tag, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", action="store_true")
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    a = ap.parse_args()
    stage(a.tag, a.aligned, a.overwrite)
