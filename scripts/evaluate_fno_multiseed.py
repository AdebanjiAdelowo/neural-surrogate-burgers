"""Research extension -- multi-seed FNO evidence (review response).

The original FNO results were single-seed. This trains/evaluates on 3 seeds (0, 1, 2), identical
architecture and training settings, and reports mean +/- std across seeds for the key metrics:
in-distribution accuracy, parameter extrapolation, and out-of-family IC generalisation.

No hyperparameter search was performed based on any test/validation/out-of-family performance --
seeds 1 and 2 use the exact same modes/width/n_layers/epochs/lr/batch_size as seed 0 (the original,
already-reported run); only the random seed differs.

Run: python scripts/evaluate_fno_multiseed.py (requires experiments/fno_checkpoint[_seed{1,2}].pt,
     produced by `python scripts/train_fno.py --epochs 200 --seed {1,2}`)
Output: report/research/fno_multiseed_results.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fno_net import FNO1d
from src.solver import solve_burgers

SEEDS = [0, 1, 2]


def rel_l2(pred, gt):
    return float(np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-12))


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_fno(seed, device):
    suffix = "" if seed == 0 else f"_seed{seed}"
    ckpt = torch.load(f"experiments/fno_checkpoint{suffix}.pt", map_location=device,
                       weights_only=True)
    model = FNO1d(ckpt["Nx"], ckpt["n_save"], modes=ckpt["modes"], width=ckpt["width"],
                   n_layers=ckpt["n_layers"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def fno_predict(model, device, u0_np):
    u0 = torch.from_numpy(u0_np.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(u0).cpu().numpy()[0]


def main():
    device = get_device()
    print(f"Device: {device}")

    d_test = np.load("data/test.npz")
    A_test, nu_test, u_test = d_test["A"], d_test["nu"], d_test["u"]
    n_save = u_test.shape[1]
    Nx = u_test.shape[2]

    d_ood = np.load("data/ood_family_test.npz")
    nu_ood, u_ood = d_ood["nu"], d_ood["u"]

    extrap_cases = [
        {"A": 2.5, "nu": 0.05, "label": "A above (2.5)"},
        {"A": 1.0, "nu": 0.15, "label": "nu above (0.15)"},
        {"A": 0.2, "nu": 0.05, "label": "A below (0.2)"},
    ]

    per_seed = {}
    for seed in SEEDS:
        model, ckpt = load_fno(seed, device)

        id_errs = [rel_l2(fno_predict(model, device, u_test[i, 0]), u_test[i])
                   for i in range(len(A_test))]

        ood_errs = [rel_l2(fno_predict(model, device, u_ood[i, 0]), u_ood[i])
                    for i in range(len(nu_ood))]

        extrap_errs = {}
        for case in extrap_cases:
            _, _, gt_e = solve_burgers(A=case["A"], nu=case["nu"], Nx=Nx, T=1.0, n_save=n_save)
            extrap_errs[case["label"]] = rel_l2(fno_predict(model, device, gt_e[0]), gt_e)

        per_seed[seed] = {
            "best_val_loss": ckpt["val_loss"], "in_distribution_mean": float(np.mean(id_errs)),
            "in_distribution_std": float(np.std(id_errs)),
            "out_of_family_ic_mean": float(np.mean(ood_errs)),
            "out_of_family_ic_std": float(np.std(ood_errs)),
            "parameter_extrapolation": extrap_errs,
        }
        print(f"seed={seed}: ID_mean={np.mean(id_errs):.4f}  OOD_IC_mean={np.mean(ood_errs):.4f}  "
              f"extrap={extrap_errs}")

    id_means = [per_seed[s]["in_distribution_mean"] for s in SEEDS]
    ood_means = [per_seed[s]["out_of_family_ic_mean"] for s in SEEDS]
    extrap_by_case = {c["label"]: [per_seed[s]["parameter_extrapolation"][c["label"]] for s in SEEDS]
                       for c in extrap_cases}

    summary = {
        "seeds": SEEDS,
        "in_distribution": {"mean_across_seeds": float(np.mean(id_means)),
                             "std_across_seeds": float(np.std(id_means)), "per_seed": id_means},
        "out_of_family_ic": {"mean_across_seeds": float(np.mean(ood_means)),
                              "std_across_seeds": float(np.std(ood_means)), "per_seed": ood_means},
        "parameter_extrapolation": {
            label: {"mean_across_seeds": float(np.mean(vals)),
                    "std_across_seeds": float(np.std(vals)), "per_seed": vals}
            for label, vals in extrap_by_case.items()
        },
        "per_seed_detail": per_seed,
    }

    print("\n=== Summary across 3 seeds (mean +/- std) ===")
    print(f"In-distribution: {summary['in_distribution']['mean_across_seeds']*100:.2f}% "
          f"+/- {summary['in_distribution']['std_across_seeds']*100:.2f}%")
    print(f"Out-of-family IC: {summary['out_of_family_ic']['mean_across_seeds']*100:.2f}% "
          f"+/- {summary['out_of_family_ic']['std_across_seeds']*100:.2f}%")
    for label, v in summary["parameter_extrapolation"].items():
        print(f"Extrapolation [{label}]: {v['mean_across_seeds']*100:.2f}% "
              f"+/- {v['std_across_seeds']*100:.2f}%")

    os.makedirs("report/research", exist_ok=True)
    with open("report/research/fno_multiseed_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved report/research/fno_multiseed_results.json")


if __name__ == "__main__":
    main()
