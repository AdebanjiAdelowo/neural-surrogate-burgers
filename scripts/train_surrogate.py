"""Stage 4 — train the MLP surrogate.

Run: python scripts/train_surrogate.py [--overfit-check] [--epochs N] [--seed S] [--device D] [--out-dir DIR]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.device import add_device_argument, device_info, select_device, synchronize
from src.surrogate_net import BurgersSurrogateMLP

SEED = 0
CHECKPOINT_PATH = "experiments/surrogate_checkpoint.pt"
RUN_META_PATH = "experiments/surrogate_run_meta.json"

# Documented normalisation ranges (must match scripts/generate_dataset.py's A_RANGE/NU_RANGE).
A_RANGE = (0.5, 2.0)
NU_RANGE = (0.01, 0.1)


def normalize_params(A, nu):
    A_norm = (A - A_RANGE[0]) / (A_RANGE[1] - A_RANGE[0])
    nu_norm = (nu - NU_RANGE[0]) / (NU_RANGE[1] - NU_RANGE[0])
    return np.stack([A_norm, nu_norm], axis=1).astype(np.float32)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(name):
    d = np.load(f"data/{name}.npz")
    params = torch.from_numpy(normalize_params(d["A"], d["nu"]))
    target = torch.from_numpy(d["u"]).float()  # (N, n_save, Nx)
    return params, target


def overfit_check(device, n_examples=4, steps=500):
    set_seed(SEED)
    params, target = load_split("train")
    params, target = params[:n_examples].to(device), target[:n_examples].to(device)

    n_save, Nx = target.shape[1], target.shape[2]
    model = BurgersSurrogateMLP(n_save, Nx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    losses = []
    for step in range(steps):
        opt.zero_grad()
        pred = model(params)
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"Overfit check: loss[0]={losses[0]:.6f} -> loss[-1]={losses[-1]:.6f} "
          f"(ratio: {losses[-1] / losses[0]:.4f})")
    return losses


def train(device, epochs=200, batch_size=16, lr=1e-3, checkpoint_path=CHECKPOINT_PATH,
          run_meta_path=RUN_META_PATH, seed=SEED, requested_device=None):
    set_seed(seed)
    p_train, y_train = load_split("train")
    p_val, y_val = load_split("val")
    p_train, y_train = p_train.to(device), y_train.to(device)
    p_val, y_val = p_val.to(device), y_val.to(device)

    n_save, Nx = y_train.shape[1], y_train.shape[2]
    model = BurgersSurrogateMLP(n_save, Nx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = p_train.shape[0]
    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    synchronize(device)
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pb, yb = p_train[idx], y_train[idx]
            opt.zero_grad()
            pred = model(pb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * pb.shape[0]
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(p_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "n_save": n_save, "Nx": Nx}, checkpoint_path)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{epochs}  train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}")

    synchronize(device)
    total_time = time.perf_counter() - t0
    print(f"Best val_loss: {best_val_loss:.6f} (checkpoint saved to {checkpoint_path})")

    # Provenance sidecar: the measured training time on THIS backend, so no result has to quote a
    # time measured elsewhere (the historical 8.23 s was measured on Apple MPS).
    run_meta = {
        "model": "BurgersSurrogateMLP", "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "seed": seed, "param_count": sum(p.numel() for p in model.parameters()),
        "device": str(device), "requested_device": requested_device,
        "device_info": device_info(device), "best_val_loss": best_val_loss,
        "total_train_time_s": total_time, "torch_version": torch.__version__, "n_train_examples": n,
    }
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Training time: {total_time:.2f} s on {device}  (run metadata: {run_meta_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit-check", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="training seed (default 0, the historical run); seeds other than 0 get a "
                             "_seed<S> file suffix")
    parser.add_argument("--out-dir", default="experiments",
                        help="directory for the checkpoint and run metadata (default: historical location)")
    add_device_argument(parser)
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")

    if args.overfit_check:
        losses = overfit_check(device)
        ratio = losses[-1] / losses[0]
        if ratio < 0.1:
            print("OVERFIT CHECK PASSED")
        else:
            print("OVERFIT CHECK FAILED — do not proceed to full training.")
            sys.exit(1)
    else:
        suffix = "" if args.seed == SEED else f"_seed{args.seed}"
        train(device, epochs=args.epochs, seed=args.seed, requested_device=args.device,
              checkpoint_path=os.path.join(args.out_dir, f"surrogate_checkpoint{suffix}.pt"),
              run_meta_path=os.path.join(args.out_dir, f"surrogate_run_meta{suffix}.json"))
