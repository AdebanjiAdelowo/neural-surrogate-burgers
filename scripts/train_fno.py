"""Research extension -- train the 1D FNO on the *existing*, already-leakage-checked
data/{train,val}.npz splits (no new data generation needed for this part: the FNO's input is the
initial-condition field u0(x) = u[:, 0, :], which is already present in the committed data
format).

Mirrors scripts/train_surrogate.py's conventions closely (same seed, optimizer, batch size,
epoch count, checkpointing-on-best-val-loss) so the two surrogates are trained under matched
conditions -- an architecture/input-representation comparison, not a training-budget comparison.

Run: python scripts/train_fno.py [--overfit-check] [--epochs N] [--device D] [--out-dir DIR]
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
from src.fno_net import FNO1d

CHECKPOINT_PATH = "experiments/fno_checkpoint.pt"
RUN_META_PATH = "experiments/fno_run_meta.json"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(name, data_prefix=""):
    d = np.load(f"data/{data_prefix}{name}.npz")
    u = d["u"].astype(np.float32)             # (N, n_save, Nx)
    u0 = torch.from_numpy(u[:, 0, :].copy())   # (N, Nx) -- the actual IC field, not (A, nu)
    target = torch.from_numpy(u)               # (N, n_save, Nx)
    nu = torch.from_numpy(d["nu"].astype(np.float32))  # (N,) -- only fed to the network if nu-aware
    return u0, target, nu


def overfit_check(device, n_examples=4, steps=500, modes=16, width=32, n_layers=4, seed=0):
    set_seed(seed)
    u0, target, _ = load_split("train")
    u0, target = u0[:n_examples].to(device), target[:n_examples].to(device)

    n_save, Nx = target.shape[1], target.shape[2]
    model = FNO1d(Nx, n_save, modes=modes, width=width, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    losses = []
    for step in range(steps):
        opt.zero_grad()
        pred = model(u0)
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"Overfit check: loss[0]={losses[0]:.6f} -> loss[-1]={losses[-1]:.6f} "
          f"(ratio: {losses[-1] / losses[0]:.4f})")
    return losses


def train(device, epochs=200, batch_size=16, lr=1e-3, modes=16, width=32, n_layers=4, seed=0,
          checkpoint_path=CHECKPOINT_PATH, run_meta_path=RUN_META_PATH, use_nu=False,
          data_prefix="", requested_device=None):
    set_seed(seed)
    u0_train, y_train, nu_train = load_split("train", data_prefix)
    u0_val, y_val, nu_val = load_split("val", data_prefix)
    u0_train, y_train, nu_train = u0_train.to(device), y_train.to(device), nu_train.to(device)
    u0_val, y_val, nu_val = u0_val.to(device), y_val.to(device), nu_val.to(device)

    n_save, Nx = y_train.shape[1], y_train.shape[2]
    model = FNO1d(Nx, n_save, modes=modes, width=width, n_layers=n_layers, use_nu=use_nu).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = u0_train.shape[0]
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
            u0b, yb = u0_train[idx], y_train[idx]
            opt.zero_grad()
            pred = model(u0b, nu_train[idx])
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * u0b.shape[0]
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(u0_val, nu_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
                        "n_save": n_save, "Nx": Nx, "modes": modes, "width": width,
                        "n_layers": n_layers, "use_nu": use_nu}, checkpoint_path)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{epochs}  train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}")

    synchronize(device)
    total_time = time.perf_counter() - t0
    print(f"Best val_loss: {best_val_loss:.6f} (checkpoint saved to {checkpoint_path})")

    # Provenance sidecar (same rationale as nerf-tensorf's fix in the remediation phase: a
    # checkpoint on disk should be traceable to the command/config/hardware that produced it).
    run_meta = {
        "model": "FNO1d", "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed,
        "modes": modes, "width": width, "n_layers": n_layers, "param_count": param_count,
        "device": str(device), "best_val_loss": best_val_loss, "total_train_time_s": total_time,
        "torch_version": torch.__version__, "n_train_examples": n, "use_nu": use_nu,
        "data_prefix": data_prefix, "requested_device": requested_device,
        "device_info": device_info(device),
    }
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"FNO param count: {param_count}  (run metadata: {run_meta_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit-check", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nu-aware", action="store_true",
                        help="add the normalised viscosity as a third input channel")
    parser.add_argument("--data-prefix", default="",
                        help="file prefix under data/, e.g. aligned_ for data/aligned_train.npz")
    parser.add_argument("--out-prefix", default="",
                        help="prefix for checkpoint/meta names (default: historical names)")
    parser.add_argument("--out-dir", default="experiments",
                        help="directory for checkpoints and run metadata (default: historical location)")
    add_device_argument(parser)
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")

    if args.overfit_check:
        losses = overfit_check(device, modes=args.modes, width=args.width, n_layers=args.n_layers)
        ratio = losses[-1] / losses[0]
        if ratio < 0.1:
            print("OVERFIT CHECK PASSED")
        else:
            print("OVERFIT CHECK FAILED -- do not proceed to full training.")
            sys.exit(1)
    else:
        seed_suffix = "" if args.seed == 0 and not args.out_prefix else f"_seed{args.seed}"
        name = f"{args.out_prefix}" if args.out_prefix else ""
        train(device, epochs=args.epochs, modes=args.modes, width=args.width,
              n_layers=args.n_layers, seed=args.seed,
              checkpoint_path=os.path.join(args.out_dir, f"{name}fno_checkpoint{seed_suffix}.pt"),
              run_meta_path=os.path.join(args.out_dir, f"{name}fno_run_meta{seed_suffix}.json"),
              use_nu=args.nu_aware, data_prefix=args.data_prefix, requested_device=args.device)
