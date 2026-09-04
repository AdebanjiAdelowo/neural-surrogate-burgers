"""Stage 4 — train the MLP surrogate.

Run: python scripts/train_surrogate.py [--overfit-check] [--epochs N]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.surrogate_net import BurgersSurrogateMLP

SEED = 0
CHECKPOINT_PATH = "experiments/surrogate_checkpoint.pt"

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


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def train(device, epochs=200, batch_size=16, lr=1e-3):
    set_seed(SEED)
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
    os.makedirs("experiments", exist_ok=True)

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
                        "val_loss": val_loss, "n_save": n_save, "Nx": Nx}, CHECKPOINT_PATH)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{epochs}  train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}")

    print(f"Best val_loss: {best_val_loss:.6f} (checkpoint saved to {CHECKPOINT_PATH})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit-check", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = get_device()
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
        train(device, epochs=args.epochs)
