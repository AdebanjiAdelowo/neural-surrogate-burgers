"""Research extension -- generate the out-of-INITIAL-CONDITION-FAMILY generalisation test set.

The existing dataset (scripts/generate_dataset.py) and its three extrapolation cases
(scripts/evaluate_comparison.py) only vary the two scalar PARAMETERS (A, nu) of a single-mode
sinusoidal family u0(x) = A*sin(2*pi*x/L). This script generates a genuinely different test: a
TWO-MODE initial-condition family, u0(x) = A1*sin(2*pi*x/L) + A2*sin(4*pi*x/L), never presented to
any method's training or basis-construction step.

Viscosity nu is kept within the ORIGINAL training range [0.01, 0.1] deliberately, so this test
isolates IC-FAMILY shift from parameter-RANGE shift (the existing extrapolation cases already
cover the latter). This is the test the MLP surrogate cannot even be meaningfully evaluated on in
its native form (it has no way to represent a two-mode IC as a 2-scalar (A, nu) pair) -- see
RESEARCH_EXTENSION.md for how each method is adapted/excluded for this specific test.

Uses src.general_ic_solver (mirrors src.solver.solve_burgers's exact numerical method; see that
module's docstring) since src.solver.solve_burgers cannot accept a non-single-mode IC.

Output: data/ood_family_test.npz -- arrays A1, A2, nu (N,), u (N, n_save, Nx). Not committed to
git (matches data/*.npz's existing .gitignore convention).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.general_ic_solver import solve_burgers_general_ic

NX = 128
L = 2 * np.pi
T_FINAL = 1.0
N_SAVE = 20
N_EXAMPLES = 20
SEED = 300_000  # disjoint from scripts/generate_dataset.py's SPLIT_SEED_OFFSETS (0/100k/200k)

# Same amplitude/viscosity SCALE as the training family (A1 acts like the original A; A2 is a
# smaller secondary mode so the field stays qualitatively "Burgers-like", not a different regime
# entirely), but nu stays in-range -- only the IC's functional FORM is new.
A1_RANGE = (0.5, 2.0)
A2_RANGE = (0.1, 0.6)
NU_RANGE = (0.01, 0.1)


def main():
    rng = np.random.default_rng(SEED)
    A1 = rng.uniform(*A1_RANGE, N_EXAMPLES).astype(np.float32)
    A2 = rng.uniform(*A2_RANGE, N_EXAMPLES).astype(np.float32)
    nu = rng.uniform(*NU_RANGE, N_EXAMPLES).astype(np.float32)

    x = np.linspace(0, L, NX, endpoint=False)
    U = np.zeros((N_EXAMPLES, N_SAVE, NX), dtype=np.float32)
    for i in range(N_EXAMPLES):
        u0 = A1[i] * np.sin(x) + A2[i] * np.sin(2 * x)
        _, _, u = solve_burgers_general_ic(u0, nu=float(nu[i]), T=T_FINAL, n_save=N_SAVE)
        U[i] = u

    os.makedirs("data", exist_ok=True)
    np.savez("data/ood_family_test.npz", A1=A1, A2=A2, nu=nu, u=U)
    print(f"ood_family_test: {N_EXAMPLES} examples, two-mode IC family "
          f"(A1 in [{A1.min():.3f},{A1.max():.3f}], A2 in [{A2.min():.3f},{A2.max():.3f}]), "
          f"nu in training range [{nu.min():.4f},{nu.max():.4f}], u shape {U.shape}")

    # Confirm no accidental parameter collision with the single-mode training family's own (A, nu)
    # pairs is even meaningful here -- this dataset uses a structurally different IC, so "leakage"
    # in the usual sense does not apply, but we do confirm the seed range is disjoint by construction.
    for split in ("train", "val", "test"):
        path = f"data/{split}.npz"
        if os.path.exists(path):
            d = np.load(path)
            assert d["u"].shape[1:] == U.shape[1:], f"grid/time mismatch vs {split}.npz"
    print("Grid/time-axis shape consistency with existing splits verified.")


if __name__ == "__main__":
    main()
