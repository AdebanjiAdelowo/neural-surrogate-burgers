# Neural Surrogate vs. POD-Galerkin ROM for the Viscous Burgers' Equation

A comparative study of a classical linear reduced-order model (POD-Galerkin) against a data-driven
neural surrogate for the 1D viscous Burgers' equation, evaluated on in-distribution accuracy,
wall-clock runtime, and generalisation to parameters outside the training range.

## Overview

Reduced-order and surrogate modelling are widely used to accelerate simulation in scientific
computing and digital-twin applications. This project compares two ways of building a fast
approximate model for a nonlinear PDE: a POD-Galerkin reduced-order model, which projects the true
governing equations onto a low-dimensional linear subspace, and a neural surrogate, which learns a
direct mapping from problem parameters to the solution field. Burgers' equation is used as the test
problem because its nonlinear advection term produces steep gradients at small viscosity, a regime
that is a standard, well-studied benchmark for both reduced-order models and operator-learning
methods (for example, the Fourier Neural Operator and DeepONet).

## Problem Formulation

Governing equation:

```
u_t + u u_x = nu * u_xx,   x in [0, L) periodic,   t in [0, T]
u(x, 0) = A * sin(2*pi*x/L)
```

The initial condition is parameterised by amplitude `A` and viscosity `nu`. Smaller `nu` produces
steeper gradients and is the more challenging regime for both reduced-order and learned models.

**Research question:** for held-out parameters, how do a POD-Galerkin ROM and a neural surrogate
compare on accuracy and runtime relative to the ground-truth solver, within the training
distribution and outside it (extrapolation)?

## Methods

- **Ground-truth solver** (`src/solver.py`): pseudo-spectral in space (FFT-based derivatives, 2/3
  dealiasing rule), integrating-factor RK4 in time. The linear diffusion term is handled exactly via
  the integrating factor, so only the nonlinear advection term is stepped explicitly.
- **POD-Galerkin ROM** (`src/pod_rom.py`): a rank-`r` spatial basis is built via SVD of a training
  snapshot ensemble. A new initial condition is projected onto the basis, and the reduced ODE is
  integrated with explicit RK4, evaluating the true Burgers right-hand side in physical space via
  the same FFT-based spectral derivatives as the ground-truth solver. This is a standard Galerkin
  projection, not a data-fit shortcut.
- **Neural surrogate** (`src/surrogate_net.py`): a small MLP mapping the normalised parameter pair
  `(A, nu)` directly to the flattened space-time solution field, trained by supervised regression on
  solver-generated trajectories. An MLP was chosen over a CNN because the input is two scalars, not
  spatially structured data. Because it performs a single forward pass with no time-stepping, it has
  a structural runtime advantage over both the solver and the ROM.

## Experimental Setup

**Dataset** (`scripts/generate_dataset.py`): parameters `(A, nu)` sampled uniformly at random from
`A in [0.5, 2.0]`, `nu in [0.01, 0.1]`, with fixed non-overlapping RNG seed ranges per split (train:
0-199, val: 100000-100039, test: 200000-200039). Grid `Nx = 128`, `T = 1.0`, 20 saved snapshots per
example. Splits: 200 train, 40 validation, 40 test examples. No `(A, nu)` pair is shared across
splits (checked programmatically at generation time). Data files are generated locally and are not
committed to the repository.

**POD-ROM:** the basis is built from the training snapshot ensemble; `r = 8` modes are used for the
comparison. This problem family is highly linearly compressible (prediction error decreases
monotonically with retained modes and is near zero by `r >= 16`), so `r = 8` is a deliberately
non-trivial accuracy baseline rather than the easiest possible case for the ROM.

**Neural surrogate training** (`scripts/train_surrogate.py`): a 3-hidden-layer MLP (width 128)
trained for 200 epochs with batch size 16 and the Adam optimiser (`lr = 1e-3`).

**Evaluation** (`scripts/evaluate_comparison.py`): relative L2 error against the ground-truth
solver and wall-clock runtime per evaluation, on the held-out test split and on three parameter
settings deliberately chosen outside the training range.

## Results

**In-distribution accuracy and runtime** (test set, n = 40, POD-ROM r = 8):

| Method | Relative L2 error (mean +/- std) | Runtime per evaluation |
|---|---|---|
| Ground-truth solver | (reference) | 6.29 ms |
| POD-ROM (r = 8) | 0.152% +/- 0.077% | 15.48 ms |
| Neural surrogate | 0.788% +/- 0.366% | 0.138 ms |

The POD-ROM is more accurate in-distribution, consistent with this problem family being highly
linearly compressible. The neural surrogate is 45.6x faster than the full solver and 112x faster
than the ROM, since it requires a single forward pass with no time-stepping.

The POD-ROM as implemented is slower than the full solver, because it evaluates the nonlinear term
via full-grid spectral derivatives at every RK4 stage (no hyper-reduction or DEIM): it reduces the
state dimension but not the cost of evaluating the right-hand side. A hyper-reduced ROM would likely
close this gap (see Limitations).

**Generalisation to out-of-training-range parameters** (three deliberately chosen extrapolation
cases; see `report/mvp_extrapolation.png`):

| Case | POD-ROM error | Neural surrogate error |
|---|---|---|
| `A = 2.5` (training max 2.0) | 0.10% | 5.45% |
| `nu = 0.15` (training max 0.10) | 0.24% | 3.05% |
| `A = 0.2` (training min 0.5) | 0.06% | 55.95% |

Across these three cases the POD-ROM's error stays below 0.24%, while the neural surrogate's error
ranges from 3.05% to 55.95%. The ROM generalises far better outside the training distribution
because it is built directly on the governing equations (Galerkin projection remains valid physics
wherever it is evaluated), while the neural surrogate is a black-box interpolator that degrades,
and at `A = 0.2` degrades severely, once evaluated outside the parameter range it was trained on
(visible as high-frequency noise in `report/mvp_extrapolation.png`).

**Summary:** for this problem, the classical POD-Galerkin ROM is both more accurate and
substantially more robust to parameter extrapolation than the neural surrogate; the surrogate's
main advantage is inference speed, which is large (over two orders of magnitude versus the ROM).

## Repository Structure

```
neural-surrogate-burgers/
├── README.md
├── ARCHITECTURE.md          mathematical formulation and design rationale
├── requirements.txt
├── src/
│   ├── solver.py            ground-truth pseudo-spectral Burgers' solver
│   ├── pod_rom.py           POD-Galerkin reduced-order model
│   └── surrogate_net.py     MLP neural surrogate
├── scripts/
│   ├── generate_dataset.py  builds train/val/test splits
│   ├── train_surrogate.py   trains the neural surrogate
│   └── evaluate_comparison.py  accuracy, runtime, and generalisation evaluation
├── data/                    generated train/val/test splits (not committed)
├── experiments/             trained checkpoint (not committed)
├── tests/                   14 tests covering the solver, ROM, and surrogate
└── report/                  evaluation figures and results
```

## Installation

Requires Python 3.11 with the packages listed in `requirements.txt`: `numpy`, `scipy`, `matplotlib`,
`torch`, `pytest`.

```bash
pip install -r requirements.txt
```

## Usage

```bash
# generate the train/val/test splits
python scripts/generate_dataset.py

# sanity-check the training pipeline on a small fixed batch
python scripts/train_surrogate.py --overfit-check

# train the neural surrogate
python scripts/train_surrogate.py

# run the accuracy, runtime, and generalisation comparison
python scripts/evaluate_comparison.py
```

`evaluate_comparison.py` writes `report/mvp_results.txt` and `report/mvp_comparison.png`.

## Tests

```bash
pytest
```

14 tests cover the solver (mass conservation, energy dissipation, viscosity trend, grid-refinement
consistency), the POD-ROM (basis orthonormality, energy-capture monotonicity, error decreasing with
retained modes, stability under extrapolated viscosity), and the surrogate network (forward-pass
shape and finiteness).

## Limitations

- The POD-ROM does not implement hyper-reduction (DEIM/EIM): it evaluates the nonlinear term on the
  full grid at every RK4 stage, so it does not provide a runtime speedup over the full solver despite
  reducing the state dimension.
- The neural surrogate is trained only on the sinusoidal single-mode initial condition family
  described above; its extrapolation behaviour outside `A in [0.5, 2.0]`, `nu in [0.01, 0.1]` is not
  representative of other initial-condition families.
- The ground-truth solver is verified via physical invariants (mass conservation, monotonic energy
  dissipation), a known shock-formation location, and grid-refinement consistency, rather than
  against an exact Cole-Hopf analytical solution.
- All results were produced on laptop-scale hardware (Apple MPS backend); no distributed or HPC
  benchmarking was performed.

## Possible Extensions

Possible extensions include hyper-reduction (DEIM/EIM) for the ROM, comparison against
operator-learning architectures such as FNO or DeepONet, uncertainty quantification, and extension
to higher-dimensional PDEs.

## References

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full mathematical formulation, system architecture,
and design rationale.
