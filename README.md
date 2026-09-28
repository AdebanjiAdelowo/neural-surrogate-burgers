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

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, \qquad x \in [0, L) \text{ periodic}, \quad t \in [0, T]$$

$$u(x, 0) = A \sin\!\left(\frac{2\pi x}{L}\right)$$

The initial condition is parameterised by amplitude $A$ and viscosity $\nu$. Smaller $\nu$ produces
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

**Dataset** (`scripts/generate_dataset.py`): parameters $(A, \nu)$ sampled uniformly at random from
$A \in [0.5, 2.0]$, $\nu \in [0.01, 0.1]$, with fixed non-overlapping RNG seed ranges per split (train:
0-199, val: 100000-100039, test: 200000-200039). Grid `Nx = 128`, `T = 1.0`, 20 saved snapshots per
example. Splits: 200 train, 40 validation, 40 test examples. No $(A, \nu)$ pair is shared across
splits (checked programmatically at generation time). Data files are generated locally and are not
committed to the repository.

**POD-ROM:** the basis is built from the training snapshot ensemble; `r = 8` modes are used for the
comparison. This problem family is highly linearly compressible: mean relative $L^2$ error on a
held-out test parameter drops from 7.55% at `r = 2` to 2.23% at `r = 4` to 0.076% at `r = 8`, and is
near zero by `r >= 16`. `r = 8` is therefore a deliberately non-trivial accuracy baseline rather
than the easiest possible case for the ROM.

**Neural surrogate training** (`scripts/train_surrogate.py`): a 3-hidden-layer MLP (width 128)
trained for 200 epochs with batch size 16 and the Adam optimiser (`lr = 1e-3`).

**Evaluation** (`scripts/evaluate_comparison.py`): relative $L^2$ error against the ground-truth
solver and wall-clock runtime per evaluation, on the held-out test split and on three parameter
settings deliberately chosen outside the training range. The relative $L^2$ error for a predicted
space-time field $\hat u$ against the solver-generated reference $u$ (both flattened over the saved
snapshots and grid points) is

$$\varepsilon = \frac{\lVert \hat u - u \rVert_2}{\lVert u \rVert_2}$$

## Results

**In-distribution accuracy and runtime** (test set, $n = 40$, POD-ROM $r = 8$):

| Method | Relative $L^2$ error (mean ± std) | Runtime per evaluation |
|---|---|---|
| Ground-truth solver | (reference) | 6.29 ms |
| POD-ROM ($r = 8$) | 0.152% ± 0.077% | 15.48 ms |
| Neural surrogate | 0.788% ± 0.366% | 0.138 ms |

The POD-ROM is more accurate in-distribution, consistent with this problem family being highly
linearly compressible. The neural surrogate is roughly 40-45x faster than the full solver on
laptop-scale hardware (measured wall-clock; the exact multiplier varies run to run, e.g. 45.6x and
~42x on two independent runs) and on the same order faster than the ROM, since it requires a single
forward pass with no time-stepping.

The plain POD-ROM is slower than the full solver, because it evaluates the nonlinear term via
full-grid spectral derivatives at every RK4 stage: it reduces the state dimension but not the cost of
evaluating the right-hand side. The hyper-reduced variant described under "Hyper-Reduction and FNO
Extension" removes that full-grid cost.

**Generalisation to out-of-training-range parameters** (three deliberately chosen extrapolation
cases; see `report/mvp_extrapolation.png`):

| Case | POD-ROM error | Neural surrogate error |
|---|---|---|
| $A = 2.5$ (training max 2.0) | 0.10% | 5.45% |
| $\nu = 0.15$ (training max 0.10) | 0.24% | 3.05% |
| $A = 0.2$ (training min 0.5) | 0.06% | 55.95% |

Across these three cases the POD-ROM's error stays below 0.24%, while the neural surrogate's error
ranges from 3.05% to 55.95%. The ROM generalises far better outside the training distribution
because it is built directly on the governing equations (Galerkin projection remains valid physics
wherever it is evaluated), while the neural surrogate is a black-box interpolator that degrades,
and at $A = 0.2$ degrades severely, once evaluated outside the parameter range it was trained on
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
├── RESEARCH_EXTENSION.md    derivation and history of the DEIM and FNO extension
├── src/
│   ├── solver.py            ground-truth pseudo-spectral Burgers' solver
│   ├── general_ic_solver.py solver for arbitrary initial conditions (out-of-family test set)
│   ├── pod_rom.py           POD-Galerkin reduced-order model
│   ├── deim.py              DEIM/local-finite-difference hyper-reduced ROM
│   ├── timestep.py          reduced-model time-step rules and stability tools
│   ├── snapshot_times.py    time-aligned snapshot saving and resampling
│   ├── metrics.py           relative L2 error
│   ├── surrogate_net.py     MLP neural surrogate
│   └── fno_net.py           1D Fourier Neural Operator (optionally viscosity-aware)
├── scripts/
│   ├── generate_dataset.py  builds train/val/test splits
│   ├── train_surrogate.py   trains the neural surrogate
│   ├── train_fno.py         trains the FNO (--nu-aware for the viscosity-aware variant)
│   ├── evaluate_comparison.py  accuracy, runtime, and generalisation evaluation
│   ├── hyperreduction_benchmark.py  40-case accuracy and timing of the reduced models
│   ├── hyperreduction_followup.py   time-step, stability, stencil and aligned-data studies
│   ├── fno_nu_ablation.py   viscosity-blind versus viscosity-aware FNO
│   └── timing_numpy_sensitivity.py  DEIM versus full-solver timing under the installed NumPy
├── data/                    generated train/val/test splits (not committed)
├── experiments/             trained checkpoints (not committed)
├── tests/                   92 tests
└── report/                  evaluation figures and results; report/audit/ holds the JSON behind
                             the hyper-reduction and FNO numbers below
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

92 tests cover the solver (mass conservation, energy dissipation, viscosity trend, grid-refinement
consistency), the POD-ROM (basis orthonormality, energy-capture monotonicity, error decreasing with
retained modes, stability under extrapolated viscosity), the DEIM/local-FD model (including a check
that no full-grid array is touched in its time loop), the FNO and its viscosity input, the time-step
rules and stability tools, time-aligned snapshot saving, the benchmark and audit code, and the
surrogate network (forward-pass shape and finiteness).

## Limitations

- The plain POD-ROM evaluates the nonlinear term on the full grid at every RK4 stage, so it is slower
  than the full solver despite reducing the state dimension. The DEIM/local-FD variant removes that
  cost but is linearly unstable on steep, low-viscosity trajectories (see below).
- The neural surrogate is trained only on the sinusoidal single-mode initial condition family
  described above; its extrapolation behaviour outside $A \in [0.5, 2.0]$, $\nu \in [0.01, 0.1]$ is
  not representative of other initial-condition families.
- The ground-truth solver is verified via physical invariants (mass conservation, monotonic energy
  dissipation), a known shock-formation location, and grid-refinement consistency, rather than
  against an exact Cole-Hopf analytical solution.
- All results were produced on laptop-scale hardware (Apple MPS backend); no distributed or HPC
  benchmarking was performed.

## Hyper-Reduction and FNO Extension

A DEIM/local-finite-difference hyper-reduced POD-Galerkin ROM and a 1D Fourier Neural Operator
(FNO) baseline extend the plain ROM and MLP surrogate above. The derivation and the history of the
extension are in [`RESEARCH_EXTENSION.md`](RESEARCH_EXTENSION.md); every number below is in
`report/audit/*.json`.

"DEIM/local-FD" means that the nonlinear term is evaluated at `m = 16` interpolation points with a
3-point finite-difference stencil on the POD modes and then interpolated by DEIM. It is not DEIM
interpolation of the spectral operator. Rank `r = 8` throughout.

**Accuracy.** On the 40 test cases (snapshots at exactly aligned times) the DEIM/local-FD model has
a mean relative $L^2$ error of 4.31%, median 0.24% and maximum 52.9%; six cases exceed 10%, all steep,
low-viscosity ones. The plain ROM has 0.047% on the same protocol (0.152% with the historical
index-wise snapshot comparison used in the Results table above). DEIM interpolation itself is not
the problem: with the exact nonlinear term it matches the plain ROM (0.047%), and point selection is
well conditioned (cond($P^T\Psi$) 4 to 7). The error comes from the local finite-difference
evaluation: along steep trajectories the frozen Jacobian of the DEIM/local-FD system has positive
real eigenvalues (up to +4.8), while the plain ROM's are all negative, so this is a linear
instability of the reduced system rather than a time-step effect (quartering the step leaves the
error unchanged). A 5-point stencil lowers the median error to 0.17% but not the tail (mean 4.44%,
maximum 55.2%) and costs 58% more per stage. The error falls with grid resolution (mean 0.41% at
Nx = 512 on the first 10 test cases).

**Runtime.** The DEIM/local-FD model uses no FFTs and its per-stage cost does not grow with Nx,
whereas the full solver is FFT-bound. At the benchmark resolution Nx = 128 their runtime ratio
therefore depends on the FFT implementation. Averaged over the 40 test cases on one Apple M3 Pro
(`scripts/timing_numpy_sensitivity.py`):

| NumPy | FFT + inverse FFT, n = 128 | DEIM/local-FD vs full solver, historical step rule | revised step rule |
|---|---|---|---|
| 2.0.0 | 7.6 µs | 1.60x faster | 1.89x faster |
| 1.26.4 | 2.4 µs | 0.78x (slower) | 0.92x (slower) |

Earlier runs under NumPy 2.0.0 gave 1.52x to 1.57x (original timing protocol) and 1.67x and 1.98x
(time-aligned protocol, historical and revised rules). The about 2.0x previously quoted was a single
test case. The advantage grows with resolution under either NumPy version. With the revised rule, on
two test cases (a steep one, A = 1.92 and nu = 0.038, and a smooth one, A = 0.86 and nu = 0.098), it
is 2.5x at Nx = 1024 and 3.7x to 5.8x at Nx = 2048 with NumPy 1.26.4
(`followup_production_validation_run1.json`), and 4.1x and 6.2x to 7.1x with NumPy 2.0.0
(`followup_scaling2_run1.json`). The plain ROM is 0.35x to 0.41x the speed of the full solver at
Nx = 128 (NumPy 2.0.0).

**Time-step rule.** The historical reduced-model rule `dt = min(0.25 dx / max|u0|, 0.4 dx^2 / (2 nu))`
carries the full-grid resolution into a system with eight degrees of freedom, so its step count grows
about as Nx². The revised rule `dt = min(0.25 dx / max|u0|, 0.5 * 2.785 / (nu * rho(K)))`, with
`rho(K)` the spectral radius of the reduced diffusion operator (2.785 is RK4's real-axis stability
limit, 0.5 a fixed safety factor), gives errors identical to the historical rule on all 40 validation
and 40 test cases (largest per-case difference 3e-7), with every case stable. It is the default of
`rom_predict` and `deim_rom_predict`; the scripts that reproduce the historical tables pass
`timestep_policy="historical"`. Using the full solver's step count with no diffusion bound is unstable
at nu = 0.15; the revised rule is stable there.

**FNO.** The FNO originally received only the initial field, not the viscosity, so no model of that
kind could do better than about 2.44% in distribution (the error of the best viscosity-blind
predictor); it reached 2.93% ± 0.10% (3 seeds). The primary FNO baseline is therefore the
viscosity-aware variant, identical except for one extra input channel (32 more parameters):

| | in-distribution (40 cases) | unseen two-mode initial-condition family (20 cases) |
|---|---|---|
| POD-Galerkin ROM | 0.152% | 1.69% |
| DEIM/local-FD ROM | 4.35% | 3.11% |
| FNO, viscosity-aware (5 seeds) | 0.90% ± 0.03% | 22.57% ± 0.92% |
| FNO, viscosity-blind (historical, 3 seeds) | 2.93% ± 0.10% | 23.41% ± 0.53% |

(Historical index-wise protocol, as in the Results table; "±" is the standard deviation over seeds.)
Feeding the viscosity-aware model shuffled viscosities raises its in-distribution error to about 3.1%,
so it does use the input. On the unseen family (`A1 sin x + A2 sin 2x`, viscosity inside the training
range) the viscosity makes little difference: the error is dominated by the change of
initial-condition family, not by the missing input. The finding is specific to this network, 200
training trajectories and one training family.

**Summary.** The plain ROM is the most accurate and generalises best but is slower than the full
solver. The DEIM/local-FD model removes the full-grid cost but is unreliable on steep, low-viscosity
cases, and at Nx = 128 its speed advantage over the full solver depends on the FFT implementation.
The FNO is the fastest per query and, given the viscosity, accurate in distribution, but it does not
transfer to a new initial-condition family.

## Possible Extensions

Further possible extensions include comparison against DeepONet, uncertainty quantification over
the fitted models, and extension to higher-dimensional PDEs.

## References

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full mathematical formulation, system architecture,
and design rationale.
