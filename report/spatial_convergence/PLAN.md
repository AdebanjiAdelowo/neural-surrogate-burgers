# Spatial convergence study: baseline and pre-registered plan

Written at the start of the study (branch `research/burgers-spatial-convergence`, from `main` at
`ed6ffdea502ab6fd65479f1e99409e21ccd59291`), before any new experiment was run. Results go to
`report/spatial_convergence/`; the report is `report/burgers_spatial_convergence_study.md`.

## 1. Verified baseline (Stage 1)

* Working tree clean; `main` = `ed6ffde` (merge of PR #3).
* Test suite: 179 passed, 1 skipped in both pinned environments (`requirements-lock.txt`, NumPy 1.26.4 /
  SciPy 1.13.1; `requirements-lock-numpy2.txt`, NumPy 2.4.6 / SciPy 1.17.1), Python 3.11.13, Apple M3 Pro.
* All numbers of this study are produced in the NumPy 1.26.4 environment unless labelled otherwise.

## 2. Numerical formulation actually implemented (read from the code, checked numerically)

* Domain x in [0, 2 pi), x_j = 2 pi j / N; IC u0 = A sin x; T = 1; 20 snapshots.
* Wavenumbers k = 2 pi fftfreq(N, 2 pi / N) (integers, Nyquist k = -N/2); unnormalised numpy FFT.
* Dealiasing: the mask |k| < (2/3) max|k| = N/3 is applied to the transform of u^2 only, before the
  derivative: N(u) = Re ifft(-0.5 i k mask fft(u^2)). The state itself is not filtered, but since the IC
  is band-limited and nothing forces modes with |k| >= N/3, those stay exactly zero: the effective
  resolved bandwidth is |k| <= ceil(N/3) - 1 (42 at N = 128, 341 at N = 1024).
* Time stepping (`src/solver.py`): Lawson integrating-factor RK4 in Fourier space, diffusion exact
  (E = exp(-nu k^2 dt/2)); dt = 0.25 dx / max|u0|, n_steps = max(ceil(T/dt), 80), rounded up to a
  multiple of 19 when `align_snapshots=True` (then snapshots are at exactly t_j = j/19). The step count
  therefore grows like N.
* Outputs are cast to float32 (relative floor about 6e-8): any convergence below about 1e-7 needs a
  float64 variant of the same scheme.
* Reduced models (`src/tensorial_pod.py`): RK4; default "reduced" policy
  dt = min(0.25 dx / max|u0|, 0.5 * 2.785 / (nu rho(L))), L = Phi^T D2 Phi. Both bounds can grow with N:
  the advective bound through dx, the diffusion bound through rho(L).

## 3. Plan

**Float64 solver and Fourier tools (new module, baseline untouched).** A float64 copy of the FOM scheme
with explicit step counts (test: float32 cast of its output equals `solve_burgers` bit for bit).
Cross-resolution errors are computed from normalised Fourier coefficients c_k = fft(u)/N on the common
mode set (coarse coefficients zero-padded; a coarse Nyquist coefficient is split equally between +N/2
and -N/2), so ||u||_L2^2 = 2 pi sum |c_k|^2 exactly for trigonometric interpolants (Parseval); no
grid-point indexing between grids.

**Exact solution (Cole-Hopf / Hopf integral).** u(x,t) = int ((x-y)/t) e^{-G/(2 nu)} dy /
int e^{-G/(2 nu)} dy, G = F0(y) + (x-y)^2/(2t), F0 = antiderivative of u0 (periodic for zero-mean u0),
evaluated by the trapezoidal rule on a window around x with a log-sum-exp shift; verified by quadrature
refinement and against the finest FOM.

**Cases (fixed now).** Corners and centre of the parameter box: (A, nu) = (0.5, 0.1), (2.0, 0.1), (0.5, 0.01),
(2.0, 0.01), (1.25, 0.055); the steepest previous test cases 5 (1.89, 0.0158), 33 (1.50, 0.0122), 0 (1.92,
0.0383); extrapolation (2.5, 0.05), (1.0, 0.15), (0.2, 0.05); plus all 40 test cases, 40 validation cases
(time-step design only) and the 20 two-mode cases.

**Stage 2, spatial.** N in {64, ..., 2048} (+4096, 8192 where needed), common fine time step, errors vs the
reference (space-time, per snapshot, final time) and consecutive E_{N,2N}.

**Stage 3, temporal and reference.** Fixed N, n_steps x {1, 2, 4, 8} against n_steps x 64; observed order
only where errors are above the spatial / float64 floor. Reference = Cole-Hopf, checked against the finest
FOM with a refined step; uncertainty = the larger of the quadrature-refinement change and the FOM
discrepancy.

**Stage 4, diagnostics.** |c_k(t)| spectra, tail energy, resolved bandwidth vs front width; mass, energy
identity dE/dt = -nu ||u_x||^2 (discrete aliasing term), max|u_x|, min/max (maximum principle), behaviour
at the front.

**Stage 5, tensorial POD.** For N in {128, 256, 512, 1024, 2048}: POD from the 200 historical training
parameters (float64 FOM, aligned), r in {4, 8, 12, 16, 24, 32}; errors vs the FOM at the same N, vs the
reference (via Fourier padding), and the POD projection errors of both; extrapolation and two-mode family.

**Stage 6, time stepping (pre-registered criteria).** Candidate policies are designed and selected on the
40 validation cases only and then evaluated once on the 40 test cases. A policy is acceptable if it is
stable on every case and its per-case error against the same-N FOM is at most 1.05 times that of the
current "reduced" policy plus 1e-6 absolute. Among acceptable policies the comparison is at matched
accuracy (runtime and RHS evaluations); solver tolerances and rejected steps are reported for adaptive
methods.
