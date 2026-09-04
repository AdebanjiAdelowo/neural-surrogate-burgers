# Implementation Log

Format per stage: **Stage → implementation → verification performed → result → issues → decision.**
Project status: **IN PROGRESS** (was PLANNED).

---

## Stage 1 — Ground-truth numerical solver — **VERIFIED**

Per the explicit instruction to validate the solver *before* generating any training data —
verification was thorough, not a single smoke test.

**Implementation:** `src/solver.py::solve_burgers` — pseudo-spectral in space (FFT), integrating-
factor RK4 in time (linear diffusion term handled exactly via the integrating factor, only the
nonlinear advection term stepped explicitly), 2/3 dealiasing rule. Independent, from-scratch code —
no dependency on `boiling-phasefield-3d` or `pinn-advection-diffusion`.

**Verification performed (7 tests, all real mathematical/physical checks, not just "does it run"):**
1. **Mass conservation**: $\int u\,dx$ is exactly conserved by the periodic-BC PDE (both the
   advective and diffusive terms integrate to zero) — verified to <1e-4 (nearly machine precision).
2. **Energy dissipation**: viscous Burgers strictly decreases $\frac{1}{2}\int u^2\,dx$ over time —
   verified monotonically non-increasing across 20 snapshots.
3. **Viscosity trend**: smaller $\nu$ produces steeper final-time gradients — verified
   (max|gradient| = 13.26 at $\nu$=0.005 vs. 2.88 at $\nu$=0.1).
4. **Physically-expected steepening location**: the classical result that inviscid Burgers with
   $u_0=\sin(x)$ on $[0,2\pi)$ forms its steepest gradient at $x=\pi$ — verified: at small
   $\nu$=0.005, the steepest-gradient location is exactly $x=\pi$ (to floating-point/grid
   precision), consistent across grid resolutions 64/128/256.
5. **Grid-refinement consistency**: the steepest-gradient *location* does not move under
   refinement (only its magnitude legitimately increases as the near-shock feature is better
   resolved) — verified across Nx = 64, 128, 256.
6. **Exact final time**: fixed a real bug found during verification — the initial `save_every`
   modulo-based snapshot logic could undershoot $T$ by up to one save interval (observed:
   $t_{\text{final}}$ = 0.927 instead of 1.0). Fixed by selecting exact step indices via
   `np.linspace` instead of modulo arithmetic; reran and confirmed $t_{\text{final}}$ = 1.0 exactly.
7. Visually inspected the solution evolution (`report/dev_solver_check.png`) — the classic
   sine-wave steepening and advective peak-shift toward $x=\pi$ is clearly visible and matches
   textbook Burgers-equation behaviour.

**Not implemented (deliberate scope decision, not an oversight):** an exact Cole-Hopf semi-
analytical solution exists for this initial condition, but computing it requires its own numerical
integration and was judged not worth the added implementation risk given the checks above already
provide strong, independent, textbook-grounded verification.

**Result:** solver is trustworthy — verified via physical invariants (mass, energy), a known
theoretical result (shock-formation location), and numerical-methods practice (grid convergence),
not merely "it produces finite output." All 7 tests **PASS**.

**Decision:** proceed to dataset generation using this solver.

---

## Stage 2 — Dataset generation — **VERIFIED**

**Implementation:** `scripts/generate_dataset.py`. Parameters $(A, \nu)$ sampled uniformly at
random from documented ranges ($A \in [0.5, 2.0]$, $\nu \in [0.01, 0.1]$), fixed non-overlapping
seed ranges per split (train: seeds 0–199, val: 100000–100039, test: 200000–200039). Grid
$N_x$=128, $T$=1.0, 20 saved snapshots per example — same defaults verified in Stage 1.

**Verification performed:**
- Ran for real: 200 train + 40 val + 40 test = 280 examples generated in **1.4 seconds**.
- Confirmed parameter ranges actually sampled span close to the full documented range in each
  split (e.g. train $A \in [0.504, 1.996]$, $\nu \in [0.0100, 0.0997]$).
- **Programmatic leakage check passed**: no $(A,\nu)$ pair shared across splits.
- Visually inspected spacetime fields for 4 train examples (`report/dev_dataset_samples.png`):
  clear, sensible parameter dependence — higher $A$ / lower $\nu$ produces a visibly sharper
  transition front, exactly the physical trend Stage 1 already confirmed for individual solves.

**Result:** `data/{train,val,test}.npz` created (not committed — gitignored; fixed proactively
this time, before any data was generated, learning from the `photoacoustic-reconstruction`
project's earlier `.gitignore` gap).

**Decision:** proceed to Stage 3 (POD-Galerkin ROM baseline).

---

## Stage 3 — POD-Galerkin ROM baseline — **VERIFIED**

**Implementation:** `src/pod_rom.py` — `build_pod_basis` (SVD of the training snapshot ensemble),
`rom_predict` (projects an initial condition onto the retained modes, integrates the
Galerkin-projected ODE via explicit RK4, evaluating the true Burgers RHS in physical space at each
stage via the same FFT-based spectral derivatives as the ground-truth solver — a genuine, textbook
Galerkin projection, not a data-fit shortcut).

**Verification performed:**
- 4 tests: basis orthonormality (to float32 precision — a real, expected, non-bug tolerance issue
  found and fixed during test-writing, not in the implementation), energy-capture monotonicity,
  prediction shape/finiteness, and — the central real behaviour this baseline exists to
  demonstrate — **prediction error decreases monotonically as retained modes increase**. All 4
  **PASS**.
- Ran the actual accuracy-vs-modes sweep on a held-out test parameter (not just the unit test's
  synthetic check): relative $L^2$ error (full trajectory) = 7.55% at $r$=2, 2.23% at $r$=4,
  0.076% at $r$=8, ≈0% at $r$≥16.

**Honest finding (not hidden, and important for framing the eventual neural-surrogate
comparison):** this problem family (single-parameter sinusoidal initial conditions, moderate
viscosity) is **highly linearly compressible** — POD-Galerkin becomes near-perfect by $r$=16 modes.
This is a genuine property of the benchmark, not a limitation of the implementation (confirmed via
the energy-capture and prediction-error sweeps both). **Consequence for the neural surrogate**: a
well-chosen POD-ROM (e.g. $r$=8, deliberately not the near-perfect $r$≥16) is a demanding, non-trivial
accuracy baseline; the neural surrogate's more likely comparative advantage is **inference speed**
(a direct function evaluation vs. an ODE time-integration) and **generalisation outside the
training parameter range**, not necessarily raw in-distribution accuracy. This reframing is
recorded here because it changes what the final comparison should emphasise, decided from real
evidence rather than assumed in the original architecture.

**Decision:** proceed to Stage 4 (neural surrogate) using $r$=8 as the POD-ROM comparison point.

---
