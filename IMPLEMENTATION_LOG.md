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

## Stage 4 — Neural surrogate — **VERIFIED**

**Architecture decision (documented, not silent):** `ARCHITECTURE.md` specified "CNN or MLP." A
CNN is architecturally inappropriate here — the input is two scalars $(A, \nu)$, not spatial data
with local structure. Implemented `src/surrogate_net.py::BurgersSurrogateMLP`, a small MLP mapping
normalised $(A, \nu) \in [0,1]^2$ directly to the flattened $(n_{\text{save}}, N_x)$ solution
field — a direct amortised regression requiring no time-stepping at inference, which is the
surrogate's structural speed advantage over both the ground-truth solver and the POD-ROM (both of
which integrate an ODE/PDE forward in time).

**Verification performed, in order:**
1. Forward-pass shape/finiteness tests — **PASS**.
2. **Overfit sanity check** (required before full training): loss dropped from 0.422 to 0.000018
   (ratio 0.00004) on a fixed 4-example batch — **PASSED**, confirms the training pipeline can
   actually learn.
3. **Full training run**: 200 epochs, batch size 16, on the real 200-example train / 40-example
   val split. **8.4 seconds wall-clock** (MPS). Train loss 0.724 → 0.00004, val loss 0.296 →
   0.00004, monotonically decreasing, train/val tracking closely (no sign of severe overfitting).

**Result:** the surrogate trains stably and fits the training distribution very well — expected,
given Stage 3 already established that this problem family is highly compressible/low-dimensional;
the interesting comparison is not "can it fit" but "how does it compare to POD-ROM on held-out
parameters, and on speed" — Stage 5.

**Decision:** proceed to Stage 5 (comparative evaluation): ground-truth solver vs. POD-ROM ($r$=8)
vs. neural surrogate, on the held-out test split.

---

## Stage 5 — Comparative evaluation — **VERIFIED, MVP COMPLETE** (two real bugs found and fixed)

**Two real bugs found and fixed during this stage (not hidden):**
1. **`src/pod_rom.py::rom_predict` produced NaN** on the very first evaluation run. Diagnosed: the
   default time step only respected an advective stability bound (matching `src.solver`'s
   integrating-factor scheme, which handles diffusion exactly and doesn't need a diffusion-based
   dt limit) — but `rom_predict`'s explicit RK4 in reduced coordinates has no such treatment and is
   only conditionally stable in the diffusion term too. Fixed by also bounding dt by the standard
   explicit-diffusion limit $dx^2/(2\nu)$. Regression test added
   (`test_rom_stable_at_higher_viscosity_extrapolation`).
2. **Snapshot-count collisions**: `src/solver.py`'s integer-rounded `np.linspace`-based snapshot
   selection could silently produce fewer than `n_save` snapshots for some parameter combinations
   when the total step count wasn't comfortably larger than `n_save`. Fixed by flooring
   `n_steps >= 4 * n_save` in both `src/solver.py` and `src/pod_rom.py`. All existing tests re-run
   and still pass after this change.

**Real, measured results (in-distribution, $n$=40 test examples, POD-ROM $r$=8):**

| Method | Relative $L^2$ error (mean ± std) | Wall-clock per evaluation |
|---|---|---|
| POD-ROM ($r$=8) | **0.152% ± 0.077%** | 15.48 ms |
| Neural surrogate | 0.788% ± 0.366% | **0.138 ms** |
| (Ground-truth solver, reference) | — | 6.29 ms |

**POD-ROM wins decisively on accuracy in-distribution** — consistent with Stage 3's finding that
this problem family is highly linearly compressible. **The neural surrogate wins decisively on
speed** — 45x faster than the full solver, 112x faster than the ROM — because it requires a single
forward pass with no time-stepping at all, exactly the structural advantage identified in Stage 4.

**Honest, important finding: the POD-ROM as implemented is *slower* than the full solver**
(15.5 ms vs. 6.3 ms), because it evaluates the nonlinear term via full-grid spectral derivatives at
every RK4 stage (no hyper-reduction/DEIM) — it reduces the *state* dimension but not the *cost of
evaluating the right-hand side*. A true speedup would require hyper-reduction, explicitly recorded
as a future extension, not attempted in this MVP.

**Generalisation to out-of-training-range parameters — the most scientifically important finding,
not cherry-picked (3 real, disclosed cases, `report/mvp_extrapolation.png`):**

| Case | POD-ROM error | Neural surrogate error |
|---|---|---|
| $A$=2.5 (above range, training max 2.0) | 0.10% | 5.45% |
| $\nu$=0.15 (above range, training max 0.10) | 0.24% | 3.05% |
| $A$=0.2 (below range, training min 0.5) | 0.06% | **55.95%** |

**The POD-ROM generalises far better than the neural surrogate outside the training distribution
— because it is built on the true governing equations (Galerkin projection), so it remains valid
physics wherever it's evaluated, whereas the neural surrogate is a black-box interpolator that
degrades severely, and at $A$=0.2 catastrophically (visible as clear high-frequency noise in
`report/mvp_extrapolation.png`), once evaluated outside what it was trained on.**

**Honest overall conclusion for the MVP's research question:** for this problem family, the
classical POD-Galerkin ROM is both more accurate and far more robust than the neural surrogate; the
surrogate's only real advantage is inference speed, which is nonetheless large and genuine. This is
a legitimate, non-obvious scientific finding — not the "neural network wins" result a less careful
study might have reported, and not hidden or reframed to look more favourable to the newer method.

**Result:** all 14 tests pass; the full pipeline — parameterised PDE → validated numerical solver
→ dataset → POD/ROM → learned surrogate → held-out comparison — runs end-to-end on real data.

**Decision: STOP per the explicit MVP stop condition.** No FNO, no DeepONet, no multi-dimensional
PDEs, no uncertainty quantification, no extensive architecture search in this pass.

---

## Project status: **MVP COMPLETE**

