# Adaptive tensorial POD integrators: baseline audit and pre-registered plan

Written at the end of Stage 1, before any new integrator was run on any data set. Branch
`research/adaptive-tensorial-pod-integrators` from `main` at `ae96509615066a71bca901d440d4a8ed510147b4`.
Results: `report/adaptive_integrators/`; report: `report/adaptive_tensorial_pod_integrators_study.md`.

## 1. Verified baseline

* `main` = `ae96509`, working tree clean. Tests: 229 passed, 1 skipped in both pinned environments
  (`requirements-lock.txt`, NumPy 1.26.4 / SciPy 1.13.1; `requirements-lock-numpy2.txt`, NumPy 2.4.6 /
  SciPy 1.17.1). All numbers of this study come from the NumPy 1.26.4 environment on the Apple M3 Pro.
* Read: `report/tensorial_pod_spectral_deim_study.md`, `report/burgers_spatial_convergence_study.md`,
  `src/tensorial_pod.py`, `src/reduced_timestep.py` (reduced CFL, reduced IF-RK4, scipy integrators),
  `scripts/spatial_convergence_study.py` (`run_policy`, stages `timestep`, `controlled`).

## 2. Operator conventions (checked numerically at N = 2048, `if_failure_run1.json` and Stage 1 log)

* The repository writes **da/dt = nu L a + Q(a, a)** with L = Phi^T D2 Phi; nu is outside L.
* L is symmetric to 1e-15 (hence normal, Henrici departure 0 up to rounding), negative definite, spectrum in
  [-rho, -1] with rho = 8.2e3, 5.8e4, 3.9e5 for r = 8, 16, 32 (the eigenvalue -1 is the sin x mode).
* Q is exactly symmetric in its last two indices; the analytic Jacobian J = nu L + sum_k (Q_ijk + Q_ikj) a_k
  agrees with a complex-step derivative to 1.6e-16 (complex step is exact up to rounding for polynomials).
* Mean mode: all POD modes have zero mean (training snapshots are zero-mean to 2e-16; r = 32 modes to
  2e-8), so the reconstructed mean is zero and mass is conserved trivially.
* The POD basis is orthonormal in the Euclidean grid norm, so coefficient-space errors equal reconstructed
  discrete-L2 field errors.

## 3. Why the reduced integrating-factor RK4 (P2) failed (`if_failure_run1.json`)

P2 of the previous study (Lawson IF-RK4, step 0.5 * 2.83 / (max|u0| sqrt(rho(L))), aligned to the 19 output
intervals) failed the criterion "error vs FOM <= 1.05 x P0's + 1e-6" at r = 16 on test cases 5 and 33. The
failures reproduce exactly (P2 1.15e-5 vs P0 2.64e-6; 5.14e-6 vs 1.22e-6). Candidate mechanisms:

| mechanism | evidence | verdict |
|---|---|---|
| genuine instability | errors fall monotonically under step refinement (9.2e-6, 1.2e-6, 1.0e-7, 7.6e-9 for test 5); all runs finite; no energy increase | ruled out |
| incorrect integrating-factor implementation | P2 converges to the independent RK4 reference (16x the default step count; self-check 16x vs 32x <= 2e-14) at observed orders 3.0, 3.5, 3.8 for test 5 | ruled out |
| floating point | a machine-epsilon perturbation of a0 changes the trajectory by <= 2e-16 | ruled out |
| snapshot-time misalignment | output indices are exact multiples of n/19; the reported time accumulates rounding (<= 1.1e-13 after 6232 steps) but the saved states are at exactly t_j = j/19 | ruled out |
| **excessive temporal truncation error from an accuracy-blind step** | the step comes from a stability (CFL) bound; at that step the time error is 9.2e-6 (test 5) and 4.3e-6 (test 33) but 1.5e-7 on the steepest validation case: it grows with front steepness while the step does not. At r = 16 the ROM's own error is about 1e-6, so the margin (1.05 x + 1e-6) is exceeded; at r = 8 the same size of time error (2.9e-5) passes because the ROM error is 1.5e-3 | **cause** |
| stiff order reduction of the Lawson scheme | present (observed orders 1.5 to 2.2 on the smoothest validation case, error then entirely in modes with h nu |lambda| > 1), but there the absolute error is <= 2e-9; on the failed cases only 9% to 15% of the error lies in stiff modes | secondary, not the cause |

Also found: the GLOBAL integrating-factor transform v = exp(-nu L t) a is unusable: exp(nu rho t) exceeds the
double-precision range (e^709) at t = 1 for every r = 16 and r = 32 case tested; integrating-factor methods
must reset the frame every step (stage-wise exponents e^{D (c_i - c_j) h} <= 1).

## 4. Data sets (fixed now)

| set | definition | role |
|---|---|---|
| training | the 200 historical training parameters; POD bases at N = 2048 from their float64 aligned FOM runs (`data/convergence_train_N2048.npz`, rebuilt deterministically if absent) | bases only |
| validation | the original 40 validation cases (seeds 100000+) | method / tolerance selection |
| stress-validation | predetermined grid A in {1.5, 1.75, 2.0} x nu in {0.010, 0.0125, 0.015, 0.0175, 0.020} (15 cases): the steep corner of the parameter box, defined by the front-width scale nu / A, not by any test error | method / tolerance selection |
| original test | the 40 historical test cases, unchanged. **Not fully independent for this question**: its P2 failures motivated the study and its results were seen in Stage 1 | reported separately |
| fresh holdout | 40 cases A ~ U(0.5, 2.0), nu ~ U(0.01, 0.1) from `numpy.random.default_rng(300000)` (A first, then nu, 40 each), plus 10 steep cases A ~ U(1.5, 2.0), nu ~ U(0.010, 0.020) from `default_rng(300001)`; drawn only in Stage 7 | final independent evaluation |

All cases use u0 = A sin x on N = 2048 points, T = 1, outputs at t_j = j/19.

## 5. Methods (fixed now)

| id | method | settings |
|---|---|---|
| B1 | reduced-CFL RK4 (P1, existing) | as in the previous study |
| B2 | reduced IF-RK4 (P2, existing) | as in the previous study |
| B3 | Radau, analytic Jacobian (existing usage) | rtol grid below |
| B3fd | Radau, finite-difference Jacobian | rtol grid below (answers "does the analytic Jacobian help") |
| E1 | RK45 (scipy) | rtol grid below |
| E2 | DOP853 (scipy) | rtol grid below |
| X1 | adaptive Lawson integrating-factor Dormand-Prince 5(4), frame reset each step, eigenbasis of L | rtol grid below |
| X2 | Krogstad ETD-RK4 with step-doubling error control (estimate / (2^4 - 1)) | rtol grid below |

Selection reasons: X1 is the adaptive form of the failed B2 (tests whether error control alone fixes it);
X2 integrates the nonlinear forcing with phi-functions instead of exponential weighting (tests whether
the exponential method's error constant is smaller than Lawson's on these problems). Exponential
Rosenbrock methods were not selected: they need phi-functions of the full Jacobian at every step (O(r^3)
per step for r <= 32) and address no separate question here. BDF is not included (Radau is the implicit
baseline; no question requires a second implicit method).

Tolerances: rtol in {1e-4, 1e-6, 1e-8, 1e-10}; atol = 1e-3 x rtol x max|a0| (scalar), the same for all
adaptive methods; error norm as in `src/adaptive_integrators.py` (RMS of e_i / (atol + rtol max(|a_i^n|,
|a_i^{n+1}|)) on the physical coefficients). Fixed-step baselines are also run with their step divided by 2,
4, 8 for Pareto curves.

## 6. Error references

* Time-integration error: relative space-time coefficient (= discrete field) error against the same reduced
  ODE solved by classical RK4 with 4 times the default step count, self-checked against 8 times on the
  stress-validation set (the previous default step already has time errors <= 4e-12 at r = 8, 16).
* Total error: against the Cole-Hopf solution of the previous study (N_ref = 8192). Its verified uncertainty
  is <= 2.2e-11 on the original test cases; for new cases the same evaluation is used without a separate
  FOM check (quadrature refinement is reported). Errors below about 1e-11 are not resolved.
* POD error: the trajectory error of the time-converged ROM against Cole-Hopf (not attributed to integrators).

## 7. Selection rule and acceptance (fixed now)

Targets on the maximum time-integration error over a case set: tau in {1e-4, 1e-6, 1e-8}. For each method,
rank r in {8, 16, 32} and target: the loosest tolerance (or largest step for B1, B2) whose maximum time error
over **validation and stress-validation together** is <= tau / 2 (a factor 2 margin for generalization).
Ties in tolerance are not possible; a method with no qualifying setting is "not eligible" for that target.
The chosen settings are written to `selection_frozen.json` and committed before the held-out sets are drawn.

Held-out evaluation (original test and fresh holdout, separately): a frozen configuration **meets** a target if
its maximum time error over the set is <= tau; the reliability rate is the fraction of cases with error <= tau.
"Fastest at a target" = lowest median end-to-end time among configurations that meet it on that set.

## 8. Timing protocol (fixed now)

As the controlled benchmark of the previous study: same machine and environment; for each case all frozen
configurations timed round-robin in one loop, 5 repetitions after a warm-up; two back-to-back runs started at
a 1-minute load average below 4. Reported separately: one-time preparation (eigendecomposition of L,
transformed tensor; phi-factors per step size are part of online time), per-trajectory setup, online
integration, reconstruction, end to end. Medians, p10/p90 over cases, within-case IQR.
