# PROPOSED replacement for the README "Hyper-Reduction and FNO Extension" section (version 2)

Status: proposal only. Not applied to README.md, not committed. Every number is in `report/audit/*.json`.
Historical numbers are kept and labelled; nothing is silently replaced. This version supersedes the first
proposal (its numbers are unchanged where the first audit measured them).

## Methods, and what each one costs

| Label | Parameter-aware? | Offline | Online, per trajectory |
|---|---|---|---|
| **FOM** | yes | none | Pseudo-spectral, integrating-factor RK4, Nx = 128, 2 FFTs per nonlinear evaluation |
| **POD-Galerkin** (r = 8) | yes | SVD, 0.03 s | Explicit RK4 on the reduced ODE; every stage reconstructs `u = Phi a` on the full grid and evaluates F with 5 FFTs |
| **POD-DEIM / local-FD** (r = 8, m = 16) | yes | + nonlinear-snapshot SVD and point selection (about 0.17 s, as recorded earlier) | F evaluated at 16 points from a 3-point finite-difference stencil on Phi rows, then interpolated; no FFT and no full-grid array in the time loop (NaN-poisoning test) |
| **FNO, historical** | **no**: input `[u0(x), x]`, no viscosity | 17 to 20 s per seed (MPS, shared machine) | one forward pass |
| **FNO, nu-aware** | yes: input `[u0(x), x, nu]` | 23 s per seed (same conditions) | one forward pass, same cost |

"DEIM" here means DEIM interpolation of values computed by a local finite-difference formula, not of the spectral
operator. Say "DEIM/local-FD".

## Why the plain ROM is slow

90% of its solver time is the full-grid right-hand side: 5 FFTs, 33 microseconds per call against 13 for the
FOM's 2 FFTs. The reduced linear algebra is about 6%. So the ROM is slower than the full solver in every case and at every grid size tested
(0.16x to 0.41x for Nx = 128 to 2048, the low end at high viscosity where its step rule binds), and its per-stage
cost grows with Nx (32 to 123 microseconds for Nx = 128 to 2048).

## What DEIM/local-FD removes, and what it does not

Per stage it removes the full-grid cost: 6 to 7 microseconds at every Nx from 128 to 2048.
End to end, the result depends on the time-step rule (steep case, test case 0):

| Nx | steps: FOM / historical rule / revised rule | DEIM speed-up over FOM, historical rule | DEIM speed-up, revised rule |
|---|---|---|---|
| 128 | 171 / 171 / 171 | 1.9x | 1.9x |
| 256 | 323 / 323 / 323 | 2.2x | 2.2x |
| 512 | 627 / 1273 / 627 | 1.4x | 2.9x |
| 1024 | 1254 / 5092 / 1254 | 1.0x | 4.1x |
| 2048 | 2508 / 20330 / 2508 | 0.84x | 7.1x |

Solver-only, aligned FOM as reference, median of interleaved repetitions; reconstruction-inclusive numbers differ by
under 10% (2 to 3 repetitions at the largest Nx). The historical rule `dt = min(0.25 dx / max|u0|, 0.4 dx^2 / (2 nu))` carries the FULL-grid resolution
into a reduced system, so its step count grows about Nx^2 (at high viscosity, 13072 steps against the FOM's 570 at
Nx = 1024). The revised rule is
`dt = min(0.25 dx / max|u0|, 0.5 * 2.785 / (nu * rho(K)))`, where `rho(K)` is the spectral radius of the reduced
diffusion operator (2.785 is RK4's real-axis stability limit; 0.5 is a fixed safety factor, not tuned). It gives
identical errors to the historical rule on all 40 validation and 40 test cases, all stable, and needs 15% fewer steps
at Nx = 128 (123 against 145 on average).

Why the constant matters: the r = 8 POD modes carry near-cutoff wavenumbers, so the reduced diffusion is genuinely
stiff (`rho(K)` = 1732 for the ROM and 1206 for DEIM/local-FD, against 4096 and 1660 for the full-grid Laplacians).
The historical constant is about 3x conservative at nu = 0.1, not arbitrary. Using the FOM's step count with no
diffusion bound at all is stable inside the training range (nu <= 0.1) but produces NaN at nu = 0.15, A = 1;
the revised rule handles that case. The multiplier study agrees with the eigenvalue limits: ROM failures start
exactly above dt = 2.785 / (nu rho) at high viscosity, and at low viscosity both models tolerate 32 times the
historical step with no change in error.

## Historical versus workload-average speed-up

| | value | what it is |
|---|---|---|
| DEIM vs FOM, historical | about 1.9x | one case (test case 0), equal step counts |
| DEIM vs FOM, 40-case average, historical rule (native protocol) | 1.52x to 1.57x | three runs, machine shared with other jobs |
| DEIM vs FOM, 40-case average, aligned protocol, historical rule | 1.67x | 0.88x to 2.30x per case, faster in 37 of 40 |
| DEIM vs FOM, 40-case average, aligned protocol, revised rule | 1.98x | 1.78x to 2.61x per case, faster in 40 of 40 |
| plain ROM vs FOM, same protocols | 0.35x to 0.41x | slower in every case |

## DEIM/local-FD accuracy limitation

Re-checked on all 40 test cases (r = 8, m = 16, time-aligned):

| What is evaluated | mean | median | max | cases above 10% |
|---|---|---|---|---|
| plain POD-Galerkin | 0.047% | 0.042% | 0.098% | 0 |
| DEIM interpolation only (exact F, full grid) | 0.047% | 0.042% | 0.103% | 0 |
| local-FD F on the full grid, no DEIM | 1.19% | 0.26% | 11.9% | 1 |
| DEIM/local-FD, 3-point stencil | 4.31% | 0.24% | 52.9% | 6 |
| DEIM/local-FD, 5-point stencil | 4.44% | 0.17% | 55.2% | 6 |

- The interpolation is not the problem: conditioning is good (cond of P^T Psi 4 to 7, `||M||_2` 4 to 8) and the exact-F
  version matches the ROM.
- The finite-difference evaluation is: along steep, low-viscosity trajectories the frozen Jacobian of the DEIM/local-FD
  system has positive real eigenvalues (up to +4.8 at m = 16, +11.8 at m = 6), while the plain ROM's are all
  negative. It is a continuous-time linear instability, and quartering the time step does not change the error at m = 16.
- A 5-point stencil lowers the median error by 28% but not the tail, and costs 58% more per stage (11.9 against
  7.5 microseconds). At Nx = 256 and 512 the mean error changes from 0.55% to 0.46% and from 0.41% to 0.38%. Higher
  resolution helps far more than a higher-order stencil.
- (r = 8, m = 12) is a separate failure: 1919% mean error under the revised rule, but 6.1% when the step is
  quartered, so that one is a time-step instability. The historical rule fails it too.
- All selected points lie in the left half of the periodic domain and the nearest to the shock is one cell away;
  adjacent points occur for every m.

## Snapshot times

Snapshots were saved at step index `floor(j n_steps / 19)`, so saved times deviate from `j/19` by up to 0.012 and
differ between methods with different step counts (23 of 40 test cases). The historical error compared by index.
An opt-in `align_snapshots=True` (default off, so all historical results are bit-identical) rounds the step count up
to a multiple of 19 and saves at exactly `j/19`.

| | historical index-wise | time-aligned |
|---|---|---|
| plain ROM, r = 8, test mean | 0.1518% | 0.047% |
| DEIM/local-FD, test mean | 4.35% | 4.31% |

Regenerating all datasets with aligned snapshots and rebuilding the basis changes nothing that matters:
ROM 0.0466% to 0.0467%, DEIM 4.311% to 4.315%, out-of-family ROM 1.675% to 1.677% and DEIM 3.105% to 3.108%, the
same 16 DEIM points, principal-angle cosines above 0.999999. Legacy snapshots can instead be time-corrected: cubic
resampling at their true times takes the legacy-versus-aligned error from 0.308% to 0.00009%.
`data/*.npz` are the provenance and stay canonical until you decide otherwise.

## Out-of-family definition

Twenty two-mode initial conditions `A1 sin x + A2 sin 2x`, `A1 in [0.5, 2]`, `A2 in [0.1, 0.6]`, viscosity inside the
training range. Bases and networks are built from single-mode data only. This is a shift in the initial-condition
family, not in viscosity or amplitude range. ROM 1.69%, DEIM 3.11% (basis-transfer limits, not learning).

## Fair comparison (40 in-distribution and 20 out-of-family held-out cases)

| | in-distribution | out-of-family | online | offline | caveat |
|---|---|---|---|---|---|
| FOM | reference | reference | 6.3 to 7.0 ms | none | Nx = 128 is itself 0.012% (median) and 3.9% (worst case) from Nx = 1024 |
| POD-Galerkin | 0.047% aligned (0.152% historical) | 1.68% | 17 to 20 ms (0.35x to 0.41x FOM) | 0.03 s | full-grid RHS; parameter-aware |
| POD-DEIM/local-FD | 4.31% mean, 0.24% median, max 52.9% | 3.11% | 3.6 to 4.2 ms (1.7x to 2.0x FOM at Nx = 128) | about 0.2 s more | linear instability on steep cases; parameter-aware |
| FNO, historical | 2.93% +/- 0.10% (3 seeds) | 23.41% +/- 0.53% | 0.48 to 0.56 ms CPU, 1.5 ms MPS, batch 1 | 17 to 20 s | given no viscosity; the best possible without it is about 2.44% |
| FNO, nu-aware | 0.90% +/- 0.03% (5 seeds) | 22.57% +/- 0.92% | 0.57 ms CPU, same cost | 23 s | same architecture and data as the historical one |

There is no single winner. The ROM is the most accurate and robust; DEIM/local-FD is the cheapest reduced model but
unreliable in the steep regime; the FNO is fastest per query and, once given the viscosity, accurate in distribution,
but fails on a new initial-condition family whether or not it knows nu.

## FNO scope

- The historical FNO was viscosity-blind. Its 2.93% is within 0.5 points of the 2.44% floor of any viscosity-blind
  predictor (the trajectory average over nu), so the in-distribution ROM-versus-FNO gap was mostly this missing input.
- Giving it nu, and changing nothing else (32 extra parameters), gives 0.90%. Feeding the nu-aware model shuffled
  viscosities raises the error back to about 3.1%, so it does use nu.
- Out-of-family it barely moves (23.23% blind retrained against 22.57% aware). The viscosity-blind floor for that set
  is 2.45%, so about 20 points of the error are due to the initial-condition-family shift, not to missing nu.
- Replicated on the aligned datasets: blind 2.946%, aware 0.883%; out-of-family 23.41% and 22.18%.
- The FNO remains a single-shot, fixed-horizon (T = 1), fixed-grid (Nx = 128) regressor trained on 200 trajectories.
