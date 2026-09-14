# Research Extension: DEIM/Local-FD Hyper-Reduction and an FNO Baseline for Parametric Burgers

**Status: implemented, tested, and put through a self-review scrutiny pass on the branch
`research/rom-deim-fno-burgers`, baseline commit `6aaa282`. Not yet committed or merged, and not
externally/third-party peer reviewed.** This is revision 2 of this document, written after a
review pass that checked the original claims mathematically and experimentally rather than
accepting them on inspection. Three
substantive corrections came out of that review and are recorded throughout: (1) the method's name
was imprecise, (2) the novelty claim was insufficiently hedged, (3) one causal explanation ("DEIM
degrades at high rank because the SVD fits noise") was tested directly and found **not** supported
by the evidence, and is retracted in favour of a different, evidence-backed mechanism.

**Novelty classification (unchanged from revision 1, reaffirmed after review): comparative/
methodological study with one modest, disclosed technical contribution, not new scientific
discovery.** Burgers' equation is an extremely standard SciML benchmark. Do not describe this
document's results as novel research without new evidence beyond what is recorded here.

## 1. Research question

> For the 1D viscous Burgers' equation with a fixed single-mode sinusoidal initial-condition
> family, how do (i) a full pseudo-spectral solver, (ii) a POD-Galerkin ROM, (iii) a DEIM/local-FD
> hyper-reduced POD-Galerkin ROM, (iv) a parameter-to-field MLP surrogate, and (v) a 1D Fourier
> Neural Operator compare on in-distribution accuracy, generalization to out-of-range viscosity and
> amplitude, generalization to a *different* initial-condition family never seen during training or
> ROM construction, and computational cost -- accounting fairly for both offline
> (basis/interpolation-point construction, training) and online (per-evaluation) cost on identical
> hardware?

## 2. Mathematical formulation

Governing equation, domain, and parameter family unchanged from the verified baseline (see
`ARCHITECTURE.md`, `README.md`):
$$u_t + u u_x = \nu u_{xx}, \quad x \in [0, 2\pi) \text{ periodic}, \quad u(x,0) = A\sin(2\pi x/L)$$
$A \in [0.5, 2.0]$, $\nu \in [0.01, 0.1]$, $N_x = 128$, $T=1.0$, 20 saved snapshots, 200 training /
40 validation / 40 test trajectories (unchanged, already leakage-checked by disjoint RNG seed
ranges).

## 3. What is mathematically implemented, precisely (Section 1 of the review)

This section replaces revision 1's informal description with a full derivation, per the review's
explicit request to check whether "DEIM adapted to a pseudo-spectral discretization" is accurate
terminology.

**3.1 The full-order nonlinear term.** $F(u) = -uu_x + \nu u_{xx}$, computed via
$u_x = \mathcal{F}^{-1}[ik\hat u]$, $u_{xx} = \mathcal{F}^{-1}[-k^2\hat u]$, and the flux dealiased:
$\widehat{-uu_x} = -\tfrac{1}{2}ik\,\mathcal{F}[u^2]\odot(\text{2/3-rule mask})$. Identical to
`src/pod_rom.py`'s `_burgers_rhs_physical`, unchanged.

**3.2 POD projection.** $u \approx \Phi a$, $\Phi \in \mathbb{R}^{N_x\times r}$ ($r=8$, SVD of
training-trajectory snapshots), Galerkin-projected dynamics $\dot a = \Phi^T F(\Phi a)$ -- exact,
given $F$ evaluated exactly on the full grid. This is the plain ROM baseline, unchanged.

**3.3 Nonlinear snapshot basis and DEIM interpolation.** A second basis
$\Psi \in \mathbb{R}^{N_x\times m}$ is built via SVD of an ensemble of $F(u)$ evaluated (exactly,
spectrally) along the training trajectories. Given $\Psi$, DEIM's greedy algorithm selects $m$
interpolation indices $\{p_1,\dots,p_m\}$ (verified line-by-line against Algorithm 1 of
Chaturantabut & Sorensen 2010 -- see `select_deim_points` in `src/deim.py`) so that
$$F(u) \approx \Psi (P^T\Psi)^{-1} P^T F(u), \qquad \text{provided } P^TF(u) \text{ -- the TRUE nonlinear-term values at the } m \text{ points -- is available.}$$
The reduced-Galerkin-DEIM operator $M = \Phi^T\Psi(P^T\Psi)^{-1} \in \mathbb{R}^{r\times m}$
(`build_deim_projector`) is exact, standard DEIM-Galerkin math, computed once offline.

**3.4 Where this implementation departs from textbook DEIM.** Obtaining the TRUE
$P^TF(\Phi a)$ for a pseudo-spectral discretization would require reconstructing $u$ on the
*entire* grid and running the full FFT-based derivative pipeline ($O(N_x\log N_x)$, independent of
$m$) -- which defeats the purpose of hyper-reduction. DEIM's own literature says nothing about how
to obtain $P^TF(u)$ cheaply when the underlying discretization is not local/sparse (finite
difference, finite element); the seminal paper's own test problems (1D FitzHugh-Nagumo, 2D
miscible viscous fingering) are both finite-difference discretized, where a point's contribution to
$F$ genuinely only depends on a few neighbours.

**What is actually computed instead** (`deim_nonlinear_at_points`, and its 4th-order variant
`deim_nonlinear_at_points_4th_order` added during review, see Section 8): $u$ is reconstructed
*exactly* (no approximation) at each DEIM point's small local stencil neighbourhood via
$u_{\text{support}} = \Phi_{\text{support}}\,a$ (this step is exact -- restricting $\Phi a$ to a
row subset is not an approximation). Then $u_x, u_{xx}$ *at the DEIM points* are estimated via a
**local finite-difference formula** (2nd-order 3-point by default, 4th-order 5-point tested in
Section 8) applied to those exact local values -- introducing genuine FD truncation error that is
**not present** in the true spectral $F$.

**3.5 The resulting reduced nonlinear operator.**
$$\dot a \approx M \cdot \tilde F_P(a), \qquad \tilde F_P(a) \in \mathbb{R}^m \text{ the LOCALLY-FD-APPROXIMATED nonlinear-term vector at the } m \text{ DEIM points (NOT the exact } P^TF(\Phi a)\text{)}.$$
$M$ is exact given $\Phi, \Psi, P$; $\tilde F_P$ is not exact.

**3.6 Is this mathematically sound?** Yes, as a *consistent approximation scheme*: as $N_x \to
\infty$ at fixed physical stencil width (i.e. $dx \to 0$), the local-FD truncation error vanishes
and the scheme converges to the (impractical) exact-DEIM limit. It is a legitimate, well-posed
numerical approximation, but it is a **composition of two distinct approximations** (DEIM's sparse
interpolatory projection, exact given its inputs; a local-FD surrogate for those inputs, not
exact), not a single, textbook DEIM approximation.

**3.7 Most precise terminology (the review's explicit ask).** "DEIM adapted to a pseudo-spectral
discretization" (revision 1's wording) is imprecise -- it can be read as implying DEIM *itself* was
mathematically adapted, when in fact DEIM's own math (point selection, the $M$ operator) is
untouched, standard DEIM-Galerkin. **The correct, precise name is: a hybrid DEIM/local
finite-difference (FD) hyper-reduction scheme** -- specifically, a DEIM-Galerkin reduced-order
model in which the interpolated nonlinear-term values are obtained via a local, low-order
finite-difference surrogate rather than exact evaluation of the underlying spectral nonlinear
operator. This document, the module docstrings, and all subsequent references use this
terminology; "genuine DEIM hyper-reduction of the spectral nonlinear term" (implying exactness)
would be **inaccurate**.

## 4. Literature positioning, corrected (Section 2 of the review)

Revision 1 stated "no literature precedent was found... specifically" without documenting the
search. A targeted re-search was performed for this revision across the query combinations: *DEIM
spectral methods*, *DEIM pseudo-spectral*, *DEIM Fourier spectral*, *hyper-reduction spectral
methods*, *POD-DEIM spectral discretization*, *empirical interpolation spectral methods*, *EIM/DEIM
Fourier methods*, *hyper-reduced Burgers spectral*, *gappy POD spectral nonlinearities*,
*GNAT/hyper-reduction spectral discretizations*, plus a follow-up on the closest hit found.

**Closest precedent identified: Mizan, Olshanskii & Timofeyev (2026), "Parametric Reduced Order
Models for the Generalized Kuramoto-Sivashinsky Equations," arXiv:2502.02718.** This paper studies
POD-DEIM ROMs for a periodic, Kuramoto-Sivashinsky-type nonlinear PDE -- structurally close to this
project's setting. Reading the paper's own methodology section (Section 3, via the arXiv PDF)
found: *"For the spatial discretization of (1) we use a uniform grid on $[0,L]$ and apply a
**finite-difference** discretization where nonlinear fluxes $F(u)=-u^2/2$ are discretized
explicitly..."* -- i.e. even in a setting that could plausibly have used a pseudo-spectral method
(common for KS-type equations elsewhere in the literature), this paper's DEIM implementation
explicitly uses finite differences, not spectral evaluation. This is a further, if indirect, data
point consistent with the finding below, not a counter-example.

Other related-but-distinct techniques found: gappy POD and GNAT (Carlberg et al.) -- least-squares
sparse-sampling alternatives to DEIM's interpolatory approach, also generally applied to local
(FE/FV) discretizations; Localized DEIM (Peherstorfer et al., SIAM J. Sci. Comput. 2014) --
multiple *regime-local* DEIM interpolants (a different notion of "local" -- local in *parameter/
state space*, not local *in the spatial derivative stencil* as this project's adaptation is).

**Corrected claim:** *I did not identify a directly matching implementation -- DEIM combined with
an inherently global (FFT-based) nonlinear-term evaluation, addressed via a local finite-difference
surrogate specifically to preserve sparse-evaluation cost -- in the literature reviewed for this
search.* This is deliberately weaker than "no precedent exists": the search, while now covering
ten query combinations plus a close-precedent deep-read, is not exhaustive, and a differently-worded
or differently-indexed paper doing the same thing may exist. The evidence supports "an
under-documented combination, closest analogues use finite differences instead," not "novel."

## 5. Methods and why each belongs in the study

| Method | Role |
|---|---|
| Ground-truth solver (`src/solver.py`, unchanged) | Reference; also the source of every other method's training/basis data |
| POD-Galerkin ROM (`src/pod_rom.py`, unchanged) | Baseline classical ROM; the audit's own disclosed limitation (no runtime speedup over the solver) is this study's central motivating question |
| DEIM/local-FD hyper-reduced ROM (`src/deim.py`, new) | Tests whether hyper-reduction closes the ROM's runtime gap while preserving its accuracy/robustness advantage |
| MLP surrogate (`src/surrogate_net.py`, unchanged) | Baseline "neural surrogate" -- a parameter-to-field regressor, structurally distinct from a true operator |
| 1D FNO (`src/fno_net.py`, new) | A genuine operator-learning architecture: input is the actual IC field, not two scalars |

## 6. Rank selection and the DEIM-rank sweep

A rank sweep ($m \in \{6,8,10,12,16,24,32\}$, `scripts/build_deim.py`) against the held-out
validation split selects $m=16$ as the rank with the lowest mean validation-trajectory error versus
the plain ROM (0.69%, vs. 1.13-2.40% at other tested ranks). This selection is deterministic and
bit-for-bit reproducible on rerun (verified: `experiments/deim_basis.npz` regenerated identically
on a fresh run of `scripts/build_deim.py`).

## 7. Failure-mechanism investigation (Section 3 of the review) -- tested, not merely plausible

Revision 1 asserted, without direct testing, that DEIM's worst failures occur in the
high-amplitude/low-viscosity (steep-gradient) regime "because the local derivative approximation
becomes inaccurate there." This revision tests that claim quantitatively.

**Method** (`scripts/diagnose_deim_mechanisms.py`, part (a)): for each of the 40 in-distribution
test examples, measured DEIM-ROM trajectory error against six candidate explanatory variables --
$A$, $\nu$, $\max|u_x|$, $\max|u_{xx}|$ (both measured over the *true* trajectory), and the
mean/max nonlinear-term approximation error at the DEIM points, itself measured at multiple real
points *along the true trajectory* (not just $t=0$, where the field is always the smooth initial
sinusoid regardless of $A$ -- an earlier version of this check used only $t=0$ and found a
*negative* correlation, which was a measurement artefact of ignoring where steepness actually
develops, not a real effect).

**Result** (Spearman rank correlation, $n=40$, robust to the heavy right-skew in trajectory error):

| Variable | Spearman $\rho$ | $p$ |
|---|---|---|
| $A$ | +0.714 | $2.3\times10^{-7}$ |
| $\nu$ | -0.374 | $0.017$ |
| $\max|u_x|$ (true trajectory) | +0.818 | $1.1\times10^{-10}$ |
| $\max|u_{xx}|$ (true trajectory) | +0.821 | $8.4\times10^{-11}$ |
| max nonlinear-term error at DEIM points | +0.895 | $7.3\times10^{-15}$ |
| **mean nonlinear-term error at DEIM points** | **+0.935** | $9.4\times10^{-19}$ |

DEIM interpolation-matrix conditioning was checked separately and ruled out as an explanation: it
is example-independent (fixed once $m$ and the training ensemble are fixed) and well-conditioned
throughout ($\text{cond}(P^T\Psi) \approx 4$-$7$ across the tested rank range -- see Section 9),
so it cannot explain *per-example* variation in trajectory error.

**Interpretation.** This is strong, statistically significant *correlational* evidence for a
coherent mechanism (steep gradients $\to$ larger local-FD nonlinear-term error $\to$ larger
trajectory error), not proof of causation from correlation alone. Section 8 below reports a direct
intervention test (a higher-order stencil, which should reduce FD truncation error specifically)
as the causal check the review asked for -- and that test produced a genuine surprise (see below).

**Worst/best cases** (`report/research/deim_mechanism_diagnostics.json`, `worst_8_cases`/
`best_5_cases`): worst cases cluster at $A>1.8$, $\nu<0.04$ (steep-gradient regime); best cases at
moderate $A$, higher $\nu$ -- consistent with the correlations above.

## 8. Higher-order stencil experiment (Section 5 of the review) -- a genuine negative result

**Question:** can the steep-gradient accuracy loss be reduced by a higher-order local derivative
stencil, without destroying DEIM's online speed advantage?

**Implementation.** Added `deim_nonlinear_at_points_4th_order` (standard periodic 4th-order central
differences, 5-point stencil) as an explicit, opt-in alternative (`stencil_order=4` parameter on
`deim_rom_predict`, default `2` preserves the original behaviour exactly -- verified bit-for-bit
identical via `tests/test_deim.py::test_deim_rom_predict_default_stencil_order_unchanged`). The
original 2nd-order implementation is kept as the baseline, unmodified in its own right.

**Result** (`scripts/compare_deim_stencil_orders.py`, same POD rank, DEIM rank $m=16$, datasets,
and test regimes as the main comparison):

| Regime | 2nd-order (baseline) | 4th-order |
|---|---|---|
| In-distribution mean | 4.35% | 4.49% |
| In-distribution max | 52.36% | 54.63% |
| Extrapolation, $A=2.5$ | 19.96% | 20.56% |
| Out-of-family IC mean | 3.11% | 3.12% |
| Worst-case ($A$=1.889, $\nu$=0.0158) | 52.36% | 54.64% |
| 2nd-worst ($A$=1.503, $\nu$=0.0122) | 23.39% | 24.34% |
| Online cost | 4.8 ms | 7.4 ms (+55%) |

**The 4th-order stencil does not help -- it is statistically indistinguishable from, and on every
tested case marginally worse than, the 2nd-order baseline, while costing 55% more online time.**

**This is a genuine, informative negative result, and it revises the mechanism claim from Section
7**: since a strictly higher-order local derivative estimate (smaller FD truncation error at fixed
$dx$, by construction) produced no accuracy improvement, **formal truncation order of the local
stencil is not the dominant error source in the steep-gradient regime.** The correlational evidence
in Section 7 (nonlinear-term error strongly predicts trajectory error) still stands, but its root
cause is evidently not simply "the finite-difference formula isn't accurate enough" -- a plausible,
not-yet-tested alternative hypothesis is that the low-rank ($r=8$) POD reconstruction itself
exhibits Gibbs-phenomenon-like oscillation near steep fronts, and *any* local stencil (2nd or 4th
order) differentiates an oscillatory reconstruction poorly regardless of formal order, since
FD truncation-error bounds assume the underlying function is smooth on the stencil's scale. This
alternative hypothesis is **not tested here** and is flagged as unconfirmed for future work, not
asserted as established.

**Answer to the review's question:** No, not via this specific intervention. The 2nd-order stencil
remains the primary/default DEIM implementation (cheaper and no less accurate).

## 9. Rank non-monotonicity: mechanism investigated and one explanation retracted (Section 4)

Revision 1 attributed the non-monotonic DEIM-rank behaviour (trajectory accuracy does not improve
monotonically with $m$) to "the nonlinear-term SVD fitting noise beyond $m\approx16$-24." This
revision tests that explanation directly and finds it **not supported**.

**Method** (`scripts/diagnose_deim_mechanisms.py`, part (b)): for each candidate rank, measured (i)
cumulative singular-value energy capture of the nonlinear-term ensemble, (ii) conditioning of the
DEIM interpolation matrix $P^T\Psi$, (iii) $\|M\|_2$ (the operator norm of the DEIM-Galerkin
reduction matrix), (iv) minimum spacing between selected DEIM points, (v) the **reduced-RHS
relative error** ($\|M\tilde F_P(a) - \Phi^TF_{\text{exact}}(a)\|/\|\Phi^TF_{\text{exact}}(a)\|$ --
the quantity that actually enters the RK4 integration, as opposed to the raw pointwise error at the
DEIM points alone), and (vi) the resulting trajectory error vs. the plain ROM.

| $m$ | Energy ratio | cond($P^T\Psi$) | $\|M\|_2$ | Min. point gap | Reduced-RHS rel. err. | Trajectory rel. err. vs. ROM |
|---|---|---|---|---|---|---|
| 6 | 0.999424 | 4.92 | 7.17 | 1 | 0.1010 | 0.0240 |
| 8 | 0.999972 | 5.84 | 7.84 | 1 | 0.1071 | 0.0154 |
| 10 | 0.999998 | 4.54 | 5.80 | 1 | 0.0708 | 0.0113 |
| 12 | 1.000000 | 4.90 | 6.21 | 1 | 0.0714 | 0.0128 |
| **16** | 1.000000 | 4.01 | **4.20** | 1 | **0.0581** | **0.0069** |
| 24 | 1.000000 | 6.90 | 5.98 | 1 | 0.0717 | 0.0098 |
| 32 | 1.000000 | 6.81 | 5.76 | 1 | 0.0740 | 0.0124 |

**What the "SVD noise" explanation predicted and what actually happened.** If SVD noise-fitting
were the mechanism, the energy-capture ratio should show a visible plateau/cliff where additional
modes stop adding real signal. It does not: energy capture saturates *smoothly* to 1.000000 by
$m=10$-12, with no discontinuity near $m=16$-24. **This explanation is retracted.**

**What actually explains the pattern.** The raw pointwise nonlinear-term error at the DEIM points
*decreases monotonically* with $m$ (not shown in the table above, verified separately in Section 7
data), as textbook DEIM intuition predicts. But the reduced-RHS error -- the quantity that
propagates into the ODE integration -- does **not** decrease monotonically, and it tracks
$\|M\|_2$ closely: both are minimised at $m=16$ (the selected rank) and both are elevated at
$m=6,8$ (worse) and mildly elevated again at $m=24,32$. Since $M = \Phi^T\Psi(P^T\Psi)^{-1}$ is a
rectangular projection from the $m$-dimensional interpolated space down to $r=8$ dimensions, its
operator norm is not guaranteed to be monotonic in $m$ -- adding an interpolation point can either
help or hurt the resulting matrix's amplification factor, depending on how the new column/row
interacts with $\Phi$'s column space. **This is the better-supported mechanism**: a non-monotonic
$\|M\|_2$ can offset the monotonically-improving pointwise error, producing the observed
non-monotonic net outcome. A full first-principles derivation of *why* $\|M\|_2$ dips specifically
at $m=16$ for this dataset (e.g. via the principal angles between $\text{range}(\Phi)$ and
$\text{range}(\Psi)$ as a function of $m$) was not attempted and is flagged as a natural follow-up,
not claimed here.

## 10. Timing verification (Section 6 of the review)

**A genuine performance bug was found and fixed during this review.** Profiling
(`scripts/diagnose_deim_mechanisms.py`'s development, and directly via `cProfile`) found
`np.searchsorted(support, neighbours)` was being recomputed identically at **every one of the
~150-2000 RK4 stages per trajectory**, even though the DEIM point/neighbour index map never changes
once the points are fixed -- accounting for roughly 18% of DEIM-ROM's online cost. This was hoisted
out of the hot loop in `deim_rom_predict` (both stencil orders); the public
`deim_nonlinear_at_points`/`_4th_order` functions (used by tests to validate correctness against
spectral ground truth) are unchanged. **Verified bit-for-bit identical output** before/after this
optimisation (two independent calls, `np.array_equal` on the full output array) -- this is a pure
performance fix, zero effect on any accuracy number reported anywhere in this document.

**A suspected MPS-synchronization bug was investigated and found NOT to be a real issue.** The
original timing methodology (`scripts/evaluate_research_extension.py`) does not call
`torch.mps.synchronize()` before stopping the timer for the MLP/FNO. A direct check (three variants:
original-style, original-style + explicit sync, device-only without the `.cpu()` transfer) found
that `.cpu().numpy()` -- already present in the original methodology, since the result needs to
reach CPU for error computation -- **already forces an implicit MPS synchronization**: explicit
`torch.mps.synchronize()` changed the measured time by <15%, well within run-to-run noise. The
original MLP/FNO numbers were not meaningfully biased by a missing synchronization call; that
specific suspicion did not hold up under test.

**Rigorous repeated-trial timing** (`scripts/verify_timing.py`: 7 independent trials of 20 calls
each per method, median-of-trials reported, MPS-synchronized where applicable): across three
separate process invocations on this machine (Apple Silicon, MPS backend where used; NumPy/CPU
for the solver/ROM/DEIM-ROM), DEIM-ROM's speedup over the plain ROM was measured at 4.59x, 4.73x,
and (one outlier run affected by background system load, ROM alone showing std=14.8ms on a 28ms
median -- clearly anomalous relative to the other two runs' std of <0.4ms, and excluded on that
basis) 6.46x; speedup over the full solver at 1.88x, 1.96x, and 1.70x in the same three runs.
**On the documented benchmark configuration and hardware described above, the optimized DEIM-ROM
achieved approximately 4.7x lower median online evaluation time than the plain POD-ROM and
approximately 2.0x lower than the full solver**, using the most recent of the two internally-
consistent (non-anomalous) runs as the reported figure. Both figures improved from revision 1's
3.6x/1.46x due to the searchsorted fix above (a real code change, not a re-measurement of
unchanged code). These multipliers are specific to this hardware and benchmark configuration and
should not be read as universal constants -- see the excluded anomalous run above for how much
single-machine timing noise can matter.

**Component breakdown** (cProfile, post-fix): `deim_rhs` (state reconstruction + local-FD
nonlinear evaluation + $M$-projection, combined) accounts for essentially all of DEIM-ROM's online
cost (0.062s of 0.082s total over 20 calls, ~76%); the remaining ~24% is RK4 bookkeeping/setup, not
further broken down (an even finer per-substep breakdown would require instrumenting inside the
hot loop, adding overhead that would itself distort the very quantity being measured -- not
attempted, judged not worth the measurement distortion for a component that is already the clear
majority cost).

## 11. FNO scope, stated precisely (Section 7 of the review)

Revision 1's wording ("FNO cannot perform temporal extrapolation") was too broad. Corrected:

- **Architecture-general claim (not tested here, not asserted):** the *Fourier Neural Operator
  architecture in general* -- used in its classical autoregressive one-step formulation, as in Li
  et al. (2021) -- is not inherently incapable of temporal extrapolation; autoregressive rollout to
  arbitrary horizons is a standard, demonstrated FNO usage pattern in the literature.
- **What was actually tested and found:** *this project's particular single-shot, full-trajectory
  regression FNO instance* -- a deliberate, disclosed design choice made to match the MLP
  surrogate's exact task framing for a clean architecture-only comparison within the available time
  budget (see `src/fno_net.py` docstring) -- has a fixed-size output layer sized for $T=1.0$'s 20
  snapshots, and *this specific instance* cannot represent a $T=2.0$ trajectory, structurally, by
  construction of its output layer. This is a property of the chosen instantiation, not a
  claim about FNOs in general.

Similarly for out-of-family generalization: the finding is **"the FNO trained on this project's
200-trajectory, single-mode-family dataset generalizes poorly to a two-mode IC family,"** not
**"FNOs generalize poorly to new IC families"** as a general architectural claim. Section 4's
literature review found the original FNO paper uses 1000 training samples from a materially richer
function space (Gaussian random fields); this project's finding is specific to its own, much
smaller and narrower, data regime.

## 12. Out-of-family IC leakage audit (Section 8 of the review)

Verified, by direct inspection of every relevant script, that the two-mode IC test set
(`data/ood_family_test.npz`, generated by `scripts/generate_ood_dataset.py`, RNG seed `300_000`,
disjoint from `train`/`val`/`test`'s `0`/`100_000`/`200_000`) is absent from:

- **POD basis construction** (`scripts/build_deim.py` loads only `data/train.npz`);
- **Nonlinear DEIM snapshot construction** (same script, same load);
- **DEIM rank ($m$) selection** (validated only against `data/val.npz`, confirmed via source read);
- **MLP training** (`scripts/train_surrogate.py`, `src/surrogate_net.py`: zero references to "ood"
  anywhere in either file, confirmed via `grep`);
- **FNO training** (`scripts/train_fno.py`, `src/fno_net.py`: zero references, confirmed via
  `grep`; `load_split` only reads `data/{train,val}.npz`);
- **FNO hyperparameter selection**: `modes=16, width=32, n_layers=4` were fixed *a priori* when the
  architecture was written, not tuned against any split's performance (confirmed: these are
  hardcoded CLI defaults, never varied across a search in any script in this repository). This
  means there is no possibility of tuning-based leakage, but it is also disclosed as a limitation:
  no systematic architecture search was performed in either direction, so the FNO's poor
  out-of-family performance cannot be fully separated from "this specific, unsearched architecture
  choice" versus "a fundamental data-diversity limit" -- the most defensible reading, given
  Section 4's literature finding on FNO's typical training-data richness, is the latter, but the
  former has not been rigorously excluded.

**Why each method can or cannot accept the new IC family** (the scientifically important structural
distinction the review asked to preserve): the ROM and DEIM-ROM operate by projecting *any* given
initial field onto $\Phi$ ($a_0 = \Phi^Tu_0$) and integrating forward -- this works for any field
of the correct grid size, single-mode or not, since nothing in the Galerkin projection assumes a
particular functional form for $u_0$. The MLP surrogate's input is *only* the 2-scalar pair
$(A,\nu)$ -- it has no mechanism to represent a two-mode field at all, so it is not merely
inaccurate on this test, it cannot be evaluated on it. The FNO's input is the actual field
$u_0(x) \in \mathbb{R}^{N_x}$ (Section on architecture, `src/fno_net.py`) -- it *can* accept the new
IC family structurally, unlike the MLP, but (Section 13 below) its *learned* behaviour on that
input generalizes poorly given this project's training-data diversity.

## 13. Multi-seed FNO evidence (Section 9 of the review)

**$n=3$ seeds total: 0 (the original run), 1, 2.** Trained two additional seeds (1, 2), identical
architecture/training settings to the original (seed 0), no hyperparameter search performed in
either direction (`scripts/train_fno.py --seed {1,2}`, ~18.5s each, cheap enough that 3 seeds was
clearly reasonable). Evaluated all three on in-distribution accuracy, the three
parameter-extrapolation cases, and out-of-family IC generalization
(`scripts/evaluate_fno_multiseed.py`).

**Statistical note:** "$\pm$" below denotes the **sample standard deviation across these 3 seeds**
(population formula, `numpy.std` default, i.e. divided by $n$ not $n{-}1$) -- it is **not** a
standard error and **not** a confidence interval. With $n=3$, this is a description of the spread
actually observed across three specific runs, not a statistically rigorous estimate of the
underlying variance; it should not be read as supporting a precise quantitative confidence
statement (e.g. "95% CI"), only the qualitative claim made below (tight vs. wide spread relative to
the mean).

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean $\pm$ std |
|---|---|---|---|---|
| In-distribution | 2.86% | 3.07% | 2.86% | **2.93% $\pm$ 0.10%** |
| Out-of-family IC | 24.10% | 23.29% | 22.83% | **23.41% $\pm$ 0.53%** |
| Extrap. $A=2.5$ | 6.13% | 6.08% | 6.30% | 6.17% $\pm$ 0.09% |
| Extrap. $\nu=0.15$ | 6.72% | 6.27% | 6.44% | 6.47% $\pm$ 0.19% |
| Extrap. $A=0.2$ | 16.64% | 10.58% | 8.85% | 12.02% $\pm$ 3.34% |

**The central out-of-family finding is robust across seeds**: 23.41% $\pm$ 0.53% is a small
relative std (~2% of the mean), confirming this is a stable, reproducible property of the training
regime, not a single unlucky initialization. In-distribution accuracy is similarly tight. The
$A=0.2$ extrapolation case shows more inter-seed variability (std $\pm$3.34 percentage points on a
12% mean) but all three individual values remain in a consistently "moderately poor" range
(8.8-16.6%), not a qualitative disagreement about whether the FNO handles this case well.

## 14. Results (final, post-review; `report/research/results.json` is the machine-readable source)

### 14.1 In-distribution accuracy (test split, $n=40$) -- unchanged from revision 1, re-verified

| Method | Relative $L^2$ error (mean $\pm$ std) |
|---|---|
| POD-ROM ($r=8$) | 0.152% $\pm$ 0.077% |
| **DEIM/local-FD-ROM ($m=16$)** | **4.35% $\pm$ 9.34%** |
| MLP surrogate | 0.79% $\pm$ 0.37% |
| FNO (seed 0; 2.93% $\pm$ 0.10% across 3 seeds) | 2.86% |

### 14.2 Parameter extrapolation

| Case | ROM | DEIM/local-FD-ROM | MLP | FNO (3-seed mean) |
|---|---|---|---|---|
| $A=2.5$ (train max 2.0) | 0.10% | 19.96% | 5.45% | 6.17% $\pm$ 0.09% |
| $\nu=0.15$ (train max 0.10) | 0.24% | 0.24% | 3.05% | 6.47% $\pm$ 0.19% |
| $A=0.2$ (train min 0.5) | 0.06% | 0.06% | 55.95% | 12.02% $\pm$ 3.34% |

### 14.3 Out-of-family IC generalisation (two-mode ICs, in-range $\nu$, $n=20$) -- central finding

| Method | Relative $L^2$ error |
|---|---|
| POD-ROM | 1.69% $\pm$ 2.54% |
| DEIM/local-FD-ROM | 3.11% $\pm$ 5.93% |
| **FNO** | **23.41% $\pm$ 0.53% (3-seed mean $\pm$ std; seed 0 alone: 24.10%)** |
| MLP | **N/A -- structurally inapplicable** (Section 12) |

### 14.4 Temporal extrapolation ($T=2.0$, $n=8$)

| Method | Relative $L^2$ error |
|---|---|
| POD-ROM | 0.17% $\pm$ 0.07% |
| DEIM/local-FD-ROM | 3.12% $\pm$ 3.88% |
| MLP, FNO (this project's specific instance) | N/A -- structural, see Section 11 |

### 14.5 Offline cost (one-time, not amortised)

Scope: basis/snapshot-construction and training wall-clock time only, on the same machine as the
online-cost measurements below. Excludes the shared dataset-generation cost (identical for every
method, already reported in the original baseline's own results) and excludes any hyperparameter
search time, since none was performed for either the DEIM rank beyond the documented sweep or the
FNO architecture (Section 12).

| Step | Time |
|---|---|
| POD basis construction (shared with plain ROM) | 0.030 s |
| DEIM nonlinear-snapshot collection | 0.111 s |
| DEIM basis + point selection ($m=16$) | 0.029 s |
| **DEIM total offline cost added on top of the plain ROM** | **~0.17 s** |
| MLP training (200 epochs) | 8.23 s |
| FNO training (200 epochs) | 18.50 s (per seed; 3 seeds trained, ~56s total) |

### 14.6 Online cost -- corrected, rigorously re-verified (Section 10)

Hardware/methodology for this table: one Apple Silicon machine, MPS backend for the MLP/FNO
(NumPy/CPU for the solver/ROM/DEIM-ROM, so no device applies there); 7 independent trials of 20
calls each per method; median-of-trials reported below; one clearly anomalous trial excluded (see
Section 10 for the exclusion criterion); `torch.mps.synchronize()` applied before stopping the
timer for MLP/FNO, confirmed not to change the measured MLP/FNO time by more than run-to-run noise.

| Method | Median time per evaluation (7 trials $\times$ 20 calls) |
|---|---|
| Ground-truth solver | 7.6 ms |
| POD-ROM ($r=8$) | 18.3 ms |
| **DEIM/local-FD-ROM ($m=16$)** | **3.9-4.0 ms** |
| MLP | 0.38-0.43 ms |
| FNO | 1.4-1.6 ms |

**On the documented benchmark configuration and hardware above, the optimized DEIM-ROM achieved
approximately 4.7x lower median online evaluation time than the plain POD-ROM and approximately
2.0x lower than the full solver** (both improved from revision 1's 3.6x/1.46x due to a real
performance fix -- the searchsorted hoisting in Section 10 -- not a re-measurement of identical
code; these multipliers are specific to this hardware/configuration, not universal constants).
This affirmatively answers the first half of this study's research question, scoped to this
benchmark: hyper-reduction *does* close the plain ROM's disclosed
runtime-vs-solver gap.

## 15. Limitations (expanded from revision 1)

- **Terminology**: "DEIM hyper-reduction" alone, without the "local-FD" qualifier, would overstate
  precision -- see Section 3.7. Use "DEIM/local-FD hyper-reduction" throughout.
- **Novelty**: the literature search (Section 4), while now covering ten query combinations and one
  close-precedent deep-read, is not exhaustive; "no directly matching implementation identified" is
  the correct strength of claim, not "no precedent exists."
- **The steep-gradient failure mechanism's root cause is not fully identified**: correlational
  evidence strongly implicates the local-FD nonlinear-term error (Section 7), but the direct
  intervention test (a higher-order stencil, Section 8) shows formal truncation order is *not* the
  dominant driver -- leaving a plausible but untested Gibbs-phenomenon-style alternative hypothesis
  for future work.
- **The rank non-monotonicity's ultimate root cause is not fully derived**: the proximate mechanism
  ($\|M\|_2$'s non-monotonicity) is measured and evidenced, but *why* $\|M\|_2$ specifically dips at
  $m=16$ for this dataset was not derived from first principles.
- **The FNO here is a single-shot, full-trajectory regressor**, not the classical autoregressive
  one-step operator (Section 11) -- a disclosed design choice, not a claim about FNOs generally.
- **The out-of-family and parameter-extrapolation FNO results reflect this project's specific
  200-trajectory, single-mode-family data budget** (Section 11), and no architecture search was
  performed (Section 12) -- the poor generalization cannot be fully separated from "this specific
  unsearched architecture" versus "a fundamental data-diversity limit," though the latter is better
  supported by the literature (Section 4).
- **Only $r=8$ (state POD rank) was tested**; a joint $(r,m)$ sweep was out of scope.
- **3 seeds for FNO** (Section 13); POD-ROM/DEIM-ROM/MLP remain single-seed/deterministic (the
  first three are deterministic given fixed data, so "seed" does not apply to them in the same way;
  the MLP was not re-run across seeds in this revision -- a remaining gap).
- Hardware: all timings on one Apple Silicon machine (MPS where applicable); no cross-hardware or
  CPU-only comparison was run. One of three independent timing-verification runs showed anomalous
  system-load-driven variance (ROM std of 14.8ms on a 28ms median) -- excluded as clearly
  non-representative, but flags that this machine's timing measurements carry real, occasionally
  large, environmental noise.

## 16. Answering the analysis questions

- **Most accurate overall**: the plain POD-ROM, by a wide margin, in-distribution and under both
  extrapolation tests, except where DEIM/local-FD-ROM ties it (parameter-range and out-of-family
  cases outside the steep-gradient regime).
- **Fastest online**: the MLP, then the FNO, then DEIM/local-FD-ROM -- but DEIM/local-FD-ROM is the
  first ROM-family method to beat the full solver's own runtime, by a verified, repeated-trial
  margin of ~2x.
- **Highest offline cost**: FNO training (18.5s/seed), roughly 2.2x the MLP's training cost and
  >100x DEIM's own offline cost.
- **Best generalisation, out-of-family IC**: the plain ROM, then DEIM/local-FD-ROM; the FNO --
  despite being architecturally capable of the task the MLP cannot even attempt -- generalises far
  worse than either classical ROM method given this project's data budget, and this finding is
  confirmed stable across 3 training seeds.
- **Fails first outside the training regime**: the MLP (55.95% at one parameter-extrapolation case)
  and the FNO (23.4% mean out-of-family, seed-stable) are both far more fragile than either
  ROM-family method; DEIM/local-FD-ROM fails specifically and predictably in the steep-gradient
  regime, correlationally well-characterized (Section 7) though not fully mechanistically resolved
  at the root-cause level (Sections 8-9).
- **Accuracy-vs-cost tradeoff**: DEIM/local-FD-ROM trades roughly 30x worse mean in-distribution
  accuracy (0.15% -> 4.35%) for a repeatedly-verified online speedup on the documented benchmark
  configuration (~4.7x vs. the plain ROM, ~2.0x vs. the solver, this hardware/config only, see
  Section 10) -- a real, disclosed, not-free tradeoff, not a strict improvement.
- **When would a classical ROM be preferable?** Whenever out-of-distribution robustness or
  in-distribution accuracy matters more than online speed, and whenever the problem stays in the
  smooth (higher-$\nu$) regime specifically for DEIM/local-FD-ROM.
- **When would operator learning (FNO) be preferable?** When online speed matters and the
  deployment distribution is known to stay within (or close to) the training distribution; this
  project's FNO instance should not be trusted to generalize to either a new IC family or a longer
  time horizon without more diverse training data or a different (autoregressive) design.
- **What conclusions are actually supported?** That hyper-reduction (in the specific DEIM/local-FD
  hybrid form implemented and tested here) can close a classical ROM's disclosed runtime
  disadvantage against the full solver, at a real, regime-dependent, and only partially
  mechanistically understood accuracy cost that a higher-order local stencil does *not* fix; and
  that architectural capacity to accept a broader input class (FNO vs. MLP) does not by itself
  confer generalization -- data diversity does, and this project's existing dataset was not
  designed to provide it, a finding now confirmed stable across three independent training seeds.

## 17. What was NOT changed (preserving the verified baseline)

`src/solver.py`, `src/pod_rom.py`, `src/surrogate_net.py`, `scripts/generate_dataset.py`,
`scripts/train_surrogate.py`, and `scripts/evaluate_comparison.py` are byte-for-byte unchanged from
the remediation-phase baseline commit `6aaa282`. The public `deim_nonlinear_at_points`/`_4th_order`
reference functions (used by tests to validate correctness against spectral ground truth) are also
unchanged in this revision; only `deim_rom_predict`'s internal hot loop was optimized (verified
bit-identical output). All new experiments write to new locations (`report/research/`,
`experiments/fno_checkpoint*.pt`, `experiments/deim_basis.npz`) rather than overwriting the
existing verified `report/mvp_results.txt` / `report/mvp_comparison.png`.

## 18. Reproduction

```bash
# 1. REQUIRED PREREQUISITES -- run README.md's Installation + Usage section IN FULL first. Every
#    script below depends on the files these steps produce (data/{train,val,test}.npz,
#    experiments/surrogate_checkpoint.pt); none of it is repeated here.
pip install -r requirements.txt
python scripts/generate_dataset.py          # -> data/{train,val,test}.npz
python scripts/train_surrogate.py --overfit-check   # sanity check (expect ratio < 0.1)
python scripts/train_surrogate.py           # -> experiments/surrogate_checkpoint.pt
python scripts/evaluate_comparison.py       # -> report/mvp_results.txt, report/mvp_comparison.png
                                             # (the original, still-verified baseline result)

# 2. Research extension, in order (requires step 1's outputs to exist first):
python scripts/build_deim.py                    # DEIM rank sweep + selection -> experiments/deim_basis.npz
python scripts/train_fno.py --overfit-check      # sanity check (expect ratio < 0.1)
python scripts/train_fno.py --epochs 200 --seed 0   # -> experiments/fno_checkpoint.pt
python scripts/train_fno.py --epochs 200 --seed 1   # -> experiments/fno_checkpoint_seed1.pt
python scripts/train_fno.py --epochs 200 --seed 2   # -> experiments/fno_checkpoint_seed2.pt
python scripts/generate_ood_dataset.py           # -> data/ood_family_test.npz
python scripts/evaluate_research_extension.py    # -> report/research/{results.json,results_summary.txt,comparison_figure.png}
python scripts/evaluate_fno_multiseed.py         # -> report/research/fno_multiseed_results.json
python scripts/compare_deim_stencil_orders.py    # -> report/research/stencil_order_comparison.json
python scripts/diagnose_deim_mechanisms.py       # -> report/research/deim_mechanism_diagnostics.json
python scripts/verify_timing.py                  # -> report/research/timing_verification.json

# Tests
pytest tests/  # 31 tests: 14 original (unchanged) + 17 new (DEIM incl. 4th-order stencil, FNO,
                # general-IC solver)
```

All steps use fixed seeds (`SEED=0` for the original data/MLP/FNO-seed-0 training, `SEED=300_000`
for the OOD dataset, `SEED=42` for the temporal-extrapolation cases, `--seed 1`/`--seed 2` for the
additional FNO seeds -- all distinct from and disjoint with the existing splits' `0/100_000/
200_000` seed ranges).

`experiments/deim_basis.npz` and `experiments/fno_run_meta*.json`/`fno_checkpoint*.pt` are
gitignored (regenerable, deterministic -- verified via bit-identical rerun); `report/research/*`
(the actual experimental results/evidence) is versioned, matching the original project's
`report/mvp_results.txt` convention.
