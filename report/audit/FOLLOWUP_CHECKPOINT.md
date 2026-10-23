# Follow-up audit checkpoint (saved 2026-09-21, session may expire)

Instruction from Adebanji: DO NOT commit or push. Do NOT modify portfolio repos (Academic_Career, Selected
Projects, master CV, Template, website, profile, applications). Historical results stay frozen. Deliver a
"second pre-commit report" with 27 numbered items, ending with exactly one of:
`READY FOR REVIEW — ROM/HYPERREDUCTION LIMITATION SUBSTANTIALLY ADDRESSED` /
`READY FOR REVIEW — LIMITATION CHARACTERISED BUT FURTHER WORK NEEDED` /
`STOPPED — NUMERICAL/METHODOLOGICAL ISSUE REQUIRES REVIEW`.

Repo: `/Users/adebanjiadelowo/Documents/GitHub/neural-surrogate-burgers`, HEAD 5f61d78, tree 68f1785, all work UNCOMMITTED.

## Follow-up scope (Adebanji's message)
A. Reduced-model timestep/stability: derive the historical rule; multiplier study (predetermined set, 5 cases
   spanning the distribution incl. steep + any unstable config); all-40-case test of historical rule vs FOM step
   count for ROM r=8 and DEIM r=8 m=16 (record failures); no per-case tuning (derive general rule, dev subset,
   then untouched test); reduced Jacobian/eigenvalue check; scaling Nx=128..1024 (opt. 2048) under historical and
   validated revised policy, per-stage vs end-to-end separated.
B. DEIM accuracy: re-verify attribution on a subset; compare local-FD stencil orders (2 vs 4) fairly (error,
   unstable cases, per-stage and trajectory runtime); diagnose unstable cases (timestep vs FD vs conditioning vs
   basis vs point placement; report cond(P^T Psi)).
C. FNO fairness: keep historical nu-blind FNO reproducible; build minimal nu-aware FNO (extra input channel), same
   everything else, same seeds; A/B on in-distribution (40) and out-of-family (20); correct the nu-blind floor;
   separate nu information from IC-family shift for the OOD set.
D. Snapshot times: implement corrected (aligned) snapshot saving separately, tests, small corrected dataset,
   report whether full regeneration changes conclusions. Do not overwrite data/*.npz.
E. Final fair comparison table (FOM, POD-Galerkin, POD-DEIM/local-FD, historical FNO, nu-aware FNO); keep historical
   and corrected metrics distinct.
F. Tests; G. compact artifacts (JSON/CSV + few figures); H. README proposal + portfolio wording proposal only.

## DONE so far (this follow-up)
1. `src/fno_net.py`: `FNO1d(..., use_nu=False)`; `build_input(u0, nu)`; nu normalised (nu-0.01)/(0.1-0.01);
   default keeps historical checkpoint layout (verified: historical ckpt loads, nu ignored when use_nu=False;
   nu-aware adds 32 params, 137,140 vs 137,108).
2. `scripts/train_fno.py`: flags `--nu-aware`, `--out-prefix`, `--data-dir` (PLAN: replace `--data-dir` by
   `--data-prefix` so aligned data can live at `data/aligned_{train,val,test}.npz`, which `.gitignore` `data/*.npz`
   covers; a subdirectory would NOT be ignored). Checkpoints are `experiments/{nu_,rerun_}fno_checkpoint_seed{s}.pt`
   (gitignored). Historical checkpoints untouched.
3. Background FNO chain (seeds 0-4, nu-aware prefix `nu_`, nu-blind retrain prefix `rerun_`), logs in the scratchpad
   `train_{nu,rerun}_{seed}.log`, done-marker `fno_chain_done`. At checkpoint time seeds 0-2 nu-aware and 0-1 rerun
   finished. If the marker is missing and processes are gone, rerun the missing seeds:
   `python scripts/train_fno.py --epochs 200 --seed S --nu-aware --out-prefix nu_` and `... --out-prefix rerun_`.
   Early result: best val MSE nu-aware 3.2e-5 / 3.6e-5 / 2.9e-5 vs nu-blind 5.4e-4 / 5.3e-4 (rerun reproduces the
   historical 5.37e-4 and 5.33e-4 exactly, so training is reproducible).
4. `src/snapshot_times.py` (new): `align_step_count`, `saved_step_indices`, `saved_times`, `intended_times`,
   `resample_in_time` (cubic spline diagnostic).
5. Solvers gained `align_snapshots=False` (src/solver.py, src/general_ic_solver.py) and reduced models gained
   `align_snapshots=False, min_steps=None` (rom/deim `*_reduced_trajectory` and `*_predict`). Verified: defaults
   bit-identical to HEAD on 32 arrays, `align_snapshots=True` == the `aligned_dt` route, 58 tests pass.
6. `src/timestep.py` (new): `historical_dt`, `fom_advective_dt`, `historical_rule_fractions_of_rk4_limit`
   (hist constant = 71% of RK4 limit for full-grid spectral Laplacian, 29% for full-grid FD3), reduced diffusion
   matrices `K` (ROM spectral, DEIM local-FD) with da/dt_diff = nu K a, `spectral_radius`, `diffusion_dt`,
   `revised_dt` (PROPOSED: min(FOM advective dt, 0.5*2.7853/(nu*rho(K)))), `rk4_amplification`,
   `rk4_max_stable_dt` (frozen-Jacobian first-crossing), `numerical_jacobian`. NOT yet tested, NOT yet used.

## TODO (in order)
1. Write `scripts/hyperreduction_followup.py` with stages (reuse helpers from `scripts/hyperreduction_benchmark.py`:
   `interleaved_timing`, `aligned_dt`, `aligned_reference`, `native_reduced_steps`, `_reduced_rk4`,
   `predict_exactF_deim`, `predict_fd_fullgrid`, `build_bases`, `deim_operator`, `_fom_ensemble`, `err_stats`, `save`):
   - `stability`: constants; rho(K) ROM and DEIM for r=8,m=16; Jacobian eigenvalues (ROM, DEIM) at 5 times along
     5 a-priori cases; frozen-Jacobian dt* vs historical / FOM / revised dt; regression case from commit 8fdbca0
     (10-trajectory basis, nu=0.15, check whether advective-only dt really is unstable for that basis).
   - `multiplier`: 5 cases = test-set ranks [0,10,20,30,39] of A/nu ascending (a-priori), multipliers 1,2,4,8,16,32
     of the historical dt, min_steps=19 + aligned, ROM r8, DEIM r8m16 and r8m12; record stable/steps/runtime/error/
     max|u|/first blow-up time.
   - `policy`: all 40 test AND 40 val cases, policies A historical, B FOM step count, R revised; ROM r8 + DEIM r8m16;
     solver-only runtime interleaved vs aligned FOM; failures listed; R evaluated on val first (dev), then test.
   - `scaling2`: Nx 128,256,512,1024 (+2048 for DEIM/FOM only), FOM (align_snapshots=True) vs ROM vs DEIM under
     A/B/R; solver-only and reconstruction-inclusive; steps; aligned error on first 10 test params; per-stage cost.
   - `stencil`: attribution re-check (exactF, FD-full-grid, DEIM) on 40 test under policy R; stencil order 2 vs 4
     (error mean/median/max, unstable count, per-stage = time/(4 steps), trajectory time) at Nx=128 (40 cases) and
     Nx=256,512 (10 cases); unstable-case diagnosis for r8 m in {6,8,10,12,16,24,32}: cond(P^T Psi), ||M||_2,
     point spacing/placement vs shock index 64, dt/4 rerun, Jacobian max Re(eig).
   - `aligned_data`: generate `data/aligned_{train,val,test}.npz` (+ aligned OOD) with `align_snapshots=True`, rebuild
     POD/DEIM, evaluate, compare with historical; resample stored snapshots to uniform times (diagnostic); optionally
     retrain FNO A/B on aligned data (3 seeds).
2. Write `scripts/fno_nu_ablation.py`: evaluate historical (0-2), `rerun_` (0-4), `nu_` (0-4) on in-dist (40) and
   OOD (20); per-seed, mean, std, median, max; inference timing (CPU, MPS, batch 1); training time from meta;
   nu-blind floor in-dist (aligned and native) and OOD floor (E_nu predictor per two-mode IC via
   `solve_burgers_general_ic`, 24 Gauss-Legendre nodes); shuffled-nu sensitivity check for the nu-aware model.
3. New tests (extend `tests/`): aligned physical times across FOM/general-IC/ROM/DEIM, resample_in_time, nu channel
   construction and determinism, timestep policy and stability/failure detection, local-FD stencil behaviour,
   existing DEIM no-full-grid guarantee still holds; all 58 existing tests must still pass.
4. Figures (few): speed-up vs Nx (historical vs revised), error vs timestep multiplier, DEIM error vs stencil order,
   nu-blind vs nu-aware FNO, historical vs aligned ROM error.
5. README proposal v2 (`report/audit/README_PROPOSAL.md` update) and portfolio wording proposal (text only).
6. Final report with the 27 items; keep historical and corrected metrics distinct: ROM 0.1518% (historical
   index-wise) vs ~0.047% (aligned); DEIM 4.35% vs 4.31%; timing 1.9x (test[0]) vs 1.52-1.57x (40-case average,
   historical step policy); revised-policy result only if validated.

## Do not lose (findings from the first audit, all in report/audit/*.json)
0.152% is the plain ROM; ~2x DEIM is one case; 4.35% is the 40-case DEIM mean (median 0.32%, max 52%); FNO gets no
nu; nu-blind floor ~2.44% vs FNO 2.93%; DEIM per-stage cost flat in Nx; end-to-end advantage falls with Nx because
of the dx^2/(2nu) reduced step rule; dtprobe showed FOM step count stable on 6 cases (3.8-3.9x at Nx=1024);
DEIM accuracy loss comes from local-FD evaluation, not DEIM interpolation.

## PROGRESS UPDATE 2 (later same session)
- FNO training chain FINISHED (nu_ and rerun_ checkpoints, seeds 0-4).
- `scripts/train_fno.py` now has `--data-prefix` (not `--data-dir`); `src/timestep.py` DEIM diffusion matrix supports stencil_order 4.
- WRITTEN: `scripts/hyperreduction_followup.py` with stages stability, multiplier, policy, scaling2, stencil (the `aligned_data`
  stage dispatch points at a not-yet-written module `scripts/hyperreduction_followup_aligned.py`: write it, or move the code
  into the main file). Stage `stability` DONE -> `report/audit/followup_stability_run1.json`.
  A background chain runs multiplier, policy, scaling2, stencil (tag run1, logs in scratchpad `fu_<stage>.log`, marker `fu_done`).
  If interrupted, rerun: `python scripts/hyperreduction_followup.py <stage> --tag run1`.
- KEY FINDINGS from `stability`: reduced diffusion rho(K): ROM 1732, DEIM/local-FD 1206 (full-grid spectral 4096, FD3 1660):
  r=8 POD modes contain near-cutoff wavenumbers, so the reduced system is genuinely stiff; historical constant is ~3x
  conservative at nu=0.1 (true limit 0.0161 vs hist 0.0048). The earlier "FOM step count is stable" probe is only true for
  nu<=0.1: at nu=0.15 (A=1) the unaligned FOM step count gives NaN for BOTH the stored and the 10-trajectory basis (limit
  2.785/(0.15*1732)=0.0107 < FOM dt 0.0122; the earlier probe passed only because aligning rounded 82 -> 95 steps). Historical rule
  is stable there. Revised policy R (dt=min(FOM dt, 0.5*2.785/(nu*rho(K)))) gives 0.0054 for that case. DEIM frozen Jacobian has
  POSITIVE real eigenvalues (+0.2..+2.5) on steep cases (test idx 0 and 33) at later snapshots while the ROM's are all negative:
  a continuous-time linear instability of the DEIM/local-FD reduced system, independent of dt.
- STILL TO DO: FNO evaluation script (`scripts/fno_nu_ablation.py`), aligned-data stage + aligned FNO retrain (optional),
  tests, figures, README/portfolio proposals, final 27-item report.

## STATUS: FOLLOW-UP COMPLETE (all stages run, 78 tests pass, nothing committed)
Everything in the TODO list was done; the second pre-commit report was delivered in chat. This file is a working note and
should NOT be committed. `followup_stencil_run1.json` was deleted (its exact-F and full-grid-FD diagnostics used a
non-aligned dt; `followup_stencil_run2.json` is the corrected one). Remaining decisions belong to Adebanji: commit boundaries,
whether to adopt `src/timestep.revised_dt` as a default, whether corrected (aligned) data become canonical, README and
Selected_Projects wording (see README_PROPOSAL.md and PORTFOLIO_WORDING_PROPOSAL.md).
