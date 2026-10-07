# Scientific Audit — Viscous Burgers Surrogates and Reduced Models

## Verdict
The viscous periodic Burgers equation is a valid benchmark for comparing intrusive ROMs and learned surrogates. POD–Galerkin, DEIM and FNO answer related but different approximation questions; they should not be ranked by one error/runtime number without conditioning on training distribution, viscosity information, timestep policy and offline cost.

## FOM
The pseudo-spectral/full solver is the reference numerical model. Dealiasing, timestep policy and time-aligned snapshots are part of the numerical contract and must be held fixed in comparisons.

## POD–Galerkin
POD gives the best linear snapshot subspace for the training ensemble in its chosen norm. Projection error and reduced-dynamics error are distinct. Good in-distribution performance does not establish generalisation to lower viscosity or unseen initial conditions.

## DEIM/local finite differences
The repository's DEIM extension is a genuine online-complexity experiment: sampled nonlinear evaluation avoids touching the full grid, and tests poison non-required grid rows to enforce that property.

The local finite-difference stencil introduces a second approximation beyond DEIM interpolation. Its order/stability therefore must be evaluated separately from POD truncation.

## Timestep correction
A reduced diffusion operator can have a spectral radius that makes a FOM-derived timestep unstable. The repository now has a revised rule tied to the reduced diffusion spectrum and tests the RK4 stability mechanism. Historical tables deliberately retain the historical rule for reproducibility.

This is the correct scholarly treatment: do not silently recompute historical results under a different stability policy.

## FNO
An FNO learns an operator/data-distribution mapping rather than projecting the PDE onto a fixed Galerkin subspace. A viscosity-aware FNO and a viscosity-blind historical FNO are therefore different models. In-distribution accuracy cannot be used to infer low-viscosity extrapolation ability.

## Runtime fairness
Online timings should distinguish:
- FOM integration;
- ROM online integration;
- DEIM/local evaluation;
- neural inference;
- offline snapshot generation/POD/DEIM construction;
- neural training.

A fast neural query does not imply lower total cost for a small number of solves.

## Classification
- Wrong PDE: **no**.
- POD–Galerkin: **appropriate intrusive ROM**.
- DEIM: **appropriate hyper-reduction experiment**.
- FNO: **appropriate learned-operator comparison**.
- Historical timestep policy: **a diagnosed numerical limitation, preserved for reproducibility**.
- Universal superiority of any method: **not established and should not be claimed**.

No new code defect was identified in this pass beyond issues the repository's follow-up audit/tests already encode.
