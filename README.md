# Neural Surrogate vs. POD-Galerkin ROM for Viscous Burgers' Equation

**Status: PLANNED — architecture defined, implementation not started.** No solver, ROM, network,
experiment, or figure in this repository is implemented or run yet.

## What this project is

A comparative study of a classical linear reduced-order model (POD-Galerkin) against a data-driven
neural surrogate for the 1D viscous Burgers' equation, evaluated on accuracy, runtime speedup, and
generalization from training-range to held-out (interpolated and extrapolated) parameters. See
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Why this project, and why it's independent

Identified during a review of `TUM_2026_Project_Portfolio_Strategy.md` as a genuinely reusable
capability gap: nothing in the existing portfolio trains across an *ensemble* of solved PDE
instances to build a model that generalizes to unseen parameters (PINN solves one instance;
`photoacoustic-reconstruction`, once built, inverts one set of observations). This project
deliberately does **not** depend on `boiling-phasefield-3d` or its data — that repository is the
development environment for a separate, real PhD project at the University of Udine with its own
supervisor-defined roadmap, and using it here would be an inappropriate, artificial dependency. This
project uses an independently-implemented Burgers'-equation solver instead — a standard benchmark in
the neural-operator/ROM literature, unconnected to any other project in this portfolio.

## Relationship to other projects in this portfolio

| Project | Research question | Relationship |
|---|---|---|
| `pinn-advection-diffusion` | Solve one instance of a *linear* PDE via physics-constrained training | Natural predecessor in difficulty (linear → nonlinear); this project does not reuse its code |
| `photoacoustic-reconstruction` | Recover an unknown source from indirect, sparse *observations* (inverse problem) | Different direction of the mapping — this project predicts forward solutions from parameters, not sources from measurements |
| `boiling-phasefield-3d` | Separate, real PhD-track solver (University of Udine) | Explicitly **not used** — see above |

## Sequencing

Built **second**, after `photoacoustic-reconstruction`'s MVP — this project's application value for
the current seven TUM/Helmholtz applications is Moderate at best (Quaini only, and Quaini's
application is already sufficient without it); its value is primarily long-term/portfolio-wide.

## Repository layout

Same standard layout as `photoacoustic-reconstruction` (`src/`, `configs/`, `data/`, `experiments/`,
`scripts/`, `tests/`, `report/`) — see `ARCHITECTURE.md` for what belongs in each once implemented.
