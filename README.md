# Neural Surrogate vs. POD-Galerkin ROM for Viscous Burgers' Equation

**Status: MVP COMPLETE.** The full pipeline — parameterised PDE → validated numerical solver →
dataset → POD-Galerkin ROM → learned surrogate → held-out comparison — runs end-to-end on real
data. See [`IMPLEMENTATION_LOG.md`](IMPLEMENTATION_LOG.md) for the stage-by-stage verification
record (including two real bugs found and fixed) and [`report/mvp_results.txt`](report/mvp_results.txt)
for full results.

## What this project is

A comparative study of a classical linear reduced-order model (POD-Galerkin) against a data-driven
neural surrogate for the 1D viscous Burgers' equation. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for
the technical plan.

## Real MVP result (measured, not projected) — a genuine, non-obvious finding

| | In-distribution rel. $L^2$ error | Runtime/eval | Out-of-range error (worst case) |
|---|---|---|---|
| POD-ROM ($r$=8) | **0.15%** | 15.5 ms | **0.10%** (still excellent) |
| Neural surrogate | 0.79% | **0.14 ms** | **55.95%** (catastrophic, see `report/mvp_extrapolation.png`) |

**The classical POD-Galerkin ROM is both more accurate and far more robust to out-of-range
parameters than the neural surrogate** — because it is built on the true governing equations
(Galerkin projection), it remains valid physics wherever evaluated, while the neural surrogate is a
black-box interpolator that degrades severely outside its training distribution. **The neural
surrogate's real, defensible advantage is inference speed** — 45× faster than the full solver,
112× faster than the ROM, since it requires a single forward pass with no time-stepping at all.

Also honestly reported: the POD-ROM **as implemented is slower than the full solver** (it evaluates
the nonlinear term via full-grid spectral derivatives each step — no hyper-reduction/DEIM — so it
reduces the state dimension but not the RHS-evaluation cost). A hyper-reduced ROM would likely close
this gap; not attempted in this MVP.

This is not the "neural network wins" result a less careful study might report — it's a genuine,
non-obvious scientific finding about when classical structure-preserving methods beat black-box
learning, arrived at by actually measuring both accuracy and generalisation rather than accuracy
alone.

## Why this project, and why it's independent

Identified during a review of `TUM_2026_Project_Portfolio_Strategy.md` as a genuinely reusable
capability gap: nothing else in the portfolio trains across an *ensemble* of solved PDE instances to
build a model that generalises to unseen parameters (PINN solves one instance;
`photoacoustic-reconstruction` inverts one set of observations). This project deliberately does
**not** depend on `boiling-phasefield-3d` — that repository is the development environment for a
separate, real PhD project at the University of Udine with its own supervisor-defined roadmap.
Uses an independently-implemented Burgers'-equation solver instead.

## Relationship to other projects in this portfolio

| Project | Research question | Relationship |
|---|---|---|
| `pinn-advection-diffusion` | Solve one instance of a *linear* PDE via physics-constrained training | Natural predecessor in difficulty (linear → nonlinear); no code reused |
| `photoacoustic-reconstruction` | Recover an unknown source from indirect, sparse *observations* (inverse problem) | Different direction of the mapping — forward solution prediction, not inverse recovery |
| `boiling-phasefield-3d` | Separate, real PhD-track solver (University of Udine) | Explicitly **not used** |

## Environment

Reuses the `photoacoustic` conda env (Python 3.11) — `numpy`, `scipy`, `matplotlib`, `torch`,
`scikit-image`, `pytest` were already installed there and cover everything this project needs; no
new environment was created.

## Repository layout

```
neural-surrogate-burgers/
├── README.md               this file
├── ARCHITECTURE.md          technical plan
├── IMPLEMENTATION_LOG.md    stage-by-stage verification record, real results, bugs found & fixed
├── requirements.txt
├── src/                     solver.py, pod_rom.py, surrogate_net.py — all implemented
├── scripts/                 generate_dataset.py, train_surrogate.py, evaluate_comparison.py
├── data/                    generated train/val/test .npz splits — gitignored, not committed
├── experiments/              trained checkpoint — gitignored, not committed
├── tests/                   14 tests, all passing
└── report/                  real figures: dev-stage verification plots + mvp_comparison.png,
                              mvp_extrapolation.png, mvp_results.txt
```

## What's next

Per the explicit MVP stop condition: FNO/DeepONet comparison, multi-dimensional PDEs, uncertainty
quantification, and hyper-reduction (to give the ROM a fair runtime comparison) are all plausible
next steps, none started in this pass. See `IMPLEMENTATION_LOG.md`'s final entry for the full
portfolio-relevance assessment.
