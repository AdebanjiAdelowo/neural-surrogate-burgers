# PROPOSED portfolio wording (text only; nothing in Academic_Career was edited)

Target: `03_Profile/Selected_Projects/Adebanji_Adelowo_Selected_Projects_LaTeX.tex`, Burgers entry (lines 212 to 223).
Submitted applications (for example the ELLIS CV) are historical snapshots and stay as they are. Their "0.152% vs 0.788%"
is correctly attributed to the plain ROM and reproducible under the historical protocol.

## Results field (replaces the current text)

> Rank-8 POD-Galerkin ROM: 0.15% mean relative L2 error on 40 held-out cases against the Nx = 128 solver (0.05% when
> snapshot times are aligned exactly). A hyper-reduced DEIM/local-finite-difference variant is 1.7x to 2.0x faster than
> the full solver at Nx = 128 (4.6x faster than the plain ROM) and up to about 7x faster at Nx = 2048 with a step-size
> rule derived from the reduced system, at a mean error of 4.3% (median 0.2%, up to 53% on steep, low-viscosity
> cases). On an unseen two-mode initial-condition family: ROM 1.7%, DEIM 3.1%, FNO 23.4% (3 seeds); giving the FNO the
> viscosity cuts its in-distribution error from 2.9% to 0.9% but leaves the out-of-family error at about 22.6%.

## Limitations field (replaces the current text)

> The plain ROM is slower than the full solver, because it evaluates the nonlinear term on the full grid at every RK4
> stage. The DEIM variant evaluates finite differences at 16 points, is linearly unstable on steep trajectories, and
> its accuracy loss shrinks with grid resolution. The original FNO was not given the viscosity. The FNO results are
> specific to this network, 200 training trajectories and one initial-condition family.

## Metric tiles

`0.15%` POD-ROM in-distribution error; `23.4%` FNO unseen-family error. Do not use "2.0x" or "4.35%" without the
qualifiers above: 2.0x is a single case (about 1.9x) or the aligned-protocol revised-rule average (1.98x); 4.35% is the
40-case mean, not a steepest-case figure.

## Claim status

| Claim | Status |
|---|---|
| POD-ROM 0.152% in-distribution | VERIFIED as the historical index-wise figure; PARTIALLY VERIFIED as physical error (0.047% aligned) |
| DEIM about 2.0x faster than the full solver | INCORRECT as a general statement; 1.52x to 1.57x (historical protocol, 40 cases), 1.67x (aligned, historical rule), 1.98x (aligned, revised rule), 0.75x to 2.6x per case |
| accuracy cost 4.35% in the steepest regime | INCORRECT: 4.35% is the 40-case mean |
| out-of-family ROM 1.69%, DEIM 3.11%, FNO 23.41% | VERIFIED (reproduced) |
| plain ROM slower than the FOM | VERIFIED (0.35x to 0.41x) |
| FNO out-of-family failure specific to setup | VERIFIED and sharpened: not caused by missing viscosity |
