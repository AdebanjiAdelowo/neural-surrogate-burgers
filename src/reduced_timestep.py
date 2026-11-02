"""Reduced-order-specific time integration for the quadratic (tensorial) Burgers ROM.

The default "reduced" policy of src/timestep.py is dt = min(0.25 dx / max|u0|, 0.5 * 2.785 / (nu rho(L))):
its advective bound uses the FULL grid spacing dx, so the reduced step count grows like N even though
the reduced system has only r degrees of freedom.

Reduced advective bound. For v = Phi b (orthonormal Phi), the spectral second derivative satisfies
D2 = D1^2 with D1 skew-symmetric, so ||D1 v||^2 = -v^T D2 v = -b^T L b <= rho(L) ||b||^2. The nonlinear
Jacobian of da/dt = Q(a, a) is J_nl(a) b = Phi^T G(2 (Phi a) * (Phi b)) with G = -(1/2) D1 masked, and
Phi^T G = (1/2) (D1_mask Phi)^T (skew), whence

    ||J_nl(a)||_2 <= ||Phi a||_inf * sqrt(rho(L))                          (tested numerically)

and ||Phi a||_inf is about max|u0| (maximum principle of the PDE, not guaranteed for the ROM). This is
the reduced analogue of the FOM's advective CFL condition with dx replaced by 1 / sqrt(rho(L)), the
smallest length scale the basis can represent. It bounds the size of the Jacobian's eigenvalues, not
their sign: eigenvalues with positive real part are unaffected by any dt, so stability is checked
empirically (pre-registered criteria, report/spatial_convergence/PLAN.md).

Policies implemented here:
  * `reduced_cfl_dt`: dt = s * min(RK4_IMAG / (max|u0| sqrt(rho(L))), RK4_REAL / (nu rho(L))), s = 0.5
    (the safety factor of the existing diffusion bound).
  * `if_rk4_trajectory`: Lawson integrating-factor RK4 in reduced coordinates (the FOM's scheme): the
    linear term nu L is integrated exactly with exp(nu L dt) (L symmetric, eigendecomposed once), so only
    the advective bound remains: dt = s * RK4_IMAG / (max|u0| sqrt(rho(L))).
  * `scipy_trajectory`: adaptive explicit (RK45, DOP853) or implicit (Radau, BDF, with the exact
    Jacobian) integration from scipy.integrate.solve_ivp at given tolerances.
"""

import numpy as np

from src.tensorial_pod import (TensorialROM, integrate_reduced, quadratic_jacobian, quadratic_rhs)
from src.timestep import RK4_IMAG_LIMIT, RK4_REAL_LIMIT

SAFETY = 0.5


def advective_rate_bound(u0_max: float, rho: float) -> float:
    """max|u0| sqrt(rho(L)): bound on ||J_nl(a)||_2 for states with ||Phi a||_inf <= max|u0|."""
    return float(u0_max) * np.sqrt(rho)


def reduced_cfl_dt(u0_max: float, nu: float, rho: float, safety: float = SAFETY) -> float:
    return safety * min(RK4_IMAG_LIMIT / max(advective_rate_bound(u0_max, rho), 1e-300),
                        RK4_REAL_LIMIT / max(nu * rho, 1e-300))


def if_rk4_dt(u0_max: float, rho: float, safety: float = SAFETY) -> float:
    return safety * RK4_IMAG_LIMIT / max(advective_rate_bound(u0_max, rho), 1e-300)


def aligned_steps(dt: float, T: float, n_save: int, min_steps: int = None) -> int:
    """Smallest multiple of n_save - 1 with T / n_steps <= dt (and >= min_steps, default 4 n_save, as in
    the existing reduced models)."""
    n = max(4 * n_save if min_steps is None else min_steps, int(np.ceil(T / dt * (1 - 1e-12))))
    return int(-(-n // (n_save - 1)) * (n_save - 1))


def rk4_trajectory(model: TensorialROM, a0: np.ndarray, nu: float, T: float, n_save: int, n_steps: int):
    """Plain RK4 (the existing reduced scheme) with a given step count; returns (t_save, a_save)."""
    return integrate_reduced(quadratic_rhs(model.Q_flat, nu * model.L), np.asarray(a0, float),
                             n_steps, T / n_steps, n_save)


def if_rk4_trajectory(model: TensorialROM, a0: np.ndarray, nu: float, T: float, n_save: int, n_steps: int):
    """Lawson integrating-factor RK4 for da/dt = Q(a, a) + nu L a with exp(nu L dt) exact.

    With w = exp(-nu L t) a the system becomes dw/dt = exp(-nu L t) Q(exp(nu L t) w, ...); classical RK4
    on w, written back in a, gives the stages below (identical in structure to src/solver.py)."""
    lam, V = np.linalg.eigh(0.5 * (model.L + model.L.T))
    dt = T / n_steps
    E = (V * np.exp(nu * lam * dt / 2)) @ V.T
    E2 = (V * np.exp(nu * lam * dt)) @ V.T
    r = model.r
    Qf = model.Q_flat

    def nl(a):
        return (Qf @ a).reshape(r, r) @ a

    save_steps = set(np.linspace(0, n_steps, n_save, dtype=int).tolist())
    a = np.asarray(a0, dtype=float)
    t_save, a_save, t = [0.0], [a.copy()], 0.0
    for step in range(1, n_steps + 1):
        k1 = nl(a)
        k2 = nl(E @ (a + dt / 2 * k1))
        k3 = nl(E @ a + dt / 2 * k2)
        k4 = nl(E2 @ a + dt * (E @ k3))
        a = E2 @ a + dt / 6 * (E2 @ k1 + 2 * (E @ (k2 + k3)) + k4)
        t += dt
        if step in save_steps:
            a_save.append(a.copy())
            t_save.append(t)
    return np.array(t_save), np.array(a_save)


def scipy_trajectory(model: TensorialROM, a0: np.ndarray, nu: float, times: np.ndarray, method: str,
                     rtol: float, atol: float):
    """solve_ivp with output at `times`. Returns (a_save, info) where info has nfev, njev, accepted
    steps and, for the explicit Runge-Kutta methods, the number of rejected steps."""
    from scipy.integrate import solve_ivp
    Lnu = nu * model.L
    f = quadratic_rhs(model.Q_flat, Lnu)
    jac = (lambda t, a: quadratic_jacobian(model.Q, Lnu, a)) if method in ("Radau", "BDF") else None

    def rhs(t, a):
        return f(a)

    kw = {"jac": jac} if jac is not None else {}
    sol = solve_ivp(rhs, (float(times[0]), float(times[-1])), np.asarray(a0, float), method=method,
                    t_eval=times, rtol=rtol, atol=atol, **kw)
    info = {"success": bool(sol.success), "nfev": int(sol.nfev), "njev": int(getattr(sol, "njev", 0)),
            "nlu": int(getattr(sol, "nlu", 0)), "message": sol.message}
    a_save = sol.y.T if sol.success else np.full((len(times), len(a0)), np.nan)
    return a_save, info


def count_steps(model: TensorialROM, a0: np.ndarray, nu: float, T: float, method: str, rtol: float,
                atol: float):
    """Accepted and rejected step counts of an explicit embedded RK solve, by stepping the solver object
    directly (RK45: 6 new RHS evaluations per attempted step with FSAL; DOP853: 12)."""
    from scipy.integrate import DOP853, RK45
    cls = {"RK45": RK45, "DOP853": DOP853}[method]
    f = quadratic_rhs(model.Q_flat, nu * model.L)
    solver = cls(lambda t, a: f(a), 0.0, np.asarray(a0, float), T, rtol=rtol, atol=atol)
    accepted = 0
    while solver.status == "running":
        solver.step()
        accepted += 1
    per_attempt = {"RK45": 6, "DOP853": 12}[method]
    attempts = (solver.nfev - 2) // per_attempt  # 2 evaluations go to the initial step selection
    return {"accepted": accepted, "rejected": int(max(attempts - accepted, 0)), "nfev": int(solver.nfev),
            "status": solver.status}
