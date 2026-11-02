"""Adaptive, error-controlled integration of the tensorial POD system da/dt = nu L a + Q(a, a).

Error norm (on the physical reduced coefficients a, the same as scipy's explicit Runge-Kutta and Radau):

    err = sqrt( mean_i ( e_i / (atol + rtol max(|a_i^n|, |a_i^{n+1}|)) )^2 ),    step accepted iff err <= 1.

Because the POD basis is orthonormal in the discrete Euclidean norm, ||a|| = ||Phi a|| (up to the fixed
grid factor dx), so coefficient errors equal reconstructed-field errors in the discrete L2 norm. A local
error tolerance does not bound the global error: the study measures the global error separately.

Drivers:
  * `adaptive_solve`: the step-size controller for the new exponential methods of
    src/exponential_integrators.py (embedded estimate for LawsonDP54, step doubling for Krogstad ETD-RK4),
    landing exactly on the output times; records accepted / rejected steps, N-evaluations and the step
    history.
  * `scipy_solve`: RK45, DOP853, Radau (analytic or finite-difference Jacobian) from scipy.integrate with
    solver statistics; accepted / rejected steps are counted by stepping the solver object.
"""

import numpy as np

from src.exponential_integrators import ETDRK4Krogstad, EigenBasis, EigenSystem, LawsonDP54, error_norm
from src.tensorial_pod import quadratic_jacobian, quadratic_rhs

SAFETY, FAC_MIN, FAC_MAX = 0.9, 0.2, 5.0


def default_atol(a0: np.ndarray, rtol: float, scale: float = 1e-3) -> float:
    """Absolute tolerance used throughout the study: scale * rtol * max|a0| (fixed in the plan)."""
    return scale * rtol * float(np.abs(a0).max())


def _initial_step(es, b0, Nb0, rtol, atol, span):
    """Hairer-Norsett-Wanner style first guess from |y| and |f| in the error norm."""
    a0 = es.to_a(b0)
    f0 = es.to_a(es.D * b0 + Nb0)
    sc = atol + rtol * np.abs(a0)
    d0, d1 = np.sqrt(np.mean((a0 / sc) ** 2)), np.sqrt(np.mean((f0 / sc) ** 2))
    h = 0.01 * d0 / d1 if d0 > 1e-5 and d1 > 1e-5 else 1e-6
    return min(h, span)


def adaptive_solve(method: str, Q: np.ndarray, L: np.ndarray, nu: float, a0: np.ndarray, times: np.ndarray,
                   rtol: float, atol: float, record_steps: bool = False, max_steps: int = 10_000_000,
                   basis: EigenBasis = None):
    """method in {"lawson_dp54", "etdrk4_doubling"}; `basis` = precomputed EigenBasis(Q, L) (one-time,
    nu-independent). Returns (a_save (len(times), r), stats)."""
    es = EigenSystem(Q, L, nu, basis=basis)
    if method == "lawson_dp54":
        integ, p_est = LawsonDP54(es), 4          # error estimate of order 4 -> exponent 1/5
    elif method == "etdrk4_doubling":
        integ, p_est = ETDRK4Krogstad(es), 4      # step doubling of an order-4 method
    else:
        raise ValueError(method)
    expo = 1.0 / (p_est + 1)
    b = es.to_b(np.asarray(a0, dtype=float))
    Nb = es.N(b)
    out = [es.to_a(b).copy()]
    st = {"accepted": 0, "rejected": 0, "n_evals": 1, "steps": [] if record_steps else None}
    t = float(times[0])
    h = _initial_step(es, b, Nb, rtol, atol, float(times[1] - times[0]))
    for t_out in times[1:]:
        while t < t_out - 1e-14 * max(1.0, abs(t_out)):
            h_try = min(h, t_out - t)
            truncated = h_try < h
            a_old = es.to_a(b)
            if method == "lawson_dp54":
                b_new, Nb_new, e_b, ne = integ.step(b, Nb, h_try)
                st["n_evals"] += ne
            else:
                b_big, ne1 = integ.step(b, Nb, h_try)
                b_half, ne2 = integ.step(b, Nb, h_try / 2)
                N_half = es.N(b_half)
                b_new, ne3 = integ.step(b_half, N_half, h_try / 2)
                e_b = (b_new - b_big) / (2 ** p_est - 1)
                st["n_evals"] += ne1 + ne2 + ne3 + 1
                Nb_new = None
            err = error_norm(es.to_a(e_b), a_old, es.to_a(b_new), rtol, atol)
            if not np.isfinite(err):
                err = 1e10
            fac = min(FAC_MAX, max(FAC_MIN, SAFETY * err ** -expo)) if err > 0 else FAC_MAX
            if err <= 1.0:
                t += h_try
                b = b_new
                Nb = Nb_new if Nb_new is not None else es.N(b)
                if Nb_new is None:
                    st["n_evals"] += 1
                st["accepted"] += 1
                if record_steps:
                    st["steps"].append((t, h_try))
                h = h if truncated and fac > 1 else h_try * fac   # a truncated step does not shrink h
            else:
                st["rejected"] += 1
                h = h_try * fac
            if st["accepted"] + st["rejected"] > max_steps:
                raise RuntimeError("step budget exceeded")
        t = float(t_out)
        out.append(es.to_a(b).copy())
    return np.array(out), st


def scipy_solve(Q: np.ndarray, L: np.ndarray, nu: float, a0: np.ndarray, times: np.ndarray, method: str,
                rtol: float, atol: float, jacobian: str = "analytic", count_steps: bool = False):
    """RK45 / DOP853 / Radau with outputs at `times` (dense output). jacobian in {"analytic", "fd"} for
    Radau ("fd": scipy's finite-difference approximation). Returns (a_save, stats)."""
    from scipy.integrate import DOP853, RK45, Radau, solve_ivp
    Lnu = nu * L
    f = quadratic_rhs(Q.reshape(Q.shape[0] ** 2, Q.shape[0]), Lnu)
    kw = {}
    if method == "Radau" and jacobian == "analytic":
        kw["jac"] = lambda t, a: quadratic_jacobian(Q, Lnu, a)
    sol = solve_ivp(lambda t, a: f(a), (float(times[0]), float(times[-1])), np.asarray(a0, float),
                    method=method, t_eval=times, rtol=rtol, atol=atol, **kw)
    st = {"success": bool(sol.success), "nfev": int(sol.nfev), "njev": int(sol.njev), "nlu": int(sol.nlu)}
    if count_steps:
        cls = {"RK45": RK45, "DOP853": DOP853, "Radau": Radau}[method]
        s = cls(lambda t, a: f(a), float(times[0]), np.asarray(a0, float), float(times[-1]), rtol=rtol, atol=atol, **kw)
        acc, hs = 0, []
        while s.status == "running":
            t_prev = s.t
            s.step()
            acc += 1
            hs.append((s.t, s.t - t_prev))
        st["accepted"] = acc
        st["steps"] = hs
        if method in ("RK45", "DOP853"):
            per = {"RK45": 6, "DOP853": 12}[method]
            st["rejected"] = int(max((s.nfev - 2) // per - acc, 0))
        else:
            st["rejected"] = None  # Radau rejections are not separable from Newton failures via nfev
    a_save = sol.y.T if sol.success else np.full((len(times), len(a0)), np.nan)
    return a_save, st
