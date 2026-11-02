"""Reduced-order time integration (src/reduced_timestep.py)."""
import numpy as np
import pytest
import scipy.linalg

from scripts.hyperreduction_benchmark import wavenumbers
from src.convergence import sine_ic, solve_burgers_f64
from src.pod_rom import build_pod_basis
from src.reduced_timestep import (advective_rate_bound, aligned_steps, count_steps, if_rk4_dt,
                                  if_rk4_trajectory, reduced_cfl_dt, rk4_trajectory, scipy_trajectory)
from src.tensorial_pod import build_tensorial_rom, quadratic_jacobian
from src.timestep import RK4_IMAG_LIMIT, RK4_REAL_LIMIT


def _model(r=8, N=64, n_examples=8):
    k, dealias = wavenumbers(N)
    rng = np.random.default_rng(0)
    ens = [solve_burgers_f64(sine_ic(rng.uniform(0.5, 2.0), N), rng.uniform(0.02, 0.1))[1]
           for _ in range(n_examples)]
    Phi, _ = build_pod_basis(np.concatenate(ens), r)
    return Phi, build_tensorial_rom(Phi, k, dealias), np.concatenate(ens)


def test_jacobian_norm_bound_holds_on_visited_and_random_states():
    """||J_nl(a)||_2 <= ||Phi a||_inf sqrt(rho(L)) (src/reduced_timestep.py docstring)."""
    Phi, m, ens = _model()
    rho = m.diffusion_radius
    states = list(ens[::7] @ Phi) + list(np.random.default_rng(1).standard_normal((30, 8)) * 3)
    for a in states:
        J = quadratic_jacobian(m.Q, 0.0 * m.L, a)
        bound = advective_rate_bound(np.abs(Phi @ a).max(), rho)
        assert np.linalg.norm(J, 2) <= bound * (1 + 1e-10)


def test_dt_formulas():
    assert reduced_cfl_dt(2.0, 0.01, 100.0) == pytest.approx(0.5 * min(RK4_IMAG_LIMIT / 20, RK4_REAL_LIMIT / 1))
    assert if_rk4_dt(2.0, 100.0) == pytest.approx(0.5 * RK4_IMAG_LIMIT / 20)
    n = aligned_steps(0.013, 1.0, 20)
    assert n % 19 == 0 and 1.0 / n <= 0.013 and n >= 80


def test_if_rk4_is_exact_for_the_linear_part():
    _, m, _ = _model()
    m0 = type(m)(Q=np.zeros_like(m.Q), L=m.L, Nx=m.Nx, diffusion_radius=m.diffusion_radius)
    a0 = np.random.default_rng(2).standard_normal(8)
    _, a = if_rk4_trajectory(m0, a0, 0.05, 1.0, 20, 19)
    np.testing.assert_allclose(a[-1], scipy.linalg.expm(0.05 * m.L) @ a0, rtol=1e-10, atol=1e-12)


def test_if_rk4_converges_at_fourth_order_to_the_rk4_solution():
    Phi, m, _ = _model()
    a0 = Phi.T @ sine_ic(1.2, 64)
    ref = rk4_trajectory(m, a0, 0.04, 1.0, 20, 19 * 400)[1]
    errs = [np.abs(if_rk4_trajectory(m, a0, 0.04, 1.0, 20, 19 * q)[1] - ref).max() for q in (8, 16, 32)]
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 3.6)


def test_if_rk4_is_stable_beyond_the_explicit_diffusion_limit():
    Phi, m, _ = _model()
    nu = 1.0  # makes the explicit diffusion limit tight enough that the aligned step still exceeds it
    dt_diff = RK4_REAL_LIMIT / (nu * m.diffusion_radius)
    n = aligned_steps(4 * dt_diff, 1.0, 20, min_steps=1)
    assert 1.0 / n > 2 * dt_diff  # the test is only meaningful beyond the RK4 limit
    a0 = Phi.T @ sine_ic(0.5, 64)
    _, a_if = if_rk4_trajectory(m, a0, nu, 1.0, 20, n)
    with np.errstate(all="ignore"):  # the explicit run is expected to overflow
        _, a_rk = rk4_trajectory(m, a0, nu, 1.0, 20, n)
    assert np.all(np.isfinite(a_if)) and np.abs(a_if).max() <= np.abs(a0).max() * 1.01
    assert not np.all(np.abs(a_rk[-1]) < 10 * np.abs(a0).max())  # plain RK4 blows up at this step


@pytest.mark.parametrize("method", ["RK45", "DOP853", "Radau"])
def test_scipy_integrators_agree_with_fine_rk4(method):
    Phi, m, _ = _model()
    a0 = Phi.T @ sine_ic(1.0, 64)
    times = np.linspace(0, 1, 20)
    ref = rk4_trajectory(m, a0, 0.05, 1.0, 20, 19 * 200)[1]
    a, info = scipy_trajectory(m, a0, 0.05, times, method, rtol=1e-9, atol=1e-11)
    assert info["success"] and np.abs(a - ref).max() <= 1e-6 * np.abs(ref).max()


def test_step_counter_accounts_for_every_rhs_evaluation():
    Phi, m, _ = _model()
    info = count_steps(m, Phi.T @ sine_ic(1.5, 64), 0.03, 1.0, "RK45", rtol=1e-6, atol=1e-9)
    assert info["status"] == "finished"
    assert info["nfev"] == 2 + 6 * (info["accepted"] + info["rejected"])
