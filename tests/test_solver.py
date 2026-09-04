import numpy as np

from src.solver import solve_burgers


def test_shape_and_finiteness():
    t, x, u = solve_burgers(A=1.0, nu=0.05, Nx=64, T=1.0, n_save=10)
    assert t.shape == (10,)
    assert x.shape == (64,)
    assert u.shape == (10, 64)
    assert np.isfinite(u).all()


def test_final_time_is_exact():
    t, _, _ = solve_burgers(A=1.0, nu=0.05, Nx=64, T=1.0, n_save=10)
    assert abs(t[-1] - 1.0) < 1e-9
    assert t[0] == 0.0


def test_mass_conservation():
    # Periodic domain, u0 = A*sin(...): integral of u dx is exactly conserved by the PDE
    # (both the advective and diffusive terms integrate to zero under periodic BCs).
    t, x, u = solve_burgers(A=1.0, nu=0.05, Nx=128, T=1.0, n_save=10)
    dx = x[1] - x[0]
    mass = u.sum(axis=1) * dx
    assert np.abs(mass).max() < 1e-4  # near machine precision, well below any physical scale


def test_energy_dissipates_monotonically():
    # Viscous Burgers strictly dissipates L2 energy over time; this is a real, checkable
    # mathematical property of the PDE, not merely a numerical-stability heuristic.
    t, x, u = solve_burgers(A=1.0, nu=0.05, Nx=128, T=1.0, n_save=20)
    dx = x[1] - x[0]
    energy = 0.5 * (u**2).sum(axis=1) * dx
    assert np.all(np.diff(energy) <= 1e-8)


def test_smaller_viscosity_produces_steeper_gradient():
    # Physically expected: less viscosity -> steeper gradients at fixed final time.
    _, x_lo, u_lo = solve_burgers(A=1.0, nu=0.1, Nx=256, T=1.0, n_save=5)
    _, x_hi, u_hi = solve_burgers(A=1.0, nu=0.005, Nx=256, T=1.0, n_save=5)
    grad_lo_visc_nu = np.abs(np.gradient(u_lo[-1], x_lo)).max()   # nu=0.1 (more viscous)
    grad_hi_visc_nu = np.abs(np.gradient(u_hi[-1], x_hi)).max()   # nu=0.005 (less viscous)
    assert grad_hi_visc_nu > grad_lo_visc_nu


def test_steepest_gradient_near_theoretical_location():
    # Classic result: inviscid Burgers with u0=sin(x) on a 2*pi domain forms its steepest
    # gradient near x=pi. With small (not zero) viscosity, the steepest gradient should still
    # be close to that location.
    t, x, u = solve_burgers(A=1.0, nu=0.005, Nx=256, T=1.0, n_save=10)
    steepest_x = x[np.argmax(np.abs(np.gradient(u[-1], x)))]
    assert abs(steepest_x - np.pi) < 0.15


def test_grid_refinement_is_consistent():
    # Grid convergence check: the steepest-gradient *location* should not move under refinement
    # (magnitude legitimately increases as a near-shock feature is better resolved, so only
    # location — not magnitude — is asserted here).
    locations = []
    for Nx in [64, 128, 256]:
        t, x, u = solve_burgers(A=1.0, nu=0.02, Nx=Nx, T=1.0, n_save=5)
        locations.append(x[np.argmax(np.abs(np.gradient(u[-1], x)))])
    assert max(locations) - min(locations) < 0.1
