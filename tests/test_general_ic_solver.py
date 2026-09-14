import numpy as np

from src.general_ic_solver import solve_burgers_general_ic
from src.solver import solve_burgers


def test_matches_solve_burgers_exactly_for_single_mode_ic():
    """This module intentionally mirrors src.solver.solve_burgers's numerical method (see module
    docstring); for the single-mode sinusoidal family both solvers should agree essentially
    bit-for-bit, since they use the identical scheme. Any divergence here would indicate the
    mirroring introduced a real discrepancy, not merely float noise.
    """
    A, nu, Nx, T, n_save = 1.3, 0.05, 128, 1.0, 20
    x = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
    u0 = A * np.sin(x)

    t_ref, x_ref, u_ref = solve_burgers(A=A, nu=nu, Nx=Nx, T=T, n_save=n_save)
    t_gen, x_gen, u_gen = solve_burgers_general_ic(u0, nu=nu, T=T, n_save=n_save)

    np.testing.assert_allclose(t_ref, t_gen, atol=1e-10)
    np.testing.assert_allclose(x_ref, x_gen, atol=1e-10)
    np.testing.assert_allclose(u_ref, u_gen, atol=1e-5)


def test_accepts_two_mode_initial_condition():
    Nx = 64
    x = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
    u0 = 1.0 * np.sin(x) + 0.5 * np.sin(2 * x)
    t_save, _, u_save = solve_burgers_general_ic(u0, nu=0.05, T=1.0, n_save=10)
    assert u_save.shape == (10, Nx)
    assert np.all(np.isfinite(u_save))
    assert t_save[0] == 0.0
