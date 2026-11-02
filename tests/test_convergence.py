"""Float64 solver, cross-resolution Fourier tools and the Cole-Hopf exact solution (src/convergence.py)."""
import numpy as np
import pytest

from src.convergence import (coeffs, default_step_count, grid, l2_norm, pad_coeffs, rel_l2,
                             rel_l2_per_snapshot, resample, restrict_coeffs, sine_ic, sine_solution,
                             solve_burgers_f64, two_mode_solution)
from src.general_ic_solver import solve_burgers_general_ic
from src.solver import solve_burgers


# ------------------------------------------------------------------------------- float64 solver
@pytest.mark.parametrize("A,nu,N", [(1.0, 0.05, 64), (1.9, 0.016, 128), (0.5, 0.1, 96)])
def test_float64_solver_is_the_baseline_scheme(A, nu, N):
    """Same operations as src/solver.py: casting to float32 reproduces it bit for bit."""
    t0, _, u32 = solve_burgers(A, nu, Nx=N, align_snapshots=True)
    t1, u64 = solve_burgers_f64(sine_ic(A, N), nu)
    np.testing.assert_array_equal(t0, t1)
    np.testing.assert_array_equal(u32, u64.astype(np.float32))
    assert u64.dtype == np.float64


def test_float64_solver_matches_general_ic_solver():
    u0 = 1.2 * np.sin(grid(64)) + 0.4 * np.sin(2 * grid(64))
    _, _, ug = solve_burgers_general_ic(u0, 0.04, align_snapshots=True)
    _, u64 = solve_burgers_f64(u0, 0.04)
    np.testing.assert_array_equal(ug, u64.astype(np.float32))


def test_step_count_must_align_with_snapshots():
    with pytest.raises(ValueError):
        solve_burgers_f64(sine_ic(1.0, 32), 0.05, n_steps=100)


def test_default_step_count_is_the_baseline_rule():
    assert default_step_count(2.0, 128) == 171 and default_step_count(2.0, 128) % 19 == 0


def test_temporal_refinement_converges_at_fourth_order_on_a_smooth_case():
    u0, nu = sine_ic(0.5, 64), 0.1
    ref = solve_burgers_f64(u0, nu, n_steps=19 * 256)[1]
    errs = [rel_l2(solve_burgers_f64(u0, nu, n_steps=19 * m)[1], ref) for m in (4, 8, 16)]
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 3.7) and np.all(orders < 4.3)


def test_float64_solver_reproducible():
    u0 = sine_ic(1.3, 64)
    np.testing.assert_array_equal(solve_burgers_f64(u0, 0.03)[1], solve_burgers_f64(u0, 0.03)[1])


# ------------------------------------------------------------------------------- Fourier tools
def _trig_poly(x, kmax, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.standard_normal(kmax + 1), rng.standard_normal(kmax + 1)
    return sum(a[k] * np.cos(k * x) + b[k] * np.sin(k * x) for k in range(kmax + 1))


@pytest.mark.parametrize("N,M", [(32, 64), (32, 96), (33, 64), (64, 256)])
def test_padding_reproduces_a_band_limited_function_exactly(N, M):
    f = lambda x: _trig_poly(x, N // 2 - 1)
    np.testing.assert_allclose(resample(f(grid(N)), M), f(grid(M)), atol=1e-12)


def test_nyquist_mode_is_split_so_the_interpolant_stays_real_and_unchanged():
    N, M = 16, 64
    u = np.cos(8 * grid(N))  # pure Nyquist mode on 16 points: (-1)^j
    up = resample(u, M)
    np.testing.assert_allclose(up, np.cos(8 * grid(M)), atol=1e-13)
    np.testing.assert_allclose(resample(up, N), u, atol=1e-13)  # restriction recovers it


@pytest.mark.parametrize("N,M", [(64, 32), (64, 33), (256, 64)])
def test_restriction_keeps_low_modes_of_a_fine_band_limited_function(N, M):
    f = lambda x: _trig_poly(x, M // 2 - 1, seed=3)
    np.testing.assert_allclose(resample(f(grid(N)), M), f(grid(M)), atol=1e-12)


def test_pad_then_restrict_is_identity():
    c = coeffs(np.random.default_rng(1).standard_normal(48))
    np.testing.assert_allclose(restrict_coeffs(pad_coeffs(c, 192), 48), c, atol=1e-15)


def test_parseval_l2_norm_matches_quadrature_of_the_interpolant():
    x = grid(64)
    u = _trig_poly(x, 20)
    np.testing.assert_allclose(l2_norm(u), np.sqrt(np.sum(u**2) * 2 * np.pi / 64), rtol=1e-13)
    assert abs(l2_norm(np.sin(x)) - np.sqrt(np.pi)) < 1e-13


def test_rel_l2_is_resolution_independent_for_band_limited_fields():
    f = lambda x: _trig_poly(x, 10, seed=5)
    g = lambda x: _trig_poly(x, 10, seed=6)
    e1 = rel_l2(f(grid(32)), g(grid(32)))
    e2 = rel_l2(f(grid(32)), g(grid(256)))
    e3 = rel_l2(f(grid(512)), g(grid(64)))
    assert abs(e1 - e2) < 1e-13 and abs(e1 - e3) < 1e-13


def test_rel_l2_equals_grid_relative_l2_on_a_common_grid():
    rng = np.random.default_rng(2)
    u, r = rng.standard_normal((5, 40)), rng.standard_normal((5, 40))
    assert abs(rel_l2(u, r) - np.linalg.norm(u - r) / np.linalg.norm(r)) < 1e-13
    np.testing.assert_allclose(rel_l2_per_snapshot(u, r),
                               np.linalg.norm(u - r, axis=1) / np.linalg.norm(r, axis=1), rtol=1e-12)


# ------------------------------------------------------------------------------- Cole-Hopf
def test_cole_hopf_initial_value_and_symmetries():
    ch = sine_solution(1.5, 0.03)
    x = grid(64)
    np.testing.assert_array_equal(ch(x, 0.0), 1.5 * np.sin(x))
    u = ch(x, 0.6)
    np.testing.assert_allclose(ch(x + 2 * np.pi, 0.6), u, atol=1e-12)        # periodic
    np.testing.assert_allclose(ch(-x, 0.6), -u, atol=1e-12)                   # odd in x
    assert abs(ch(np.array([np.pi]), 0.6)[0]) < 1e-12                         # front at x = pi
    assert np.abs(u).max() <= 1.5 + 1e-12                                     # maximum principle


@pytest.mark.parametrize("A,nu", [(0.5, 0.1), (2.0, 0.01)])
def test_cole_hopf_is_insensitive_to_quadrature_refinement(A, nu):
    x, times = grid(256), (0.05, 0.4, 1.0)
    u1 = np.array([sine_solution(A, nu)(x, t) for t in times])
    u2 = np.array([sine_solution(A, nu, step_fraction=1 / 80)(x, t) for t in times])
    assert rel_l2(u1, u2) < 1e-13


def test_cole_hopf_agrees_with_the_resolved_pseudo_spectral_solution():
    """Two independent methods: Hopf integral vs the FOM on a resolved grid with a small step (N = 256:
    at N = 64 the exact solution's coefficient at the dealiasing cutoff k = N/3 is still about 5e-5)."""
    A, nu, N = 1.0, 0.08, 256
    t, u = solve_burgers_f64(sine_ic(A, N), nu, n_steps=19 * 64)
    ch = sine_solution(A, nu).trajectory(N, t)
    assert rel_l2(u, ch) < 1e-10


def test_two_mode_cole_hopf_agrees_with_the_pseudo_spectral_solution():
    A1, A2, nu, N = 1.0, 0.4, 0.08, 256
    u0 = A1 * np.sin(grid(N)) + A2 * np.sin(2 * grid(N))
    t, u = solve_burgers_f64(u0, nu, n_steps=19 * 64)
    assert rel_l2(u, two_mode_solution(A1, A2, nu).trajectory(N, t)) < 1e-10


def test_pseudo_spectral_error_against_cole_hopf_decays_spectrally_with_n():
    A, nu = 1.0, 0.08
    errs = []
    for N in (64, 96, 128):
        t, u = solve_burgers_f64(sine_ic(A, N), nu, n_steps=19 * 64)
        errs.append(rel_l2(u, sine_solution(A, nu).trajectory(N, t)))
    assert errs[0] / errs[1] > 20 and errs[1] / errs[2] > 20  # faster than any low algebraic order


def test_cole_hopf_conserves_mass_and_dissipates_energy():
    ch = sine_solution(1.8, 0.02)
    U = ch.trajectory(512, np.linspace(0, 1, 6))
    np.testing.assert_allclose(U.sum(axis=1) * 2 * np.pi / 512, 0.0, atol=1e-12)
    E = np.sum(U**2, axis=1)
    assert np.all(np.diff(E) < 0)
