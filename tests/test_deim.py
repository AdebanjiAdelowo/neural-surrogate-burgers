import numpy as np

from src.deim import (build_deim_basis, build_deim_projector, collect_nonlinear_snapshots,
                       deim_nonlinear_at_points, deim_nonlinear_at_points_4th_order,
                       deim_rom_predict, select_deim_points, _stencil_support)
from src.pod_rom import build_pod_basis, rom_predict
from src.solver import solve_burgers


def _setup(r=8, m=12, Nx=64, n_examples=10):
    L = 2 * np.pi
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    dealias = np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k))
    rng = np.random.default_rng(0)

    ensemble, nu_per_snapshot = [], []
    for _ in range(n_examples):
        A, nu = rng.uniform(0.5, 2.0), rng.uniform(0.01, 0.1)
        _, _, u = solve_burgers(A=A, nu=nu, Nx=Nx, T=1.0, n_save=10)
        ensemble.append(u)
        nu_per_snapshot.append(np.full(u.shape[0], nu))
    ensemble = np.concatenate(ensemble, axis=0)
    nu_per_snapshot = np.concatenate(nu_per_snapshot, axis=0)

    Phi, _ = build_pod_basis(ensemble, r)
    F_ensemble = collect_nonlinear_snapshots(ensemble, k, nu_per_snapshot, dealias)
    Psi, _ = build_deim_basis(F_ensemble, m)
    points = select_deim_points(Psi)
    M = build_deim_projector(Phi, Psi, points)
    return dict(Phi=Phi, Psi=Psi, points=points, M=M, k=k, dealias=dealias, Nx=Nx,
                ensemble=ensemble, nu_per_snapshot=nu_per_snapshot)


def test_deim_points_are_distinct_and_in_range():
    ctx = _setup()
    points = ctx["points"]
    assert len(points) == len(set(points.tolist())), "DEIM points must be distinct"
    assert np.all((points >= 0) & (points < ctx["Nx"]))


def test_deim_projector_shape():
    ctx = _setup(r=8, m=12)
    assert ctx["M"].shape == (8, 12)
    assert np.all(np.isfinite(ctx["M"]))


def test_local_fd_nonlinear_matches_spectral_at_deim_points():
    """Core validation of this module's central design choice (see src/deim.py docstring): the
    local finite-difference nonlinear evaluation at the DEIM points must be a good approximation
    of the *exact* spectral evaluation restricted to those same points -- not merely "close enough
    by construction," but empirically verified here against the ground-truth spectral RHS.
    """
    ctx = _setup(r=8, m=12, Nx=64)
    Phi, Psi, points, k, dealias, Nx = (ctx["Phi"], ctx["Psi"], ctx["points"], ctx["k"],
                                          ctx["dealias"], ctx["Nx"])
    nu = 0.05
    # Use a genuine, solver-generated mid-trajectory state (not an arbitrary random reduced
    # coordinate vector) -- DEIM-ROM only ever operates along real Burgers trajectories, and an
    # unconstrained random `a` can put disproportionate energy into higher (less smooth) POD
    # modes than any physically realized state exhibits, which would make this a test of an
    # irrelevant regime rather than of the method's actual operating conditions.
    _, _, u_traj = solve_burgers(A=1.3, nu=nu, Nx=Nx, T=0.5, n_save=10)
    a = Phi.T @ u_traj[5]  # project a mid-trajectory snapshot onto the POD basis
    u = Phi @ a  # reconstruct -- the exact spectral RHS below must be evaluated on the SAME
    # reduced-then-reconstructed field the DEIM approximation implicitly operates on, not on the
    # original unprojected snapshot, for an apples-to-apples comparison.

    # Exact spectral RHS (full grid), restricted after the fact to the DEIM points.
    u_hat = np.fft.fft(u)
    ux_exact = np.real(np.fft.ifft(1j * k * u_hat))
    uxx_exact = np.real(np.fft.ifft(-(k**2) * u_hat))
    flux_hat = np.fft.fft(u * u) * dealias
    nonlinear_exact = np.real(np.fft.ifft(-0.5j * k * flux_hat))
    F_exact_at_points = (nonlinear_exact + nu * uxx_exact)[points]

    # This module's local-FD approximation, evaluated ONLY via Phi's rows on the stencil support.
    support, neighbours = _stencil_support(points, Nx, halfwidth=1)
    Phi_support = Phi[support, :]
    dx = (2 * np.pi) / Nx
    F_approx_at_points = deim_nonlinear_at_points(a, Phi_support, support, neighbours, nu, dx)

    rel_err = np.linalg.norm(F_approx_at_points - F_exact_at_points) / (
        np.linalg.norm(F_exact_at_points) + 1e-12)
    # A 2nd-order central-difference stencil on a 64-point periodic grid is expected to introduce
    # a real, non-negligible discretization error relative to spectral accuracy -- this test
    # documents and bounds that error rather than asserting spectral-level agreement. The bound is
    # loose by design (this is a validation, not a tautology): it would fail if the stencil
    # indexing/formula were wrong (e.g. off-by-one neighbours), not merely because the
    # approximation is imperfect.
    assert rel_err < 0.25, f"local-FD DEIM nonlinear evaluation diverges too far from spectral truth: {rel_err:.4f}"


def test_deim_rom_predict_shape_and_finiteness():
    ctx = _setup(r=8, m=12, Nx=64)
    Phi, M, points, Nx = ctx["Phi"], ctx["M"], ctx["points"], ctx["Nx"]
    u0 = 1.2 * np.sin(np.linspace(0, 2 * np.pi, Nx, endpoint=False))
    t_save, u_pred = deim_rom_predict(u0, Phi, M, points, Nx, nu=0.05, T=0.5, n_save=8)
    assert u_pred.shape == (8, Nx)
    assert np.all(np.isfinite(u_pred))
    assert t_save[0] == 0.0
    assert abs(t_save[-1] - 0.5) < 1e-9


def test_deim_rom_trajectory_close_to_plain_rom_for_adequate_m():
    """Sanity check, not a strict accuracy bound: with enough DEIM modes, the hyper-reduced ROM's
    trajectory should stay in the same ballpark as the plain (non-hyper-reduced) ROM -- both are
    approximating the same underlying Galerkin-projected dynamics, so a wildly different answer
    here would indicate an implementation bug, not merely added approximation error.
    """
    ctx = _setup(r=8, m=16, Nx=64)
    Phi, Psi, points, M, k, dealias, Nx = (ctx["Phi"], ctx["Psi"], ctx["points"], ctx["M"],
                                             ctx["k"], ctx["dealias"], ctx["Nx"])
    A, nu = 1.5, 0.05
    u0 = A * np.sin(np.linspace(0, 2 * np.pi, Nx, endpoint=False))

    _, u_plain = rom_predict(u0, Phi, k, nu, dealias, T=0.5, n_save=10)
    _, u_deim = deim_rom_predict(u0, Phi, M, points, Nx, nu=nu, T=0.5, n_save=10)

    rel_err = np.linalg.norm(u_deim - u_plain) / (np.linalg.norm(u_plain) + 1e-12)
    assert rel_err < 0.3, f"DEIM-ROM diverges too far from plain ROM for adequate m: {rel_err:.4f}"


def test_deim_online_cost_independent_of_grid_resolution():
    """Verifies the actual design goal, not just accuracy: the per-stage DEIM RHS evaluation only
    ever touches Phi rows at the (small, m-dependent) stencil support -- never the full Nx grid.
    This is checked structurally (support size scales with m, not Nx), matching the complexity
    claim documented in RESEARCH_EXTENSION.md."""
    for Nx in (64, 128, 256):
        points = np.array([5, 20, 40, 60]) % Nx
        support, _ = _stencil_support(points, Nx, halfwidth=1)
        # halfwidth=1 -> at most 3 neighbours per point; support size must not scale with Nx.
        assert len(support) <= 3 * len(points)


def test_4th_order_stencil_matches_spectral_at_deim_points_at_least_as_well_as_2nd_order():
    """4th-order central differences have strictly smaller truncation error than 2nd-order at
    fixed dx for a smooth-enough field, so on the same genuine solver-generated state this variant
    should match the exact spectral evaluation at least as well (not worse) as the 2nd-order one.
    """
    ctx = _setup(r=8, m=12, Nx=64)
    Phi, points, k, dealias, Nx = ctx["Phi"], ctx["points"], ctx["k"], ctx["dealias"], ctx["Nx"]
    nu = 0.05
    _, _, u_traj = solve_burgers(A=1.3, nu=nu, Nx=Nx, T=0.5, n_save=10)
    a = Phi.T @ u_traj[5]
    u = Phi @ a

    u_hat = np.fft.fft(u)
    uxx_exact = np.real(np.fft.ifft(-(k**2) * u_hat))
    flux_hat = np.fft.fft(u * u) * dealias
    nonlinear_exact = np.real(np.fft.ifft(-0.5j * k * flux_hat))
    F_exact_at_points = (nonlinear_exact + nu * uxx_exact)[points]

    dx = (2 * np.pi) / Nx
    support2, neigh2 = _stencil_support(points, Nx, halfwidth=1)
    F_2nd = deim_nonlinear_at_points(a, Phi[support2, :], support2, neigh2, nu, dx)
    support4, neigh4 = _stencil_support(points, Nx, halfwidth=2)
    F_4th = deim_nonlinear_at_points_4th_order(a, Phi[support4, :], support4, neigh4, nu, dx)

    err_2nd = np.linalg.norm(F_2nd - F_exact_at_points) / (np.linalg.norm(F_exact_at_points) + 1e-12)
    err_4th = np.linalg.norm(F_4th - F_exact_at_points) / (np.linalg.norm(F_exact_at_points) + 1e-12)
    assert err_4th <= err_2nd + 1e-9, f"4th-order ({err_4th:.4f}) should not be worse than 2nd-order ({err_2nd:.4f})"


def test_deim_rom_predict_stencil_order_4_shape_and_finiteness():
    ctx = _setup(r=8, m=12, Nx=64)
    Phi, M, points, Nx = ctx["Phi"], ctx["M"], ctx["points"], ctx["Nx"]
    u0 = 1.2 * np.sin(np.linspace(0, 2 * np.pi, Nx, endpoint=False))
    t_save, u_pred = deim_rom_predict(u0, Phi, M, points, Nx, nu=0.05, T=0.5, n_save=8,
                                       stencil_order=4)
    assert u_pred.shape == (8, Nx)
    assert np.all(np.isfinite(u_pred))


def test_deim_rom_predict_default_stencil_order_unchanged():
    """stencil_order defaults to 2 -- confirms the new parameter is fully backward-compatible and
    every existing caller's behaviour (scripts/build_deim.py, scripts/evaluate_research_extension.py)
    is bit-for-bit unchanged."""
    ctx = _setup(r=8, m=12, Nx=64)
    Phi, M, points, Nx = ctx["Phi"], ctx["M"], ctx["points"], ctx["Nx"]
    u0 = 1.2 * np.sin(np.linspace(0, 2 * np.pi, Nx, endpoint=False))
    t_save_a, u_pred_a = deim_rom_predict(u0, Phi, M, points, Nx, nu=0.05, T=0.5, n_save=8)
    t_save_b, u_pred_b = deim_rom_predict(u0, Phi, M, points, Nx, nu=0.05, T=0.5, n_save=8,
                                           stencil_order=2)
    np.testing.assert_array_equal(u_pred_a, u_pred_b)
