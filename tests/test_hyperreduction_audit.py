"""Correctness tests for the hyper-reduction audit (scripts/hyperreduction_benchmark.py).

Small grids (Nx = 32..128) and a handful of trajectories, so the whole file runs in a few seconds.
"""
import numpy as np
import pytest

from scripts.generate_dataset import generate_split
from scripts.hyperreduction_benchmark import (_reduced_rk4, aligned_dt, aligned_reference,
                                               fom_steps, native_reduced_steps, pareto_mask,
                                               rng_free_config, wavenumbers)
from src.deim import (_stencil_support, build_deim_basis, build_deim_projector,
                       collect_nonlinear_snapshots, deim_nonlinear_at_points, deim_rom_predict,
                       deim_rom_reduced_trajectory, select_deim_points)
from src.metrics import relative_l2
from src.pod_rom import (_burgers_rhs_physical, build_pod_basis, rom_predict,
                          rom_reduced_trajectory)
from src.solver import solve_burgers

L = 2 * np.pi


def _ctx(r=8, m=12, Nx=64, n_examples=12, seed=0):
    k, dealias = wavenumbers(Nx)
    rng = np.random.default_rng(seed)
    ens, nus = [], []
    for _ in range(n_examples):
        A, nu = rng.uniform(0.5, 2.0), rng.uniform(0.01, 0.1)
        u = solve_burgers(A=A, nu=nu, Nx=Nx, T=1.0, n_save=10)[2]
        ens.append(u.astype(np.float64))
        nus.append(np.full(u.shape[0], nu))
    ens, nus = np.concatenate(ens), np.concatenate(nus)
    Phi, _ = build_pod_basis(ens, r)
    F = collect_nonlinear_snapshots(ens, k, nus, dealias)
    Psi, _ = build_deim_basis(F, m)
    pts = select_deim_points(Psi)
    M = build_deim_projector(Phi, Psi, pts)
    return dict(Phi=Phi, Psi=Psi, points=pts, M=M, k=k, dealias=dealias, Nx=Nx, r=r, m=m)


# ------------------------------------------------------------------ bases and operator shapes
def test_pod_and_deim_bases_are_orthonormal_to_machine_precision():
    c = _ctx()
    np.testing.assert_allclose(c["Phi"].T @ c["Phi"], np.eye(c["r"]), atol=1e-12)
    np.testing.assert_allclose(c["Psi"].T @ c["Psi"], np.eye(c["m"]), atol=1e-12)


@pytest.mark.parametrize("r,m", [(4, 6), (8, 12), (6, 16)])
def test_reduced_operator_and_trajectory_dimensions(r, m):
    c = _ctx(r=r, m=m)
    assert c["Phi"].shape == (c["Nx"], r) and c["Psi"].shape == (c["Nx"], m)
    assert c["M"].shape == (r, m)
    support, neigh = _stencil_support(c["points"], c["Nx"], 1)
    assert neigh.shape == (m, 3) and len(support) <= 3 * m
    u0 = 1.2 * np.sin(np.linspace(0, L, c["Nx"], endpoint=False))
    a0 = c["Phi"].T @ u0
    t, a, n = deim_rom_reduced_trajectory(a0, np.abs(u0).max(), c["Phi"], c["M"], c["points"],
                                          c["Nx"], 0.05, 0.5, 8)
    assert a.shape == (8, r) and t.shape == (8,) and n >= 32
    t2, a2, n2 = rom_reduced_trajectory(a0, np.abs(u0).max(), c["Phi"], c["k"], 0.05,
                                        c["dealias"], 0.5, 8)
    assert a2.shape == (8, r)


# ------------------------------------------------------------------- DEIM interpolation maths
@pytest.mark.parametrize("m", [6, 12, 16])
def test_deim_points_unique_and_in_range(m):
    c = _ctx(m=m)
    p = c["points"]
    assert len(p) == m and len(set(p.tolist())) == m
    assert p.min() >= 0 and p.max() < c["Nx"]


def test_deim_reproduces_vectors_in_span_of_psi_exactly():
    """Defining property of DEIM: for f in range(Psi), Psi (P^T Psi)^{-1} P^T f = f, and therefore
    the Galerkin-DEIM operator gives M @ f[points] = Phi^T f."""
    c = _ctx()
    rng = np.random.default_rng(1)
    f = c["Psi"] @ rng.standard_normal(c["m"])
    interp = c["Psi"] @ np.linalg.solve(c["Psi"][c["points"], :], f[c["points"]])
    np.testing.assert_allclose(interp, f, atol=1e-10)
    np.testing.assert_allclose(c["M"] @ f[c["points"]], c["Phi"].T @ f, atol=1e-10)


def test_deim_with_full_identity_basis_equals_plain_galerkin_rhs():
    """Limiting case m = Nx: with Psi = I every grid point is an interpolation point, P = I and
    M = Phi^T, so the exact-F DEIM right-hand side must equal the plain Galerkin right-hand side."""
    Nx, r = 32, 5
    k, dealias = wavenumbers(Nx)
    rng = np.random.default_rng(2)
    Phi = np.linalg.qr(rng.standard_normal((Nx, r)))[0]
    Psi = np.eye(Nx)
    pts = select_deim_points(Psi)
    assert sorted(pts.tolist()) == list(range(Nx))
    M = build_deim_projector(Phi, Psi, pts)
    np.testing.assert_allclose(M, Phi.T, atol=1e-12)
    a = rng.standard_normal(r) * 0.3
    F = _burgers_rhs_physical(Phi @ a, k, 0.05, dealias)
    np.testing.assert_allclose(M @ F[pts], Phi.T @ F, atol=1e-12)


def test_local_fd_nonlinear_is_second_order_accurate_on_a_smooth_field():
    """Limiting case: for a smooth (low-wavenumber) field the local-FD evaluation must converge to
    the spectral one at second order as the grid is refined (ratio ~4 per doubling)."""
    errs = []
    for Nx in (64, 128, 256):
        k, dealias = wavenumbers(Nx)
        x = np.linspace(0, L, Nx, endpoint=False)
        u = np.sin(x) + 0.3 * np.cos(2 * x)
        # exact-in-the-limit reference: the spectral RHS of the same sampled field
        F_spec = _burgers_rhs_physical(u, k, 0.05, dealias)
        pts = np.arange(3, Nx, 7)
        # "basis" = identity on the support, so Phi_support @ a is just u sampled there
        support, neigh = _stencil_support(pts, Nx, 1)
        F_fd = deim_nonlinear_at_points(np.ones(1), u[support, None], support, neigh, 0.05,
                                        L / Nx)
        errs.append(np.linalg.norm(F_fd - F_spec[pts]) / np.linalg.norm(F_spec[pts]))
    assert errs[2] < 5e-3
    assert 3.0 < errs[0] / errs[1] < 5.0 and 3.0 < errs[1] / errs[2] < 5.0


# --------------------------------------------- no full-grid work in the hyper-reduced time loop
def _poisoned(c, halfwidth=1):
    """NaN out every Phi row outside the stencil support of the given half-width (1 for the 3-point
    stencil, 2 for the 5-point one)."""
    support, _ = _stencil_support(c["points"], c["Nx"], halfwidth)
    Phi_p = c["Phi"].copy()
    mask = np.ones(c["Nx"], dtype=bool)
    mask[support] = False
    Phi_p[mask, :] = np.nan
    return Phi_p, int(mask.sum())


@pytest.mark.parametrize("order", [2, 4])
def test_deim_time_loop_reads_only_stencil_support_rows(order):
    """Every row of Phi outside the DEIM stencil support is replaced with NaN. If the timed online
    loop touched any full-grid quantity the NaNs would reach the trajectory; it must instead be
    bit-identical to the clean run."""
    c = _ctx(Nx=128, r=8, m=12)
    Phi = c["Phi"]
    u0 = 1.4 * np.sin(np.linspace(0, L, 128, endpoint=False))
    a0 = Phi.T @ u0  # the initial projection is the one full-grid step, done outside the loop
    args = (np.abs(u0).max(),)
    kw = dict(M=c["M"], points=c["points"], Nx=128, nu=0.04, T=0.5, n_save=6, stencil_order=order)
    clean = deim_rom_reduced_trajectory(a0, *args, Phi, **kw)
    Phi_p, n_poisoned = _poisoned(c, halfwidth=1 if order == 2 else 2)
    assert n_poisoned > 128 - 5 * c["m"]  # most of the grid really is poisoned
    poisoned = deim_rom_reduced_trajectory(a0, *args, Phi_p, **kw)
    assert np.all(np.isfinite(poisoned[1]))
    np.testing.assert_array_equal(clean[1], poisoned[1])


def test_deim_time_loop_performs_no_fft(monkeypatch):
    c = _ctx(Nx=64, m=12)
    u0 = 1.0 * np.sin(np.linspace(0, L, 64, endpoint=False))
    a0 = c["Phi"].T @ u0

    def boom(*a, **k):
        raise AssertionError("FFT called inside the hyper-reduced time loop")

    for name in ("fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "fftn", "ifftn"):
        monkeypatch.setattr(np.fft, name, boom)
    _, a, _ = deim_rom_reduced_trajectory(a0, np.abs(u0).max(), c["Phi"], c["M"], c["points"], 64,
                                          0.05, 0.3, 5)
    assert np.all(np.isfinite(a))


def test_poison_check_detects_the_plain_rom_full_grid_dependence():
    """Positive control for the test above: the plain ROM's loop DOES need the full grid, so the
    same poisoning must break it (otherwise the poison check would prove nothing)."""
    c = _ctx(Nx=64, m=12)
    Phi_p, _ = _poisoned(c)
    u0 = 1.0 * np.sin(np.linspace(0, L, 64, endpoint=False))
    a0 = c["Phi"].T @ u0
    # explicit dt: the reduced step policy itself reads the full Phi (spectral radius of the reduced
    # diffusion operator), so pin dt to exercise the RHS loop alone
    _, a, _ = rom_reduced_trajectory(a0, np.abs(u0).max(), Phi_p, c["k"], 0.05, c["dealias"],
                                     0.3, 5, dt=0.01)
    assert not np.all(np.isfinite(a))


def test_predict_wrappers_equal_reduced_loop_plus_reconstruction():
    c = _ctx(Nx=64, m=12)
    u0 = 1.3 * np.sin(np.linspace(0, L, 64, endpoint=False)).astype(np.float32)
    _, u_p = deim_rom_predict(u0, c["Phi"], c["M"], c["points"], 64, nu=0.05, T=0.5, n_save=8)
    _, a, _ = deim_rom_reduced_trajectory(c["Phi"].T @ u0, np.abs(u0).max(), c["Phi"], c["M"],
                                          c["points"], 64, 0.05, 0.5, 8)
    np.testing.assert_array_equal(u_p, (a @ c["Phi"].T).astype(np.float32))
    _, u_r = rom_predict(u0, c["Phi"], c["k"], 0.05, c["dealias"], T=0.5, n_save=8)
    _, a_r, _ = rom_reduced_trajectory(c["Phi"].T @ u0, np.abs(u0).max(), c["Phi"], c["k"], 0.05,
                                       c["dealias"], 0.5, 8)
    np.testing.assert_array_equal(u_r, (a_r @ c["Phi"].T).astype(np.float32))


# ------------------------------------------------------------------------------------- metric
def test_relative_l2_definition():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((5, 16)).astype(np.float32) + 3
    assert relative_l2(ref, ref) == 0.0
    assert relative_l2(1.1 * ref, ref) == pytest.approx(0.1, rel=1e-5)
    assert relative_l2(0.0 * ref, ref) == pytest.approx(1.0, rel=1e-6)
    pred = ref + 0.01 * rng.standard_normal(ref.shape).astype(np.float32)
    assert relative_l2(pred, ref) == pytest.approx(
        np.linalg.norm(pred - ref) / np.linalg.norm(ref), rel=1e-5)
    assert relative_l2(5 * pred, 5 * ref) == pytest.approx(relative_l2(pred, ref), rel=1e-5)


# ------------------------------------------- benchmark determinism and time-aligned protocol
def test_benchmark_data_and_config_are_deterministic():
    a1, n1, u1 = generate_split("test", 40)
    a2, n2, u2 = generate_split("test", 40)
    assert np.array_equal(a1, a2) and np.array_equal(n1, n2) and np.array_equal(u1, u2)
    assert rng_free_config() == rng_free_config()


def test_reduced_predictions_are_bit_reproducible():
    c = _ctx(Nx=64, m=12)
    u0 = 1.3 * np.sin(np.linspace(0, L, 64, endpoint=False)).astype(np.float32)
    runs = [deim_rom_predict(u0, c["Phi"], c["M"], c["points"], 64, nu=0.05, T=0.5, n_save=8)[1]
            for _ in range(2)]
    np.testing.assert_array_equal(runs[0], runs[1])


@pytest.mark.parametrize("A,nu", [(1.92, 0.038), (0.86, 0.098), (0.66, 0.026)])
def test_aligned_protocol_gives_uniform_snapshot_times(A, nu):
    n_native = fom_steps(A, nu, 128, 1.0, 20)
    t, _, _ = solve_burgers(A=A, nu=nu, Nx=128, T=1.0, n_save=20, dt=aligned_dt(n_native))
    np.testing.assert_allclose(t, np.linspace(0, 1, 20), atol=1e-12)
    # the native protocol is NOT uniform whenever 19 does not divide the step count -- the reason
    # index-wise comparison across methods with different step counts is misleading
    t_native, _, _ = solve_burgers(A=A, nu=nu, Nx=128, T=1.0, n_save=20)
    if n_native % 19:
        assert np.abs(t_native - np.linspace(0, 1, 20)).max() > 1e-4


def test_aligned_fom_is_temporally_converged():
    ref = aligned_reference(1.5, 0.03, 128)
    fine = solve_burgers(A=1.5, nu=0.03, Nx=128, T=1.0, n_save=20,
                         dt=aligned_dt(fom_steps(1.5, 0.03, 128, 1.0, 20), 4))[2]
    assert relative_l2(ref, fine) < 1e-4


@pytest.mark.parametrize("nu,A", [(0.01, 2.0), (0.09, 0.6), (0.04, 1.2)])
def test_native_step_formula_matches_src_loops(nu, A):
    c = _ctx(Nx=64, m=12)
    u0 = (A * np.sin(np.linspace(0, L, 64, endpoint=False))).astype(np.float32)
    a0 = c["Phi"].T @ u0
    n_formula = native_reduced_steps(np.abs(u0).max(), 64, nu)
    _, _, n_rom = rom_reduced_trajectory(a0, np.abs(u0).max(), c["Phi"], c["k"], nu, c["dealias"],
                                         1.0, 20, timestep_policy="historical")
    _, _, n_deim = deim_rom_reduced_trajectory(a0, np.abs(u0).max(), c["Phi"], c["M"],
                                               c["points"], 64, nu, 1.0, 20,
                                               timestep_policy="historical")
    assert n_formula == n_rom == n_deim


def test_diagnostic_rk4_driver_matches_src_reduced_loop():
    c = _ctx(Nx=64, m=12)
    u0 = (1.2 * np.sin(np.linspace(0, L, 64, endpoint=False))).astype(np.float32)
    a0 = c["Phi"].T @ u0
    _, a_src, n_src = rom_reduced_trajectory(a0, np.abs(u0).max(), c["Phi"], c["k"], 0.03,
                                             c["dealias"], 1.0, 10)
    rhs = lambda a: c["Phi"].T @ _burgers_rhs_physical(c["Phi"] @ a, c["k"], 0.03, c["dealias"])
    a_cp, n_cp = _reduced_rk4(rhs, a0, np.abs(u0).max(), 64, 0.03, 1.0, 10)
    assert n_src == n_cp
    np.testing.assert_array_equal(a_src, a_cp)


def test_pareto_mask_treats_near_equal_times_as_ties():
    err = np.array([0.001, 0.05, 0.04, 0.5, np.nan])
    tim = np.array([17.0, 3.60, 3.57, 3.58, 1.0])
    assert pareto_mask(err, tim).tolist() == [True, False, True, False, False]
