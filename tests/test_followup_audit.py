"""Tests for the follow-up audit: aligned snapshot times, nu-aware FNO input, time-step policy and
stability tools, local-FD stencil behaviour, and the DEIM no-full-grid guarantee with the new options."""
import numpy as np
import pytest
import torch

from scripts.hyperreduction_benchmark import wavenumbers
from scripts.hyperreduction_followup import assess, make_ctx, run_red
from src.deim import (_stencil_support, build_deim_basis, build_deim_projector,
                       collect_nonlinear_snapshots, deim_nonlinear_at_points,
                       deim_nonlinear_at_points_4th_order, deim_rom_predict,
                       deim_rom_reduced_trajectory, select_deim_points)
from src.fno_net import FNO1d
from src.general_ic_solver import solve_burgers_general_ic
from src.metrics import relative_l2
from src.pod_rom import _burgers_rhs_physical, build_pod_basis, rom_predict, rom_reduced_trajectory
from src.snapshot_times import (align_step_count, intended_times, resample_in_time, saved_step_indices,
                                 saved_times)
from src.solver import solve_burgers
from src.timestep import (RK4_IMAG_LIMIT, RK4_REAL_LIMIT, diffusion_dt, fom_advective_dt,
                           historical_dt, historical_rule_fractions_of_rk4_limit, numerical_jacobian,
                           reduced_diffusion_matrix_deim, reduced_diffusion_matrix_rom, revised_dt,
                           rk4_amplification, rk4_max_stable_dt, spectral_radius)

L = 2 * np.pi


def _small_ctx(Nx=64, r=8, m=12):
    k, dealias = wavenumbers(Nx)
    rng = np.random.default_rng(0)
    ens, nus = [], []
    for _ in range(12):
        A, nu = rng.uniform(0.5, 2.0), rng.uniform(0.01, 0.1)
        u = solve_burgers(A=A, nu=nu, Nx=Nx, T=1.0, n_save=10)[2].astype(np.float64)
        ens.append(u)
        nus.append(np.full(u.shape[0], nu))
    ens, nus = np.concatenate(ens), np.concatenate(nus)
    Phi, _ = build_pod_basis(ens, r)
    Psi, _ = build_deim_basis(collect_nonlinear_snapshots(ens, k, nus, dealias), m)
    pts = select_deim_points(Psi)
    return make_ctx(Phi, build_deim_projector(Phi, Psi, pts), pts, Nx)


# ------------------------------------------------------------------- aligned snapshot times
@pytest.mark.parametrize("A,nu", [(1.92, 0.038), (0.86, 0.098), (0.66, 0.026), (1.3, 0.05)])
def test_aligned_fom_saves_at_exactly_the_requested_times(A, nu):
    for Nx in (64, 128):
        t, _, _ = solve_burgers(A=A, nu=nu, Nx=Nx, T=1.0, n_save=20, align_snapshots=True)
        np.testing.assert_allclose(t, intended_times(20, 1.0), atol=1e-12)
    t2, _, _ = solve_burgers_general_ic(1.2 * np.sin(np.linspace(0, L, 64, endpoint=False)), nu,
                                        T=1.0, n_save=20, align_snapshots=True)
    np.testing.assert_allclose(t2, intended_times(20, 1.0), atol=1e-12)


def test_reduced_models_share_the_aligned_times_whatever_their_step_count():
    C = _small_ctx()
    u0 = 1.1 * np.sin(np.linspace(0, L, 64, endpoint=False))
    a0 = C["Phi"].T @ u0
    times = []
    for dt in (None, 0.004, 0.011):  # three different step counts
        t_r, _, n_r = rom_reduced_trajectory(a0, np.abs(u0).max(), C["Phi"], C["k"], 0.05, C["dealias"],
                                             1.0, 20, dt, align_snapshots=True)
        t_d, _, n_d = deim_rom_reduced_trajectory(a0, np.abs(u0).max(), C["Phi"], C["M"], C["pts"], 64,
                                                  0.05, 1.0, 20, dt, align_snapshots=True)
        assert n_r % 19 == 0 and n_d % 19 == 0
        times += [t_r, t_d]
    for t in times:
        np.testing.assert_allclose(t, intended_times(20, 1.0), atol=1e-12)


def test_default_behaviour_is_the_historical_nonuniform_rule():
    """The default is untouched: times deviate from j/19 when 19 does not divide n_steps, and follow
    src.snapshot_times.saved_times exactly."""
    A, nu = 0.86, 0.098
    t, _, _ = solve_burgers(A=A, nu=nu, Nx=128, T=1.0, n_save=20)  # 80 steps
    assert np.abs(t - intended_times(20, 1.0)).max() > 1e-3
    np.testing.assert_allclose(t, saved_times(80, 20, 1.0), atol=1e-12)


def test_align_step_count_properties():
    for n in (1, 18, 19, 20, 80, 157, 1253):
        m = align_step_count(n, 20)
        assert m % 19 == 0 and n <= m < n + 19
    idx = saved_step_indices(align_step_count(157, 20), 20)
    assert np.all(np.diff(idx) == align_step_count(157, 20) // 19)


def test_resampling_legacy_snapshots_recovers_the_aligned_solution():
    """Legacy (non-uniform-time) snapshots, cubic-resampled at their true times, agree with the
    aligned solve far better than the raw index-wise comparison does."""
    for A, nu in ((1.5, 0.05), (0.86, 0.098)):
        t, _, u = solve_burgers(A=A, nu=nu, Nx=128, T=1.0, n_save=20)
        _, _, ua = solve_burgers(A=A, nu=nu, Nx=128, T=1.0, n_save=20, align_snapshots=True)
        raw = relative_l2(u, ua)
        fixed = relative_l2(resample_in_time(t, u, intended_times(20, 1.0)).astype(np.float32), ua)
        assert raw > 1e-3 and fixed < 1e-4 and fixed < raw / 20


def test_predict_wrappers_accept_the_new_options_without_changing_defaults():
    C = _small_ctx()
    u0 = (1.3 * np.sin(np.linspace(0, L, 64, endpoint=False))).astype(np.float32)
    d0 = deim_rom_predict(u0, C["Phi"], C["M"], C["pts"], 64, nu=0.05, T=0.5, n_save=8)[1]
    d1 = deim_rom_predict(u0, C["Phi"], C["M"], C["pts"], 64, nu=0.05, T=0.5, n_save=8,
                          align_snapshots=False, min_steps=None)[1]
    np.testing.assert_array_equal(d0, d1)
    r0 = rom_predict(u0, C["Phi"], C["k"], 0.05, C["dealias"], T=0.5, n_save=8)[1]
    r1 = rom_predict(u0, C["Phi"], C["k"], 0.05, C["dealias"], T=0.5, n_save=8, align_snapshots=False)[1]
    np.testing.assert_array_equal(r0, r1)


def test_historical_policy_equals_explicit_historical_dt_bit_for_bit():
    C = _small_ctx()
    u0 = (1.3 * np.sin(np.linspace(0, L, 64, endpoint=False))).astype(np.float32)
    dt = historical_dt(np.abs(u0).max(), 64, 0.05)
    a0, u0max = C["Phi"].T @ u0, np.abs(u0).max()
    for name, call in (("rom", lambda **kw: rom_reduced_trajectory(
            a0, u0max, C["Phi"], C["k"], 0.05, C["dealias"], 1.0, 20, **kw)),
            ("deim", lambda **kw: deim_rom_reduced_trajectory(
                a0, u0max, C["Phi"], C["M"], C["pts"], 64, 0.05, 1.0, 20, **kw))):
        _, a_hist, n_hist = call(timestep_policy="historical")
        _, a_dt, n_dt = call(dt=dt)
        assert n_hist == n_dt
        np.testing.assert_array_equal(a_hist, a_dt)


# ------------------------------------------------------------------------ nu-aware FNO input
def test_nu_channel_construction_and_determinism():
    m = FNO1d(32, 6, modes=4, width=8, n_layers=2, use_nu=True)
    u0 = torch.randn(3, 32)
    nu = torch.tensor([0.01, 0.055, 0.1])
    x = m.build_input(u0, nu)
    assert x.shape == (3, 3, 32) and m.lift.in_channels == 3
    torch.testing.assert_close(x[:, 0], u0)
    torch.testing.assert_close(x[:, 1], m.x_grid.expand(3, -1))
    expected = ((nu - 0.01) / 0.09).reshape(3, 1).expand(3, 32)
    torch.testing.assert_close(x[:, 2], expected)
    assert torch.equal(m.build_input(u0, nu), x)  # deterministic
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(u0, nu), m(u0, nu))
        assert not torch.allclose(m(u0, nu), m(u0, nu.flip(0)))  # output depends on nu


def test_blind_fno_is_unchanged_and_ignores_nu():
    blind = FNO1d(32, 6, modes=4, width=8, n_layers=2)
    aware = FNO1d(32, 6, modes=4, width=8, n_layers=2, use_nu=True)
    assert blind.lift.in_channels == 2 and blind.build_input(torch.zeros(2, 32)).shape == (2, 2, 32)
    n_b, n_a = sum(p.numel() for p in blind.parameters()), sum(p.numel() for p in aware.parameters())
    assert n_a - n_b == 8  # one extra input channel x width 8 lift weights
    kb, ka = blind.state_dict(), aware.state_dict()
    assert set(kb) == set(ka) and [k for k in kb if kb[k].shape != ka[k].shape] == ["lift.weight"]
    blind.eval()
    u0 = torch.randn(2, 32)
    with torch.no_grad():
        assert torch.equal(blind(u0), blind(u0, torch.tensor([0.02, 0.09])))
    with pytest.raises(ValueError):
        aware.build_input(u0)  # nu-aware network refuses to run without nu


# --------------------------------------------------------------------- timestep and stability
def test_historical_rule_constants_and_fractions():
    dx = L / 128
    assert historical_dt(np.float32(1.0), 128, 0.1) == pytest.approx(0.4 * dx**2 / 0.2, rel=1e-6)
    assert historical_dt(np.float32(2.0), 128, 0.01) == pytest.approx(0.25 * dx / 2.0, rel=1e-6)
    f = historical_rule_fractions_of_rk4_limit()
    assert f["vs_fullgrid_spectral_laplacian"] == pytest.approx(0.7087, abs=1e-3)
    assert f["vs_fullgrid_fd3_laplacian"] == pytest.approx(0.2872, abs=1e-3)


def test_rk4_stability_limits_match_theory():
    assert rk4_max_stable_dt(np.array([-100.0])) == pytest.approx(RK4_REAL_LIMIT / 100, rel=0.01)
    assert rk4_max_stable_dt(np.array([10.0j, -10.0j])) == pytest.approx(RK4_IMAG_LIMIT / 10, rel=0.01)
    assert rk4_max_stable_dt(np.array([-1.0, 0.5])) == 0.0  # linearly unstable: no step size helps
    assert abs(rk4_amplification(-2.7)) < 1 < abs(rk4_amplification(-2.9))


def test_stability_detector_flags_blowup_and_nonfinite():
    Phi = np.eye(4, 3)
    C = {"Phi": Phi}
    ref = np.ones((5, 4), dtype=np.float32)
    good = assess(np.ones((5, 3)), C, ref, 1.0)
    assert good["stable"] and good["finite"]
    bad = assess(np.full((5, 3), 1e3), C, ref, 1.0)
    assert not bad["stable"] and bad["first_bad_snapshot"] == 0
    nan = np.ones((5, 3))
    nan[3:] = np.nan
    r = assess(nan, C, ref, 1.0)
    assert not r["stable"] and not r["finite"] and r["first_bad_snapshot"] == 3


def test_reduced_diffusion_matrices_and_policy():
    Nx = 64
    k, _ = wavenumbers(Nx)
    x = np.linspace(0, L, Nx, endpoint=False)
    Phi = np.stack([np.sin(x), np.cos(x), np.sin(3 * x)], axis=1) / np.sqrt(Nx / 2)
    K = reduced_diffusion_matrix_rom(Phi, k)
    np.testing.assert_allclose(np.sort(np.linalg.eigvals(K).real), [-9.0, -1.0, -1.0], atol=1e-10)
    assert spectral_radius(K) == pytest.approx(9.0)
    assert diffusion_dt(0.1, K) == pytest.approx(0.5 * RK4_REAL_LIMIT / 0.9)
    # DEIM/local-FD version with every grid point selected: second-order accurate diffusion
    pts = np.arange(Nx)
    Kd = reduced_diffusion_matrix_deim(Phi, Phi.T, pts, Nx)
    np.testing.assert_allclose(np.sort(np.linalg.eigvals(Kd).real), np.sort(np.linalg.eigvals(K).real),
                               rtol=0.05)
    K4 = reduced_diffusion_matrix_deim(Phi, Phi.T, pts, Nx, stencil_order=4)
    assert np.max(np.abs(np.sort(np.linalg.eigvals(K4).real) - np.sort(np.linalg.eigvals(K).real))) < \
        np.max(np.abs(np.sort(np.linalg.eigvals(Kd).real) - np.sort(np.linalg.eigvals(K).real)))
    # policy: never above the FOM step, never above the diffusion bound, tighter for larger nu
    u0max = np.float32(1.5)
    Ks = 1000.0 * K  # a stiff reduced diffusion, so that the diffusion bound can bind
    d = [revised_dt(u0max, Nx, nu, Ks) for nu in (0.01, 0.05, 0.2, 2.0)]
    assert all(v <= fom_advective_dt(u0max, Nx) + 1e-15 for v in d)
    assert d[-1] < d[0] and d[-1] <= diffusion_dt(2.0, Ks) + 1e-15
    # with the mild K above the advective (FOM) bound is the binding one for every nu
    assert revised_dt(u0max, Nx, 2.0, K) == pytest.approx(fom_advective_dt(u0max, Nx))


def test_numerical_jacobian_of_a_quadratic_map_is_exact():
    rng = np.random.default_rng(0)
    B, Q = rng.standard_normal((4, 4)), rng.standard_normal((4, 4, 4))
    f = lambda a: B @ a + np.einsum("ijk,j,k->i", Q, a, a)
    a = rng.standard_normal(4)
    J_exact = B + np.einsum("ijk,k->ij", Q, a) + np.einsum("ijk,j->ik", Q, a)
    np.testing.assert_allclose(numerical_jacobian(f, a), J_exact, atol=1e-6)


def test_reduced_dt_rule_makes_the_stiff_reduced_diffusion_stable_where_the_fom_step_is_not():
    """Scalar model of the regression behind the historical rule: a reduced diffusion eigenvalue
    -nu*rho with dt at the FOM step is unstable (|R| > 1) while the revised step is stable."""
    K = -np.diag([1.0, 1732.0])  # rho(K) ~ the r=8 POD basis' value
    nu, dt_fom = 0.15, 0.0122
    assert abs(rk4_amplification(-dt_fom * nu * 1732.0)) > 1  # FOM step unstable
    assert abs(rk4_amplification(-revised_dt(np.float32(1.0), 128, nu, K) * nu * 1732.0)) < 1


# ------------------------------------------------------------------------- local-FD stencils
def test_fourth_order_stencil_converges_faster_on_a_smooth_field():
    errs = {2: [], 4: []}
    for Nx in (32, 64, 128):
        k, dealias = wavenumbers(Nx)
        x = np.linspace(0, L, Nx, endpoint=False)
        u = np.sin(x) + 0.3 * np.cos(2 * x)
        F = _burgers_rhs_physical(u, k, 0.05, dealias)
        pts = np.arange(3, Nx, 5)
        for order, fn, hw in ((2, deim_nonlinear_at_points, 1), (4, deim_nonlinear_at_points_4th_order, 2)):
            support, neigh = _stencil_support(pts, Nx, hw)
            Fh = fn(np.ones(1), u[support, None], support, neigh, 0.05, L / Nx)
            errs[order].append(np.linalg.norm(Fh - F[pts]) / np.linalg.norm(F[pts]))
    assert 3 < errs[2][0] / errs[2][1] < 5 and 3 < errs[2][1] / errs[2][2] < 5
    assert errs[4][0] / errs[4][1] > 10 and errs[4][1] / errs[4][2] > 10
    assert errs[4][2] < errs[2][2]


# ------------------------------------------------------- DEIM no-full-grid guarantee (new options)
def test_no_full_grid_touch_with_aligned_snapshots_and_min_steps():
    C = _small_ctx(Nx=64, m=12)
    support, _ = _stencil_support(C["pts"], 64, 1)
    Phi_p = C["Phi"].copy()
    mask = np.ones(64, dtype=bool)
    mask[support] = False
    Phi_p[mask] = np.nan
    u0 = 1.2 * np.sin(np.linspace(0, L, 64, endpoint=False))
    a0 = C["Phi"].T @ u0
    kw = dict(align_snapshots=True, min_steps=19)
    clean = deim_rom_reduced_trajectory(a0, np.abs(u0).max(), C["Phi"], C["M"], C["pts"], 64, 0.05, 1.0, 20,
                                        0.02, **kw)
    poisoned = deim_rom_reduced_trajectory(a0, np.abs(u0).max(), Phi_p, C["M"], C["pts"], 64, 0.05, 1.0, 20,
                                           0.02, **kw)
    assert np.all(np.isfinite(poisoned[1]))
    np.testing.assert_array_equal(clean[1], poisoned[1])
