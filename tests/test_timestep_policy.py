"""The canonical reduced-model time-step policy ("reduced") and the explicit historical one."""
import inspect

import numpy as np
import pytest

from scripts.hyperreduction_benchmark import native_reduced_steps, wavenumbers
from scripts.hyperreduction_followup import make_ctx
from src.deim import (build_deim_basis, build_deim_projector, collect_nonlinear_snapshots,
                       deim_rom_predict, deim_rom_reduced_trajectory, select_deim_points)
from src.pod_rom import build_pod_basis, rom_predict, rom_reduced_trajectory
from src.solver import solve_burgers
from src.timestep import (DEFAULT_POLICY, POLICIES, RK4_REAL_LIMIT, check_policy, fom_advective_dt,
                           historical_dt, reduced_dt, spectral_radius)

L = 2 * np.pi
FUNCS = (rom_predict, deim_rom_predict, rom_reduced_trajectory, deim_rom_reduced_trajectory)


def _ctx(Nx=64, r=8, m=12):
    k, dealias = wavenumbers(Nx)
    rng = np.random.default_rng(0)
    ens, nus = [], []
    for _ in range(12):
        u = solve_burgers(A=rng.uniform(0.5, 2.0), nu=rng.uniform(0.01, 0.1), Nx=Nx, T=1.0,
                          n_save=10)[2].astype(np.float64)
        ens.append(u)
        nus.append(np.full(u.shape[0], 0.05))
    ens, nus = np.concatenate(ens), np.concatenate(nus)
    Phi, _ = build_pod_basis(ens, r)
    Psi, _ = build_deim_basis(collect_nonlinear_snapshots(ens, k, nus, dealias), m)
    pts = select_deim_points(Psi)
    return make_ctx(Phi, build_deim_projector(Phi, Psi, pts), pts, Nx)


def _traj(kind, C, u0, nu, T=1.0, **kw):
    a0, u0max = C["Phi"].T @ u0, np.abs(u0).max()
    if kind == "rom":
        return rom_reduced_trajectory(a0, u0max, C["Phi"], C["k"], nu, C["dealias"], T, 20, **kw)
    return deim_rom_reduced_trajectory(a0, u0max, C["Phi"], C["M"], C["pts"], C["Nx"], nu, T, 20, **kw)


def test_default_policy_is_the_reduced_operator_policy_everywhere():
    assert DEFAULT_POLICY == "reduced" and POLICIES == ("reduced", "historical")
    for f in FUNCS:
        assert inspect.signature(f).parameters["timestep_policy"].default == "reduced"


@pytest.mark.parametrize("kind", ["rom", "deim"])
def test_default_uses_the_spectral_radius_of_the_reduced_diffusion_operator(kind):
    C = _ctx()
    u0 = (1.2 * np.sin(np.linspace(0, L, 64, endpoint=False))).astype(np.float32)
    nu, T = 5.0, 5.0  # large nu and long horizon: the diffusion bound must bind and exceed the floor
    rho = spectral_radius(C["K"][kind])
    dt_expected = reduced_dt(np.abs(u0).max(), 64, nu, rho)
    assert dt_expected < fom_advective_dt(np.abs(u0).max(), 64)  # diffusion binds
    n_expected = max(80, int(np.ceil(T / dt_expected)))
    assert n_expected > 80
    with np.errstate(all="ignore"):
        n_default = _traj(kind, C, u0, nu, T)[2]
        n_override = _traj(kind, C, u0, nu, T, diffusion_radius=rho)[2]
        n_stiffer = _traj(kind, C, u0, nu, T, diffusion_radius=10 * rho)[2]
    assert n_default == n_expected == n_override
    assert n_stiffer > n_default  # the step responds to the reduced spectral radius


@pytest.mark.parametrize("nu,A", [(0.01, 2.0), (0.09, 0.6), (0.04, 1.2)])
def test_historical_policy_reproduces_the_legacy_step_counts(nu, A):
    C = _ctx()
    u0 = (A * np.sin(np.linspace(0, L, 64, endpoint=False))).astype(np.float32)
    expected = native_reduced_steps(np.abs(u0).max(), 64, nu)
    for kind in ("rom", "deim"):
        with np.errstate(all="ignore"):  # the small test basis may overflow; only the step count matters
            assert _traj(kind, C, u0, nu, timestep_policy="historical")[2] == expected
    # ... and the historical dt is the full-grid diffusion rule
    assert historical_dt(np.abs(u0).max(), 64, nu) == pytest.approx(
        min(0.25 * (L / 64) / A, 0.4 * (L / 64) ** 2 / (2 * nu)), rel=1e-6)


def test_reduced_policy_never_takes_more_steps_than_needed_at_high_resolution():
    """The historical rule's step count grows ~Nx^2 through dx; the reduced one is bounded by the
    FOM's step count (advective) for a fixed basis-independent diffusion radius."""
    nu, u0max = 0.098, np.float32(0.86)
    rho = 1732.0
    hist = [historical_dt(u0max, Nx, nu) for Nx in (128, 512, 2048)]
    red = [reduced_dt(u0max, Nx, nu, rho) for Nx in (128, 512, 2048)]
    assert hist[0] / hist[2] > 200 > red[0] / red[2]  # ~256 (Nx^2) against ~16 (Nx)


def test_nu_015_regression_case():
    """The case behind the historical rule: with a small-basis ROM at nu = 0.15 the FOM's own step
    count is unstable, while both named policies are stable."""
    rng = np.random.default_rng(0)
    ens = [solve_burgers(A=rng.uniform(0.5, 2.0), nu=rng.uniform(0.01, 0.1), Nx=128, T=1.0, n_save=10)[2]
           for _ in range(10)]
    Phi, _ = build_pod_basis(np.concatenate(ens, axis=0), r=8)
    k, dealias = wavenumbers(128)
    gt = solve_burgers(A=1.0, nu=0.15, Nx=128, T=1.0, n_save=10)[2]
    u0 = gt[0]
    kw = dict(T=1.0, n_save=10)
    with np.errstate(all="ignore"):
        fom_step = rom_predict(u0, Phi, k, 0.15, dealias, dt=fom_advective_dt(np.abs(u0).max(), 128), **kw)[1]
        default = rom_predict(u0, Phi, k, 0.15, dealias, **kw)[1]
        hist = rom_predict(u0, Phi, k, 0.15, dealias, timestep_policy="historical", **kw)[1]
    assert not np.all(np.isfinite(fom_step))  # what the historical rule was protecting against
    for u in (default, hist):
        assert np.all(np.isfinite(u)) and np.abs(u).max() < 1.5
    assert np.linalg.norm(default - gt) / np.linalg.norm(gt) < 5e-3


@pytest.mark.parametrize("f", FUNCS)
def test_invalid_policy_name_fails_clearly(f):
    C = _ctx()
    u0 = (1.0 * np.sin(np.linspace(0, L, 64, endpoint=False))).astype(np.float32)
    with pytest.raises(ValueError, match="unknown timestep_policy 'fast'.*reduced.*historical"):
        if f is rom_predict:
            f(u0, C["Phi"], C["k"], 0.05, C["dealias"], 1.0, 10, timestep_policy="fast")
        elif f is deim_rom_predict:
            f(u0, C["Phi"], C["M"], C["pts"], 64, 0.05, 1.0, 10, timestep_policy="fast")
        elif f is rom_reduced_trajectory:
            f(C["Phi"].T @ u0, 1.0, C["Phi"], C["k"], 0.05, C["dealias"], 1.0, 10, timestep_policy="fast")
        else:
            f(C["Phi"].T @ u0, 1.0, C["Phi"], C["M"], C["pts"], 64, 0.05, 1.0, 10, timestep_policy="fast")
    with pytest.raises(ValueError):
        check_policy("Historical")  # names are exact


def test_explicit_dt_overrides_the_policy():
    C = _ctx()
    u0 = (1.0 * np.sin(np.linspace(0, L, 64, endpoint=False))).astype(np.float32)
    n = [_traj("rom", C, u0, 0.05, dt=0.01, timestep_policy=p)[2] for p in POLICIES]
    assert n[0] == n[1] == 100


def test_reduced_policy_reads_only_stencil_rows_for_deim():
    """Step selection for DEIM (spectral radius of the local-FD diffusion operator) must not need the
    full grid either: poison every Phi row outside the stencil support."""
    from src.deim import _stencil_support
    C = _ctx(Nx=128, m=12)
    support, _ = _stencil_support(C["pts"], 128, 1)
    Phi_p = C["Phi"].copy()
    mask = np.ones(128, dtype=bool)
    mask[support] = False
    Phi_p[mask] = np.nan
    u0 = 1.3 * np.sin(np.linspace(0, L, 128, endpoint=False))
    a0 = C["Phi"].T @ u0
    clean = deim_rom_reduced_trajectory(a0, np.abs(u0).max(), C["Phi"], C["M"], C["pts"], 128, 0.05, 1.0, 20)
    pois = deim_rom_reduced_trajectory(a0, np.abs(u0).max(), Phi_p, C["M"], C["pts"], 128, 0.05, 1.0, 20)
    assert clean[2] == pois[2]
    np.testing.assert_array_equal(clean[1], pois[1])
