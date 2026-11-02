"""Energy-conserving correction and quadratic folding (src/energy_conserving.py)."""
import numpy as np
import pytest

from scripts.hyperreduction_benchmark import wavenumbers
from src.deim import (build_deim_basis, build_deim_projector, collect_nonlinear_snapshots,
                       deim_rom_reduced_trajectory, select_deim_points)
from src.energy_conserving import (energy_conserving_tensor, energy_production,
                                   energy_production_ratio, fold_hybrid_localfd, max_symmetric_eig,
                                   quadratic_model, symmetrized_part,
                                   weighted_energy_conserving_tensor)
from src.pod_rom import build_pod_basis
from src.solver import solve_burgers
from src.tensorial_pod import build_tensorial_rom, tensorial_reduced_trajectory, tensorial_rhs
from src.timestep import reduced_diffusion_matrix_deim

L = 2 * np.pi


def _ctx(r=8, m=12, Nx=64):
    k, dealias = wavenumbers(Nx)
    rng = np.random.default_rng(0)
    ens, nus = [], []
    for _ in range(10):
        A, nu = rng.uniform(0.5, 2.0), rng.uniform(0.01, 0.1)
        u = solve_burgers(A=A, nu=nu, Nx=Nx, T=1.0, n_save=10)[2].astype(np.float64)
        ens.append(u)
        nus.append(np.full(u.shape[0], nu))
    ens, nus = np.concatenate(ens), np.concatenate(nus)
    Phi, _ = build_pod_basis(ens, r)
    Psi, _ = build_deim_basis(collect_nonlinear_snapshots(ens, k, nus, dealias), m)
    pts = select_deim_points(Psi)
    return Phi, build_deim_projector(Phi, Psi, pts), pts, k, dealias, Nx


def _random_tensor(r=6, seed=0):
    Q = np.random.default_rng(seed).standard_normal((r, r, r))
    return 0.5 * (Q + Q.transpose(0, 2, 1))


def test_energy_conserving_tensor_has_zero_cubic_form():
    Q = _random_tensor()
    Qe = energy_conserving_tensor(Q)
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.standard_normal(6)
        assert abs(energy_production(Qe, a)) <= 1e-12 * np.linalg.norm(a) ** 3 * np.abs(Q).max()
    assert abs(energy_production(Q, a)) > 1e-3  # the uncorrected tensor does produce energy


def test_correction_keeps_jk_symmetry_is_idempotent_and_orthogonal():
    Q = _random_tensor()
    Qe = energy_conserving_tensor(Q)
    np.testing.assert_allclose(Qe, Qe.transpose(0, 2, 1), atol=1e-14)
    np.testing.assert_allclose(energy_conserving_tensor(Qe), Qe, atol=1e-14)
    np.testing.assert_allclose(symmetrized_part(Qe), 0.0, atol=1e-14)
    assert abs(np.sum(symmetrized_part(Q) * Qe)) <= 1e-12 * np.sum(Q * Q)  # orthogonal projection


def test_correction_leaves_an_already_conservative_tensor_unchanged():
    Qe = energy_conserving_tensor(_random_tensor(seed=2))
    np.testing.assert_allclose(energy_conserving_tensor(Qe), Qe, atol=1e-14)
    assert energy_production_ratio(Qe) < 1e-14


def test_weighted_correction_is_energy_conserving_and_jk_symmetric():
    Q = _random_tensor()
    sigma = np.array([10.0, 2.0, 0.5, 0.2, 0.08, 0.03])
    Qw = weighted_energy_conserving_tensor(Q, sigma)
    np.testing.assert_allclose(symmetrized_part(Qw), 0.0, atol=1e-12)
    np.testing.assert_allclose(Qw, Qw.transpose(0, 2, 1), atol=1e-14)


def test_weighted_correction_with_uniform_weights_is_the_frobenius_projection():
    Q = _random_tensor()
    np.testing.assert_allclose(weighted_energy_conserving_tensor(Q, np.ones(6)),
                               energy_conserving_tensor(Q), atol=1e-13)


def test_weighted_correction_minimises_the_weighted_change():
    """Any other energy-conserving, (j,k)-symmetric tensor is at least as far from Q in the weighted
    norm (the feasible set is a linear subspace, so compare against feasible perturbations)."""
    Q = _random_tensor()
    sigma = np.array([10.0, 2.0, 0.5, 0.2, 0.08, 0.03])
    W = (sigma[None, :, None] * sigma[None, None, :]) ** 2 * np.ones((6, 1, 1))
    Qw = weighted_energy_conserving_tensor(Q, sigma)
    best = np.sum(W * (Qw - Q) ** 2)
    rng = np.random.default_rng(8)
    for _ in range(20):
        D = energy_conserving_tensor(_random_tensor(seed=int(rng.integers(1e6))))  # feasible direction
        other = Qw + 1e-2 * D
        assert np.abs(symmetrized_part(other)).max() < 1e-12
        assert np.sum(W * (other - Q) ** 2) >= best - 1e-12


def test_energy_nonincreasing_along_energy_conserving_trajectory():
    """With Sym(Q) = 0 and K symmetric negative semidefinite, |a|^2 cannot grow (continuous time);
    RK4 with a stable step must reproduce that to its truncation error."""
    Phi, _, _, k, dealias, Nx = _ctx()
    tens = build_tensorial_rom(Phi, k, dealias)
    assert max_symmetric_eig(tens.L) <= 1e-9 * np.abs(tens.L).max()
    model = quadratic_model(energy_conserving_tensor(tens.Q), tens.L, Nx)
    a0 = Phi.T @ (1.8 * np.sin(np.linspace(0, L, Nx, endpoint=False)))
    _, a, _ = tensorial_reduced_trajectory(a0, 1.8, model, 0.01, 1.0, 60, align_snapshots=True)
    E = 0.5 * np.sum(a**2, axis=1)
    assert np.all(np.diff(E) <= 1e-9 * E[0])


def test_exact_galerkin_tensor_energy_production_is_small_but_nonzero():
    """The dealiased pseudo-spectral product is not exactly energy conserving (aliasing), so the
    exact Galerkin Q has a small symmetric part; quantify rather than assume it is zero."""
    Phi, _, _, k, dealias, _ = _ctx()
    ratio = energy_production_ratio(build_tensorial_rom(Phi, k, dealias).Q)
    assert 0.0 < ratio < 0.1


@pytest.mark.parametrize("order", [2, 4])
def test_folded_hybrid_reproduces_src_deim_trajectory(order):
    Phi, M, pts, k, dealias, Nx = _ctx()  # order 4 uses the same M with a 5-point stencil
    Q, K = fold_hybrid_localfd(Phi, M, pts, Nx, stencil_order=order)
    np.testing.assert_allclose(K, reduced_diffusion_matrix_deim(Phi, M, pts, Nx, stencil_order=order),
                               atol=1e-10 * np.abs(K).max())
    model = quadratic_model(Q, K, Nx)
    u0 = 1.2 * np.sin(np.linspace(0, L, Nx, endpoint=False))
    a0 = Phi.T @ u0
    _, a_ref, n_ref = deim_rom_reduced_trajectory(a0, 1.2, Phi, M, pts, Nx, 0.04, 1.0, 20,
                                                  stencil_order=order, align_snapshots=True)
    _, a_f, n_f = tensorial_reduced_trajectory(a0, 1.2, model, 0.04, 1.0, 20, align_snapshots=True)
    assert n_ref == n_f
    assert np.abs(a_ref - a_f).max() <= 1e-10 * np.abs(a_ref).max()


def test_energy_production_matches_rhs_inner_product():
    Phi, _, _, k, dealias, _ = _ctx()
    tens = build_tensorial_rom(Phi, k, dealias)
    a = np.random.default_rng(3).standard_normal(8)
    lhs = a @ tensorial_rhs(tens, a, 0.0)
    assert abs(lhs - energy_production(tens.Q, a)) <= 1e-12 * max(1.0, abs(lhs))
