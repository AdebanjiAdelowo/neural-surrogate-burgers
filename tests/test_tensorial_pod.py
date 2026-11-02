"""Tensorial POD-Galerkin (src/tensorial_pod.py): equivalence with src/pod_rom.py.

The tensorial model is an exact re-association of the plain ROM's discrete reduced vector field, so
every comparison here is held to floating-point rounding, not to a modelling tolerance.
"""
import numpy as np
import pytest

from scripts.hyperreduction_benchmark import wavenumbers
from src.pod_rom import _burgers_rhs_physical, build_pod_basis, rom_predict, rom_reduced_trajectory
from src.solver import solve_burgers
from src.tensorial_pod import (apply_nonlinear_operator, build_quadratic_tensor,
                               build_tensorial_rom, quadratic_jacobian, reduced_step_count,
                               tensorial_predict, tensorial_reduced_trajectory, tensorial_rhs)
from src.timestep import numerical_jacobian

L = 2 * np.pi


def _basis(r=8, Nx=64, n_examples=10, seed=0):
    k, dealias = wavenumbers(Nx)
    rng = np.random.default_rng(seed)
    ens = []
    for _ in range(n_examples):
        A, nu = rng.uniform(0.5, 2.0), rng.uniform(0.01, 0.1)
        ens.append(solve_burgers(A=A, nu=nu, Nx=Nx, T=1.0, n_save=10)[2].astype(np.float64))
    Phi, _ = build_pod_basis(np.concatenate(ens), r)
    return Phi, k, dealias


def _original_rhs(Phi, k, dealias, a, nu):
    return Phi.T @ _burgers_rhs_physical(Phi @ a, k, nu, dealias)


# ------------------------------------------------------------------ operator and tensor
def test_nonlinear_operator_matches_solver_convention():
    """G(u*u) equals the nonlinear part of src.pod_rom._burgers_rhs_physical (nu = 0)."""
    Nx = 64
    k, dealias = wavenumbers(Nx)
    u = np.random.default_rng(1).standard_normal(Nx)
    np.testing.assert_allclose(apply_nonlinear_operator(u * u, k, dealias),
                               _burgers_rhs_physical(u, k, 0.0, dealias), rtol=0, atol=1e-13)


def test_nonlinear_operator_on_stack_equals_columnwise():
    Nx = 32
    k, dealias = wavenumbers(Nx)
    W = np.random.default_rng(2).standard_normal((Nx, 3, 4))
    G = apply_nonlinear_operator(W, k, dealias)
    for j in range(3):
        for c in range(4):
            np.testing.assert_allclose(G[:, j, c], apply_nonlinear_operator(W[:, j, c], k, dealias),
                                       atol=1e-14)


def test_quadratic_tensor_is_symmetric_in_last_two_indices():
    Phi, k, dealias = _basis()
    Q = build_quadratic_tensor(Phi, k, dealias)
    assert Q.shape == (8, 8, 8)
    np.testing.assert_array_equal(Q, Q.transpose(0, 2, 1))


def test_quadratic_tensor_entries_match_direct_definition():
    Phi, k, dealias = _basis(r=4)
    Q = build_quadratic_tensor(Phi, k, dealias)
    for i, j, kk in [(0, 0, 0), (1, 2, 3), (3, 1, 0), (2, 2, 1)]:
        direct = Phi[:, i] @ apply_nonlinear_operator(Phi[:, j] * Phi[:, kk], k, dealias)
        assert abs(Q[i, j, kk] - direct) <= 1e-13 * max(1.0, abs(direct))


def test_diffusion_matrix_is_symmetric_negative_semidefinite():
    Phi, k, dealias = _basis()
    Lr = build_tensorial_rom(Phi, k, dealias).L
    np.testing.assert_allclose(Lr, Lr.T, atol=1e-10 * np.abs(Lr).max())
    assert np.linalg.eigvalsh(0.5 * (Lr + Lr.T)).max() <= 1e-10 * np.abs(Lr).max()


# ------------------------------------------------------------------ RHS equivalence
@pytest.mark.parametrize("r", [2, 4, 8, 12])
def test_tensorial_rhs_equals_original_galerkin_rhs(r):
    Phi, k, dealias = _basis(r=r)
    model = build_tensorial_rom(Phi, k, dealias)
    rng = np.random.default_rng(r)
    for _ in range(25):
        a = rng.standard_normal(r) * rng.uniform(0.1, 5.0)
        nu = rng.uniform(1e-3, 0.3)
        ref = _original_rhs(Phi, k, dealias, a, nu)
        got = tensorial_rhs(model, a, nu)
        assert np.linalg.norm(got - ref) <= 1e-12 * np.linalg.norm(ref)


def test_tensorial_rhs_equivalence_on_odd_grid():
    """Odd Nx has no Nyquist mode; the derivation must not depend on even Nx."""
    Phi, k, dealias = _basis(r=6, Nx=63)
    model = build_tensorial_rom(Phi, k, dealias)
    a = np.random.default_rng(3).standard_normal(6)
    ref = _original_rhs(Phi, k, dealias, a, 0.04)
    assert np.linalg.norm(tensorial_rhs(model, a, 0.04) - ref) <= 1e-12 * np.linalg.norm(ref)


def test_analytic_jacobian_matches_finite_differences():
    Phi, k, dealias = _basis()
    model = build_tensorial_rom(Phi, k, dealias)
    a = np.random.default_rng(4).standard_normal(8)
    J = quadratic_jacobian(model.Q, 0.03 * model.L, a)
    Jfd = numerical_jacobian(lambda b: _original_rhs(Phi, k, dealias, b, 0.03), a)
    np.testing.assert_allclose(J, Jfd, rtol=0, atol=1e-6 * np.abs(J).max())


def test_online_rhs_performs_no_fft(monkeypatch):
    Phi, k, dealias = _basis()
    model = build_tensorial_rom(Phi, k, dealias)

    def boom(*a, **kw):
        raise AssertionError("FFT called in the tensorial online stage")

    for name in ("fft", "ifft", "rfft", "irfft"):
        monkeypatch.setattr(np.fft, name, boom)
    a0 = np.random.default_rng(5).standard_normal(8)
    _, a, _ = tensorial_reduced_trajectory(a0, 1.0, model, 0.05, 0.2, 5)
    assert np.all(np.isfinite(a))


# ------------------------------------------------------------------ trajectory equivalence
@pytest.mark.parametrize("A,nu", [(0.6, 0.09), (1.9, 0.012), (1.2, 0.04)])
@pytest.mark.parametrize("policy", ["reduced", "historical"])
def test_trajectory_equals_original_rom(A, nu, policy):
    """Same step count, same saved times, states equal to rounding (a0 in float64 for both, see
    test_float32_initial_state_explains_larger_difference)."""
    Phi, k, dealias = _basis()
    Nx = Phi.shape[0]
    model = build_tensorial_rom(Phi, k, dealias)
    u0 = A * np.sin(2 * np.pi * np.linspace(0, L, Nx, endpoint=False) / L)
    a0 = Phi.T @ u0
    t_o, a_o, n_o = rom_reduced_trajectory(a0, np.abs(u0).max(), Phi, k, nu, dealias, 1.0, 20,
                                           align_snapshots=True, timestep_policy=policy)
    t_t, a_t, n_t = tensorial_reduced_trajectory(a0, np.abs(u0).max(), model, nu, 1.0, 20,
                                                 align_snapshots=True, timestep_policy=policy)
    assert n_o == n_t
    np.testing.assert_array_equal(t_o, t_t)
    assert np.abs(a_o - a_t).max() <= 1e-12 * np.abs(a_o).max()


def test_float32_initial_state_explains_larger_difference():
    """With float32 Phi and float32 u0 (the stored basis and data files), a0 = Phi^T u0 is float32
    and the ORIGINAL ROM evaluates its very first RK4 stage Phi @ a0 in float32. That, not the
    tensor, is the source of the ~1e-9 trajectory differences seen with stored data: promoting a0 to
    float64 makes both models agree to rounding again."""
    Phi64, k, dealias = _basis()
    Phi = Phi64.astype(np.float32)
    Nx = Phi.shape[0]
    model = build_tensorial_rom(Phi, k, dealias)
    u0 = (1.5 * np.sin(np.linspace(0, L, Nx, endpoint=False))).astype(np.float32)
    a0 = Phi.T @ u0
    assert a0.dtype == np.float32
    args = (np.abs(u0).max(),)
    _, a_o32, _ = rom_reduced_trajectory(a0, *args, Phi, k, 0.03, dealias, 1.0, 20)
    _, a_o64, _ = rom_reduced_trajectory(a0.astype(np.float64), *args, Phi, k, 0.03, dealias, 1.0, 20)
    _, a_t, _ = tensorial_reduced_trajectory(a0, *args, model, 0.03, 1.0, 20)
    scale = np.abs(a_o64).max()
    assert np.abs(a_o64 - a_t).max() <= 1e-12 * scale
    assert np.abs(a_o32 - a_o64).max() > 1e3 * np.abs(a_o64 - a_t).max()  # float32 stage is visible
    assert np.abs(a_o32 - a_o64).max() <= 1e-6 * scale                   # ... but tiny


def test_diffusion_matrix_is_float64_accurate_for_a_float32_basis():
    """Regression: NumPy >= 2 transforms float32 input in single precision, which once made L from
    the stored float32 basis accurate to only ~1e-8. L must not depend on the basis dtype."""
    Phi64, k, dealias = _basis()
    Phi32 = Phi64.astype(np.float32)
    L32 = build_tensorial_rom(Phi32, k, dealias).L
    L_ref = build_tensorial_rom(Phi32.astype(np.float64), k, dealias).L
    assert np.abs(L32 - L_ref).max() <= 1e-13 * np.abs(L_ref).max()


def test_predict_equals_original_rom_predict_to_float32_output():
    Phi, k, dealias = _basis()
    Nx = Phi.shape[0]
    model = build_tensorial_rom(Phi, k, dealias)
    u0 = 1.3 * np.sin(np.linspace(0, L, Nx, endpoint=False))
    t_o, u_o = rom_predict(u0, Phi, k, 0.02, dealias, T=1.0, n_save=20, align_snapshots=True)
    t_t, u_t = tensorial_predict(u0, Phi, model, 0.02, T=1.0, n_save=20, align_snapshots=True)
    np.testing.assert_array_equal(t_o, t_t)
    assert u_t.dtype == np.float32 and u_t.shape == u_o.shape
    np.testing.assert_allclose(u_t, u_o, rtol=0, atol=2e-6 * np.abs(u_o).max())


def test_step_count_rule_matches_original():
    Phi, k, dealias = _basis()
    model = build_tensorial_rom(Phi, k, dealias)
    for policy in ("reduced", "historical"):
        for nu in (0.01, 0.05, 0.15):
            u0max = 1.7
            n, _ = reduced_step_count(u0max, model.Nx, nu, 1.0, 20, align_snapshots=True,
                                      timestep_policy=policy,
                                      diffusion_radius=model.diffusion_radius)
            _, _, n_o = rom_reduced_trajectory(np.zeros(8), u0max, Phi, k, nu, dealias, 1.0, 20,
                                               align_snapshots=True, timestep_policy=policy)
            assert n == n_o and n % 19 == 0


def test_reduced_mass_is_conserved_by_the_tensorial_dynamics():
    """Periodic Burgers conserves mass. The mass of u = Phi a is c^T a with c = Phi^T 1 dx; since
    1^T G = 0 and 1^T D2 = 0 (zero-wavenumber symbols), c^T F_r(a) = 0 holds whenever 1 lies in
    span(Phi). With a POD basis it only holds to the extent 1 is represented; check the identity
    on a basis that contains the constant mode."""
    Phi, k, dealias = _basis(r=7)
    Nx = Phi.shape[0]
    ones = np.ones((Nx, 1)) / np.sqrt(Nx)
    B, _ = np.linalg.qr(np.hstack([ones, Phi]))
    model = build_tensorial_rom(B, k, dealias)
    c = B.T @ np.ones(Nx)
    a = np.random.default_rng(6).standard_normal(8)
    assert abs(c @ tensorial_rhs(model, a, 0.05)) <= 1e-11 * np.linalg.norm(a) ** 2


def test_tensorial_trajectory_is_bit_reproducible():
    Phi, k, dealias = _basis()
    model = build_tensorial_rom(Phi, k, dealias)
    a0 = np.random.default_rng(7).standard_normal(8)
    r1 = tensorial_reduced_trajectory(a0, 1.0, model, 0.02, 1.0, 20)[1]
    r2 = tensorial_reduced_trajectory(a0, 1.0, model, 0.02, 1.0, 20)[1]
    np.testing.assert_array_equal(r1, r2)
