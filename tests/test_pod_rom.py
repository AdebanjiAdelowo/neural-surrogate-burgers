import numpy as np

from src.pod_rom import build_pod_basis, energy_captured, rom_predict
from src.solver import solve_burgers


def _setup(r=8):
    Nx = 64
    L = 2 * np.pi
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    dealias = np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k))
    rng = np.random.default_rng(0)
    ensemble = []
    for _ in range(10):
        A, nu = rng.uniform(0.5, 2.0), rng.uniform(0.01, 0.1)
        _, _, u = solve_burgers(A=A, nu=nu, Nx=Nx, T=1.0, n_save=10)
        ensemble.append(u)
    ensemble = np.concatenate(ensemble, axis=0)
    Phi, S = build_pod_basis(ensemble, r)
    return Phi, S, k, dealias, Nx


def test_basis_orthonormal():
    # atol=1e-6, not 1e-8: the snapshot ensemble is float32 (src.solver's output dtype), so
    # orthonormality from SVD only holds to float32 precision (~1e-7), not float64 machine
    # epsilon -- this is expected, not a bug in build_pod_basis.
    Phi, _, _, _, _ = _setup(r=8)
    gram = Phi.T @ Phi
    np.testing.assert_allclose(gram, np.eye(8), atol=1e-6)


def test_energy_captured_is_increasing_and_bounded():
    _, S, _, _, _ = _setup(r=8)
    vals = [energy_captured(S, r) for r in [1, 2, 4, 8]]
    assert all(0 <= v <= 1.0 + 1e-9 for v in vals)
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def test_rom_prediction_shape_and_finiteness():
    Phi, S, k, dealias, Nx = _setup(r=8)
    _, _, u_gt = solve_burgers(A=1.0, nu=0.05, Nx=Nx, T=1.0, n_save=10)
    t, u_pred = rom_predict(u_gt[0], Phi, k, 0.05, dealias, T=1.0, n_save=10)
    assert u_pred.shape == u_gt.shape
    assert np.isfinite(u_pred).all()


def test_rom_stable_at_higher_viscosity_extrapolation():
    # Regression test for a real bug found during Stage 5 evaluation: rom_predict's default dt
    # only respected an advective stability bound, not the diffusion stability bound needed by
    # explicit RK4 in reduced coordinates (src.solver avoids this via an integrating factor,
    # rom_predict does not). This produced NaN at nu=0.15 (outside the [0.01,0.1] training range).
    # Fixed by also bounding dt by the standard explicit-diffusion limit dx^2/(2*nu).
    Nx = 128
    L = 2 * np.pi
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    dealias = np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k))
    rng = np.random.default_rng(0)
    ensemble = []
    for _ in range(10):
        A, nu = rng.uniform(0.5, 2.0), rng.uniform(0.01, 0.1)
        _, _, u = solve_burgers(A=A, nu=nu, Nx=Nx, T=1.0, n_save=10)
        ensemble.append(u)
    Phi, _ = build_pod_basis(np.concatenate(ensemble, axis=0), r=8)

    _, _, gt = solve_burgers(A=1.0, nu=0.15, Nx=Nx, T=1.0, n_save=10)  # nu above training range
    _, u_pred = rom_predict(gt[0], Phi, k, 0.15, dealias, T=1.0, n_save=10)
    assert np.isfinite(u_pred).all()


def test_more_modes_reduces_prediction_error():
    # The central real behaviour this baseline exists to demonstrate: accuracy improves as
    # retained modes increase.
    Nx = 64
    L = 2 * np.pi
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    dealias = np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k))
    rng = np.random.default_rng(1)
    ensemble = []
    for _ in range(15):
        A, nu = rng.uniform(0.5, 2.0), rng.uniform(0.01, 0.1)
        _, _, u = solve_burgers(A=A, nu=nu, Nx=Nx, T=1.0, n_save=10)
        ensemble.append(u)
    ensemble = np.concatenate(ensemble, axis=0)
    Phi_full, _ = build_pod_basis(ensemble, r=16)

    _, _, u_gt = solve_burgers(A=1.2, nu=0.03, Nx=Nx, T=1.0, n_save=10)

    errors = []
    for r in [2, 4, 8, 16]:
        _, u_pred = rom_predict(u_gt[0], Phi_full[:, :r], k, 0.03, dealias, T=1.0, n_save=10)
        errors.append(np.linalg.norm(u_pred - u_gt) / np.linalg.norm(u_gt))
    assert all(errors[i] >= errors[i + 1] - 1e-6 for i in range(len(errors) - 1))
