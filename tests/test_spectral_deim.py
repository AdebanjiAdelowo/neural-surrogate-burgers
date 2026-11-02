"""Spectral-consistent DEIM (src/spectral_deim.py): sampled-operator equivalence, interpolation,
point selection and the two exactness limits that tie it to the plain Galerkin ROM."""
import numpy as np
import pytest

from scripts.hyperreduction_benchmark import wavenumbers
from src.deim import collect_nonlinear_snapshots
from src.pod_rom import _burgers_rhs_physical, build_pod_basis, rom_reduced_trajectory
from src.solver import solve_burgers
from src.spectral_deim import (build_basis, build_spectral_deim, folded_tensor, greedy_oversample,
                               interpolation_conditioning, interpolation_pinv, nonlinear_snapshots,
                               qdeim_points, sampled_diffusion, sampled_nonlinear,
                               sampled_quadratic_tensor, select_points, spectral_deim_predict,
                               spectral_deim_reduced_trajectory, spectral_deim_rhs)
from src.tensorial_pod import build_tensorial_rom, mode_products_image, tensorial_rhs

L = 2 * np.pi


def _ens(Nx=64, n_examples=10, seed=0):
    rng = np.random.default_rng(seed)
    ens, nus = [], []
    for _ in range(n_examples):
        A, nu = rng.uniform(0.5, 2.0), rng.uniform(0.01, 0.1)
        u = solve_burgers(A=A, nu=nu, Nx=Nx, T=1.0, n_save=10)[2].astype(np.float64)
        ens.append(u)
        nus.append(np.full(u.shape[0], nu))
    return np.concatenate(ens), np.concatenate(nus)


def _ctx(r=8, m=12, Nx=64, target="nonlinear"):
    k, dealias = wavenumbers(Nx)
    ens, nus = _ens(Nx)
    Phi, _ = build_pod_basis(ens, r)
    snaps = (nonlinear_snapshots(ens, k, dealias) if target == "nonlinear"
             else collect_nonlinear_snapshots(ens, k, nus, dealias))
    Psi, _ = build_basis(snaps, m)
    return Phi, Psi, k, dealias


# ------------------------------------------------------------------ sampled operator equivalence
@pytest.mark.parametrize("Nx,r", [(64, 4), (64, 8), (63, 6), (128, 12)])
def test_sampled_tensor_reproduces_sampled_spectral_nonlinear_term(Nx, r):
    k, dealias = wavenumbers(Nx)
    ens, _ = _ens(Nx, n_examples=6)
    Phi, _ = build_pod_basis(ens, r)
    rng = np.random.default_rng(Nx + r)
    pts = np.sort(rng.choice(Nx, size=min(Nx, 3 * r), replace=False))
    pts = np.unique(np.concatenate([pts, [0, Nx - 1]]))  # include both ends of the periodic grid
    C = sampled_quadratic_tensor(Phi, pts, k, dealias)
    for _ in range(20):
        a = rng.standard_normal(r) * rng.uniform(0.1, 5.0)
        ref = _burgers_rhs_physical(Phi @ a, k, 0.0, dealias)[pts]
        got = (C.reshape(len(pts) * r, r) @ a).reshape(len(pts), r) @ a
        assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max()


def test_sampled_full_rhs_reproduces_sampled_spectral_rhs():
    Nx, r = 64, 8
    k, dealias = wavenumbers(Nx)
    Phi, _, _, _ = _ctx(r=r, Nx=Nx)
    pts = np.arange(0, Nx, 5)
    C, Ds = sampled_quadratic_tensor(Phi, pts, k, dealias), sampled_diffusion(Phi, pts, k)
    rng = np.random.default_rng(1)
    for nu in (0.005, 0.05, 0.2):
        a = rng.standard_normal(r)
        ref = _burgers_rhs_physical(Phi @ a, k, nu, dealias)[pts]
        got = (C.reshape(-1, r) @ a).reshape(len(pts), r) @ a + nu * Ds @ a
        assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max()


def test_sampled_tensor_rows_are_rows_of_full_image():
    Phi, _, k, dealias = _ctx()
    pts = np.array([3, 0, 63, 17])
    np.testing.assert_array_equal(sampled_quadratic_tensor(Phi, pts, k, dealias),
                                  mode_products_image(Phi, k, dealias)[pts])


def test_model_sampled_nonlinear_matches_full_order_at_selected_points():
    Phi, Psi, k, dealias = _ctx()
    model = build_spectral_deim(Phi, Psi, select_points(Psi, method="qdeim"), k, dealias)
    a = np.random.default_rng(2).standard_normal(8)
    ref = _burgers_rhs_physical(Phi @ a, k, 0.0, dealias)[model.points]
    assert np.abs(sampled_nonlinear(model, a) - ref).max() <= 1e-12 * np.abs(ref).max()


# ------------------------------------------------------------------ interpolation
@pytest.mark.parametrize("s_factor", [1.0, 1.5, 2.0])
def test_interpolation_is_exact_for_vectors_in_span_of_psi(s_factor):
    _, Psi, _, _ = _ctx(m=10)
    s = int(np.ceil(s_factor * 10))
    pts = select_points(Psi, s=s, method="qdeim" if s == 10 else "qdeim+greedy")
    c = np.random.default_rng(3).standard_normal(10)
    f = Psi @ c
    recon = Psi @ (interpolation_pinv(Psi, pts) @ f[pts])
    np.testing.assert_allclose(recon, f, atol=1e-11 * np.abs(f).max())


def test_oversampled_reconstruction_is_the_least_squares_solution():
    _, Psi, _, _ = _ctx(m=8)
    pts = select_points(Psi, s=16, method="qdeim+greedy")
    g = np.random.default_rng(4).standard_normal(Psi.shape[0])  # NOT in span(Psi)
    coef = interpolation_pinv(Psi, pts) @ g[pts]
    np.testing.assert_allclose(coef, np.linalg.pinv(Psi[pts]) @ g[pts], atol=1e-10)
    resid = g[pts] - Psi[pts] @ coef
    np.testing.assert_allclose(Psi[pts].T @ resid, 0.0, atol=1e-10)  # normal-equation optimality


def test_interpolation_pinv_is_inverse_when_square():
    _, Psi, _, _ = _ctx(m=8)
    pts = select_points(Psi, method="deim")
    np.testing.assert_allclose(interpolation_pinv(Psi, pts), np.linalg.inv(Psi[pts]), atol=1e-9)


# ------------------------------------------------------------------ point selection
@pytest.mark.parametrize("method,s", [("deim", 12), ("qdeim", 12), ("qdeim+greedy", 18),
                                      ("qdeim+greedy", 24)])
def test_points_distinct_in_range_and_well_posed(method, s):
    _, Psi, _, _ = _ctx(m=12)
    pts = select_points(Psi, s=s, method=method)
    assert len(pts) == s and len(set(pts.tolist())) == s
    assert pts.min() >= 0 and pts.max() < Psi.shape[0]
    assert np.isfinite(interpolation_conditioning(Psi, pts)["cond"])


def test_qdeim_recovers_support_of_coordinate_basis():
    Psi = np.zeros((20, 3))
    Psi[[4, 11, 17], [0, 1, 2]] = 1.0
    assert set(qdeim_points(Psi).tolist()) == {4, 11, 17}


def test_qdeim_points_are_the_pivoted_qr_pivots():
    import scipy.linalg
    _, Psi, _, _ = _ctx(m=12)
    piv = scipy.linalg.qr(Psi.T, pivoting=True)[2]
    np.testing.assert_array_equal(qdeim_points(Psi), piv[:12])


def test_greedy_oversampling_never_decreases_sigma_min_and_is_deterministic():
    _, Psi, _, _ = _ctx(m=8)
    base = qdeim_points(Psi)
    pts = greedy_oversample(Psi, base, 16)
    np.testing.assert_array_equal(pts[:8], base)
    sig = [interpolation_conditioning(Psi, pts[:j])["sigma_min"] for j in range(8, 17)]
    assert all(sig[j + 1] >= sig[j] - 1e-14 for j in range(len(sig) - 1))
    np.testing.assert_array_equal(pts, greedy_oversample(Psi, base, 16))


def test_extended_selection_uses_s_modes_of_the_larger_basis():
    Phi, _, k, dealias = _ctx()
    ens, _ = _ens(64)
    Psi_ext, _ = build_basis(nonlinear_snapshots(ens, k, dealias), 24)
    pts = select_points(Psi_ext[:, :8], s=16, method="qdeim_extended", Psi_extended=Psi_ext)
    np.testing.assert_array_equal(pts, qdeim_points(Psi_ext[:, :16]))


def test_selection_rejects_invalid_requests():
    _, Psi, _, _ = _ctx(m=8)
    with pytest.raises(ValueError):
        select_points(Psi, s=6, method="qdeim+greedy")
    with pytest.raises(ValueError):
        select_points(Psi, s=12, method="deim")
    with pytest.raises(ValueError):
        select_points(Psi, method="nope")


# ------------------------------------------------------------------ exactness limits
def test_basis_spanning_the_nonlinear_image_reproduces_tensorial_galerkin_exactly():
    """N(Phi a) lies in span{G(phi_j phi_k)} (at most r(r+1)/2 vectors). With Psi an orthonormal
    basis of that span and its Q-DEIM points, DEIM interpolation is exact, so the spectral DEIM
    model equals the tensorial (= original) Galerkin RHS to rounding."""
    Nx, r = 64, 5
    Phi, _, k, dealias = _ctx(r=r, Nx=Nx)
    image = mode_products_image(Phi, k, dealias)
    j, kk = np.triu_indices(r)
    U, S, _ = np.linalg.svd(image[:, j, kk], full_matrices=False)
    Psi = U[:, S > 1e-10 * S[0]]
    model = build_spectral_deim(Phi, Psi, qdeim_points(Psi), k, dealias, target="nonlinear")
    tens = build_tensorial_rom(Phi, k, dealias)
    a = np.random.default_rng(5).standard_normal(r)
    ref = tensorial_rhs(tens, a, 0.03)
    assert np.linalg.norm(spectral_deim_rhs(model, a, 0.03) - ref) <= 1e-10 * np.linalg.norm(ref)
    np.testing.assert_allclose(folded_tensor(model), tens.Q, atol=1e-10 * np.abs(tens.Q).max())


@pytest.mark.parametrize("target", ["nonlinear", "full"])
def test_identity_basis_all_points_equals_plain_galerkin(target):
    Nx, r = 32, 6
    Phi, _, k, dealias = _ctx(r=r, Nx=Nx)
    model = build_spectral_deim(Phi, np.eye(Nx), np.arange(Nx), k, dealias, target=target)
    a = np.random.default_rng(6).standard_normal(r)
    ref = Phi.T @ _burgers_rhs_physical(Phi @ a, k, 0.04, dealias)
    assert np.linalg.norm(spectral_deim_rhs(model, a, 0.04) - ref) <= 1e-11 * np.linalg.norm(ref)


def test_identity_basis_trajectory_equals_plain_rom_trajectory():
    Nx, r = 32, 6
    Phi, _, k, dealias = _ctx(r=r, Nx=Nx)
    model = build_spectral_deim(Phi, np.eye(Nx), np.arange(Nx), k, dealias, target="nonlinear")
    u0 = 1.4 * np.sin(np.linspace(0, L, Nx, endpoint=False))
    a0 = Phi.T @ u0
    _, a_o, n_o = rom_reduced_trajectory(a0, 1.4, Phi, k, 0.03, dealias, 1.0, 20, align_snapshots=True)
    _, a_d, n_d = spectral_deim_reduced_trajectory(a0, 1.4, model, 0.03, 1.0, 20, align_snapshots=True)
    assert n_o == n_d
    assert np.abs(a_o - a_d).max() <= 1e-11 * np.abs(a_o).max()


def test_full_target_diffusion_is_interpolated_nonlinear_target_is_exact():
    Phi, Psi, k, dealias = _ctx()
    pts = select_points(Psi, method="qdeim")
    nl = build_spectral_deim(Phi, Psi, pts, k, dealias, target="nonlinear")
    np.testing.assert_array_equal(nl.Lin, build_tensorial_rom(Phi, k, dealias).L)
    _, Psi_f, _, _ = _ctx(target="full")
    full = build_spectral_deim(Phi, Psi_f, select_points(Psi_f, method="qdeim"), k, dealias, target="full")
    np.testing.assert_allclose(full.Lin, full.M @ sampled_diffusion(Phi, full.points, k))


# ------------------------------------------------------------------ online stage
def test_online_stage_performs_no_fft(monkeypatch):
    Phi, Psi, k, dealias = _ctx()
    model = build_spectral_deim(Phi, Psi, select_points(Psi, method="qdeim"), k, dealias)

    def boom(*a, **kw):
        raise AssertionError("FFT called in the spectral DEIM online stage")

    for name in ("fft", "ifft", "rfft", "irfft"):
        monkeypatch.setattr(np.fft, name, boom)
    _, a, _ = spectral_deim_reduced_trajectory(np.random.default_rng(7).standard_normal(8), 1.0,
                                               model, 0.05, 0.2, 5)
    assert np.all(np.isfinite(a))


def test_predict_shape_dtype_and_reproducibility():
    Phi, Psi, k, dealias = _ctx()
    model = build_spectral_deim(Phi, Psi, select_points(Psi, s=18, method="qdeim+greedy"), k, dealias)
    u0 = 1.1 * np.sin(np.linspace(0, L, 64, endpoint=False))
    t1, u1 = spectral_deim_predict(u0, Phi, model, 0.05, T=1.0, n_save=20, align_snapshots=True)
    t2, u2 = spectral_deim_predict(u0, Phi, model, 0.05, T=1.0, n_save=20, align_snapshots=True)
    assert u1.shape == (20, 64) and u1.dtype == np.float32 and np.isfinite(u1).all()
    np.testing.assert_allclose(np.diff(t1), 1.0 / 19, rtol=1e-12)
    np.testing.assert_array_equal(u1, u2)
