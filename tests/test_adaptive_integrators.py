"""Exponential integrators, adaptive drivers and stiffness diagnostics (src/exponential_integrators.py,
src/adaptive_integrators.py, src/reduced_stiffness.py), verified first on problems with known solutions."""
import decimal
import math

import numpy as np
import pytest
import scipy.linalg

from scripts.hyperreduction_benchmark import wavenumbers
from src.adaptive_integrators import adaptive_solve, default_atol, scipy_solve
from src.convergence import sine_ic, solve_burgers_f64
from src.exponential_integrators import ETDRK4Krogstad, EigenSystem, LawsonDP54, phi, phi_matrix
from src.pod_rom import build_pod_basis
from src.reduced_stiffness import (diffusion_spectrum, henrici_departure, jacobian_diagnostics,
                                   transient_amplification)
from src.reduced_timestep import rk4_trajectory
from src.tensorial_pod import build_tensorial_rom


# ------------------------------------------------------------------------------- phi functions
def _phi_ref(k, z, digits=60):
    """phi_k(z) in 60-digit decimal arithmetic via the Taylor series (exact reference)."""
    decimal.getcontext().prec = digits
    z = decimal.Decimal(repr(float(z)))
    s, term, m = decimal.Decimal(0), decimal.Decimal(1) / math.factorial(k), 0
    while True:
        s += term
        m += 1
        term = term * z / (m + k)
        if abs(term) < decimal.Decimal(10) ** (-digits + 5) and m > 10:
            break
    return float(s)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("z", [0.0, -1e-12, -1e-6, -0.3, -0.999, -1.001, -3.0, -20.0, 0.7])
def test_phi_matches_high_precision_reference(k, z):
    if abs(z) > 5:
        pytest.skip("Taylor reference loses digits for large |z|; checked by the recurrence test")
    got = phi(k, np.array([z]))[0]
    assert abs(got - _phi_ref(k, z)) <= 2e-15 * max(1.0, abs(_phi_ref(k, z)))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_phi_large_negative_argument_and_recurrence(k):
    z = np.array([-20.0, -500.0, -1e5])
    # phi_k(z) -> -1/z * phi_{k-1}... asymptotically phi_k(z) ~ -1/((k-1)! z) for z -> -inf
    np.testing.assert_allclose(phi(k, z) * (-z) * math.factorial(k - 1), 1.0, rtol=0.2)
    zz = np.array([-0.5, -2.0, -40.0])
    np.testing.assert_allclose(phi(k, zz), (phi(k - 1, zz) - 1.0 / math.factorial(k - 1)) / zz, rtol=1e-12)


def test_phi_is_accurate_on_both_sides_of_the_series_switch():
    for k in (1, 2, 3):
        for z in (-1.0 + 1e-12, -1.0 - 1e-12, 1.0 - 1e-12, 1.0 + 1e-12):
            assert abs(phi(k, np.array([z]))[0] - _phi_ref(k, z)) < 2e-15


def test_phi_matrix_agrees_with_diagonal_phi_for_symmetric_matrices():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((6, 6))
    A = -(B @ B.T) - 0.1 * np.eye(6)
    lam, V = np.linalg.eigh(A)
    for k, Pk in enumerate(phi_matrix(A, 3), start=1):
        np.testing.assert_allclose(Pk, (V * phi(k, lam)) @ V.T, atol=1e-12 * np.abs(Pk).max())


def test_phi_matrix_handles_a_nonnormal_jordan_block_without_eigenvalue_division():
    lam = -2.0
    A = np.array([[lam, 1.0], [0.0, lam]])
    P1 = phi_matrix(A, 1)[0]
    dphi1 = (lam * math.exp(lam) - math.exp(lam) + 1.0) / lam**2   # d/dlam (e^lam - 1)/lam
    np.testing.assert_allclose(P1, [[phi(1, np.array([lam]))[0], dphi1], [0.0, phi(1, np.array([lam]))[0]]],
                               atol=1e-13)


# ------------------------------------------------------------------------------- eigen-system
def _sym_system(r=5, seed=0, stiff=1.0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((r, r))
    L = -(B @ B.T) * stiff - np.eye(r)
    Q = rng.standard_normal((r, r, r)) * 0.1
    Q = 0.5 * (Q + Q.transpose(0, 2, 1))
    return Q, L


def test_eigensystem_preserves_the_vector_field_and_exponential():
    Q, L = _sym_system()
    es = EigenSystem(Q, L, nu=0.3)
    a = np.random.default_rng(1).standard_normal(5)
    Qa = np.einsum("ijk,j,k->i", Q, a, a)
    np.testing.assert_allclose(es.to_a(es.N(es.to_b(a))), Qa, atol=1e-13)
    np.testing.assert_allclose(es.to_a(np.exp(0.7 * es.D) * es.to_b(a)), scipy.linalg.expm(0.7 * 0.3 * L) @ a,
                               atol=1e-12)


def test_eigensystem_rejects_a_nonsymmetric_operator():
    Q, L = _sym_system()
    L[0, 1] += 1.0
    with pytest.raises(ValueError):
        EigenSystem(Q, L, nu=1.0)


# ------------------------------------------------------------------------------- manufactured problems
def _riccati(d=np.array([-1.0, -3.0, -0.5]), a0=np.array([0.4, -0.3, 0.2])):
    """da_i/dt = d_i a_i + a_i^2 (diagonal L, Q_iii = 1): exact a_i(t) = 1 / ((1/a0 + 1/d) e^{-dt} - 1/d)."""
    r = len(d)
    Q = np.zeros((r, r, r))
    for i in range(r):
        Q[i, i, i] = 1.0
    def exact(t):
        with np.errstate(over="ignore"):  # e^{-d t} -> inf for very stiff d gives the exact limit 0
            return 1.0 / ((1 / a0 + 1 / d) * np.exp(-d * t) - 1 / d)
    return Q, np.diag(d), a0, exact


def test_linear_problems_are_integrated_exactly():
    Q, L = _sym_system(stiff=50.0)
    Q[:] = 0.0
    a0 = np.random.default_rng(2).standard_normal(5)
    times = np.linspace(0, 1, 6)
    exact = np.array([scipy.linalg.expm(t * L) @ a0 for t in times])
    for method in ("lawson_dp54", "etdrk4_doubling"):
        a, st = adaptive_solve(method, Q, L, 1.0, a0, times, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(a, exact, atol=1e-12)


@pytest.mark.parametrize("cls,order", [(LawsonDP54, 5), (ETDRK4Krogstad, 4)])
def test_fixed_step_convergence_order_on_the_riccati_problem(cls, order):
    Q, L, a0, exact = _riccati()
    es = EigenSystem(Q, L, 1.0)
    errs = []
    for n in (8, 16, 32):
        integ, b, h = cls(es), es.to_b(a0), 1.0 / n
        Nb = es.N(b)
        for _ in range(n):
            res = integ.step(b, Nb, h)
            b = res[0]
            Nb = es.N(b)
        errs.append(np.abs(es.to_a(b) - exact(1.0)).max())
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > order - 0.3), orders


def test_exponential_methods_stay_accurate_on_a_stiff_riccati_problem():
    """d = -1e4 is stiff for explicit RK; the exponential methods need no stability-limited steps."""
    Q, L, a0, exact = _riccati(d=np.array([-1e4, -2.0, -0.5]), a0=np.array([0.4, -0.3, 0.2]))
    times = np.array([0.0, 0.5, 1.0])
    for method in ("lawson_dp54", "etdrk4_doubling"):
        a, st = adaptive_solve(method, Q, L, 1.0, a0, times, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(a[-1], exact(1.0), atol=1e-8)
        assert st["accepted"] < 200
    _, st_rk = scipy_solve(Q, L, 1.0, a0, times, "RK45", rtol=1e-8, atol=1e-12)
    assert st_rk["nfev"] > 10 * 200  # the explicit method is stability-limited


@pytest.mark.parametrize("method", ["lawson_dp54", "etdrk4_doubling"])
def test_global_error_follows_the_tolerance_on_the_riccati_problem(method):
    # one output interval so that the tolerance, not the output spacing, limits the step
    Q, L, a0, exact = _riccati(d=np.array([-1.0, -3.0, -0.5]), a0=np.array([-2.0, -1.5, -1.0]))
    times = np.array([0.0, 1.0])
    errs = []
    for rtol in (1e-4, 1e-6, 1e-8):
        a, _ = adaptive_solve(method, Q, L, 1.0, a0, times, rtol=rtol, atol=1e-3 * rtol)
        errs.append(np.abs(a[-1] - exact(1.0)).max())
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-7


def test_embedded_error_estimate_tracks_the_true_local_error():
    Q, L, a0, exact = _riccati()
    es = EigenSystem(Q, L, 1.0)
    integ = LawsonDP54(es)
    b = es.to_b(a0)
    ratios = []
    for h in (0.2, 0.1, 0.05):
        b_new, _, e, _ = integ.step(b, es.N(b), h)
        true5 = np.abs(es.to_a(b_new) - exact(h)).max()
        ratios.append(np.abs(es.to_a(e)).max() / max(true5, 1e-300))
    assert all(r > 1.0 for r in ratios)  # the 4th-order estimate over-estimates the 5th-order solution's error


def test_steps_are_rejected_and_counted_when_the_first_step_is_too_large():
    # a0 < 0 with d < 0: the Riccati solution decays (a0 > -d would blow up in finite time)
    Q, L, a0, _ = _riccati(d=np.array([-50.0, -2.0, -0.5]), a0=np.array([-4.0, -3.0, -2.0]))
    a, st = adaptive_solve("lawson_dp54", Q, L, 1.0, a0, np.array([0.0, 1.0]), rtol=1e-10, atol=1e-13,
                           record_steps=True)
    assert st["n_evals"] == 1 + 6 * (st["accepted"] + st["rejected"])
    assert len(st["steps"]) == st["accepted"]
    assert abs(sum(h for _, h in st["steps"]) - 1.0) < 1e-12


def test_outputs_land_exactly_on_nonuniform_requested_times():
    Q, L, a0, exact = _riccati()
    times = np.array([0.0, 0.013, 0.3, 0.31, 1.0])
    for method in ("lawson_dp54", "etdrk4_doubling"):
        a, st = adaptive_solve(method, Q, L, 1.0, a0, times, rtol=1e-10, atol=1e-13, record_steps=True)
        np.testing.assert_allclose(a, np.array([exact(t) for t in times]), atol=1e-9)
        ends = [t for t, _ in st["steps"]]
        for t in times[1:]:
            assert min(abs(np.array(ends) - t)) < 1e-14


# ------------------------------------------------------------------------------- Burgers ROM
def _burgers_rom(r=8, N=64):
    k, dealias = wavenumbers(N)
    rng = np.random.default_rng(0)
    ens = [solve_burgers_f64(sine_ic(rng.uniform(0.5, 2), N), rng.uniform(0.02, 0.1))[1] for _ in range(6)]
    Phi, _ = build_pod_basis(np.concatenate(ens), r)
    return Phi, build_tensorial_rom(Phi, k, dealias)


@pytest.mark.parametrize("method", ["lawson_dp54", "etdrk4_doubling", "RK45", "DOP853", "Radau", "Radau_fd"])
def test_all_methods_agree_with_a_high_accuracy_reduced_solution(method):
    Phi, m = _burgers_rom()
    a0 = Phi.T @ sine_ic(1.3, 64)
    times = np.linspace(0, 1, 20)
    ref = rk4_trajectory(m, a0, 0.04, 1.0, 20, 19 * 400)[1]
    rtol = 1e-10
    atol = default_atol(a0, rtol)
    if method in ("lawson_dp54", "etdrk4_doubling"):
        a, _ = adaptive_solve(method, m.Q, m.L, 0.04, a0, times, rtol, atol)
    else:
        name, jac = (method.split("_") + ["analytic"])[:2]
        a, st = scipy_solve(m.Q, m.L, 0.04, a0, times, name, rtol, atol, jacobian="fd" if jac == "fd" else "analytic")
        assert st["success"]
    assert np.linalg.norm(a - ref) / np.linalg.norm(ref) < 1e-8


def test_reduced_trajectories_keep_zero_mean_and_do_not_gain_energy_at_tight_tolerance():
    Phi, m = _burgers_rom()
    a0 = Phi.T @ sine_ic(1.5, 64)
    a, _ = adaptive_solve("lawson_dp54", m.Q, m.L, 0.03, a0, np.linspace(0, 1, 20), 1e-8, default_atol(a0, 1e-8))
    u = a @ Phi.T
    np.testing.assert_allclose(u.mean(axis=1), 0.0, atol=1e-12)
    E = 0.5 * np.sum(a**2, axis=1)
    assert np.all(np.diff(E) <= 1e-10 * E[0])


def test_adaptive_solutions_are_bit_reproducible():
    Phi, m = _burgers_rom()
    a0 = Phi.T @ sine_ic(1.1, 64)
    t = np.linspace(0, 1, 20)
    for method in ("lawson_dp54", "etdrk4_doubling"):
        a1, _ = adaptive_solve(method, m.Q, m.L, 0.05, a0, t, 1e-7, default_atol(a0, 1e-7))
        a2, _ = adaptive_solve(method, m.Q, m.L, 0.05, a0, t, 1e-7, default_atol(a0, 1e-7))
        np.testing.assert_array_equal(a1, a2)


def test_radau_reports_jacobian_and_factorization_counts():
    Phi, m = _burgers_rom()
    a0 = Phi.T @ sine_ic(1.1, 64)
    _, sa = scipy_solve(m.Q, m.L, 0.05, a0, np.linspace(0, 1, 20), "Radau", 1e-6, default_atol(a0, 1e-6), "analytic")
    _, sf = scipy_solve(m.Q, m.L, 0.05, a0, np.linspace(0, 1, 20), "Radau", 1e-6, default_atol(a0, 1e-6), "fd")
    # scipy does not include the RHS calls of its finite-difference Jacobian in nfev; their cost appears
    # only in the runtime (measured in the study)
    assert sa["success"] and sf["success"] and sa["njev"] >= 1 and sf["njev"] >= 1 and sa["nlu"] >= 1


# ------------------------------------------------------------------------------- stiffness diagnostics
def test_stiffness_diagnostics_on_known_matrices():
    S = -np.diag([1.0, 10.0, 100.0])
    d = diffusion_spectrum(S)
    assert d["rho"] == pytest.approx(100.0) and d["stiffness_ratio"] == pytest.approx(100.0)
    assert d["henrici_departure"] < 1e-14 and d["symmetry_defect"] == 0.0
    J = np.array([[-1.0, 50.0], [0.0, -2.0]])          # stable but strongly nonnormal
    assert henrici_departure(J) > 0.9
    assert transient_amplification(J) > 5.0             # transient growth despite negative eigenvalues


def test_jacobian_diagnostics_numerical_abscissa_bounds_spectral_abscissa():
    Phi, m = _burgers_rom()
    a = Phi.T @ sine_ic(1.8, 64)
    jd = jacobian_diagnostics(m.Q, m.L, 0.02, a, transient=True)
    assert jd["numerical_abscissa"] >= jd["spectral_abscissa"] - 1e-12
    assert jd["transient_amplification"] >= 1.0 - 1e-12


def test_precomputed_basis_gives_identical_results():
    from src.exponential_integrators import EigenBasis
    Phi, m = _burgers_rom()
    a0 = Phi.T @ sine_ic(1.2, 64)
    t = np.linspace(0, 1, 20)
    basis = EigenBasis(m.Q, m.L)
    for method in ("lawson_dp54", "etdrk4_doubling"):
        a1, _ = adaptive_solve(method, m.Q, m.L, 0.05, a0, t, 1e-7, default_atol(a0, 1e-7))
        a2, _ = adaptive_solve(method, m.Q, m.L, 0.05, a0, t, 1e-7, default_atol(a0, 1e-7), basis=basis)
        np.testing.assert_array_equal(a1, a2)
