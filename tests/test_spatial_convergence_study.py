"""Study-level checks for scripts/spatial_convergence_study.py: conservation and maximum-principle
diagnostics of the float64 FOM, grid consistency, POD orthogonality at higher resolution, the policy
dispatcher and the error bookkeeping. No data files are needed (CI has none)."""
import numpy as np
import pytest

from scripts.hyperreduction_benchmark import wavenumbers
from scripts.spatial_convergence_study import (CASES, POLICIES, TIMES, err_record, project, run_policy)
from src.convergence import grid, rel_l2, resample, sine_ic, sine_solution, solve_burgers_f64
from src.pod_rom import build_pod_basis
from src.tensorial_pod import build_tensorial_rom


def test_snapshot_times_are_the_aligned_solver_times():
    t, _ = solve_burgers_f64(sine_ic(1.0, 32), 0.05)
    np.testing.assert_allclose(t, TIMES, atol=1e-13)


def test_mass_is_conserved_and_energy_decreases_on_a_resolved_grid():
    u = solve_burgers_f64(sine_ic(1.5, 256), 0.03)[1]
    dx = 2 * np.pi / 256
    np.testing.assert_allclose(dx * u.sum(axis=1), 0.0, atol=1e-13)
    assert np.all(np.diff(0.5 * dx * np.sum(u**2, axis=1)) < 0)
    assert np.abs(u).max() <= 1.5 * (1 + 1e-12)  # maximum principle holds when resolved


def test_under_resolved_steep_case_violates_the_maximum_principle():
    """Documents the Gibbs behaviour found in the study: at N = 64 the A = 2, nu = 0.01 front is not
    resolved and max|u| exceeds max|u0| (at N = 2048 it does not: diagnostics_run1.json, too slow
    for the unit tests)."""
    u_coarse = solve_burgers_f64(sine_ic(2.0, 64), 0.01)[1]
    assert np.abs(u_coarse).max() > 2.0 * 1.05


def test_same_physical_grid_points_across_resolutions():
    """Every second point of the 2N grid is a point of the N grid (spatial-grid consistency)."""
    np.testing.assert_allclose(grid(128)[::2], grid(64), atol=1e-15)


def test_exact_solution_restricted_to_coarse_grid_matches_direct_evaluation():
    ch = sine_solution(1.0, 0.08)
    fine, coarse = ch(grid(512), 0.5), ch(grid(128), 0.5)
    np.testing.assert_allclose(resample(fine, 128), coarse, atol=1e-12)


def test_float64_pod_basis_is_orthonormal_at_high_resolution():
    rng = np.random.default_rng(0)
    ens = np.concatenate([solve_burgers_f64(sine_ic(rng.uniform(0.5, 2), 512), rng.uniform(0.02, 0.1))[1]
                          for _ in range(4)])
    Phi, _ = build_pod_basis(ens, 16)
    np.testing.assert_allclose(Phi.T @ Phi, np.eye(16), atol=1e-13)
    u = ens[7]
    p = project(u, Phi)
    assert abs(np.dot(u - p, p)) <= 1e-10 * np.dot(u, u)  # projection residual is orthogonal


def test_err_record_fields_are_consistent():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((20, 64))
    u = ref + 1e-3 * rng.standard_normal((20, 64))
    rec = err_record(u, ref)
    assert rec["space_time"] == pytest.approx(rel_l2(u, ref))
    assert rec["final"] == pytest.approx(rec["per_snapshot"][-1])
    assert rec["max_snapshot"] == pytest.approx(np.max(rec["per_snapshot"][1:]))


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_runs_and_agrees_with_the_current_policy(policy):
    rng = np.random.default_rng(2)
    ens = np.concatenate([solve_burgers_f64(sine_ic(rng.uniform(0.5, 2), 64), rng.uniform(0.03, 0.1))[1]
                          for _ in range(6)])
    Phi, _ = build_pod_basis(ens, 8)
    m = build_tensorial_rom(Phi, *wavenumbers(64))
    u0 = sine_ic(1.2, 64)
    a_ref, _ = run_policy("P0_reduced", m, Phi, u0, 0.05)
    a, info = run_policy(policy, m, Phi, u0, 0.05)
    assert a.shape == (20, 8) and info["rhs_evals"] > 0
    assert np.abs(a - a_ref).max() <= 1e-3 * np.abs(a_ref).max()


def test_case_list_is_the_pre_registered_one():
    assert len(CASES) == 11 and CASES["corner_highA_lownu"] == (2.0, 0.01)
