"""Classical reduced-order model: POD-Galerkin projection.

Method: a spatial POD basis Phi (Nx x r) is built by SVD of a snapshot ensemble drawn from the
training set. A new initial condition is projected onto Phi to get reduced coordinates a(0); the
reduced ODE da/dt = Phi^T f(Phi a) is integrated with explicit RK4, where f is Burgers' RHS
(-u u_x + nu u_xx) evaluated in physical space via the same FFT-based spectral derivatives as the
ground-truth solver (src/solver.py) -- a standard, textbook Galerkin-projection ROM, not a
"non-intrusive"/data-fit shortcut. This is the classical linear-subspace baseline the neural
surrogate must beat, not a strawman.
"""

import numpy as np

from src.snapshot_times import align_step_count
from src.timestep import (DEFAULT_POLICY, check_policy, historical_dt, reduced_diffusion_matrix_rom,
                          reduced_dt, spectral_radius)


def build_pod_basis(u_ensemble: np.ndarray, r: int):
    """Build a rank-r POD basis from a snapshot ensemble.

    Args:
        u_ensemble: (n_snapshots, Nx) -- flattened (example, time) snapshots.
        r: number of modes to retain.

    Returns:
        Phi: (Nx, r) orthonormal POD basis.
        singular_values: full singular value spectrum (for energy-capture diagnostics).
    """
    U, S, _ = np.linalg.svd(u_ensemble.T, full_matrices=False)  # U: (Nx, n_snapshots)
    Phi = U[:, :r]
    return Phi, S


def energy_captured(singular_values: np.ndarray, r: int) -> float:
    """Fraction of ensemble variance captured by the first r POD modes."""
    return float(np.sum(singular_values[:r] ** 2) / np.sum(singular_values**2))


def _burgers_rhs_physical(u: np.ndarray, k: np.ndarray, nu: float, dealias: np.ndarray) -> np.ndarray:
    u_hat = np.fft.fft(u)
    ux = np.real(np.fft.ifft(1j * k * u_hat))
    uxx = np.real(np.fft.ifft(-(k**2) * u_hat))
    flux_hat = np.fft.fft(u * u) * dealias
    nonlinear = np.real(np.fft.ifft(-0.5j * k * flux_hat))  # = -(u u_x), dealiased
    return nonlinear + nu * uxx


def rom_reduced_trajectory(a0: np.ndarray, u0_max, Phi: np.ndarray, k: np.ndarray, nu: float,
                            dealias: np.ndarray, T: float, n_save: int, dt: float = None,
                            align_snapshots: bool = False, min_steps: int = None,
                            timestep_policy: str = DEFAULT_POLICY, diffusion_radius: float = None):
    """Time-step the POD-Galerkin reduced ODE from reduced initial coordinates a0.

    This is the "solver-only" online path of `rom_predict`: no initial projection and no final
    reconstruction of the saved states onto the full grid (those two steps live in `rom_predict`).
    Every RHS evaluation inside still reconstructs u = Phi @ a on the FULL grid and evaluates the
    FFT-based nonlinear term there -- the full-grid cost this ROM does not remove.

    Args:
        a0: (r,) reduced initial coordinates, Phi.T @ u0.
        u0_max: max|u0| of the full-grid initial condition (sets the advective step bound; passed
            through unchanged from the caller so `rom_predict` stays bit-identical).

    Returns:
        t_save: (n_save,), a_save: (n_save, r) reduced states, n_steps: RK4 steps taken.
    """
    check_policy(timestep_policy)
    a = a0
    if dt is None:
        Nx = Phi.shape[0]
        if timestep_policy == "historical":
            # Unlike src.solver (which handles diffusion exactly via an integrating factor), this
            # explicit RK4 in reduced coordinates has no such treatment, so it is only conditionally
            # stable in the diffusion term too. Bug found during Stage 5 evaluation: using only the
            # advective bound (matching src.solver's dt) produced NaN via an unstable diffusion step.
            # Fixed by also respecting the standard explicit-diffusion stability limit
            # dt < dx^2 / (2*nu), using the full grid's dx even though the ROM never touches the full
            # grid directly. That rule scales the step count like Nx^2; see src/timestep.py.
            dt = historical_dt(u0_max, Nx, nu)
        else:
            rho = (spectral_radius(reduced_diffusion_matrix_rom(Phi, k))
                   if diffusion_radius is None else diffusion_radius)
            dt = reduced_dt(u0_max, Nx, nu, rho)
    n_steps = max(4 * n_save if min_steps is None else min_steps, int(np.ceil(T / dt)))
    if align_snapshots:
        n_steps = align_step_count(n_steps, n_save)
    dt = T / n_steps
    save_steps = set(np.linspace(0, n_steps, n_save, dtype=int).tolist())

    def galerkin_rhs(a_vec):
        u = Phi @ a_vec
        f = _burgers_rhs_physical(u, k, nu, dealias)
        return Phi.T @ f

    t_save = [0.0]
    a_save = [a.copy()]
    t = 0.0
    for step in range(1, n_steps + 1):
        k1 = galerkin_rhs(a)
        k2 = galerkin_rhs(a + dt / 2 * k1)
        k3 = galerkin_rhs(a + dt / 2 * k2)
        k4 = galerkin_rhs(a + dt * k3)
        a = a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if step in save_steps:
            a_save.append(a.copy())
            t_save.append(t)

    return np.array(t_save), np.array(a_save), n_steps


def rom_predict(u0: np.ndarray, Phi: np.ndarray, k: np.ndarray, nu: float,
                 dealias: np.ndarray, T: float, n_save: int, dt: float = None,
                 align_snapshots: bool = False, min_steps: int = None,
                 timestep_policy: str = DEFAULT_POLICY, diffusion_radius: float = None):
    """Predict a trajectory via POD-Galerkin projection, starting from u0.

    Returns:
        t_save: (n_save,), u_pred: (n_save, Nx) -- reconstructed physical-space predictions.
    """
    a = Phi.T @ u0  # project initial condition onto the POD basis
    t_save, a_save, _ = rom_reduced_trajectory(a, np.abs(u0).max(), Phi, k, nu, dealias, T,
                                                n_save, dt, align_snapshots, min_steps,
                                                timestep_policy, diffusion_radius)
    u_pred = a_save @ Phi.T  # (n_save, Nx)
    return t_save, u_pred.astype(np.float32)
