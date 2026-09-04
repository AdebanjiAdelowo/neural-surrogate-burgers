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


def rom_predict(u0: np.ndarray, Phi: np.ndarray, k: np.ndarray, nu: float,
                 dealias: np.ndarray, T: float, n_save: int, dt: float = None):
    """Predict a trajectory via POD-Galerkin projection, starting from u0.

    Returns:
        t_save: (n_save,), u_pred: (n_save, Nx) -- reconstructed physical-space predictions.
    """
    a = Phi.T @ u0  # project initial condition onto the POD basis
    if dt is None:
        dt = 0.25 * (2 * np.pi / len(u0)) / (np.abs(u0).max() + 1e-8)
    n_steps = max(1, int(np.ceil(T / dt)))
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

    a_save = np.array(a_save)
    u_pred = a_save @ Phi.T  # (n_save, Nx)
    return np.array(t_save), u_pred.astype(np.float32)
