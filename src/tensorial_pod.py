"""Tensorial POD-Galerkin ROM: the plain POD-Galerkin model of src/pod_rom.py with its quadratic
nonlinearity precomputed as a third-order tensor, so no FFT and no full-grid field appear online.

Derivation (exact, discrete). src.pod_rom._burgers_rhs_physical evaluates

    f(u) = N(u) + nu D2 u,     N(u) = Re ifft( -0.5 i k * mask * fft(u * u) ),
                               D2 u = Re ifft( -k^2 * fft(u) ),

with k the FFT wavenumbers and `mask` the 2/3-rule dealiasing mask. Both maps are LINEAR (over the
reals) in their argument: N(u) = G (u * u) and D2 u = D2 u, with G = Re F^-1 diag(-0.5 i k mask) F and
D2 = Re F^-1 diag(-k^2) F real circulant Nx x Nx matrices (taking the real part is real-linear; the
Nyquist mode, the only one whose symbol is not conjugate-symmetric, is removed by the mask). With
u = Phi a and the Galerkin RHS Phi^T f(Phi a),

    da_i/dt = sum_{j,k} Q_ijk a_j a_k + nu sum_j L_ij a_j,
    Q_ijk = phi_i^T G (phi_j * phi_k),     L = Phi^T D2 Phi,

which is the SAME reduced vector field, re-associated: it differs from src.pod_rom only by
floating-point rounding (tests/test_tensorial_pod.py). Q is symmetric in (j, k). The tensor is built
offline by applying the existing FFT operator to the r(r+1)/2 distinct mode products; online, the
RHS costs O(r^3) and never touches the Nx-point grid.

Time stepping, the step-size policies and snapshot alignment are those of src.pod_rom (same explicit
RK4, same `timestep_policy` rules; the "reduced" policy uses rho(L), the same matrix the plain ROM
uses, so both models take the same number of steps).
"""

from dataclasses import dataclass

import numpy as np

from src.snapshot_times import align_step_count
from src.timestep import (DEFAULT_POLICY, check_policy, historical_dt, reduced_diffusion_matrix_rom,
                          reduced_dt, spectral_radius)


def apply_nonlinear_operator(w: np.ndarray, k: np.ndarray, dealias: np.ndarray) -> np.ndarray:
    """G w = Re ifft(-0.5 i k mask fft(w)) along axis 0: the dealiased -(1/2) d/dx of src.pod_rom,
    applied to w = u*u (or to any stack of grid functions, one per trailing index)."""
    shape = (-1,) + (1,) * (w.ndim - 1)
    w_hat = np.fft.fft(w, axis=0) * dealias.reshape(shape)
    return np.real(np.fft.ifft(-0.5j * k.reshape(shape) * w_hat, axis=0))


def mode_products_image(Phi: np.ndarray, k: np.ndarray, dealias: np.ndarray) -> np.ndarray:
    """G (phi_j * phi_k) for every pair, shape (Nx, r, r), symmetric in the last two indices.

    Only the r(r+1)/2 distinct products are transformed; the rest are filled by symmetry."""
    Phi = np.asarray(Phi, dtype=np.float64)
    Nx, r = Phi.shape
    j, kk = np.triu_indices(r)
    image = np.empty((Nx, r, r))
    image[:, j, kk] = apply_nonlinear_operator(Phi[:, j] * Phi[:, kk], k, dealias)
    image[:, kk, j] = image[:, j, kk]
    return image


def build_quadratic_tensor(Phi: np.ndarray, k: np.ndarray, dealias: np.ndarray) -> np.ndarray:
    """Q (r, r, r), Q_ijk = phi_i^T G (phi_j * phi_k)."""
    image = mode_products_image(Phi, k, dealias)
    return np.einsum("xi,xjk->ijk", np.asarray(Phi, dtype=np.float64), image)


@dataclass
class TensorialROM:
    """Offline operators of the tensorial model (everything the online stage needs)."""
    Q: np.ndarray            # (r, r, r) quadratic tensor
    L: np.ndarray            # (r, r) reduced spectral Laplacian Phi^T D2 Phi
    Nx: int                  # full-grid size (only used by the step-size rules)
    diffusion_radius: float  # rho(L), for the "reduced" step-size policy

    @property
    def r(self) -> int:
        return self.Q.shape[0]

    @property
    def Q_flat(self) -> np.ndarray:
        """Q reshaped to (r*r, r) so that (Q_flat @ a).reshape(r, r) @ a = Q(a, a)."""
        return self.Q.reshape(self.r * self.r, self.r)


def build_tensorial_rom(Phi: np.ndarray, k: np.ndarray, dealias: np.ndarray) -> TensorialROM:
    """Offline stage: Q and L from the POD basis and the solver's own spectral operators.

    L (and so rho(L)) is computed by src.timestep.reduced_diffusion_matrix_rom on the basis exactly
    as given, so the "reduced" step-size rule picks the same dt as src.pod_rom for the same basis."""
    L = reduced_diffusion_matrix_rom(Phi, k)
    return TensorialROM(Q=build_quadratic_tensor(Phi, k, dealias), L=L, Nx=Phi.shape[0],
                        diffusion_radius=spectral_radius(L))


def quadratic_rhs(Q_flat: np.ndarray, Lnu: np.ndarray):
    """Online RHS a -> Q(a, a) + Lnu a for a quadratic reduced model (Lnu = nu * L, fixed per
    trajectory). Two small mat-vecs; no FFT, no full-grid array."""
    r = Lnu.shape[0]

    def rhs(a):
        return ((Q_flat @ a).reshape(r, r) + Lnu) @ a
    return rhs


def tensorial_rhs(model: TensorialROM, a: np.ndarray, nu: float) -> np.ndarray:
    """F_r(a) = Q(a, a) + nu L a (convenience form of `quadratic_rhs` for a single evaluation)."""
    return quadratic_rhs(model.Q_flat, nu * model.L)(a)


def quadratic_jacobian(Q: np.ndarray, Lnu: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Exact Jacobian of a -> Q(a, a) + Lnu a:  J_ij = sum_k (Q_ijk + Q_ikj) a_k + Lnu_ij."""
    return np.einsum("ijk,k->ij", Q + Q.transpose(0, 2, 1), a) + Lnu


def reduced_step_count(u0_max, Nx: int, nu: float, T: float, n_save: int, dt: float = None,
                       align_snapshots: bool = False, min_steps: int = None,
                       timestep_policy: str = DEFAULT_POLICY, diffusion_radius: float = None):
    """(n_steps, dt) by exactly the rule of src.pod_rom.rom_reduced_trajectory."""
    check_policy(timestep_policy)
    if dt is None:
        if timestep_policy == "historical":
            dt = historical_dt(u0_max, Nx, nu)
        else:
            dt = reduced_dt(u0_max, Nx, nu, diffusion_radius)
    n_steps = max(4 * n_save if min_steps is None else min_steps, int(np.ceil(T / dt)))
    if align_snapshots:
        n_steps = align_step_count(n_steps, n_save)
    return n_steps, T / n_steps


def integrate_reduced(rhs, a0: np.ndarray, n_steps: int, dt: float, n_save: int):
    """Classical explicit RK4 with snapshots at the step indices of the existing solvers
    (src.snapshot_times.saved_step_indices). Returns t_save (n_save,), a_save (n_save, r)."""
    save_steps = set(np.linspace(0, n_steps, n_save, dtype=int).tolist())
    a = a0
    t_save = [0.0]
    a_save = [a.copy()]
    t = 0.0
    for step in range(1, n_steps + 1):
        k1 = rhs(a)
        k2 = rhs(a + dt / 2 * k1)
        k3 = rhs(a + dt / 2 * k2)
        k4 = rhs(a + dt * k3)
        a = a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if step in save_steps:
            a_save.append(a.copy())
            t_save.append(t)
    return np.array(t_save), np.array(a_save)


def tensorial_reduced_trajectory(a0: np.ndarray, u0_max, model: TensorialROM, nu: float, T: float,
                                 n_save: int, dt: float = None, align_snapshots: bool = False,
                                 min_steps: int = None, timestep_policy: str = DEFAULT_POLICY):
    """Solver-only online path (no projection of u0, no reconstruction), analogue of
    src.pod_rom.rom_reduced_trajectory.

    Returns:
        t_save: (n_save,), a_save: (n_save, r) reduced states, n_steps: RK4 steps taken.
    """
    n_steps, dt = reduced_step_count(u0_max, model.Nx, nu, T, n_save, dt, align_snapshots,
                                     min_steps, timestep_policy, model.diffusion_radius)
    t_save, a_save = integrate_reduced(quadratic_rhs(model.Q_flat, nu * model.L), a0, n_steps, dt,
                                       n_save)
    return t_save, a_save, n_steps


def tensorial_predict(u0: np.ndarray, Phi: np.ndarray, model: TensorialROM, nu: float, T: float,
                      n_save: int, dt: float = None, align_snapshots: bool = False,
                      min_steps: int = None, timestep_policy: str = DEFAULT_POLICY):
    """End-to-end prediction (project u0, integrate, reconstruct), analogue of
    src.pod_rom.rom_predict.

    Returns:
        t_save: (n_save,), u_pred: (n_save, Nx) float32 reconstructed predictions.
    """
    a0 = Phi.T @ u0
    t_save, a_save, _ = tensorial_reduced_trajectory(a0, np.abs(u0).max(), model, nu, T, n_save, dt,
                                                     align_snapshots, min_steps, timestep_policy)
    return t_save, (a_save @ Phi.T).astype(np.float32)
