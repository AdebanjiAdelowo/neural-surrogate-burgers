"""Stiffness diagnostics of the tensorial POD system f(a) = nu L a + Q(a, a).

Jacobian (src.tensorial_pod.quadratic_jacobian): J_ij(a) = nu L_ij + sum_k (Q_ijk + Q_ikj) a_k, which is
nu L_ij + 2 sum_k Q_ijk a_k for the (j, k)-symmetric tensor stored by the repository.

Quantities:
  * spectral abscissa alpha(J) = max Re lambda(J): sign of the frozen linearisation's growth rate;
  * numerical abscissa omega(J) = lambda_max((J + J^T)/2) = d/dt ||e^{Jt}|| at t = 0: the initial growth
    rate of perturbations in the 2-norm; omega > 0 >= alpha signals possible transient growth;
  * Henrici departure from normality dep(J) = sqrt(||J||_F^2 - sum |lambda_i|^2) / ||J||_F (0 for normal);
  * transient amplification max_t ||e^{Jt}||_2 on a time grid (frozen J; > 1 is transient growth).
None of these is a nonlinear stability guarantee: they describe the frozen linearisation at given states.
"""

import numpy as np
import scipy.linalg

from src.tensorial_pod import quadratic_jacobian


def henrici_departure(A: np.ndarray) -> float:
    ev = np.linalg.eigvals(A)
    fro2 = float(np.sum(np.abs(A) ** 2))
    return float(np.sqrt(max(fro2 - float(np.sum(np.abs(ev) ** 2)), 0.0) / fro2)) if fro2 > 0 else 0.0


def diffusion_spectrum(L: np.ndarray) -> dict:
    """Spectrum of the reduced Laplacian L (nu multiplies it in the ODE)."""
    lam = np.linalg.eigvals(L)
    re = np.sort(lam.real)
    return {"eigenvalues": re, "max_imag": float(np.abs(lam.imag).max()), "rho": float(np.abs(lam).max()),
            "slowest_decay": float(-re[-1]), "fastest_decay": float(-re[0]),
            "stiffness_ratio": float(re[0] / re[-1]), "symmetry_defect": float(np.linalg.norm(L - L.T) / np.linalg.norm(L)),
            "henrici_departure": henrici_departure(L)}


def transient_amplification(J: np.ndarray, t_max: float = 1.0, n: int = 40) -> float:
    ts = np.geomspace(1e-4, t_max, n)
    return float(max(np.linalg.norm(scipy.linalg.expm(J * t), 2) for t in ts))


def jacobian_diagnostics(Q: np.ndarray, L: np.ndarray, nu: float, a: np.ndarray, transient: bool = False) -> dict:
    J = quadratic_jacobian(Q, nu * L, a)
    Jn = quadratic_jacobian(Q, 0.0 * L, a)
    ev = np.linalg.eigvals(J)
    out = {"spectral_abscissa": float(ev.real.max()),
           "numerical_abscissa": float(np.linalg.eigvalsh(0.5 * (J + J.T)).max()),
           "spectral_radius": float(np.abs(ev).max()), "max_abs_imag": float(np.abs(ev.imag).max()),
           "henrici_departure": henrici_departure(J), "nonlinear_part_norm": float(np.linalg.norm(Jn, 2)),
           "diffusion_part_norm": float(nu * np.linalg.norm(L, 2))}
    if transient:
        out["transient_amplification"] = transient_amplification(J)
    return out
