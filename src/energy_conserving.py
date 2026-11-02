"""Quadratic-form analysis and an energy-conserving correction of quadratic reduced Burgers models.

Every reduced model in this repository is a quadratic ODE

    da/dt = Q(a, a) + nu K a,        Q(a, a)_i = sum_{j,k} Q_ijk a_j a_k,

  * tensorial POD-Galerkin (src/tensorial_pod.py):  Q exact, K = L = Phi^T D2 Phi;
  * spectral DEIM (src/spectral_deim.py):           Q = M C (folded), K = L or M D_s;
  * hybrid DEIM/local-FD (src/deim.py):             Q, K from the FD stencil rows (`fold_hybrid_localfd`;
    -u_c (u_r - u_l) / (2 dx) is a product of two linear functions of a, so this is exact too).

Energy. With orthonormal Phi, the reduced energy is E = |a|^2 / 2 (the discrete L2 energy of Phi a
up to the factor dx), and

    dE/dt = a^T Q(a, a) + nu a^T K a.

The continuous convective term conserves energy (int u * u u_x dx = 0), so the energy can only grow
through (i) the cubic form a^T Q(a, a), which depends only on the FULLY SYMMETRIZED part Sym(Q), and
(ii) the symmetric part of K if it is not negative semidefinite. For the exact Galerkin model K = L is
symmetric negative semidefinite, and Sym(Q) is non-zero only through the aliasing error of the
pseudo-spectral product (measured, not assumed, in the study).

Correction. `energy_conserving_tensor(Q) = Q - Sym(Q)` is the orthogonal projection (Frobenius inner
product) of Q onto the tensors whose cubic form vanishes identically: the nearest tensor that has the
continuous energy-conservation property, keeping the (j, k) symmetry. It changes only the energy-
producing part of Q and adds no dissipation. If in addition the symmetric part of K is negative
semidefinite, dE/dt = nu a^T K a <= 0 for every state: the continuous-time reduced dynamics cannot gain
energy, whatever the approximation error elsewhere in Q.
"""

import itertools

import numpy as np

from src.deim import _stencil_support
from src.tensorial_pod import TensorialROM
from src.timestep import spectral_radius

L_DOMAIN = 2 * np.pi

# central-difference weights over offsets -h..h (h = 1 for order 2, h = 2 for order 4), as in src/deim.py
_FD_WEIGHTS = {2: (np.array([-0.5, 0.0, 0.5]), np.array([1.0, -2.0, 1.0])),
               4: (np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0,
                   np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0)}


def symmetrized_part(Q: np.ndarray) -> np.ndarray:
    """Full symmetrization over all 6 index permutations."""
    return sum(np.transpose(Q, p) for p in itertools.permutations(range(3))) / 6.0


def energy_conserving_tensor(Q: np.ndarray) -> np.ndarray:
    """Q - Sym(Q): the Frobenius-nearest tensor with a^T Q(a, a) = 0 for all a."""
    return Q - symmetrized_part(Q)


def weighted_energy_conserving_tensor(Q: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Minimum-change energy-conserving tensor in the norm sum_ijk (sigma_j sigma_k)^2 dQ_ijk^2.

    `energy_conserving_tensor` is optimal in the plain Frobenius norm, which ignores how much each
    entry matters: Q_ijk multiplies a_j a_k, and the reduced coordinates of a POD basis differ in size
    by orders of magnitude. With sigma_i the typical (RMS) size of a_i, (sigma_j sigma_k)^2 dQ_ijk^2 is
    the expected squared change of the RHS for states with independent components of that size, so
    this correction moves energy-producing error away from the entries that dominate the dynamics.

    The constraint Sym(Q') = 0 splits into one linear equation per index multiset {i, j, k}: the sum
    of Q' over the distinct permutations of (i, j, k) vanishes. Each is solved in closed form
    (Lagrange multiplier). With uniform sigma this reduces to `energy_conserving_tensor`; (j, k)
    symmetry is preserved because the two entries swapped by it carry the same weight."""
    sigma = np.asarray(sigma, dtype=np.float64)
    r = Q.shape[0]
    Qn = np.array(Q, dtype=np.float64, copy=True)
    for orbit in itertools.combinations_with_replacement(range(r), 3):
        entries = sorted(set(itertools.permutations(orbit)))
        w = np.array([(sigma[e[1]] * sigma[e[2]]) ** 2 for e in entries])
        q = np.array([Qn[e] for e in entries])
        q_new = q - (q.sum() / np.sum(1.0 / w)) / w
        for e, v in zip(entries, q_new):
            Qn[e] = v
    return Qn


def symmetrize_jk(Q: np.ndarray) -> np.ndarray:
    """Same quadratic form, symmetric in its last two indices."""
    return 0.5 * (Q + Q.transpose(0, 2, 1))


def energy_production(Q: np.ndarray, a: np.ndarray) -> float:
    """a^T Q(a, a): the rate at which the quadratic term changes |a|^2 / 2."""
    return float(np.einsum("ijk,i,j,k->", Q, a, a, a))


def energy_production_ratio(Q: np.ndarray) -> float:
    """||Sym(Q)||_F / ||Q||_F: size of the energy-producing part relative to the whole tensor."""
    return float(np.linalg.norm(symmetrized_part(Q)) / np.linalg.norm(Q))


def max_symmetric_eig(K: np.ndarray) -> float:
    """Largest eigenvalue of (K + K^T)/2: the linear term's worst-case energy growth rate per unit nu."""
    return float(np.linalg.eigvalsh(0.5 * (K + K.T)).max())


def quadratic_model(Q: np.ndarray, K: np.ndarray, Nx: int) -> TensorialROM:
    """A quadratic model integrable with src.tensorial_pod's trajectory/predict functions (the
    "reduced" step policy uses rho(K) of THIS model's linear operator)."""
    return TensorialROM(Q=np.ascontiguousarray(Q), L=K, Nx=Nx, diffusion_radius=spectral_radius(K))


def fold_hybrid_localfd(Phi: np.ndarray, M: np.ndarray, points: np.ndarray, Nx: int,
                        stencil_order: int = 2):
    """(Q, K) of src/deim.py's DEIM/local-FD model:  M [ -u_c * (D1 u)_p + nu (D2 u)_p ] with u = Phi a
    on the stencil rows. Q is returned symmetrized in (j, k); its quadratic form, and so the RHS,
    equals src.deim's exactly (tests/test_energy_conserving.py)."""
    hw = 1 if stencil_order == 2 else 2
    support, neigh = _stencil_support(np.asarray(points), Nx, hw)
    rows = np.asarray(Phi, dtype=np.float64)[neigh]          # (m, 2hw+1, r)
    dx = L_DOMAIN / Nx
    w1, w2 = _FD_WEIGHTS[stencil_order]
    D1 = np.einsum("o,por->pr", w1, rows) / dx                 # (m, r) first derivative rows
    D2 = np.einsum("o,por->pr", w2, rows) / dx**2              # (m, r) second derivative rows
    centre = rows[:, hw, :]                                    # (m, r) u at the DEIM points
    C = -centre[:, :, None] * D1[:, None, :]                   # (m, r, r)
    M64 = np.asarray(M, dtype=np.float64)
    return symmetrize_jk(np.einsum("ip,pjk->ijk", M64, C)), M64 @ D2
