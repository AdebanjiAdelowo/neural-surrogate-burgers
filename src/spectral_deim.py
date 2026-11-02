"""Spectral-consistent DEIM for the pseudo-spectral POD-Galerkin Burgers ROM.

Unlike src/deim.py (DEIM interpolation of values computed by LOCAL finite differences at the DEIM
points), the sampled values here are exactly those of the solver's own discrete spectral operator,
P^T f(Phi a), obtained without any online FFT through a precomputed sampled quadratic tensor.

Derivation. With N(u) = G (u * u) (G the dealiased spectral -(1/2) d/dx, see src/tensorial_pod.py)
and u = Phi a,

    [P^T N(Phi a)]_p = sum_{j,k} C_pjk a_j a_k,     C_pjk = [G (phi_j * phi_k)]_{x_p},
    [P^T D2 Phi a]_p = sum_j (D_s)_pj a_j,           D_s = P^T D2 Phi.

C (s x r x r) and D_s (s x r) are built offline by applying the existing FFT operators to the mode
products / modes on the full grid and keeping the s sampled rows: they reproduce the sampled
full-order operator to rounding (tests/test_spectral_deim.py).

DEIM / gappy-POD reconstruction with a basis Psi (Nx x m) and s >= m sample points P:

    N(u) ~ Psi (P^T Psi)^+ P^T N(u),      M = Phi^T Psi (P^T Psi)^+   (r x s, offline),

with the pseudo-inverse from a least-squares solve (never the normal equations); for s = m it is
(P^T Psi)^-1 and this is classical DEIM.

Two explicit choices of what DEIM approximates (`target`), kept fixed per experiment:
  "nonlinear"  DEIM on N only (Psi from N-snapshots); diffusion exact in reduced coordinates:
                   da/dt = M C(a, a) + nu L a,                L = Phi^T D2 Phi.
  "full"       DEIM on the complete RHS f = N + nu D2 u (Psi from f-snapshots, the choice made by
               src/deim.py), so the diffusion is interpolated as well:
                   da/dt = M [C(a, a) + nu D_s a].
Both are quadratic reduced models; the online cost is O(s r^2 + r s), and no full-grid array or
FFT is used online. `folded_tensor` gives the equivalent r x r x r tensor M C (useful for
analysis: a DEIM model of a quadratic term is a tensorial model with an approximated tensor).

Point selection: classical greedy DEIM (src.deim.select_deim_points), Q-DEIM (column-pivoted QR of
Psi^T, Drmac & Gugercin 2016), and two deterministic oversampling rules for s > m (`select_points`).
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from src.deim import select_deim_points
from src.tensorial_pod import (apply_nonlinear_operator, integrate_reduced, mode_products_image,
                               reduced_step_count)
from src.timestep import DEFAULT_POLICY, reduced_diffusion_matrix_rom, spectral_radius

TARGETS = ("nonlinear", "full")
SELECTIONS = ("deim", "qdeim", "qdeim+greedy", "qdeim_extended")


# ------------------------------------------------------------------ snapshots and bases
def nonlinear_snapshots(u_ensemble: np.ndarray, k: np.ndarray, dealias: np.ndarray) -> np.ndarray:
    """N(u) = G(u * u) for each row of u_ensemble (n, Nx): the nonlinear term alone (no viscosity)."""
    U = np.asarray(u_ensemble, dtype=np.float64).T
    return apply_nonlinear_operator(U * U, k, dealias).T


def build_basis(snapshots: np.ndarray, m: int):
    """Leading m left singular vectors of the (n, Nx) snapshot matrix, and all singular values."""
    U, S, _ = np.linalg.svd(np.asarray(snapshots, dtype=np.float64).T, full_matrices=False)
    return U[:, :m], S


# ------------------------------------------------------------------ point selection
def qdeim_points(Psi: np.ndarray) -> np.ndarray:
    """Q-DEIM: the first m column pivots of a column-pivoted QR of Psi^T (m x Nx)."""
    m = Psi.shape[1]
    _, _, piv = scipy.linalg.qr(np.asarray(Psi).T, mode="economic", pivoting=True)
    return np.asarray(piv[:m], dtype=int)


def greedy_oversample(Psi: np.ndarray, points: np.ndarray, s: int) -> np.ndarray:
    """Add points one at a time, each maximising sigma_min(Psi[points, :]) (the quantity that
    controls the gappy-POD error constant ||(P^T Psi)^+||_2 = 1 / sigma_min). Exhaustive search over
    the remaining grid points; ties go to the lowest index (np.argmax), so the rule is deterministic."""
    pts = list(int(p) for p in points)
    Nx = Psi.shape[0]
    while len(pts) < s:
        best, best_val = -1, -1.0
        for p in range(Nx):
            if p in pts:
                continue
            val = np.linalg.svd(Psi[pts + [p], :], compute_uv=False)[-1]
            if val > best_val:
                best, best_val = p, val
        pts.append(best)
    return np.array(pts, dtype=int)


def select_points(Psi: np.ndarray, s: int = None, method: str = "qdeim",
                  Psi_extended: np.ndarray = None) -> np.ndarray:
    """Sample points for a DEIM basis Psi (Nx x m), s >= m of them.

    deim            classical greedy DEIM (s must equal m);
    qdeim           Q-DEIM (s must equal m);
    qdeim+greedy    Q-DEIM's m points, then `greedy_oversample` up to s;
    qdeim_extended  Q-DEIM pivots of the first s columns of a LARGER snapshot basis Psi_extended
                    (s points from s modes), used with the m-column Psi for reconstruction.
    """
    m = Psi.shape[1]
    s = m if s is None else int(s)
    if s < m:
        raise ValueError(f"need s >= m sample points (s={s}, m={m})")
    if method in ("deim", "qdeim") and s != m:
        raise ValueError(f"{method} selects exactly m points; use an oversampling method for s > m")
    if method == "deim":
        return select_deim_points(Psi)
    if method == "qdeim":
        return qdeim_points(Psi)
    if method == "qdeim+greedy":
        return greedy_oversample(Psi, qdeim_points(Psi), s)
    if method == "qdeim_extended":
        if Psi_extended is None or Psi_extended.shape[1] < s:
            raise ValueError("qdeim_extended needs Psi_extended with at least s columns")
        return qdeim_points(Psi_extended[:, :s])
    raise ValueError(f"unknown selection {method!r}; valid: {SELECTIONS}")


def interpolation_pinv(Psi: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(P^T Psi)^+ (m x s) by a least-squares solve (SVD-based LAPACK gelsd), not normal equations."""
    PtPsi = np.asarray(Psi, dtype=np.float64)[points, :]
    return np.linalg.lstsq(PtPsi, np.eye(len(points)), rcond=None)[0]


def interpolation_conditioning(Psi: np.ndarray, points: np.ndarray) -> dict:
    sv = np.linalg.svd(np.asarray(Psi, dtype=np.float64)[points, :], compute_uv=False)
    return {"cond": float(sv[0] / sv[-1]), "pinv_norm": float(1.0 / sv[-1]), "sigma_min": float(sv[-1])}


# ------------------------------------------------------------------ sampled spectral operators
def sampled_quadratic_tensor(Phi: np.ndarray, points: np.ndarray, k: np.ndarray,
                             dealias: np.ndarray) -> np.ndarray:
    """C (s, r, r): rows `points` of G(phi_j * phi_k), built with the solver's FFT operator."""
    return np.ascontiguousarray(mode_products_image(Phi, k, dealias)[points])


def sampled_diffusion(Phi: np.ndarray, points: np.ndarray, k: np.ndarray) -> np.ndarray:
    """D_s (s, r): rows `points` of D2 Phi (spectral second derivative of each mode)."""
    Phi = np.asarray(Phi, dtype=np.float64)
    D2Phi = np.real(np.fft.ifft(-(k**2)[:, None] * np.fft.fft(Phi, axis=0), axis=0))
    return np.ascontiguousarray(D2Phi[points])


@dataclass
class SpectralDEIM:
    """Offline operators of a spectral-consistent DEIM model."""
    target: str
    points: np.ndarray       # (s,) sample points
    M: np.ndarray            # (r, s) Phi^T Psi (P^T Psi)^+
    C: np.ndarray            # (s, r, r) sampled quadratic tensor
    Lin: np.ndarray          # (r, r) linear operator multiplying nu: L ("nonlinear") or M D_s ("full")
    Nx: int
    diffusion_radius: float  # rho(Lin), for the "reduced" step-size policy
    conditioning: dict

    @property
    def r(self) -> int:
        return self.M.shape[0]

    @property
    def s(self) -> int:
        return self.M.shape[1]

    @property
    def C_flat(self) -> np.ndarray:
        return self.C.reshape(self.s * self.r, self.r)


def build_spectral_deim(Phi: np.ndarray, Psi: np.ndarray, points: np.ndarray, k: np.ndarray,
                        dealias: np.ndarray, target: str = "nonlinear") -> SpectralDEIM:
    """Offline stage. Psi must be a basis for the quantity named by `target` (N-snapshots for
    "nonlinear", full-RHS snapshots for "full")."""
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; valid: {TARGETS}")
    points = np.asarray(points, dtype=int)
    Phi64 = np.asarray(Phi, dtype=np.float64)
    M = (Phi64.T @ np.asarray(Psi, dtype=np.float64)) @ interpolation_pinv(Psi, points)
    C = sampled_quadratic_tensor(Phi, points, k, dealias)
    if target == "nonlinear":
        # same matrix as the plain ROM, in float64 (see src.tensorial_pod.build_tensorial_rom), and
        # the same step-size radius as src.pod_rom computes on the basis as given
        Lin = reduced_diffusion_matrix_rom(Phi64, k)
        radius = spectral_radius(reduced_diffusion_matrix_rom(Phi, k))
    else:
        Lin = M @ sampled_diffusion(Phi, points, k)
        radius = spectral_radius(Lin)
    return SpectralDEIM(target=target, points=points, M=M, C=C, Lin=Lin, Nx=Phi.shape[0],
                        diffusion_radius=radius,
                        conditioning=interpolation_conditioning(Psi, points))


def sampled_nonlinear(model: SpectralDEIM, a: np.ndarray) -> np.ndarray:
    """P^T N(Phi a) from the sampled tensor: (s,) values, no FFT."""
    r, s = model.r, model.s
    return (model.C_flat @ a).reshape(s, r) @ a


def deim_rhs_fn(model: SpectralDEIM, nu: float):
    """Online RHS a -> M C(a, a) + nu Lin a (no FFT, no full-grid array)."""
    r, s = model.r, model.s
    C_flat, M, Lnu = model.C_flat, model.M, nu * model.Lin

    def rhs(a):
        return M @ ((C_flat @ a).reshape(s, r) @ a) + Lnu @ a
    return rhs


def spectral_deim_rhs(model: SpectralDEIM, a: np.ndarray, nu: float) -> np.ndarray:
    return deim_rhs_fn(model, nu)(a)


def folded_tensor(model: SpectralDEIM) -> np.ndarray:
    """The equivalent (r, r, r) quadratic tensor M C of the DEIM model."""
    return np.einsum("is,sjk->ijk", model.M, model.C)


def spectral_deim_reduced_trajectory(a0: np.ndarray, u0_max, model: SpectralDEIM, nu: float, T: float,
                                     n_save: int, dt: float = None, align_snapshots: bool = False,
                                     min_steps: int = None, timestep_policy: str = DEFAULT_POLICY):
    """Solver-only online path. "reduced" step policy: rho(Lin), the model's own diffusion operator
    (equal to the plain ROM's for target "nonlinear").

    Returns:
        t_save: (n_save,), a_save: (n_save, r), n_steps.
    """
    n_steps, dt = reduced_step_count(u0_max, model.Nx, nu, T, n_save, dt, align_snapshots,
                                     min_steps, timestep_policy, model.diffusion_radius)
    t_save, a_save = integrate_reduced(deim_rhs_fn(model, nu), a0, n_steps, dt, n_save)
    return t_save, a_save, n_steps


def spectral_deim_predict(u0: np.ndarray, Phi: np.ndarray, model: SpectralDEIM, nu: float, T: float,
                          n_save: int, dt: float = None, align_snapshots: bool = False,
                          min_steps: int = None, timestep_policy: str = DEFAULT_POLICY):
    """End-to-end prediction (projection, integration, reconstruction)."""
    a0 = Phi.T @ u0
    t_save, a_save, _ = spectral_deim_reduced_trajectory(a0, np.abs(u0).max(), model, nu, T, n_save,
                                                         dt, align_snapshots, min_steps,
                                                         timestep_policy)
    return t_save, (a_save @ Phi.T).astype(np.float32)
