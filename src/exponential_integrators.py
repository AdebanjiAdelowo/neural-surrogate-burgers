"""Matrix functions and exponential / integrating-factor integrators for the tensorial POD system

    da/dt = nu L a + Q(a, a)        (L = Phi^T D2 Phi symmetric negative definite, src/tensorial_pod.py)

Eigenbasis. L = V Lambda V^T with V orthogonal (L is symmetric, checked at construction; otherwise the
general-matrix path is used). In b = V^T a the system is db/dt = D b + Qt(b, b) with D = nu Lambda diagonal
and Qt_ijk = sum V_pi Q_pqr V_qj V_rk, so exp(h nu L) and phi_k(h nu L) are diagonal and the 2-norm of b
equals that of a (and of the reconstructed field).

phi-functions. phi_0(z) = e^z, phi_{k+1}(z) = (phi_k(z) - 1/k!) / z, phi_k(0) = 1/k!. Near z = 0 the
recurrence cancels catastrophically; there a Taylor series phi_k(z) = sum_m z^m / (m + k)! is used. For
non-diagonalizable or nonnormal matrices `phi_matrix` evaluates phi_1..phi_k(A) from one exponential of the
augmented block matrix [[A, I, 0..], [0, 0, I, ..], ..] (no division by eigenvalues).

Integrators (all in the eigen-coordinates; outputs land exactly on the requested times):
  * `LawsonDP54`: integrating-factor (Lawson) Dormand-Prince 5(4) with the frame reset every step, written
    with combined exponents e^{D (c_i - c_j) h} (all <= 0, so nothing overflows; the GLOBAL transform
    e^{-nu L t} a overflows double precision once nu rho(L) t > 709). Embedded error estimate mapped to
    the physical coefficients a.
  * `ETDRK4Krogstad`: Krogstad's exponential RK4 (fixed step), with `adaptive_etd` adding step-doubling
    error control (estimate (y_{h/2,h/2} - y_h) / (2^4 - 1), the classical order of the method).
"""

import math

import numpy as np
import scipy.linalg

N_TAYLOR = 25       # Taylor terms for |z| < 1: error below 1/(25 + k)! ~ 1e-25
TAYLOR_RADIUS = 1.0


# ------------------------------------------------------------------------------- phi functions
def phi(k: int, z: np.ndarray) -> np.ndarray:
    """phi_k(z) elementwise for real or complex z (k >= 0), stable near 0."""
    z = np.asarray(z)
    if k == 0:
        return np.exp(z)
    small = np.abs(z) < TAYLOR_RADIUS
    out = np.empty(z.shape, dtype=np.result_type(z, float))
    if small.any():
        zs = z[small]
        acc = np.zeros_like(zs, dtype=out.dtype)
        for m in range(N_TAYLOR - 1, -1, -1):  # Horner: sum z^m / (m+k)!
            acc = acc * zs + 1.0 / math.factorial(m + k)
        out[small] = acc
    if (~small).any():
        zl = z[~small]
        p = np.exp(zl)
        for j in range(k):
            p = (p - 1.0 / math.factorial(j)) / zl
        out[~small] = p
    return out


def phi_matrix(A: np.ndarray, k: int):
    """[phi_1(A), ..., phi_k(A)] for a general square matrix via one exponential of the (k+1) n block
    matrix [[A, I, 0, ...], [0, 0, I, ...], ..., [0, ..., 0]] (Saad 1992 / Sidje 1998 augmentation)."""
    n = A.shape[0]
    big = np.zeros(((k + 1) * n, (k + 1) * n), dtype=np.result_type(A, float))
    big[:n, :n] = A
    for j in range(k):
        big[j * n:(j + 1) * n, (j + 1) * n:(j + 2) * n] = np.eye(n)
    E = scipy.linalg.expm(big)
    return [E[:n, (j + 1) * n:(j + 2) * n] for j in range(k)]


# ------------------------------------------------------------------------------- eigen-coordinates
class EigenBasis:
    """One-time preparation for a POD basis (independent of nu): eigendecomposition of the symmetric L and
    the tensor in eigen-coordinates. Raises if L is not symmetric (then use `phi_matrix`)."""

    def __init__(self, Q: np.ndarray, L: np.ndarray, sym_tol: float = 1e-10):
        asym = np.linalg.norm(L - L.T) / np.linalg.norm(L)
        if asym > sym_tol:
            raise ValueError(f"L is not symmetric (relative asymmetry {asym:.1e}); use phi_matrix")
        self.lam, self.V = np.linalg.eigh(0.5 * (L + L.T))
        self.r = r = len(self.lam)
        Qt = np.einsum("pi,pqs,qj,sk->ijk", self.V, Q, self.V, self.V, optimize=True)
        self.Qt_flat = np.ascontiguousarray(Qt.reshape(r * r, r))


class EigenSystem:
    """da/dt = nu L a + Q(a, a) rewritten in the eigenbasis of the symmetric L (per trajectory: only
    D = nu Lambda depends on nu; pass a precomputed `basis` to keep the one-time work out of online time)."""

    def __init__(self, Q: np.ndarray, L: np.ndarray, nu: float, sym_tol: float = 1e-10, basis: EigenBasis = None):
        basis = basis if basis is not None else EigenBasis(Q, L, sym_tol)
        self.V, self.lam, self.nu, self.r, self.Qt_flat = basis.V, basis.lam, nu, basis.r, basis.Qt_flat
        self.D = nu * self.lam                              # diagonal linear part (all < 0 here)

    def N(self, b):
        r = self.r
        return (self.Qt_flat @ b).reshape(r, r) @ b

    def to_b(self, a):
        return self.V.T @ a

    def to_a(self, b):
        return b @ self.V.T if b.ndim == 2 else self.V @ b


# ------------------------------------------------------------------------------- error norm
def error_norm(e_a: np.ndarray, a_old: np.ndarray, a_new: np.ndarray, rtol: float, atol) -> float:
    """sqrt(mean((e_i / (atol_i + rtol max(|a_i^n|, |a_i^{n+1}|)))^2)) on the PHYSICAL coefficients."""
    sc = atol + rtol * np.maximum(np.abs(a_old), np.abs(a_new))
    return float(np.sqrt(np.mean((e_a / sc) ** 2)))


# ------------------------------------------------------------------------------- Lawson DP5(4)
_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
_A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
      [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


class LawsonDP54:
    """Integrating-factor Dormand-Prince 5(4). One step from b_n with step h:
         Y_i = e^{D c_i h} b_n + h sum_{j<i} a_ij e^{D (c_i - c_j) h} N(Y_j)
         b_{n+1} = e^{D h} b_n + h sum_j b_j e^{D (1 - c_j) h} N(Y_j)
       (all exponents <= 0 since c_j <= c_i); the embedded estimate uses the 4th-order weights, and
       FSAL holds (Y_7 = b_{n+1})."""

    order = 5

    def __init__(self, es: EigenSystem):
        self.es = es

    def step(self, b, Nb, h):
        D = self.es.D
        Ns = [Nb]
        for i in range(1, 7):
            y = np.exp(D * _C[i] * h) * b
            for j, aij in enumerate(_A[i]):
                if aij:
                    y = y + h * aij * np.exp(D * (_C[i] - _C[j]) * h) * Ns[j]
            Ns.append(self.es.N(y))
        b_new = y                                   # stage 7 equals the 5th-order solution (FSAL)
        err = h * sum((_B5[j] - _B4[j]) * np.exp(D * (1 - _C[j]) * h) * Ns[j] for j in range(7))
        return b_new, Ns[6], err, 6                  # 6 new N evaluations per attempted step


# ------------------------------------------------------------------------------- Krogstad ETD-RK4
class ETDRK4Krogstad:
    """Krogstad's exponential RK4 for db/dt = D b + N(b) (D diagonal):
         a = e^{hD/2} b + h/2 phi1(hD/2) N(b)
         c1 = a + h phi2(hD/2) (N(a) - N(b))
         c2 = e^{hD} b + h phi1(hD) N(b) + 2h phi2(hD) (N(c1) - N(b))
         b+ = e^{hD} b + h [phi1 N(b) + phi2 (-3N(b) + 2N(a) + 2N(c1) - N(c2))
                            + phi3 (4N(b) - 4N(a) - 4N(c1) + 4N(c2))]      (phi_k at hD)
       phi-factors are cached per step size."""

    order = 4

    def __init__(self, es: EigenSystem):
        self.es = es
        self._cache = {}

    def _coef(self, h):
        c = self._cache.get(h)
        if c is None:
            z, z2 = h * self.es.D, 0.5 * h * self.es.D
            c = (np.exp(z2), phi(1, z2), phi(2, z2), np.exp(z), phi(1, z), phi(2, z), phi(3, z))
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[h] = c
        return c

    def step(self, b, Nb, h):
        e2, p1h, p2h, e1, p1, p2, p3 = self._coef(h)
        N = self.es.N
        a = e2 * b + 0.5 * h * p1h * Nb
        Na = N(a)
        c1 = a + h * p2h * (Na - Nb)
        Nc1 = N(c1)
        c2 = e1 * b + h * p1 * Nb + 2 * h * p2 * (Nc1 - Nb)
        Nc2 = N(c2)
        b_new = e1 * b + h * (p1 * Nb + p2 * (-3 * Nb + 2 * Na + 2 * Nc1 - Nc2)
                              + p3 * (4 * Nb - 4 * Na - 4 * Nc1 + 4 * Nc2))
        return b_new, 3  # 3 new N evaluations (N(b) supplied)
