"""Time-step policies for the explicit-RK4 reduced models, and the stability tools behind them.

POLICIES (selected with `timestep_policy=` in src.pod_rom / src.deim; an explicit `dt` overrides both):
  "reduced"    (DEFAULT) dt = min(FOM advective dt, 0.5 * 2.785 / (nu * rho(K))), K the reduced diffusion
               operator of the actual model (`revised_dt` / `reduced_dt` below).
  "historical" the original full-grid rule, kept bit-identical for reproducing published numbers.

The reduced models (src.pod_rom.rom_predict, src.deim.deim_rom_predict) integrate
    da/dt = nu K a + N(a)                                   (K: reduced diffusion, N: quadratic)
with classical explicit RK4, whose stability region contains the negative real axis up to
z = -2.7853 and the imaginary axis up to |z| = 2.8284.

HISTORICAL RULE (policy "historical"; see `historical_dt`):

    dt = min( 0.25 dx / (max|u0| + 1e-8),  0.4 dx^2 / (2 nu + 1e-12) ),    n_steps = max(80, ceil(T/dt))

It was copied from the full-grid solver's reasoning, not derived for the reduced systems:
  * 0.25 dx / max|u0| is an advective CFL number for the full grid (same as src.solver);
  * 0.4 dx^2 / (2 nu) is the classic explicit-diffusion bound of a full-grid method. Relative to the
    RK4 limit it is 71% of the limit for the full-grid SPECTRAL Laplacian (eigenvalue nu (pi/dx)^2)
    and 29% of the limit for a full-grid 3-point FD Laplacian (4 nu / dx^2). It contains dx, the
    resolution of the FULL grid, although a rank-r reduced system has only r eigenvalues.
It was added in commit 8fdbca0 after the reduced ODE produced NaN at nu = 0.15 with a ROM built from
10 training trajectories; no stability analysis of the reduced system accompanied it.

DERIVED DIAGNOSTIC (`reduced_diffusion_matrix_*`, `diffusion_dt`): the diffusion part of each reduced
system is LINEAR in a, nu K a, with K an r x r matrix computable offline from the basis. Its spectral
radius rho(K) replaces the full-grid 2/dx^2 in the diffusion bound. The "reduced" policy (the default)
uses it with a safety factor of 0.5 of the RK4 real-axis limit (a fixed, a-priori constant, not tuned).
"""

import numpy as np

RK4_REAL_LIMIT = 2.785293  # largest |z| on the negative real axis inside RK4's stability region
RK4_IMAG_LIMIT = 2.828427  # largest |z| on the imaginary axis (= 2 sqrt 2)
DEFAULT_SAFETY = 0.5
POLICIES = ("reduced", "historical")
DEFAULT_POLICY = "reduced"


def check_policy(policy: str) -> str:
    """Validate a `timestep_policy` name (fails clearly rather than silently falling back)."""
    if policy not in POLICIES:
        raise ValueError(f"unknown timestep_policy {policy!r}; valid policies are {POLICIES}")
    return policy


def historical_dt(u0_max, Nx: int, nu: float, L: float = 2 * np.pi) -> float:
    """The historical reduced-model step size (identical formula, incl. dtype behaviour, to
    src.pod_rom.rom_reduced_trajectory / src.deim.deim_rom_reduced_trajectory)."""
    dx = L / Nx
    return min(0.25 * dx / (u0_max + 1e-8), 0.4 * dx**2 / (2 * nu + 1e-12))


def fom_advective_dt(u0_max, Nx: int, L: float = 2 * np.pi) -> float:
    """The FOM's step size (src.solver): advective bound only; diffusion is exact there."""
    return 0.25 * (L / Nx) / (u0_max + 1e-8)


def historical_rule_fractions_of_rk4_limit() -> dict:
    """How the historical diffusion constant sits relative to the RK4 real-axis limit, for the two
    full-grid Laplacians it could have been derived from (returns fractions of that limit)."""
    # dt_hist = 0.4 dx^2/(2 nu); limit dt = RK4_REAL_LIMIT / (nu * lam_max_grid)
    spectral = 0.4 / (2) * (np.pi**2) / RK4_REAL_LIMIT      # lam_max = (pi/dx)^2
    fd3 = 0.4 / (2) * 4.0 / RK4_REAL_LIMIT                  # lam_max = 4/dx^2
    return {"vs_fullgrid_spectral_laplacian": float(spectral), "vs_fullgrid_fd3_laplacian": float(fd3)}


# ------------------------------------------------------------------ reduced diffusion operators
def reduced_diffusion_matrix_rom(Phi: np.ndarray, k: np.ndarray) -> np.ndarray:
    """K (r x r) with the plain ROM's diffusion term equal to nu * K @ a:
    K = Phi^T D2 Phi, D2 the spectral second derivative used in src.pod_rom._burgers_rhs_physical."""
    D2Phi = np.real(np.fft.ifft(-(k**2)[:, None] * np.fft.fft(Phi, axis=0), axis=0))
    return Phi.T @ D2Phi


def reduced_diffusion_matrix_deim(Phi: np.ndarray, M: np.ndarray, points: np.ndarray, Nx: int,
                                  L: float = 2 * np.pi, stencil_order: int = 2) -> np.ndarray:
    """K (r x r) with the DEIM/local-FD model's diffusion term equal to nu * K @ a:
    K = M @ FD2_at_points(Phi_support)  -- the same 3-point (order 2) or 5-point (order 4) stencil
    src.deim uses."""
    from src.deim import _stencil_support
    hw = 1 if stencil_order == 2 else 2
    support, neigh = _stencil_support(points, Nx, hw)
    idx = np.searchsorted(support, neigh)
    U = Phi[support, :]
    dx = L / Nx
    if stencil_order == 2:
        fd2 = (U[idx[:, 2]] - 2 * U[idx[:, 1]] + U[idx[:, 0]]) / dx**2  # (m, r)
    else:
        fd2 = (-U[idx[:, 4]] + 16 * U[idx[:, 3]] - 30 * U[idx[:, 2]] + 16 * U[idx[:, 1]]
               - U[idx[:, 0]]) / (12 * dx**2)
    return M @ fd2


def spectral_radius(K: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(K))))


def diffusion_dt(nu: float, K: np.ndarray, safety: float = DEFAULT_SAFETY) -> float:
    """Largest dt keeping dt * nu * rho(K) at `safety` times the RK4 real-axis limit."""
    return safety * RK4_REAL_LIMIT / (nu * spectral_radius(K) + 1e-300)


def reduced_dt(u0_max, Nx: int, nu: float, rho: float, safety: float = DEFAULT_SAFETY) -> float:
    """The canonical reduced-model step: the FOM's advective bound, tightened only if the reduced
    diffusion operator (spectral radius `rho`) demands it. No tuned constant: `safety` = 0.5 of the
    RK4 real-axis stability limit."""
    return min(fom_advective_dt(u0_max, Nx), safety * RK4_REAL_LIMIT / (nu * rho + 1e-300))


def revised_dt(u0_max, Nx: int, nu: float, K: np.ndarray, safety: float = DEFAULT_SAFETY) -> float:
    """`reduced_dt` from the reduced diffusion matrix K (rho = its spectral radius)."""
    return reduced_dt(u0_max, Nx, nu, spectral_radius(K), safety)


# ------------------------------------------------------------------------- RK4 stability tools
def rk4_amplification(z: np.ndarray) -> np.ndarray:
    return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def rk4_max_stable_dt(eigs: np.ndarray, dt_max: float = 10.0, n: int = 6000) -> float:
    """Largest dt (on a geometric grid, first crossing) with max_i |R(dt * lambda_i)| <= 1 for the
    frozen-Jacobian eigenvalues `eigs`. 0.0 if some eigenvalue has positive real part (the frozen
    linearisation is unstable for every dt)."""
    eigs = np.asarray(eigs, dtype=complex)
    if np.max(eigs.real) > 1e-9 * max(1.0, np.max(np.abs(eigs))):
        return 0.0
    grid = np.geomspace(1e-7, dt_max, n)
    bad = np.max(np.abs(rk4_amplification(grid[:, None] * eigs[None, :])), axis=1) > 1.0 + 1e-12
    if not bad.any():
        return float(dt_max)
    first = int(np.argmax(bad))
    return float(grid[max(first - 1, 0)])


def numerical_jacobian(rhs, a: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of a reduced right-hand side (r x r); the RHS is quadratic in a,
    so central differences are exact up to rounding."""
    r = a.size
    J = np.empty((r, r))
    for j in range(r):
        step = h * max(1.0, abs(a[j]))
        e = np.zeros(r)
        e[j] = step
        J[:, j] = (rhs(a + e) - rhs(a - e)) / (2 * step)
    return J
