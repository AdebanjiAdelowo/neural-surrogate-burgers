"""A hybrid DEIM/local finite-difference (FD) hyper-reduction scheme for the pseudo-spectral
POD-Galerkin Burgers ROM.

Research-extension module (see RESEARCH_EXTENSION.md, Section 3 "What is mathematically
implemented, precisely" for the full derivation and terminology justification). Not part of the
original verified MVP (src/solver.py, src/pod_rom.py, src/surrogate_net.py are untouched by this
file).

Precise terminology (reviewed): the DEIM interpolatory projection itself (point selection via
select_deim_points, the reduction operator M in build_deim_projector) is exact, standard
DEIM-Galerkin math (verified against Chaturantabut & Sorensen 2010, Algorithm 1). What is NOT
exact is the nonlinear-term VALUE fed into that projection at the interpolation points -- see
below. Calling this module "DEIM hyper-reduction" alone would overstate precision; "hybrid
DEIM/local-FD hyper-reduction" is accurate.

Why this is not a mechanical drop-in of textbook DEIM
-------------------------------------------------------
The classical DEIM speed-up (Chaturantabut & Sorensen, 2010, SIAM J. Sci. Comput. 32(5)) assumes
the full-order nonlinear term F(u) can be evaluated at m << Nx selected grid points *without*
touching the rest of the grid -- true for a finite-difference/finite-element discretization, where
each grid point's contribution to F depends only on a few local neighbours. The seminal paper's own
test problems (1D FitzHugh-Nagumo, 2D miscible viscous fingering) are both finite-difference
discretized; a literature search (ten query combinations, see RESEARCH_EXTENSION.md Section 4) did
not identify a directly matching implementation combining DEIM with a genuinely global (FFT-based)
nonlinear-term evaluation. The closest precedent found (Mizan, Olshanskii & Timofeyev 2026,
arXiv:2502.02718, POD-DEIM for the periodic Kuramoto-Sivashinsky equation) explicitly uses a
finite-difference discretization for the nonlinear flux, not pseudo-spectral -- consistent with,
though not proof of, this combination being uncommon.

`src/pod_rom.py`'s nonlinear term is evaluated via FFT-based spectral derivatives
(`_burgers_rhs_physical`), which is a *global* operation: reconstructing u at even a single grid
point formally requires all Fourier coefficients. Naively "evaluating F at m points" does not by
itself avoid the full-grid cost if the derivatives feeding into F are still computed spectrally.

The adaptation implemented here: reduce the *derivative operator* used for the DEIM-point nonlinear
evaluation to a local, second-order central finite-difference stencil restricted to the DEIM
points' immediate neighbours, while the POD basis, the Galerkin projection, and the ground-truth
solver's own spectral method are all left untouched. This achieves genuine O(m) reconstruction +
O(m) local-derivative cost per RK4 stage, independent of Nx -- but it introduces a new source of
error (finite-difference truncation error at the DEIM points, versus the ground-truth spectral
accuracy) that must be validated empirically, not assumed away. See tests/test_deim.py for that
validation (`test_local_fd_nonlinear_matches_spectral_at_deim_points`) and RESEARCH_EXTENSION.md
Section "DEIM validation" for the resulting numbers.
"""

import numpy as np

from src.snapshot_times import align_step_count
from src.timestep import (DEFAULT_POLICY, check_policy, historical_dt, reduced_diffusion_matrix_deim,
                          reduced_dt, spectral_radius)


def collect_nonlinear_snapshots(u_ensemble: np.ndarray, k: np.ndarray, nu_values: np.ndarray,
                                 dealias: np.ndarray) -> np.ndarray:
    """Compute Burgers' RHS F(u) = -u u_x + nu u_xx at each snapshot, via the exact spectral method.

    This is the "nonlinear-term ensemble" DEIM's basis is built from -- computed directly from the
    already-generated state snapshots (u_ensemble), so no change to src/solver.py or
    scripts/generate_dataset.py is needed.

    Args:
        u_ensemble: (n_snapshots, Nx) flattened (example, time) state snapshots.
        k: (Nx,) spectral wavenumbers, matching src.pod_rom's convention.
        nu_values: (n_snapshots,) viscosity for each snapshot (one value per *example*, broadcast
            across that example's time snapshots by the caller).
        dealias: (Nx,) boolean 2/3-rule mask, matching src.pod_rom's convention.

    Returns:
        F_ensemble: (n_snapshots, Nx) nonlinear-term snapshots.
    """
    n, Nx = u_ensemble.shape
    F = np.empty_like(u_ensemble)
    for i in range(n):
        u = u_ensemble[i]
        u_hat = np.fft.fft(u)
        ux = np.real(np.fft.ifft(1j * k * u_hat))
        uxx = np.real(np.fft.ifft(-(k**2) * u_hat))
        flux_hat = np.fft.fft(u * u) * dealias
        nonlinear = np.real(np.fft.ifft(-0.5j * k * flux_hat))  # -(u u_x), dealiased
        F[i] = nonlinear + nu_values[i] * uxx
    return F


def build_deim_basis(F_ensemble: np.ndarray, m: int):
    """Build a rank-m DEIM basis Psi from a nonlinear-term snapshot ensemble (same SVD recipe as
    the state POD basis in src.pod_rom.build_pod_basis, applied to F instead of u)."""
    U, S, _ = np.linalg.svd(F_ensemble.T, full_matrices=False)  # U: (Nx, n_snapshots)
    Psi = U[:, :m]
    return Psi, S


def select_deim_points(Psi: np.ndarray) -> np.ndarray:
    """Greedy DEIM interpolation-point selection (Algorithm 1, Chaturantabut & Sorensen 2010).

    Returns:
        points: (m,) int array of selected grid-point indices, in selection order.
    """
    Nx, m = Psi.shape
    p0 = int(np.argmax(np.abs(Psi[:, 0])))
    points = [p0]
    U = Psi[:, :1]
    for l in range(1, m):
        u_l = Psi[:, l]
        PT_U = U[points, :]          # (l, l)
        PT_ul = u_l[points]          # (l,)
        c = np.linalg.solve(PT_U, PT_ul)
        residual = u_l - U @ c
        p_l = int(np.argmax(np.abs(residual)))
        points.append(p_l)
        U = np.hstack([U, u_l[:, None]])
    return np.array(points, dtype=int)


def build_deim_projector(Phi: np.ndarray, Psi: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Offline: precompute M = Phi^T @ Psi @ (P^T Psi)^{-1}, shape (r, m).

    Online, the reduced RHS becomes  da/dt ~= M @ F(Phi a)[points]  -- F evaluated at only the m
    DEIM points, never on the full Nx grid.
    """
    PT_Psi = Psi[points, :]              # (m, m)
    deim_matrix = Psi @ np.linalg.inv(PT_Psi)   # (Nx, m); Nx-sized only because we also need the
    # points' local neighbourhoods below -- the *evaluation* cost (see deim_rom_predict) never
    # touches the full Nx grid; only this small, offline, one-time (Nx, m) matrix does.
    M = Phi.T @ deim_matrix              # (r, m)
    return M


def _stencil_support(points: np.ndarray, Nx: int, halfwidth: int = 1) -> np.ndarray:
    """Union of each DEIM point's local neighbour indices (periodic), for reconstructing u locally
    without touching the full grid. halfwidth=1 -> standard 3-point central-difference stencil."""
    offsets = np.arange(-halfwidth, halfwidth + 1)
    neighbours = (points[:, None] + offsets[None, :]) % Nx  # (m, 2*halfwidth+1)
    support = np.unique(neighbours.ravel())
    return support, neighbours


def deim_nonlinear_at_points(a: np.ndarray, Phi_support: np.ndarray, support: np.ndarray,
                              neighbours: np.ndarray, nu: float, dx: float) -> np.ndarray:
    """Evaluate F(Phi @ a) at the m DEIM points only, via local 2nd-order central differences.

    Cost: O(support_size * r) to reconstruct u on the (small) stencil support, then O(m) for the
    finite-difference formulas -- independent of Nx.

    Args:
        a: (r,) reduced coordinates.
        Phi_support: (support_size, r) -- POD basis rows restricted to the stencil support
            (precomputed offline once; independent of a).
        support: (support_size,) grid indices covered by Phi_support (for indexing into
            `neighbours`, which is expressed in the same global-index space).
        neighbours: (m, 3) grid indices of each DEIM point's [left, centre, right] neighbours.
        nu: viscosity.
        dx: grid spacing.

    Returns:
        F_at_points: (m,) nonlinear-term values at the m DEIM points.
    """
    u_support = Phi_support @ a  # (support_size,)
    # Map each neighbour's global grid index to its position within `support` (support is sorted).
    idx_in_support = np.searchsorted(support, neighbours)  # (m, 3)
    u_left, u_centre, u_right = (u_support[idx_in_support[:, 0]],
                                  u_support[idx_in_support[:, 1]],
                                  u_support[idx_in_support[:, 2]])
    ux = (u_right - u_left) / (2 * dx)
    uxx = (u_right - 2 * u_centre + u_left) / (dx**2)
    return -u_centre * ux + nu * uxx


def deim_nonlinear_at_points_4th_order(a: np.ndarray, Phi_support: np.ndarray, support: np.ndarray,
                                        neighbours: np.ndarray, nu: float, dx: float) -> np.ndarray:
    """Evaluate F(Phi @ a) at the m DEIM points via local 4TH-order central differences (5-point
    stencil), as an alternative to `deim_nonlinear_at_points`'s 2nd-order (3-point) formula.

    Standard periodic 4th-order central differences:
        u_x  ~= (-u[i+2] + 8 u[i+1] - 8 u[i-1] + u[i-2]) / (12 dx)
        u_xx ~= (-u[i+2] + 16 u[i+1] - 30 u[i] + 16 u[i-1] - u[i-2]) / (12 dx^2)

    Args mirror `deim_nonlinear_at_points`, except `neighbours` must have 5 columns
    (from `_stencil_support(points, Nx, halfwidth=2)`), not 3.

    Rationale (see RESEARCH_EXTENSION.md "Higher-order stencil experiment"): the 2nd-order
    variant's truncation error was found to correlate strongly with trajectory error specifically
    in the steep-gradient (large |u_x|/|u_xx|) regime. A 4th-order stencil has strictly smaller
    truncation error at fixed dx (error ~ O(dx^4) vs O(dx^2)) for smooth-enough fields, so this
    tests directly whether that is the dominant error source or whether some other mechanism
    (e.g. the DEIM-Galerkin projection matrix's own operator norm, see RESEARCH_EXTENSION.md
    "Rank non-monotonicity") explains a comparable share of the error.
    """
    u_support = Phi_support @ a  # (support_size,)
    idx_in_support = np.searchsorted(support, neighbours)  # (m, 5)
    u_m2, u_m1, u_0, u_p1, u_p2 = (u_support[idx_in_support[:, 0]], u_support[idx_in_support[:, 1]],
                                    u_support[idx_in_support[:, 2]], u_support[idx_in_support[:, 3]],
                                    u_support[idx_in_support[:, 4]])
    ux = (-u_p2 + 8 * u_p1 - 8 * u_m1 + u_m2) / (12 * dx)
    uxx = (-u_p2 + 16 * u_p1 - 30 * u_0 + 16 * u_m1 - u_m2) / (12 * dx**2)
    return -u_0 * ux + nu * uxx


def deim_rom_reduced_trajectory(a0: np.ndarray, u0_max, Phi: np.ndarray, M: np.ndarray,
                                 points: np.ndarray, Nx: int, nu: float, T: float, n_save: int,
                                 dt: float = None, stencil_order: int = 2,
                                 align_snapshots: bool = False, min_steps: int = None,
                                 timestep_policy: str = DEFAULT_POLICY,
                                 diffusion_radius: float = None):
    """Time-step the DEIM/local-FD hyper-reduced ODE from reduced initial coordinates a0.

    This is the "solver-only" online path of `deim_rom_predict`: no initial projection and no
    final reconstruction onto the full grid. `Phi` is read ONLY through `Phi[support, :]`, the rows
    on the DEIM points' stencil support (at most (2*halfwidth+1)*m of the Nx rows), so nothing in
    the time-stepping loop touches the full grid (see tests/test_hyperreduction_audit.py, which
    poisons every other row of Phi with NaN and checks the trajectory is unchanged).

    Args:
        a0: (r,) reduced initial coordinates, Phi.T @ u0.
        u0_max: max|u0| of the full-grid initial condition (sets the advective step bound; passed
            through unchanged from the caller so `deim_rom_predict` stays bit-identical).
        dt, align_snapshots, min_steps, timestep_policy: as in src.pod_rom.rom_reduced_trajectory. The
        "reduced" policy (default) uses the spectral radius of THIS model's reduced diffusion operator
        (local-FD stencil of the given order); "historical" is the original full-grid rule.
        diffusion_radius: precomputed rho(K) for the "reduced" policy (computed if None).

    Returns:
        t_save: (n_save,), a_save: (n_save, r) reduced states, n_steps: RK4 steps taken.
    """
    a = a0
    dx = (2 * np.pi) / Nx
    halfwidth = 1 if stencil_order == 2 else 2
    support, neighbours = _stencil_support(points, Nx, halfwidth=halfwidth)
    Phi_support = Phi[support, :]  # (support_size, r) -- precomputed once per call; independent
    # of the RK4 loop below, so this setup cost is paid once, not per stage.
    # Profiling (see RESEARCH_EXTENSION.md "Timing verification") found deim_nonlinear_at_points's
    # internal np.searchsorted(support, neighbours) call was being repeated identically at every
    # one of the ~150-2000 RK4 stages per trajectory, even though `support`/`neighbours` (and
    # hence this index map) never change once the DEIM points are fixed -- ~18% of DEIM-ROM's
    # online cost was this single redundant computation. Hoisted out of the hot loop here; the
    # public deim_nonlinear_at_points/_4th_order functions (used by tests to validate correctness
    # against spectral ground truth) are left as-is, unaffected by this optimisation.
    idx_in_support = np.searchsorted(support, neighbours)  # (m, 3) or (m, 5)

    check_policy(timestep_policy)
    if dt is None:
        if timestep_policy == "historical":
            dt = historical_dt(u0_max, Nx, nu)
        else:
            rho = (spectral_radius(reduced_diffusion_matrix_deim(Phi, M, points, Nx,
                                                                 stencil_order=stencil_order))
                   if diffusion_radius is None else diffusion_radius)
            dt = reduced_dt(u0_max, Nx, nu, rho)
    n_steps = max(4 * n_save if min_steps is None else min_steps, int(np.ceil(T / dt)))
    if align_snapshots:
        n_steps = align_step_count(n_steps, n_save)
    dt = T / n_steps
    save_steps = set(np.linspace(0, n_steps, n_save, dtype=int).tolist())

    if stencil_order == 2:
        def deim_rhs(a_vec):
            u_support = Phi_support @ a_vec
            u_left = u_support[idx_in_support[:, 0]]
            u_centre = u_support[idx_in_support[:, 1]]
            u_right = u_support[idx_in_support[:, 2]]
            ux = (u_right - u_left) / (2 * dx)
            uxx = (u_right - 2 * u_centre + u_left) / (dx**2)
            F_points = -u_centre * ux + nu * uxx
            return M @ F_points
    else:
        def deim_rhs(a_vec):
            u_support = Phi_support @ a_vec
            u_m2 = u_support[idx_in_support[:, 0]]
            u_m1 = u_support[idx_in_support[:, 1]]
            u_0 = u_support[idx_in_support[:, 2]]
            u_p1 = u_support[idx_in_support[:, 3]]
            u_p2 = u_support[idx_in_support[:, 4]]
            ux = (-u_p2 + 8 * u_p1 - 8 * u_m1 + u_m2) / (12 * dx)
            uxx = (-u_p2 + 16 * u_p1 - 30 * u_0 + 16 * u_m1 - u_m2) / (12 * dx**2)
            F_points = -u_0 * ux + nu * uxx
            return M @ F_points

    t_save = [0.0]
    a_save = [a.copy()]
    t = 0.0
    for step in range(1, n_steps + 1):
        k1 = deim_rhs(a)
        k2 = deim_rhs(a + dt / 2 * k1)
        k3 = deim_rhs(a + dt / 2 * k2)
        k4 = deim_rhs(a + dt * k3)
        a = a + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if step in save_steps:
            a_save.append(a.copy())
            t_save.append(t)

    return np.array(t_save), np.array(a_save), n_steps


def deim_rom_predict(u0: np.ndarray, Phi: np.ndarray, M: np.ndarray, points: np.ndarray,
                      Nx: int, nu: float, T: float, n_save: int, dt: float = None,
                      stencil_order: int = 2, align_snapshots: bool = False,
                      min_steps: int = None, timestep_policy: str = DEFAULT_POLICY,
                      diffusion_radius: float = None):
    """DEIM-hyper-reduced POD-Galerkin prediction -- drop-in analogue of src.pod_rom.rom_predict,
    but the nonlinear term is never evaluated on the full grid.

    Args:
        stencil_order: 2 (default, original/baseline -- 3-point central differences) or 4
            (5-point central differences, see `deim_nonlinear_at_points_4th_order`). Default
            preserves the exact original behaviour of every existing caller.

    Returns:
        t_save: (n_save,), u_pred: (n_save, Nx) reconstructed physical-space predictions.
    """
    a = Phi.T @ u0
    t_save, a_save, _ = deim_rom_reduced_trajectory(a, np.abs(u0).max(), Phi, M, points, Nx, nu,
                                                     T, n_save, dt, stencil_order,
                                                     align_snapshots, min_steps,
                                                     timestep_policy, diffusion_radius)
    u_pred = a_save @ Phi.T  # (n_save, Nx) -- reconstruction is O(n_save * r * Nx), same as
    # the plain ROM's reconstruction cost; only the *time-stepping* RHS evaluations are hyper-
    # reduced. This matches how DEIM speed-ups are conventionally reported in the literature
    # (the online RHS-evaluation cost, not final-output reconstruction).
    return t_save, u_pred.astype(np.float32)
