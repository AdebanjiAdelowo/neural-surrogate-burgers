"""Tools for the spatial / temporal convergence study (report/burgers_spatial_convergence_study.md).

Three independent pieces, none of which changes the verified baseline solvers:

1. `solve_burgers_f64`: the same pseudo-spectral Lawson integrating-factor RK4 scheme as
   src/solver.py (identical operations, so its float32 cast is bit-identical to `solve_burgers`, tested),
   but returning float64 snapshots, accepting any initial field and an explicit step count. The baseline
   casts its output to float32, which puts a ~6e-8 relative floor under any convergence measurement.

2. Fourier-space comparison across resolutions. With c_k = fft(u)/N (normalised coefficients) the
   trigonometric interpolant of grid values u_j is p(x) = sum_k c_k e^{ikx}, and Parseval gives
   ||p||_L2^2 = 2 pi sum |c_k|^2. A coarse field is compared with a fine one by zero-padding its
   coefficients (`pad_coeffs`); for even N the Nyquist coefficient c_{-N/2} (whose trigonometric term is
   real, c cos(N x / 2)) is split equally between k = +N/2 and k = -N/2, which keeps the interpolant real
   and unchanged. Restriction to M < N keeps |k| < M/2 and sets the M-grid Nyquist coefficient to
   c_{M/2} + c_{-M/2}. Errors are therefore L2 errors of the trigonometric interpolants, computed exactly,
   never by indexing one grid into another.

3. The exact solution by the Cole-Hopf transformation, evaluated through the Hopf integral

       u(x, t) = int ((x - y)/t) w(y) dy / int w(y) dy,    w(y) = exp(-G(x, y, t) / (2 nu)),
       G = F0(y) + (x - y)^2 / (2 t),                       F0(y) = int_0^y u0(s) ds,

   over y in R. For a zero-mean periodic u0, F0 is periodic and the formula is the periodic solution. The
   Fourier-series form of the heat solution phi is catastrophically ill-conditioned here (phi spans
   e^0 to e^{-A/nu}, about e^{-200}, with its minimum exactly at the front); the integral form is
   evaluated instead with a log-sum-exp shift (the largest weight is 1) on a window and step chosen a
   priori from the bounds below, so it is well conditioned for every tested (A, nu).

   Window: G/(2 nu) - min G/(2 nu) >= s^2/(4 nu t) - R/(2 nu) with R = max F0 - min F0, so all weights
   beyond |s| = W = sqrt(t (2 R + 4 nu * TAIL)) are below exp(-TAIL). Step: the integrand varies on the
   scales sqrt(2 nu t) (Gaussian) and sqrt(2 nu / max|u0'|) (curvature of F0 / (2 nu)); the step is a
   fraction of the smaller one. Both are checked by refinement in the tests and in the study.
"""

import numpy as np

from src.snapshot_times import align_step_count

L_DOMAIN = 2 * np.pi


# ------------------------------------------------------------------------------- float64 solver
def grid(N: int, L: float = L_DOMAIN) -> np.ndarray:
    return np.linspace(0, L, N, endpoint=False)


def sine_ic(A: float, N: int, L: float = L_DOMAIN) -> np.ndarray:
    """A sin(2 pi x / L), computed exactly as src/solver.py does."""
    return A * np.sin(2 * np.pi * grid(N, L) / L)


def default_step_count(u0_max: float, N: int, n_save: int = 20, T: float = 1.0,
                       L: float = L_DOMAIN) -> int:
    """The baseline FOM step count with `align_snapshots=True` (src/solver.py)."""
    dt = 0.25 * (L / N) / (abs(u0_max) + 1e-8)
    n_steps = max(max(1, int(np.ceil(T / dt))), 4 * n_save)
    return align_step_count(n_steps, n_save)


def solve_burgers_f64(u0: np.ndarray, nu: float, T: float = 1.0, n_save: int = 20,
                      n_steps: int = None, L: float = L_DOMAIN):
    """src/solver.py's scheme in float64 for an arbitrary initial field.

    n_steps: number of RK4 steps (must be a multiple of n_save - 1 so that the snapshots are at exactly
        t_j = j T / (n_save - 1)); default: the baseline aligned step count for max|u0|.

    Returns:
        t_save (n_save,), u_save (n_save, N) float64.
    """
    u0 = np.asarray(u0, dtype=np.float64)
    N = u0.shape[0]
    if n_steps is None:
        n_steps = default_step_count(np.abs(u0).max(), N, n_save, T, L)
    if n_steps % (n_save - 1):
        raise ValueError(f"n_steps={n_steps} is not a multiple of n_save - 1 = {n_save - 1}")
    k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    dealias = (np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k)))
    dt = T / n_steps
    E = np.exp(-nu * k**2 * dt / 2)
    E2 = np.exp(-nu * k**2 * dt)

    def nonlinear(v_hat):
        u = np.real(np.fft.ifft(v_hat))
        flux_hat = np.fft.fft(u * u)
        flux_hat *= dealias
        return -0.5j * k * flux_hat

    save_steps = set(np.linspace(0, n_steps, n_save, dtype=int).tolist())
    v = np.fft.fft(u0)
    t_save, u_save, t = [0.0], [u0.copy()], 0.0
    for step in range(1, n_steps + 1):
        a = nonlinear(v)
        b = nonlinear(E * (v + dt / 2 * a))
        c = nonlinear(E * v + dt / 2 * b)
        d = nonlinear(E2 * v + dt * E * c)
        v = E2 * v + dt / 6 * (E2 * a + 2 * E * (b + c) + d)
        t += dt
        if step in save_steps:
            u_save.append(np.real(np.fft.ifft(v)))
            t_save.append(t)
    return np.array(t_save), np.array(u_save)


# ------------------------------------------------------------------------------- Fourier tools
def coeffs(u: np.ndarray) -> np.ndarray:
    """Normalised Fourier coefficients c_k = fft(u)/N along the last axis (numpy ordering)."""
    u = np.asarray(u, dtype=np.float64)
    return np.fft.fft(u, axis=-1) / u.shape[-1]


def pad_coeffs(c: np.ndarray, M: int) -> np.ndarray:
    """Coefficients of the same trigonometric interpolant on M >= N points (Nyquist split equally)."""
    N = c.shape[-1]
    if M < N:
        raise ValueError("pad_coeffs needs M >= N; use restrict_coeffs")
    if M == N:
        return c.copy()
    out = np.zeros(c.shape[:-1] + (M,), dtype=complex)
    h = N // 2
    if N % 2:  # odd N: modes -h..h, no Nyquist
        out[..., :h + 1] = c[..., :h + 1]
        out[..., M - h:] = c[..., N - h:]
    else:
        out[..., :h] = c[..., :h]
        out[..., M - h + 1:] = c[..., N - h + 1:]
        out[..., h] = c[..., h] / 2
        out[..., M - h] = c[..., h] / 2
    return out


def restrict_coeffs(c: np.ndarray, M: int) -> np.ndarray:
    """Coefficients on M <= N points: modes |k| < M/2 kept; for even M the Nyquist coefficient is
    c_{M/2} + c_{-M/2} (the value the M-point grid sees of those two modes)."""
    N = c.shape[-1]
    if M > N:
        raise ValueError("restrict_coeffs needs M <= N; use pad_coeffs")
    if M == N:
        return c.copy()
    out = np.zeros(c.shape[:-1] + (M,), dtype=complex)
    h = M // 2
    if M % 2:
        out[..., :h + 1] = c[..., :h + 1]
        out[..., M - h:] = c[..., N - h:]
    else:
        out[..., :h] = c[..., :h]
        out[..., M - h + 1:] = c[..., N - h + 1:]
        out[..., h] = c[..., h] + c[..., N - h]
    return out


def resample(u: np.ndarray, M: int) -> np.ndarray:
    """Trigonometric interpolation (M > N) or Fourier restriction (M < N) of grid values to M points."""
    c = coeffs(u)
    N = c.shape[-1]
    c2 = pad_coeffs(c, M) if M >= N else restrict_coeffs(c, M)
    return np.real(np.fft.ifft(c2 * M, axis=-1))


def _common(u, ref):
    cu, cr = coeffs(u), coeffs(ref)
    M = max(cu.shape[-1], cr.shape[-1])
    return pad_coeffs(cu, M), pad_coeffs(cr, M)


def l2_norm(u: np.ndarray, axis=None, L: float = L_DOMAIN) -> np.ndarray:
    """L2 norm of the trigonometric interpolant(s); axis=None sums over all leading axes too."""
    sq = L * np.sum(np.abs(coeffs(u)) ** 2, axis=-1)
    return np.sqrt(np.sum(sq) if axis is None else sq)


def rel_l2(u: np.ndarray, ref: np.ndarray) -> float:
    """||u - ref|| / ||ref|| over every snapshot (space-time), grids may differ."""
    cu, cr = _common(u, ref)
    return float(np.sqrt(np.sum(np.abs(cu - cr) ** 2) / np.sum(np.abs(cr) ** 2)))


def rel_l2_per_snapshot(u: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Relative L2 error of each snapshot (leading axis = time); NaN where ||ref(t)|| = 0."""
    cu, cr = _common(u, ref)
    num = np.sum(np.abs(cu - cr) ** 2, axis=-1)
    den = np.sum(np.abs(cr) ** 2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(num / den)


# ------------------------------------------------------------------------------- Cole-Hopf
TAIL = 60.0  # weights below exp(-TAIL) (about 1e-26) are dropped


class HopfColeSolution:
    """Exact periodic Burgers solution for a zero-mean initial condition given by its antiderivative.

    F0: callable, F0(y) = int_0^y u0(s) ds (periodic). F0_range = max F0 - min F0, du0_max = max|u0'|
    (bounds used for the a-priori window and step; see the module docstring)."""

    def __init__(self, F0, u0, nu: float, F0_range: float, du0_max: float, step_fraction: float = 1 / 40):
        self.F0, self.u0, self.nu = F0, u0, nu
        self.F0_range, self.du0_max, self.step_fraction = F0_range, du0_max, step_fraction

    def window_and_step(self, t: float):
        W = np.sqrt(t * (2 * self.F0_range + 4 * self.nu * TAIL))
        scale = min(np.sqrt(2 * self.nu * t), np.sqrt(2 * self.nu / max(self.du0_max, 1e-300)))
        return W, scale * self.step_fraction

    def __call__(self, x: np.ndarray, t: float, chunk: int = 256) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if t == 0:
            return self.u0(x)
        W, h = self.window_and_step(t)
        n = int(np.ceil(W / h))
        s = np.arange(-n, n + 1) * h                       # y = x + s
        out = np.empty_like(x)
        for i in range(0, x.size, chunk):
            xi = x[i:i + chunk, None]
            E = (self.F0(xi + s[None, :]) + s[None, :] ** 2 / (2 * t)) / (2 * self.nu)
            w = np.exp(-(E - E.min(axis=1, keepdims=True)))
            out[i:i + chunk] = np.sum((-s[None, :] / t) * w, axis=1) / np.sum(w, axis=1)
        return out

    def trajectory(self, N: int, times, L: float = L_DOMAIN) -> np.ndarray:
        x = grid(N, L)
        return np.array([self(x, float(t)) for t in times])


def sine_solution(A: float, nu: float, step_fraction: float = 1 / 40) -> HopfColeSolution:
    """u0 = A sin x on [0, 2 pi): F0 = A (1 - cos y)."""
    return HopfColeSolution(lambda y: A * (1 - np.cos(y)), lambda x: A * np.sin(x), nu,
                            F0_range=2 * abs(A), du0_max=abs(A), step_fraction=step_fraction)


def two_mode_solution(A1: float, A2: float, nu: float, step_fraction: float = 1 / 40) -> HopfColeSolution:
    """u0 = A1 sin x + A2 sin 2x: F0 = A1 (1 - cos y) + (A2 / 2) (1 - cos 2y). F0_range is bounded by
    2|A1| + |A2| (an upper bound suffices for the window), max|u0'| by |A1| + 2|A2|."""
    return HopfColeSolution(lambda y: A1 * (1 - np.cos(y)) + 0.5 * A2 * (1 - np.cos(2 * y)),
                            lambda x: A1 * np.sin(x) + A2 * np.sin(2 * x), nu,
                            F0_range=2 * abs(A1) + abs(A2), du0_max=abs(A1) + 2 * abs(A2),
                            step_fraction=step_fraction)
