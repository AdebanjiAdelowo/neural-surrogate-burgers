"""Research extension -- ground-truth Burgers' solver for an ARBITRARY initial-condition field.

`src.solver.solve_burgers` only accepts the two-scalar family u0(x) = A*sin(2*pi*x/L); the
out-of-family generalisation experiment (see RESEARCH_EXTENSION.md) needs a ground-truth reference
for initial conditions outside that family (e.g. a two-mode sinusoid), which the FNO can accept as
input but the MLP surrogate structurally cannot.

Rather than modifying `src.solver.solve_burgers` (the verified baseline, left untouched throughout
this research extension -- see RESEARCH_EXTENSION.md "What was NOT changed"), this module
INTENTIONALLY MIRRORS its exact numerical method (pseudo-spectral FFT derivatives, integrating-
factor RK4, 2/3 dealiasing rule -- identical to src/solver.py line for line where the method
itself is unchanged) so that ground-truth trajectories for the new IC family are produced by the
same verified scheme, not a different, uncompared one. The only change is the initial condition:
an arbitrary array u0, not a computed A*sin(...) expression.
"""

import numpy as np


def solve_burgers_general_ic(u0: np.ndarray, nu: float, L: float = 2 * np.pi, T: float = 1.0,
                              n_save: int = 20, dt: float = None):
    """Solve Burgers' equation for an arbitrary periodic initial condition u0, returning n_save
    snapshots. Numerical method identical to src.solver.solve_burgers (see module docstring).

    Args:
        u0: (Nx,) initial condition, sampled on x = linspace(0, L, Nx, endpoint=False).

    Returns:
        t_save, x, u_save -- same conventions as src.solver.solve_burgers.
    """
    Nx = u0.shape[0]
    x = np.linspace(0, L, Nx, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    dealias = (np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k)))

    v = np.fft.fft(u0)

    if dt is None:
        dt = 0.25 * (L / Nx) / (np.abs(u0).max() + 1e-8)
    n_steps = max(1, int(np.ceil(T / dt)))
    n_steps = max(n_steps, 4 * n_save)
    dt = T / n_steps

    Lop = -nu * k**2
    E = np.exp(Lop * dt / 2)
    E2 = np.exp(Lop * dt)

    def nonlinear(v_hat):
        u = np.real(np.fft.ifft(v_hat))
        flux_hat = np.fft.fft(u * u)
        flux_hat *= dealias
        return -0.5j * k * flux_hat

    save_steps = set(np.linspace(0, n_steps, n_save, dtype=int).tolist())

    t_save = [0.0]
    u_save = [u0.astype(np.float32).copy()]
    t = 0.0

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

    return np.array(t_save), x, np.array(u_save).astype(np.float32)
