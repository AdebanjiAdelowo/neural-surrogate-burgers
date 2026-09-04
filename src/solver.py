"""Ground-truth numerical solver for the 1D viscous Burgers' equation.

    u_t + u u_x = nu u_xx,   x in [0, L) periodic,   u(x,0) = A sin(2*pi*x/L)

Method: pseudo-spectral in space (FFT), integrating-factor RK4 in time — the linear diffusion term
is handled exactly via the integrating factor exp(nu * (ik)^2 * dt), so the scheme is not limited
by the diffusion term's stiff time-step restriction; only the nonlinear advection term is stepped
via RK4. This is the standard textbook approach for stiff PDEs of this type (e.g. Trefethen,
"Spectral Methods in MATLAB", the ETDRK-lite scheme used for KdV/Burgers-type equations) —
independent, from-scratch code, not reused from `pinn-advection-diffusion` or
`boiling-phasefield-3d`. Aliasing from the quadratic nonlinearity is controlled with the standard
2/3 dealiasing rule.

Not implemented: an implicit/exact Cole-Hopf semi-analytical solution (Burgers admits one for this
IC, but it requires its own numerical integration and was judged not worth the extra implementation
risk for this MVP). Verification instead uses grid-refinement convergence, the two exact conserved
quantities of the periodic problem (mass, and energy dissipation direction), and a physically
expected steepening-location check — see IMPLEMENTATION_LOG.md Stage "solver validation".
"""

import numpy as np


def solve_burgers(A: float, nu: float, Nx: int = 128, L: float = 2 * np.pi,
                   T: float = 1.0, n_save: int = 20, dt: float = None):
    """Solve Burgers' equation for u0(x) = A*sin(2*pi*x/L), returning n_save snapshots.

    Returns:
        t_save: (n_save,) times, including t=0 and t=T.
        x: (Nx,) spatial grid.
        u_save: (n_save, Nx) solution snapshots.
    """
    x = np.linspace(0, L, Nx, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)
    dealias = (np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k)))

    u0 = A * np.sin(2 * np.pi * x / L)
    v = np.fft.fft(u0)

    if dt is None:
        # Advective CFL-style bound (diffusion is unconditionally stable here via the
        # integrating factor, so only the nonlinear term constrains dt).
        dt = 0.25 * (L / Nx) / (abs(A) + 1e-8)
    n_steps = max(1, int(np.ceil(T / dt)))
    # Ensure comfortably more steps than requested snapshots: found during Stage 5 evaluation
    # (neural-surrogate-burgers) that when n_steps was not >> n_save, the integer-rounded
    # np.linspace(0, n_steps, n_save) snapshot-selection could collide onto fewer than n_save
    # distinct step indices for some (A, nu) combinations, silently under-producing snapshots.
    n_steps = max(n_steps, 4 * n_save)
    dt = T / n_steps  # adjust so n_steps evenly divides [0, T]

    Lop = -nu * k**2  # linear operator eigenvalues (diffusion)
    E = np.exp(Lop * dt / 2)
    E2 = np.exp(Lop * dt)

    def nonlinear(v_hat):
        u = np.real(np.fft.ifft(v_hat))
        flux_hat = np.fft.fft(u * u)
        flux_hat *= dealias
        return -0.5j * k * flux_hat  # spectral d/dx of -(1/2) u^2

    # Save snapshots at n_save evenly-spaced *step indices* (not modulo-based), so the final
    # snapshot always lands exactly on step n_steps (t=T), fixing an earlier version that could
    # undershoot T by up to one save interval.
    save_steps = set(np.linspace(0, n_steps, n_save, dtype=int).tolist())

    t_save = [0.0]
    u_save = [u0.copy()]
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
