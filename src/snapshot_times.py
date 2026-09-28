"""Physical times at which the solvers save snapshots, and how to make them uniform.

Historical rule (still the default in every solver): snapshots are stored at the integer step
indices floor(j * n_steps / (n_save - 1)), j = 0..n_save-1. Whenever (n_save - 1) does not divide
n_steps the saved times deviate from the intended j/(n_save-1) * T by up to one step, and two runs
with different step counts (the FOM and a reduced model with a different dt) save at DIFFERENT
physical times, so comparing them snapshot-by-snapshot compares different times.

Aligned rule (opt-in, `align_snapshots=True` in the solvers): n_steps is rounded UP to a multiple of
(n_save - 1), so every saved index is an exact multiple of q = n_steps / (n_save - 1) and the saved
times are exactly j/(n_save-1) * T for every method, whatever its dt.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def align_step_count(n_steps: int, n_save: int) -> int:
    """Smallest multiple of (n_save - 1) that is >= n_steps."""
    d = n_save - 1
    return int(-(-int(n_steps) // d) * d)


def saved_step_indices(n_steps: int, n_save: int) -> np.ndarray:
    """Step indices at which the solvers save (the historical rule; identical when aligned)."""
    return np.linspace(0, n_steps, n_save, dtype=int)


def saved_times(n_steps: int, n_save: int, T: float) -> np.ndarray:
    """Physical times of the saved snapshots for a run of n_steps steps of size T / n_steps."""
    return saved_step_indices(n_steps, n_save) * (T / n_steps)


def intended_times(n_save: int, T: float) -> np.ndarray:
    return np.linspace(0.0, T, n_save)


def resample_in_time(t_src: np.ndarray, u_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Cubic-spline resampling of a (n_save, Nx) trajectory from its actual saved times onto other
    times, for comparing legacy (non-uniform-time) snapshots with aligned ones. Interpolation error
    is not zero (a steepening front is not band-limited in time); use it as a diagnostic, not to
    replace generating aligned data."""
    return CubicSpline(np.asarray(t_src, dtype=float), np.asarray(u_src, dtype=float), axis=0)(t_dst)
