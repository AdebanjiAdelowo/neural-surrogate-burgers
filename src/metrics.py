"""Error metric shared by the hyper-reduction audit and its tests."""

import numpy as np


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    """||pred - ref||_2 / ||ref||_2, both flattened over every saved snapshot and grid point.

    Same definition (including the 1e-12 guard) as `rel_l2` in
    scripts/evaluate_research_extension.py, whose float32 arithmetic on float32 inputs is kept so
    historical numbers reproduce to the printed precision.
    """
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))
