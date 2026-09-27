"""Out-of-distribution gate.

Computes the squared Mahalanobis distance from the pooled workspace to the nearest centroid
under a tied precision matrix (Podolskiy et al., arXiv 2101.03778), and converts it to a
conformal p-value against held-out in-distribution distances (Bates et al., arXiv 2104.08279).
"""

import numpy as np
import numpy.typing as npt

Floats = npt.NDArray[np.float64]


def nearest_distance(workspace: Floats, centroids: Floats, precision: Floats) -> float:
    """Return the squared Mahalanobis distance to the nearest centroid, clipped at zero.

    Shapes: `workspace` [W], `centroids` [C, W], `precision` [W, W].
    """
    features = workspace[None, :]
    projected = features @ precision
    own = np.einsum("ij,ij->i", projected, features)
    centre = np.einsum("ij,jk,ik->i", centroids, precision, centroids)
    cross = projected @ centroids.T
    nearest = np.maximum((own[:, None] - 2 * cross + centre[None, :]).min(axis=1), 0.0)
    return float(nearest[0])


def p_value(reference: Floats, distance: float) -> float:
    """Return `(1 + #{reference >= distance}) / (n + 1)`; `reference` must be sorted."""
    count = len(reference)
    above = count - int(np.searchsorted(reference, distance, side="left"))
    return (1 + above) / (count + 1)
