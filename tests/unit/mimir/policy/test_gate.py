import numpy as np
import pytest

from mimir.policy.gate import nearest_distance, p_value


def test_nearest_distance_is_the_minimum_mahalanobis_distance() -> None:
    centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
    precision = np.diag([1.0, 4.0])
    workspace = np.array([9.0, 1.0])
    # To the second centroid: (-1)^2 * 1 + 1^2 * 4 = 5; to the first: 81 + 4 = 85.
    assert nearest_distance(workspace, centroids, precision) == pytest.approx(5.0)


def test_nearest_distance_is_zero_at_a_centroid_and_never_negative() -> None:
    centroids = np.array([[1.0, 2.0, 3.0]])
    assert nearest_distance(np.array([1.0, 2.0, 3.0]), centroids, np.eye(3)) == 0.0


def test_p_value_counts_reference_distances_at_or_above() -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    assert p_value(reference, 0.5) == 5 / 5
    assert p_value(reference, 2.0) == 4 / 5
    assert p_value(reference, 2.5) == 3 / 5
    assert p_value(reference, 9.0) == 1 / 5


def test_p_value_with_an_empty_reference_is_one() -> None:
    assert p_value(np.array([]), 3.0) == 1.0
