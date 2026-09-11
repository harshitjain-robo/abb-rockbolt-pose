"""PCA recovers the bolt axis and its two ends, and the angle helpers score
orientation. Both have to respect the fact that a bolt has no head or tail, so
an axis and its negation describe the same line."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from geometry import (axis_angle_deg, geodesic_angle_deg, get_endpoints_pca,
                      principal_axis, quaternion_axis, unit)


class FakeCloud:
    """Stands in for an Open3D cloud, which only needs to expose .points here."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)


def rod(length=1.0, n=400, axis=(0, 0, 1), centre=(0, 0, 0), jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(-length / 2, length / 2, n)
    pts = t[:, None] * unit(axis)[None, :] + np.asarray(centre, dtype=float)
    if jitter:
        pts = pts + rng.normal(0, jitter, pts.shape)
    return FakeCloud(pts)


def test_length_of_a_known_rod():
    _, _, length = get_endpoints_pca(rod(length=0.8))
    assert length == pytest.approx(0.8, abs=1e-9)


def test_endpoints_are_the_two_extremes():
    p1, p2, _ = get_endpoints_pca(rod(length=1.0, centre=(1, 2, 3)))
    assert np.allclose(sorted([p1[2], p2[2]]), [2.5, 3.5])


def test_midpoint_of_the_endpoints_is_the_centre():
    p1, p2, _ = get_endpoints_pca(rod(length=0.6, centre=(0.4, -0.2, 1.1)))
    assert np.allclose((p1 + p2) / 2, [0.4, -0.2, 1.1], atol=1e-9)


@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0.3, -0.7, 0.6)])
def test_principal_axis_finds_the_direction_the_rod_points(axis):
    assert axis_angle_deg(principal_axis(rod(axis=axis)), axis) < 1e-6


def test_principal_axis_survives_noise():
    noisy = rod(length=0.6, axis=(0.2, 0.9, -0.3), jitter=0.004, seed=3)
    assert axis_angle_deg(principal_axis(noisy), (0.2, 0.9, -0.3)) < 2.0


def test_a_cloud_with_too_few_points_returns_zeros():
    p1, p2, length = get_endpoints_pca(FakeCloud([[0, 0, 0], [1, 1, 1]]))
    assert length == 0.0 and np.allclose(p1, 0) and np.allclose(p2, 0)


def test_axis_angle_ignores_which_end_is_which():
    """A bolt has no head or tail, so a and -a are the same line."""
    assert axis_angle_deg([0, 0, 1], [0, 0, -1]) == pytest.approx(0.0)
    assert axis_angle_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)


def test_axis_angle_matches_a_known_rotation():
    turned = Rotation.from_euler("y", 30, degrees=True).apply([0, 0, 1.0])
    assert axis_angle_deg(turned, [0, 0, 1]) == pytest.approx(30.0, abs=1e-6)


def test_geodesic_angle_is_zero_for_the_same_rotation():
    q = Rotation.from_euler("xyz", [12, -8, 40], degrees=True).as_quat()
    assert geodesic_angle_deg(q, q) == pytest.approx(0.0, abs=1e-6)


def test_geodesic_angle_treats_q_and_minus_q_as_equal():
    """A plain MSE on components scores these as different, which is what makes
    the shipped loss misleading. The metric must not repeat that mistake."""
    q = Rotation.from_euler("xyz", [12, -8, 40], degrees=True).as_quat()
    assert geodesic_angle_deg(q, -np.asarray(q)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("degrees", [15, 45, 90, 150])
def test_geodesic_angle_recovers_a_known_turn(degrees):
    a = Rotation.identity().as_quat()
    b = Rotation.from_euler("z", degrees, degrees=True).as_quat()
    assert geodesic_angle_deg(a, b) == pytest.approx(degrees, abs=1e-6)


def test_quaternion_axis_turns_a_rotation_into_the_bolt_direction():
    q = Rotation.from_euler("y", 90, degrees=True).as_quat()
    assert np.allclose(quaternion_axis(q), [1, 0, 0], atol=1e-9)


def test_unit_leaves_a_zero_vector_alone():
    assert np.allclose(unit([0, 0, 0]), [0, 0, 0])
