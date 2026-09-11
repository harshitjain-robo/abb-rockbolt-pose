"""The synthetic bolts are what the pose network learns from, so the properties
that matter are the ranges they span and, more importantly, whether the rotation
label they carry is recoverable from the cloud at all."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import config
from generate_synthetic_data import (add_rock_occlusions, apply_pose,
                                     generate_random_pose, generate_rockbolt)


def chamfer_mm(a, b):
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return 0.5 * (d.min(1).mean() + d.min(0).mean()) * 1000


def test_bolt_has_the_requested_length_along_its_axis():
    np.random.seed(0)
    bolt = generate_rockbolt(0.7, 0.02, bend=False)
    assert bolt[:, 2].max() - bolt[:, 2].min() == pytest.approx(0.7, abs=1e-9)


def test_straight_bolt_sits_within_its_radius():
    np.random.seed(0)
    bolt = generate_rockbolt(0.5, 0.02, bend=False)
    radial = np.linalg.norm(bolt[:, :2], axis=1)
    assert radial.max() < 0.02 + 5 * 0.002


def test_bending_pushes_the_bolt_off_its_axis():
    np.random.seed(0)
    straight = generate_rockbolt(0.8, 0.02, bend=False)
    np.random.seed(0)
    bent = generate_rockbolt(0.8, 0.02, bend=True)
    assert np.abs(bent[:, 0]).max() > np.abs(straight[:, 0]).max() + 0.02


def test_occlusion_adds_the_requested_fraction_of_points():
    np.random.seed(0)
    bolt = generate_rockbolt(0.5, 0.02, bend=False)
    with_rock = add_rock_occlusions(bolt, 0.3)
    assert len(with_rock) == len(bolt) + int(len(bolt) * 0.3)


def test_random_poses_stay_inside_the_configured_ranges():
    np.random.seed(0)
    for _ in range(50):
        quat, trans = generate_random_pose()
        assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-9)
        assert np.all(np.abs(trans) <= 0.2 + 1e-9)
        assert np.all(np.abs(Rotation.from_quat(quat).as_euler("xyz", degrees=True)) <= 45 + 1e-6)


def test_apply_pose_moves_the_bolt_where_the_label_says():
    np.random.seed(0)
    bolt = generate_rockbolt(0.6, 0.02, bend=False)
    quat = Rotation.from_euler("y", 90, degrees=True).as_quat()
    moved = apply_pose(bolt, quat, np.array([1.0, 0.0, 0.0]))
    # +z becomes +x, so the spread along x should now match the bolt length.
    assert moved[:, 0].max() - moved[:, 0].min() == pytest.approx(0.6, abs=1e-6)


def test_spinning_a_straight_bolt_does_not_change_the_cloud():
    """The label is not a function of the input for straight bolts.

    This is why the rotation head collapses toward the identity: half the
    training set carries a target the cloud cannot determine. src/rotation_study.py
    measures the consequence.
    """
    np.random.seed(11)
    first = generate_rockbolt(0.6, 0.02, bend=False)[::5]
    np.random.seed(22)
    second = generate_rockbolt(0.6, 0.02, bend=False)[::5]

    base = Rotation.from_euler("xyz", [20, -15, 35], degrees=True)
    floor = chamfer_mm(base.apply(first), base.apply(second))
    for angle in (30, 90, 180):
        spun = base * Rotation.from_euler("z", angle, degrees=True)
        assert chamfer_mm(base.apply(first), spun.apply(second)) < floor * 1.25


def test_spinning_a_bent_bolt_does_change_the_cloud():
    """The bend breaks the symmetry, so for bent bolts the label is recoverable."""
    np.random.seed(11)
    first = generate_rockbolt(0.6, 0.02, bend=True)[::5]
    np.random.seed(22)
    second = generate_rockbolt(0.6, 0.02, bend=True)[::5]

    base = Rotation.from_euler("xyz", [20, -15, 35], degrees=True)
    floor = chamfer_mm(base.apply(first), base.apply(second))
    spun = base * Rotation.from_euler("z", 90, degrees=True)
    assert chamfer_mm(base.apply(first), spun.apply(second)) > floor * 3
