"""Back-projection turns a masked pixel into a 3D point, and every downstream
number depends on it. The expected values here come from the pinhole equations
rather than from whatever the function currently returns."""

import numpy as np
import pytest

from geometry import backproject

FOCAL = 600.0
CX, CY = 320.0, 240.0
K = np.array([[FOCAL, 0, CX], [0, FOCAL, CY], [0, 0, 1]], dtype=np.float64)


def scene(shape=(480, 640)):
    return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=bool)


def test_principal_point_lands_on_the_optical_axis():
    depth, mask = scene()
    depth[int(CY), int(CX)] = 2.0
    mask[int(CY), int(CX)] = True
    assert np.allclose(backproject(depth, mask, K, 1.0), [[0.0, 0.0, 2.0]])


def test_offset_pixel_matches_the_pinhole_equation():
    """A pixel 60 px right and 30 px down at 2 m sits at (60/600*2, 30/600*2, 2)."""
    depth, mask = scene()
    depth[int(CY) + 30, int(CX) + 60] = 2.0
    mask[int(CY) + 30, int(CX) + 60] = True
    assert np.allclose(backproject(depth, mask, K, 1.0), [[0.2, 0.1, 2.0]])


def test_depth_scale_is_applied():
    depth, mask = scene()
    depth[int(CY), int(CX)] = 1000.0
    mask[int(CY), int(CX)] = True
    got = backproject(depth, mask, K, 0.001)
    assert np.allclose(got, [[0.0, 0.0, 1.0]])


def test_points_scale_linearly_with_depth():
    depth, mask = scene()
    for row, z in ((int(CY) + 30, 1.0), (int(CY) + 30, 2.0)):
        depth[row, int(CX) + 60] = z
        mask[row, int(CX) + 60] = True
        got = backproject(depth, mask, K, 1.0)
        assert np.allclose(got[0], [0.1 * z, 0.05 * z, z])
        mask[row, int(CX) + 60] = False


def test_pixels_outside_the_mask_are_ignored():
    depth, mask = scene()
    depth[:] = 2.0
    mask[100, 100] = True
    assert backproject(depth, mask, K, 1.0).shape == (1, 3)


def test_zero_and_near_zero_depth_is_dropped():
    """Invalid depth comes back as zero, and must not become a point at the camera."""
    depth, mask = scene()
    depth[100, 100] = 0.0
    depth[100, 101] = 0.0005
    depth[100, 102] = 2.0
    mask[100, 100:103] = True
    assert backproject(depth, mask, K, 1.0).shape == (1, 3)


def test_empty_mask_gives_an_empty_but_correctly_shaped_array():
    """Open3D needs (N, 3), so an empty result must still have three columns."""
    depth, mask = scene()
    got = backproject(depth, mask, K, 1.0)
    assert got.shape == (0, 3)


def test_points_come_out_in_row_major_order():
    depth, mask = scene()
    depth[:] = 1.0
    mask[10, 20] = mask[10, 21] = mask[11, 20] = True
    got = backproject(depth, mask, K, 1.0)
    rows = [(10, 20), (10, 21), (11, 20)]
    expected = [[(u - CX) / FOCAL, (v - CY) / FOCAL, 1.0] for v, u in rows]
    assert np.allclose(got, expected)


@pytest.mark.parametrize("density", [0.01, 0.2, 1.0])
def test_every_masked_valid_pixel_produces_exactly_one_point(density):
    rng = np.random.default_rng(0)
    depth = np.full((60, 80), 2.0)
    mask = rng.random((60, 80)) < density
    assert len(backproject(depth, mask, K, 1.0)) == int(mask.sum())
