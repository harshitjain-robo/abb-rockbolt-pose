"""Point cloud geometry for the rockbolt pipeline.

get_endpoints_pca lives here rather than inside infer_pipeline because that
module builds a detector and loads the trained weights at import time. Keeping
the geometry separate means anything that only needs the shape of a cloud can
import it without pulling a detector into memory.
"""

import numpy as np


def get_endpoints_pca(pcd):
    """Returns the two ends of the bolt and the distance between them.

    A bolt is long and thin, so the direction its points are most spread along
    is its axis. That is the eigenvector of the point covariance with the
    largest eigenvalue. Projecting every point onto that axis and taking the two
    extremes gives the ends, and their separation is the length.
    """
    points = np.asarray(pcd.points)
    if len(points) < 3:
        return np.zeros(3), np.zeros(3), 0.0
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    proj = points @ axis
    pt1 = points[np.argmin(proj)]
    pt2 = points[np.argmax(proj)]
    length = np.linalg.norm(pt2 - pt1)
    return pt1, pt2, length


def backproject(depth, mask, K, depth_scale):
    """Lifts the masked depth pixels into camera-frame 3D points.

    Standard pinhole back-projection: a pixel at (u, v) with depth z sits at
    ((u - cx) * z / fx, (v - cy) * z / fy, z). Only pixels inside the mask are
    kept, so the result is the bolt rather than the whole scene.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # np.nonzero walks the mask in row-major order, the same order the old
    # pixel-by-pixel loop did, so the points come out in the same sequence.
    v, u = np.nonzero(mask)
    z = depth[v, u] * depth_scale
    keep = z > 0.001
    u, v, z = u[keep], v[keep], z[keep]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack((x, y, z), axis=1)


def principal_axis(pcd):
    """Unit vector along the direction the cloud is most spread in."""
    points = np.asarray(pcd.points)
    centered = points - points.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
    return eigvecs[:, np.argmax(eigvals)]


def unit(v):
    """Normalises a vector, leaving a zero vector alone."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def quaternion_axis(quat):
    """The bolt axis a rotation implies. Bolts are generated along +z."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(unit(quat)).apply([0.0, 0.0, 1.0])


def axis_angle_deg(a, b):
    """Angle between two directions, in degrees.

    A bolt has no head or tail, so a and -a are the same line and the cosine
    is taken absolute.
    """
    cos = abs(float(np.dot(unit(a), unit(b))))
    return float(np.degrees(np.arccos(min(1.0, cos))))


def geodesic_angle_deg(q_pred, q_true):
    """Rotation angle between two quaternions, in degrees.

    Absolute dot product, since q and -q are the same rotation. Scoring them
    as different is what makes an MSE on raw components misleading.
    """
    cos = abs(float(np.dot(unit(q_pred), unit(q_true))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, cos))))
