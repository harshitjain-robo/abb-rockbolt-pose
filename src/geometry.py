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


def principal_axis(pcd):
    """Unit vector along the direction the cloud is most spread in."""
    points = np.asarray(pcd.points)
    centered = points - points.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))
    return eigvecs[:, np.argmax(eigvals)]
