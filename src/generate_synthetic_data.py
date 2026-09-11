"""Generates synthetic rockbolt point clouds with known 6-DoF pose.

Each sample is a cylinder with radial jitter, half of them bent, placed at a
random pose and padded with scattered points that stand in for the surrounding
rock. Writes one PLY per sample plus a CSV of the poses that produced them,
which together are the training set for the pose network.

Real clouds carry no pose ground truth, so this is what the network learns from.
"""

import os
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

import config

# --- Configuration ---
num_samples = config.SYNTH_NUM_SAMPLES
output_dir = config.SYNTHETIC_DIR
pcd_dir = config.SYNTHETIC_PCD_DIR

# --- Parameter space ---
length_range = config.SYNTH_LENGTH_RANGE  # meters
radius_range = config.SYNTH_RADIUS_RANGE  # meters
rock_density = config.SYNTH_OCCLUSION_DENSITY  # fraction of occlusion points

# --- Bolt geometry ---
def generate_rockbolt(length, radius, bend=False):
    """Points on a cylinder of the given length and radius, along +z.

    The radius is jittered so the surface is not perfectly smooth, and the bend
    displaces the axis sinusoidally to imitate a bolt that has been deformed in
    handling.
    """
    num_points = config.SYNTH_POINTS_PER_BOLT
    z = np.linspace(0, length, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    r = np.random.normal(loc=radius, scale=0.002, size=num_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    if bend:
        x += 0.05 * np.sin(z * 5)  # Apply sinusoidal bend
    return np.vstack((x, y, z)).T

# --- Rock occlusion ---
def add_rock_occlusions(points, density):
    """Adds scattered points near the bolt to stand in for surrounding rock.

    Real masks are never clean, so training on bare cylinders alone would not
    match what the segmentation stage hands over at inference.
    """
    num_rocks = int(len(points) * density)
    sampled_pts = points[np.random.choice(len(points), num_rocks)]
    noise = np.random.uniform(-0.05, 0.05, size=(num_rocks, 3))
    return np.vstack((points, sampled_pts + noise))

# --- Random pose ---
def generate_random_pose():
    euler_deg = np.random.uniform(-45, 45, size=3)
    rotation = R.from_euler('xyz', euler_deg, degrees=True)
    translation = np.random.uniform(-0.2, 0.2, size=3)
    quat = rotation.as_quat()  # [x, y, z, w]
    return quat, translation

# --- Apply a pose to a cloud ---
def apply_pose(points, quat, translation):
    rotation = R.from_quat(quat)
    return rotation.apply(points) + translation

# --- Generation loop ---
def main():
    import open3d as o3d
    import pandas as pd

    os.makedirs(pcd_dir, exist_ok=True)
    pose_records = []

    for i in range(num_samples):
        length = np.random.uniform(*length_range)
        radius = np.random.uniform(*radius_range)
        bent = random.choice([True, False])

        bolt = generate_rockbolt(length, radius, bend=bent)
        bolt = add_rock_occlusions(bolt, rock_density)

        quat, trans = generate_random_pose()
        transformed = apply_pose(bolt, quat, trans)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transformed)
        filename = f"bolt_{i:04d}.ply"
        o3d.io.write_point_cloud(os.path.join(pcd_dir, filename), pcd)

        pose_records.append({
            "filename": filename,
            "Tx": trans[0], "Ty": trans[1], "Tz": trans[2],
            "Qx": quat[0], "Qy": quat[1], "Qz": quat[2], "Qw": quat[3]
        })

    # --- Pose metadata ---
    pd.DataFrame(pose_records).to_csv(
        os.path.join(output_dir, "pose_metadata_train.csv"), index=False)

    print(f"Wrote {num_samples} synthetic clouds and pose metadata to {output_dir}")


if __name__ == "__main__":
    main()
