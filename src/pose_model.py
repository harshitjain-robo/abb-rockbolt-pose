"""Pose regression network, dataset and loss for the rockbolt pipeline.

The network is a PointNet-style encoder: one MLP applied independently to every
point, then a symmetric global max pool, then separate heads for rotation (a
unit quaternion) and translation. There are no set abstraction layers, so this
is the original PointNet formulation rather than PointNet++. The class name
below is the one it carried in the project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import open3d as o3d

# --- Network ---
class PointNet2PoseRegression(nn.Module):
    """Regresses a unit quaternion and a translation from an unordered cloud.

    Input is [B, N, 3]. Because the same MLP is applied to every point and the
    points are then reduced by a max, the output does not depend on the order
    the points arrive in, which is what makes it usable on raw clouds.
    """

    def __init__(self):
        super(PointNet2PoseRegression, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2))
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3))
        self.mlp3 = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4))

        self.fc_rot = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 4))  # Quaternion
        self.fc_trans = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, x):
        # x: [B, N, 3]
        B, N, _ = x.shape
        x = x.view(B * N, 3)
        x = self.mlp1(x)           # [B*N, 64]
        x = self.mlp2(x)           # [B*N, 128]
        x = self.mlp3(x)           # [B*N, 256]
        x = x.view(B, N, 256)      # [B, N, 256]
        x = torch.max(x, dim=1)[0]  # global max pool over the N points -> [B, 256]

        rot = self.fc_rot(x)
        rot = F.normalize(rot, p=2, dim=1)  # Normalize to unit quaternion
        trans = self.fc_trans(x)
        return rot, trans

# --- Dataset ---
class SyntheticPoseDataset(Dataset):
    """Loads the synthetic clouds listed in the pose metadata CSV.

    Every cloud is resampled to num_points so clouds of different sizes can be
    batched together. Sampling draws with replacement only when a cloud holds
    fewer points than num_points.
    """

    def __init__(self, csv_file, num_points=1024):
        self.df = pd.read_csv(csv_file)
        self.num_points = num_points
        self.base_dir = csv_file.replace("pose_metadata_train.csv", "pcd")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = f"{self.base_dir}/{row['filename']}"
        pcd = o3d.io.read_point_cloud(file_path)
        pts = np.asarray(pcd.points)
        if pts.shape[0] < self.num_points:
            idxs = np.random.choice(pts.shape[0], self.num_points, replace=True)
        else:
            idxs = np.random.choice(pts.shape[0], self.num_points, replace=False)
        pts = pts[idxs]

        Q = np.array([row['Qx'], row['Qy'], row['Qz'], row['Qw']], dtype=np.float32)  # Quaternion
        T = np.array([row['Tx'], row['Ty'], row['Tz']], dtype=np.float32)

        return torch.tensor(pts, dtype=torch.float32), torch.tensor(Q), torch.tensor(T)

# --- Loss ---
def pose_loss(quat_pred, trans_pred, quat_gt, trans_gt):
    """Unweighted sum of the quaternion MSE and the translation MSE.

    Both quaternions are renormalised before the comparison. The comparison is
    on raw components, so a prediction of -q scores as wrong even though it is
    the same rotation as q.
    """
    quat_pred = F.normalize(quat_pred, p=2, dim=1)
    quat_gt = F.normalize(quat_gt, p=2, dim=1)
    loss_rot = F.mse_loss(quat_pred, quat_gt)
    loss_trans = F.mse_loss(trans_pred, trans_gt)
    return loss_rot + loss_trans

# --- Metrics helper. Not called by either pipeline, kept for evaluation work ---
def compute_metrics(quat_pred, trans_pred, quat_gt, trans_gt):
    rot_error = torch.norm(quat_pred - quat_gt, dim=1).mean().item()
    trans_error = torch.norm(trans_pred - trans_gt, dim=1).mean().item()
    return rot_error, trans_error
