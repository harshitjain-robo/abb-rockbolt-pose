"""Runs the trained models over an RGB-D sequence and records a pose per bolt.

Per frame: Mask R-CNN segments the bolts, the masked RGB-D pair becomes a point
cloud, PCA over that cloud gives the bolt axis and from it the two endpoints and
the length, and the pose network predicts a rotation and a translation relative
to the midpoint of those endpoints. Writes a CSV row per instance alongside
overlay images and the clouds themselves.

Note that this reads a different camera and a different depth format from the
training pipeline. Both are described in config.py.
"""

import os
import cv2
import numpy as np
import torch
import open3d as o3d
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from scipy.spatial.transform import Rotation as R
from geometry import get_endpoints_pca
from pose_model import PointNet2PoseRegression

import config

# --- Paths ---
color_dir = config.INFER_COLOR_DIR
depth_dir = config.INFER_DEPTH_DIR
mask_dir = config.INFER_MASK_DIR
mask_vis_dir = config.INFER_MASK_VIS_DIR
pcd_dir = config.INFER_PCD_DIR
csv_path = config.INFER_CSV
vis_dir = config.INFER_POSE_VIS_DIR

for _d in (mask_dir, mask_vis_dir, pcd_dir, vis_dir):
    os.makedirs(_d, exist_ok=True)

# --- Camera ---
fx, fy, cx, cy = config.INFER_FX, config.INFER_FY, config.INFER_CX, config.INFER_CY
intrinsics = o3d.camera.PinholeCameraIntrinsic(config.INFER_WIDTH, config.INFER_HEIGHT, fx, fy, cx, cy)

# --- Models ---
maskrcnn_weights = str(config.MASKRCNN_WEIGHTS)
pointnet_ckpt = str(config.POSE_WEIGHTS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pointnet = PointNet2PoseRegression().to(device)
pointnet.load_state_dict(torch.load(pointnet_ckpt))
pointnet.eval()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config.MASKRCNN_CONFIG))
cfg.MODEL.WEIGHTS = maskrcnn_weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.SCORE_THRESH
cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.NUM_CLASSES
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

# --- Utilities ---
def create_pointcloud(color, depth, mask):
    """Builds a point cloud from the part of the RGB-D frame inside the mask.

    Depth outside the mask is zeroed so Open3D drops it, and the final filter
    removes anything closer than 1 cm, which is where invalid depth ends up.
    """
    masked_depth = np.where(mask, depth, 0).astype(np.uint16)
    color_o3d = o3d.geometry.Image(color)
    depth_o3d = o3d.geometry.Image(masked_depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, depth_scale=config.INFER_DEPTH_SCALE)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    return pcd.select_by_index([i for i, pt in enumerate(pcd.points) if pt[2] > 0.01])

def compute_iou(mask1, mask2):
    """Intersection over union of two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def draw_pose_overlay(image, center, quat, pt1, pt2, length):
    """Draws the estimate back onto the frame.

    Red is the translation the network predicted, white is the midpoint of the
    two PCA endpoints, green are the endpoints, and the three arrows are the
    predicted rotation drawn as 5 cm axes.
    """

    def project(pt):
        x, y, z = pt
        if z <= 0:
            return (0, 0)
        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)
        return (u, v)

    vis = image.copy()
    c = project(center)
    p1 = project(pt1)
    p2 = project(pt2)
    bolt_c = project((pt1 + pt2) / 2)

    cv2.circle(vis, c, 6, (0, 0, 255), -1)
    cv2.circle(vis, p1, 6, (0, 255, 0), -1)
    cv2.circle(vis, p2, 6, (0, 255, 0), -1)
    cv2.circle(vis, bolt_c, 6, (255, 255, 255), -1)
    cv2.putText(vis, "Centroid", (bolt_c[0] + 5, bolt_c[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    rotation = R.from_quat(quat)
    tip_x = project(center + 0.05 * rotation.apply([1, 0, 0]))
    tip_y = project(center + 0.05 * rotation.apply([0, 1, 0]))
    tip_z = project(center + 0.05 * rotation.apply([0, 0, 1]))

    cv2.arrowedLine(vis, c, tip_x, (255, 0, 0), 2)
    cv2.arrowedLine(vis, c, tip_y, (0, 255, 255), 2)
    cv2.arrowedLine(vis, c, tip_z, (0, 255, 0), 2)

    return vis

# --- Pipeline ---
image_files = sorted([f for f in os.listdir(color_dir) if f.lower().endswith(".png")])

with open(csv_path, "w") as f:
    f.write("filename,tx,ty,tz,bolt_cx,bolt_cy,bolt_cz,Qx,Qy,Qz,Qw,x1,y1,z1,x2,y2,z2,length\n")
    for idx, fname in enumerate(image_files):
        base_name = os.path.splitext(fname)[0]
        color = cv2.imread(os.path.join(color_dir, fname))

        depth_path = os.path.join(depth_dir, base_name + ".npy")
        depth = np.load(depth_path)

        outputs = predictor(color)
        instances = outputs["instances"].to("cpu")
        if not instances.has("pred_masks") or len(instances) == 0:
            continue

        masks = instances.pred_masks.numpy()
        boxes = instances.pred_boxes.tensor.numpy()

        keep = []
        for i, m in enumerate(masks):
            if all(compute_iou(m, masks[j]) < config.MASK_IOU_DEDUP for j in keep):
                keep.append(i)

        for i in keep:
            mask = masks[i]
            box = boxes[i]
            score = float(instances.scores[i]) * 100

            np.save(os.path.join(mask_dir, f"{base_name}_mask_{i}.npy"), mask)
            mask_vis = color.copy()
            mask_vis[mask] = (255, 255, 0)
            cv2.rectangle(mask_vis, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (255, 255, 0), 2)
            cv2.putText(mask_vis, f"rockbolt: {score:.1f}%", tuple(box[:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imwrite(os.path.join(mask_vis_dir, f"{base_name}_mask_{i}.png"), mask_vis)

            pcd = create_pointcloud(color, depth, mask)
            if len(pcd.points) < config.MIN_POINTS_FOR_INFERENCE:
                print(f"[SKIP] Too few points in {base_name} - mask {i}")
                continue

            pts = np.asarray(pcd.points)
            pt1, pt2, length = get_endpoints_pca(pcd)
            bolt_center = (pt1 + pt2) / 2

            # np.asarray over an Open3D cloud is a view onto the same buffer,
            # so translating the cloud also shifts pts. That is deliberate: the
            # network sees a cloud centred on the endpoint midpoint, matching
            # how the synthetic training clouds were framed. The midpoint is
            # added back to the prediction below to return to camera frame.
            pcd.translate(-bolt_center)
            idxs = np.random.choice(len(pts), config.NUM_POINTS, replace=len(pts) < config.NUM_POINTS)
            sample = torch.tensor(pts[idxs], dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                rot_pred, trans_pred = pointnet(sample)

            quat = rot_pred.squeeze().cpu().numpy()
            quat = quat / np.linalg.norm(quat)
            translation = trans_pred.squeeze().cpu().numpy() + bolt_center
            pcd.translate(bolt_center)

            vis_img = draw_pose_overlay(color, translation, quat, pt1, pt2, length)
            cv2.imwrite(os.path.join(vis_dir, f"{base_name}_pose_{i}.png"), vis_img)
            o3d.io.write_point_cloud(os.path.join(pcd_dir, f"{base_name}_pcd_{i}.ply"), pcd)

            f.write(f"{base_name},{translation[0]:.4f},{translation[1]:.4f},{translation[2]:.4f}," +
                    f"{bolt_center[0]:.4f},{bolt_center[1]:.4f},{bolt_center[2]:.4f}," +
                    f"{quat[0]:.4f},{quat[1]:.4f},{quat[2]:.4f},{quat[3]:.4f}," +
                    f"{pt1[0]:.4f},{pt1[1]:.4f},{pt1[2]:.4f},{pt2[0]:.4f},{pt2[1]:.4f},{pt2[2]:.4f},{length:.3f}\n")

print(f"\nProcessed {len(image_files)} frames. Results written to {csv_path}")
