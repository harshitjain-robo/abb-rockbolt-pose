"""Training and evaluation pipeline for rockbolt pose estimation.

Runs the whole thing in order: fine-tune Mask R-CNN on the annotated frames,
predict masks for the test split and for the full set, back-project those masks
into point clouds using the depth maps and the camera intrinsics, train the pose
network on the synthetic clouds, then run it over the real ones.

Every path and constant comes from config.py.
"""

import os
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetMapper
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
import albumentations as A
import open3d as o3d
from geometry import backproject
from pose_model import PointNet2PoseRegression, SyntheticPoseDataset, pose_loss
import pandas as pd

import config


# --- Paths ---
train_images = config.TRAIN_IMAGES
val_images = config.VAL_IMAGES
test_images = config.TEST_IMAGES
all_images = config.ALL_IMAGES
all_depth = config.ALL_DEPTH

train_json = config.TRAIN_JSON
val_json = config.VAL_JSON
test_json = config.TEST_JSON
pose_csv_file = str(config.POSE_CSV)

mask_output_test = config.MASK_OUT_TEST
mask_output_test_vis = config.MASK_OUT_TEST_VIS
mask_output_all = config.MASK_OUT_ALL
output_pc_folder = config.PC_OUT_DIR
pointnet_test_pc_folder = config.PC_OUT_TEST_DIR
pointnet_ckpt_folder = config.POSE_CKPT_DIR
pointnet_output_csv = config.POSE_OUT_CSV

for _d in (mask_output_test, mask_output_test_vis, mask_output_all,
           output_pc_folder, pointnet_test_pc_folder, pointnet_ckpt_folder):
    os.makedirs(_d, exist_ok=True)

# Selected once at import so the helpers below can use it whether this module
# is run as a script or imported.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Albumentations augmentation ---
# build_augmentations and custom_train_mapper are not connected to training.
# DefaultTrainer builds its own loader with Detectron2's stock DatasetMapper, so
# the model trained under Detectron2's defaults, which are a shortest-edge
# resize and a horizontal flip. Connecting this chain would also mean passing
# the masks and boxes into the transform, since the geometric operations here
# move the image and would otherwise leave the labels behind.
def build_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
        A.MotionBlur(p=0.2),
        A.GaussianBlur(p=0.2),
        A.GaussNoise(p=0.2),
        A.Resize(640, 640)
    ])



# --- Custom train mapper, see the note above ---
def custom_train_mapper(dataset_dict):
    dataset_dict = dataset_dict.copy()
    image = cv2.imread(dataset_dict["file_name"])
    aug = build_augmentations()
    transformed = aug(image=image)
    dataset_dict["image"] = transformed["image"]
    return DatasetMapper(is_train=True)(dataset_dict)



# --- Register the COCO splits with Detectron2 ---
def setup_datasets():
    register_coco_instances("rockbolt_train", {}, str(train_json), str(train_images))
    register_coco_instances("rockbolt_val", {}, str(val_json), str(val_images))
    register_coco_instances("rockbolt_test", {}, str(test_json), str(test_images))
    MetadataCatalog.get("rockbolt_train").thing_classes = ["rockbolt"]
    MetadataCatalog.get("rockbolt_val").thing_classes = ["rockbolt"]
    MetadataCatalog.get("rockbolt_test").thing_classes = ["rockbolt"]


# --- Inference thresholds ---
# Not called anywhere. train_mask_rcnn sets the score threshold and class count
# itself, but it never sets NMS_THRESH_TEST, so inference ran on Detectron2's
# default of 0.5 rather than the 0.4 here. Overlapping masks were removed by the
# IoU check in run_mask_rcnn_on_folder instead.
def update_inference_cfg(cfg):
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.SCORE_THRESH  # high confidence
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.NUM_CLASSES  # rock_bolts
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.NMS_THRESH  # non-maximum suppression
    return cfg


# --- Mask R-CNN training ---
def train_mask_rcnn():
    global cfg
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config.MASKRCNN_CONFIG))

    cfg.DATASETS.TRAIN = ("rockbolt_train",)
    cfg.DATASETS.TEST = ("rockbolt_val",)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config.MASKRCNN_CONFIG)
    cfg.SOLVER.IMS_PER_BATCH = config.IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = config.BASE_LR
    cfg.SOLVER.MAX_ITER = config.MAX_ITER
    cfg.SOLVER.CHECKPOINT_PERIOD = config.CHECKPOINT_PERIOD
    cfg.TEST.EVAL_PERIOD = config.EVAL_PERIOD
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.ROI_BATCH_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.SCORE_THRESH  # Raise threshold for better precision
    cfg.INPUT.MIN_SIZE_TRAIN = (config.INPUT_SIZE,)
    cfg.INPUT.MAX_SIZE_TRAIN = config.INPUT_SIZE
    cfg.INPUT.MIN_SIZE_TEST = config.INPUT_SIZE
    cfg.INPUT.MAX_SIZE_TEST = config.INPUT_SIZE
    cfg.OUTPUT_DIR = str(config.MASKRCNN_DIR)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


# --- Mask overlap check ---
def compute_iou(mask1, mask2):
    """Intersection over union of two boolean masks."""
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0




# --- Mask R-CNN inference over a folder ---
def run_mask_rcnn_on_folder(input_dir, output_dir, vis_dir=None):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)

    for img_name in os.listdir(input_dir):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        image_path = os.path.join(input_dir, img_name)
        image = cv2.imread(image_path)
        outputs = predictor(image)
        pred_masks = outputs["instances"].pred_masks.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy().astype(int)

        kept_masks = []
        for idx, mask in enumerate(pred_masks):
            if scores[idx] < config.SCORE_THRESH:
                continue
            if any(compute_iou(mask, m) > config.MASK_IOU_DEDUP for m in kept_masks):
                continue
            kept_masks.append(mask)

            mask_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_mask_{idx}.png")
            cleaned = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            cv2.imwrite(mask_path, cleaned)

            if vis_dir:
                vis = image.copy()
                vis[mask] = [255, 255, 0]
                x1, y1, x2, y2 = boxes[idx]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"rockbolt: {scores[idx] * 100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                vis_out = os.path.join(vis_dir, f"{os.path.splitext(img_name)[0]}_vis_{idx}.png")
                cv2.imwrite(vis_out, vis)



# --- Mask and depth to point cloud ---
def convert_mask_to_pointcloud(mask_folder, depth_folder, output_folder):
    K = np.array(config.TRAIN_INTRINSICS)

    for filename in os.listdir(mask_folder):
        if not filename.endswith(".png"):
            continue

        mask_path = os.path.join(mask_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[SKIP] Couldn't read mask: {mask_path}")
            continue

        full_base = filename.split("_mask_")[0]
        base = full_base.split(".")[0].replace("_png", "")
        depth_file = next((f for f in os.listdir(depth_folder) if base in f), None)
        if not depth_file:
            print(f"[SKIP] No depth file found for {base}")
            continue

        depth_path = os.path.join(depth_folder, depth_file)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"[SKIP] Couldn't read depth: {depth_path}")
            continue

        # Handle 3-channel images mistakenly read
        if len(depth.shape) == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

        # An 8-bit depth PNG spreads the range over 256 levels, so it converts
        # with TRAIN_DEPTH_SCALE_8BIT. A 16-bit one is read as millimetres.
        if depth.dtype == np.uint8:
            depth_scale = config.TRAIN_DEPTH_SCALE_8BIT
        else:
            depth_scale = config.TRAIN_DEPTH_SCALE_16BIT

        # Resize mask if needed to match depth resolution
        if mask.shape != depth.shape:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Optional: dilate mask to increase points
        mask_binary = (mask > 127).astype(np.uint8)
        mask_binary = cv2.dilate(mask_binary, np.ones((3, 3), np.uint8), iterations=1)

        points = backproject(depth, mask_binary.astype(bool), K, depth_scale)

        if points.shape[0] == 0:
            print(f"[SKIP] No valid points in {filename}")
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_file = os.path.join(output_folder, f"{base}_pc.ply")
        o3d.io.write_point_cloud(pcd_file, pcd)
        print(f"[OK] Point cloud saved: {pcd_file}")



# --- Pose network training ---
def train_pointnet():
    dataset = SyntheticPoseDataset(pose_csv_file, num_points=config.NUM_POINTS)
    dataloader = DataLoader(dataset, batch_size=config.POSE_BATCH_SIZE, shuffle=True)
    model = PointNet2PoseRegression().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.POSE_LR)

    for epoch in range(config.POSE_EPOCHS):
        model.train()
        total_loss = 0
        for points, R_gt, T_gt in dataloader:
            points, R_gt, T_gt = points.to(device), R_gt.to(device), T_gt.to(device)
            optimizer.zero_grad()
            R_pred, T_pred = model(points)
            loss = pose_loss(R_pred, T_pred, R_gt, T_gt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{config.POSE_EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")
        if (epoch + 1) % config.POSE_CKPT_EVERY == 0:
            ckpt = os.path.join(pointnet_ckpt_folder, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"[INFO] Saved: {ckpt}")



# --- Pose network inference on real clouds ---
def test_pointnet_on_real_pcd(input_folder, output_csv):
    model = PointNet2PoseRegression().to(device)
    model.load_state_dict(torch.load(config.POSE_WEIGHTS))
    model.eval()

    results = []
    for file in os.listdir(input_folder):
        if not file.endswith(".ply"):
            continue
        pcd = o3d.io.read_point_cloud(os.path.join(input_folder, file))
        pts = np.asarray(pcd.points)
        if pts.shape[0] < config.NUM_POINTS:
            continue
        idx = np.random.choice(pts.shape[0], config.NUM_POINTS, replace=False)
        pts_sampled = torch.tensor(pts[idx], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            R_pred, T_pred = model(pts_sampled)
        R_pred = R_pred.squeeze().cpu().numpy().tolist()
        T_pred = T_pred.squeeze().cpu().numpy().tolist()
        results.append({"filename": file, "Tx": T_pred[0], "Ty": T_pred[1], "Tz": T_pred[2],
                        "Qx": R_pred[0], "Qy": R_pred[1], "Qz": R_pred[2], "Qw": R_pred[3]})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Pose results saved to {output_csv}")



# --- Entry point ---
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    setup_datasets()
    train_mask_rcnn()
    run_mask_rcnn_on_folder(test_images, mask_output_test, mask_output_test_vis)
    run_mask_rcnn_on_folder(all_images, mask_output_all)
    convert_mask_to_pointcloud(mask_output_all, all_depth, pointnet_test_pc_folder)
    train_pointnet()
    test_pointnet_on_real_pcd(pointnet_test_pc_folder, pointnet_output_csv)
    print("\nTraining and testing pipeline completed.")
