"""Paths, camera parameters and hyperparameters for the rockbolt pose pipeline.

Both pipelines read every path and constant from here. Override the two roots
with the ROCKBOLT_DATA and ROCKBOLT_OUTPUT environment variables, or edit the
defaults below.

Dataset layout expected under DATA_ROOT:

    train/images, train/_annotations.coco.json
    valid/images, valid/_annotations.coco.json
    test/images,  test/_annotations.coco.json
    _total_images, _total_depth
    synthetic_dataset/pcd, synthetic_dataset/pose_metadata_train.csv
    realtime/color, realtime/depth
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(os.environ.get("ROCKBOLT_DATA", REPO_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("ROCKBOLT_OUTPUT", REPO_ROOT / "outputs"))

# --- Mask R-CNN training data (COCO format) ---
TRAIN_IMAGES = DATA_ROOT / "train" / "images"
VAL_IMAGES = DATA_ROOT / "valid" / "images"
TEST_IMAGES = DATA_ROOT / "test" / "images"
TRAIN_JSON = DATA_ROOT / "train" / "_annotations.coco.json"
VAL_JSON = DATA_ROOT / "valid" / "_annotations.coco.json"
TEST_JSON = DATA_ROOT / "test" / "_annotations.coco.json"

# --- Full unsplit set, used to harvest masks for point cloud extraction ---
ALL_IMAGES = DATA_ROOT / "_total_images"
ALL_DEPTH = DATA_ROOT / "_total_depth"

# --- Synthetic point clouds for pose regression ---
SYNTHETIC_DIR = DATA_ROOT / "synthetic_dataset"
SYNTHETIC_PCD_DIR = SYNTHETIC_DIR / "pcd"
POSE_CSV = SYNTHETIC_DIR / "pose_metadata_train.csv"

# --- Inference input (RGB png plus aligned depth npy) ---
INFER_COLOR_DIR = DATA_ROOT / "realtime" / "color"
INFER_DEPTH_DIR = DATA_ROOT / "realtime" / "depth"

# --- Outputs ---
MASKRCNN_DIR = OUTPUT_ROOT / "mask-rcnn_output"
MASK_OUT_TEST = OUTPUT_ROOT / "test_masks"
MASK_OUT_TEST_VIS = OUTPUT_ROOT / "test_mask_vis"
MASK_OUT_ALL = OUTPUT_ROOT / "_total_masks"
PC_OUT_DIR = OUTPUT_ROOT / "pointclouds"
PC_OUT_TEST_DIR = OUTPUT_ROOT / "test_pointclouds"
POSE_CKPT_DIR = OUTPUT_ROOT / "pointnet-output"
POSE_OUT_CSV = OUTPUT_ROOT / "predicted_pose_output.csv"

INFER_OUT_ROOT = OUTPUT_ROOT / "realtime"
INFER_MASK_DIR = INFER_OUT_ROOT / "mask"
INFER_MASK_VIS_DIR = INFER_OUT_ROOT / "mask_vis"
INFER_PCD_DIR = INFER_OUT_ROOT / "pointclouds"
INFER_POSE_VIS_DIR = INFER_OUT_ROOT / "pose_vis"
INFER_CSV = INFER_OUT_ROOT / "pose_results.csv"

# --- Trained weights consumed by the inference pipeline ---
MASKRCNN_WEIGHTS = MASKRCNN_DIR / "model_final.pth"
POSE_WEIGHTS = POSE_CKPT_DIR / "checkpoint_epoch_50.pth"

# --- Camera used for the training split (320x240 depth export) ---
TRAIN_INTRINSICS = ((230.6816, 0.0, 166.752),
                    (0.0, 230.4453, 117.7266),
                    (0.0, 0.0, 1.0))

# The training-split depth maps are 8-bit PNGs, so a pixel value is a fraction
# of whatever range the export was normalised over rather than a distance.
# TRAIN_DEPTH_RANGE_M is that range, and a pixel then converts straight to
# metres. Change it to match the export if the working distance is different.
TRAIN_DEPTH_RANGE_M = 1.0
TRAIN_DEPTH_SCALE_8BIT = TRAIN_DEPTH_RANGE_M / 255
# 16-bit depth PNGs are read as millimetres, the usual convention.
TRAIN_DEPTH_SCALE_16BIT = 1.0 / 1000

# --- Camera used for the inference split (640x480 RGB-D, depth as npy) ---
INFER_WIDTH, INFER_HEIGHT = 640, 480
INFER_FX, INFER_FY = 609.8589, 610.0
INFER_CX, INFER_CY = 330.8136, 237.9314
INFER_DEPTH_SCALE = 4000.0

# --- Mask R-CNN ---
MASKRCNN_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
NUM_CLASSES = 1
SCORE_THRESH = 0.95
NMS_THRESH = 0.4
MASK_IOU_DEDUP = 0.7
IMS_PER_BATCH = 2
BASE_LR = 0.00025
MAX_ITER = 1000
CHECKPOINT_PERIOD = 200
EVAL_PERIOD = 200
ROI_BATCH_PER_IMAGE = 64
INPUT_SIZE = 640

# --- Pose regression ---
NUM_POINTS = 1024
POSE_EPOCHS = 50
POSE_BATCH_SIZE = 16
POSE_LR = 0.001
POSE_CKPT_EVERY = 10
MIN_POINTS_FOR_INFERENCE = 30

# --- Synthetic generator ---
SYNTH_NUM_SAMPLES = 500
SYNTH_LENGTH_RANGE = (0.3, 1.0)
SYNTH_RADIUS_RANGE = (0.01, 0.03)
SYNTH_OCCLUSION_DENSITY = 0.3
SYNTH_POINTS_PER_BOLT = 2048
