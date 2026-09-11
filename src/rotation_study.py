#!/usr/bin/env python3
"""Why the learned rotation collapsed, measured rather than asserted.

Two experiments, both reproducible from this repository alone. Neither needs the
dataset, the trained weights, or anything the pipeline recorded during the
project, because the bolts are procedural and the model is small.

1. Identifiability. Spin a bolt about its own long axis and measure how much the
   observed cloud changes, against the change you get from simply redrawing the
   random sample. For a straight bolt the two are the same size, which means the
   rotation label is not a function of the input and no model can recover it.

2. What is recoverable. Train the network in pose_model.py on clouds from
   generate_synthetic_data.py, then score its rotation against PCA on the same
   clouds. PCA has no parameters and sees no training data.

    python src/rotation_study.py
    python src/rotation_study.py --train 500 --test 150 --epochs 50 --device cpu
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

import config
from generate_synthetic_data import (add_rock_occlusions, apply_pose,
                                     generate_random_pose, generate_rockbolt)
from geometry import axis_angle_deg, geodesic_angle_deg, principal_axis, unit
from pose_model import PointNet2PoseRegression, pose_loss

BOLT_AXIS = np.array([0.0, 0.0, 1.0])      # bolts are generated along +z
SPIN_ANGLES = (30, 90, 180)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_points(cloud, num_points, rng):
    idx = rng.choice(len(cloud), num_points, replace=len(cloud) < num_points)
    return cloud[idx]


def chamfer_mm(a, b):
    """Mean nearest-neighbour distance both ways, in millimetres."""
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return 0.5 * (d.min(1).mean() + d.min(0).mean()) * 1000


def identifiability(length=0.6, radius=0.02, stride=5):
    """Does spinning a bolt about its own axis change what the camera sees?"""
    print("Experiment 1: can the rotation label be recovered from the cloud?\n")
    header = f"  {'bolt':<10}{'redraw only':>15}" + "".join(
        f"{'spin ' + str(a) + ' deg':>15}" for a in SPIN_ANGLES)
    print(header)

    results = {}
    base = Rotation.from_euler("xyz", [20, -15, 35], degrees=True)
    for bend in (False, True):
        np.random.seed(11)
        # Stride rather than truncate, so the sample spans the whole bolt
        # and a bend near the far end is not left out of the comparison.
        first = generate_rockbolt(length, radius, bend=bend)[::stride]
        np.random.seed(22)
        second = generate_rockbolt(length, radius, bend=bend)[::stride]

        # Same bolt, new random draw, no spin. This is the noise floor.
        row = [chamfer_mm(base.apply(first), base.apply(second))]
        for angle in SPIN_ANGLES:
            spun = base * Rotation.from_euler("z", angle, degrees=True)
            row.append(chamfer_mm(base.apply(first), spun.apply(second)))

        label = "bent" if bend else "straight"
        results[label] = row
        print(f"  {label:<10}" + "".join(f"{v:>12.2f} mm" for v in row))

    floor, spun = results["straight"][0], results["straight"][1:]
    if max(spun) < floor * 1.25:
        print("\n  For a straight bolt every spin sits at the redraw noise floor. The\n"
              "  same cloud is consistent with any rotation about the bolt axis, so\n"
              "  half the training set carries a label the input cannot determine.")
    return results


def make_dataset(count, num_points, rng):
    clouds, quats, axes = [], [], []
    for i in range(count):
        length = np.random.uniform(*config.SYNTH_LENGTH_RANGE)
        radius = np.random.uniform(*config.SYNTH_RADIUS_RANGE)
        bolt = generate_rockbolt(length, radius, bend=bool(i % 2))
        bolt = add_rock_occlusions(bolt, config.SYNTH_OCCLUSION_DENSITY)

        quat, trans = generate_random_pose()
        cloud = apply_pose(bolt, quat, trans)

        clouds.append(sample_points(cloud, num_points, rng))
        quats.append(quat)
        axes.append(Rotation.from_quat(quat).apply(BOLT_AXIS))

    return (torch.tensor(np.array(clouds), dtype=torch.float32),
            np.array(quats), np.array(axes))


def train(model, clouds, quats, epochs, batch, lr, device):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    targets = torch.tensor(quats, dtype=torch.float32)
    zeros = torch.zeros(batch, 3, device=device)
    for _ in range(epochs):
        order = torch.randperm(len(clouds))
        for start in range(0, len(clouds), batch):
            picked = order[start:start + batch]
            if len(picked) < 2:
                continue
            quat_pred, trans_pred = model(clouds[picked].to(device))
            loss = pose_loss(quat_pred, trans_pred,
                             targets[picked].to(device), zeros[:len(picked)])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model


def recovery(args, rng):
    """Train the shipped model, then score it against PCA on the same clouds."""
    print("\n\nExperiment 2: what part of the orientation is recoverable?\n")

    train_clouds, train_quats, _ = make_dataset(args.train, config.NUM_POINTS, rng)
    test_clouds, test_quats, test_axes = make_dataset(args.test, config.NUM_POINTS, rng)

    model = PointNet2PoseRegression().to(args.device)
    train(model, train_clouds, train_quats, args.epochs,
          config.POSE_BATCH_SIZE, config.POSE_LR, args.device)
    model.eval()
    with torch.no_grad():
        predicted, _ = model(test_clouds.to(args.device))
    predicted = F.normalize(predicted, dim=1).cpu().numpy()

    scores = {
        "learned quaternion, full orientation": [
            geodesic_angle_deg(p, t) for p, t in zip(predicted, test_quats)],
        "learned quaternion, bolt axis only": [
            axis_angle_deg(Rotation.from_quat(unit(p)).apply(BOLT_AXIS), a)
            for p, a in zip(predicted, test_axes)],
        "PCA on the cloud, no learning": [
            axis_angle_deg(principal_axis_of(c.numpy()), a)
            for c, a in zip(test_clouds, test_axes)],
    }

    print(f"  {'method':<40}{'median':>11}{'90th pct':>12}")
    for name, values in scores.items():
        values = np.array(values)
        print(f"  {name:<40}{np.median(values):>8.2f} deg{np.percentile(values, 90):>9.2f} deg")
    print(f"\n  Guessing at random would score about 90 deg on full orientation.")
    return scores


def principal_axis_of(points):
    """principal_axis takes an Open3D cloud, this takes the array directly."""
    centred = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(centred.T))
    return vectors[:, np.argmax(values)]


def draw_figure(identity_results, scores, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    labels = ["redraw\nonly"] + [f"spin\n{a} deg" for a in SPIN_ANGLES]
    width = 0.38
    positions = np.arange(len(labels))
    ax.bar(positions - width / 2, identity_results["straight"], width,
           label="straight bolt", color="tab:blue")
    ax.bar(positions + width / 2, identity_results["bent"], width,
           label="bent bolt", color="tab:orange")
    ax.axhline(identity_results["straight"][0], color="tab:blue", ls="--", lw=1,
               label="noise floor, straight")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("change in the observed cloud (mm)")
    ax.set_title("Spinning a straight bolt does not change what the camera sees")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    names = list(scores)
    data = [np.array(scores[n]) for n in names]
    parts = ax2.boxplot(data, vert=False, widths=0.55, showfliers=False,
                        patch_artist=True)
    for patch, colour in zip(parts["boxes"], ["tab:red", "tab:orange", "tab:green"]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.65)
    ax2.set_yticklabels(["learned quaternion\nfull orientation",
                         "learned quaternion\naxis only",
                         "PCA, no learning\naxis only"], fontsize=8)
    ax2.set_xlabel("angular error (degrees)")
    ax2.set_title("Orientation error on held-out synthetic bolts")
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  figure written to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=int, default=config.SYNTH_NUM_SAMPLES)
    p.add_argument("--test", type=int, default=150)
    p.add_argument("--epochs", type=int, default=config.POSE_EPOCHS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--figure", type=Path,
                   default=config.REPO_ROOT / "docs" / "results" / "rotation_study.png")
    p.add_argument("--no-figure", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    identity_results = identifiability()
    scores = recovery(args, rng)

    if not args.no_figure:
        draw_figure(identity_results, scores, args.figure)


if __name__ == "__main__":
    main()
