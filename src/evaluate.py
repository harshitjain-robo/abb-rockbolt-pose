"""Scores the pipeline's position estimates against hand-measured ground truth.

The inference pipeline writes two positions per detected bolt: the midpoint of
the two PCA endpoints, and the translation the pose network predicted. They are
not the same number and they do not agree, so this script asks which one you
want scored and says so in its output.

Ground truth is a CSV, one row per ground-truthed bolt, positions in metres in
the camera frame:

    filename,x,y,z
    001,-0.1435,-0.0004,0.9056

Add an instance column if a frame holds more than one ground-truthed bolt.
Without one, each ground truth row is matched to the nearest unused detection in
its frame, and the summary says that is what happened, so the figure is not read
as stricter than it is.

    python src/evaluate.py --predictions outputs/realtime/pose_results.csv \
                           --ground-truth data/realtime/ground_truth.csv
"""

import argparse

import numpy as np
import pandas as pd

# --- Which columns hold which estimate ---
ESTIMATE_COLUMNS = {
    "centroid": ["bolt_cx", "bolt_cy", "bolt_cz"],
    "translation": ["tx", "ty", "tz"],
}


def frame_key(value):
    """Normalises a frame name so '001', '001.png' and 001 all compare equal.

    The prediction CSV writes the base name while a hand-typed ground truth file
    often keeps the extension, and a spreadsheet will happily turn 001 into the
    integer 1.
    """
    text = str(value).strip()
    if "." in text:
        text = text.rsplit(".", 1)[0]
    return text.zfill(3) if text.isdigit() else text


def load_predictions(path, estimate):
    """Reads the inference CSV and returns one row per detected instance."""
    df = pd.read_csv(path)
    columns = ESTIMATE_COLUMNS[estimate]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} has no {', '.join(missing)} column. "
                         f"Is it the CSV written by infer_pipeline.py?")

    out = pd.DataFrame({
        "frame": df["filename"].map(frame_key),
        "x": df[columns[0]],
        "y": df[columns[1]],
        "z": df[columns[2]],
    })
    # Detections keep the order infer_pipeline wrote them in, so the nth row for
    # a frame is that frame's nth instance.
    out["instance"] = out.groupby("frame").cumcount()
    return out


def load_ground_truth(path):
    """Reads the measured positions. The instance column is optional."""
    df = pd.read_csv(path)
    for column in ("filename", "x", "y", "z"):
        if column not in df.columns:
            raise SystemExit(f"{path} needs a {column} column. "
                             f"Expected header: filename,x,y,z[,instance]")

    out = pd.DataFrame({
        "frame": df["filename"].map(frame_key),
        "x": df["x"],
        "y": df["y"],
        "z": df["z"],
    })
    out["instance"] = df["instance"] if "instance" in df.columns else np.nan
    return out


def match(predictions, ground_truth):
    """Pairs each ground truth row with a detection from the same frame.

    Returns the pairs, the ground truth rows whose frame produced no detection
    at all, and whether any pairing had to fall back to nearest distance.
    """
    pairs = []
    unmatched = []
    used_nearest = False

    for frame, truth_rows in ground_truth.groupby("frame", sort=True):
        candidates = predictions[predictions["frame"] == frame].copy()

        if candidates.empty:
            unmatched.extend(truth_rows["frame"].tolist())
            continue

        for _, truth in truth_rows.iterrows():
            if candidates.empty:
                unmatched.append(frame)
                continue

            if not pd.isna(truth["instance"]):
                picked = candidates[candidates["instance"] == int(truth["instance"])]
                if picked.empty:
                    unmatched.append(frame)
                    continue
                index = picked.index[0]
            else:
                # No instance given, so take the closest detection still going
                # spare. Removing it afterwards stops two ground truth rows in
                # one frame from both claiming the same detection.
                used_nearest = True
                offsets = candidates[["x", "y", "z"]].to_numpy() - \
                    np.array([truth["x"], truth["y"], truth["z"]])
                index = candidates.index[int(np.argmin(np.linalg.norm(offsets, axis=1)))]

            prediction = candidates.loc[index]
            candidates = candidates.drop(index)

            error = np.array([prediction["x"] - truth["x"],
                              prediction["y"] - truth["y"],
                              prediction["z"] - truth["z"]])
            pairs.append({
                "frame": frame,
                "instance": int(prediction["instance"]),
                "pred_x": prediction["x"], "pred_y": prediction["y"], "pred_z": prediction["z"],
                "true_x": truth["x"], "true_y": truth["y"], "true_z": truth["z"],
                "error_x": error[0], "error_y": error[1], "error_z": error[2],
                "error": float(np.linalg.norm(error)),
            })

    return pd.DataFrame(pairs), unmatched, used_nearest


def report(pairs, unmatched, used_nearest, estimate):
    """Prints the per-instance table and the summary that goes in the README."""
    label = {"centroid": "PCA endpoint midpoint",
             "translation": "pose network translation"}[estimate]
    print(f"Scoring the {label} against ground truth.\n")

    if pairs.empty:
        print("Nothing matched. Check that the frame names line up.")
        return

    print(f"{'frame':<8}{'inst':<6}{'predicted (m)':<28}{'measured (m)':<28}{'error (m)':>10}")
    for _, row in pairs.iterrows():
        predicted = f"{row['pred_x']:>7.3f}{row['pred_y']:>8.3f}{row['pred_z']:>8.3f}"
        measured = f"{row['true_x']:>7.3f}{row['true_y']:>8.3f}{row['true_z']:>8.3f}"
        print(f"{row['frame']:<8}{row['instance']:<6}{predicted:<28}{measured:<28}{row['error']:>10.3f}")

    errors = pairs["error"].to_numpy()
    print(f"\n{len(errors)} instances matched across {pairs['frame'].nunique()} frames")
    print(f"mean error    {errors.mean():.3f} m")
    print(f"median error  {np.median(errors):.3f} m")
    print(f"worst error   {errors.max():.3f} m  (frame {pairs.loc[errors.argmax(), 'frame']})")
    print(f"per axis mean absolute error   "
          f"x {pairs['error_x'].abs().mean():.3f} m   "
          f"y {pairs['error_y'].abs().mean():.3f} m   "
          f"z {pairs['error_z'].abs().mean():.3f} m")

    if unmatched:
        frames = ", ".join(sorted(set(unmatched)))
        print(f"\nNo detection above the score threshold in: {frames}")
        print("Those frames are excluded from the mean, so quote the detection "
              "rate alongside it.")

    if used_nearest:
        print("\nGround truth carried no instance column, so each measurement "
              "was paired with the nearest detection in its frame.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True,
                        help="pose_results.csv written by infer_pipeline.py")
    parser.add_argument("--ground-truth", required=True,
                        help="CSV of measured positions: filename,x,y,z[,instance]")
    parser.add_argument("--estimate", choices=sorted(ESTIMATE_COLUMNS), default="centroid",
                        help="which position to score (default: centroid)")
    parser.add_argument("--out", help="optional CSV to write the per-instance errors to")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions, args.estimate)
    ground_truth = load_ground_truth(args.ground_truth)
    pairs, unmatched, used_nearest = match(predictions, ground_truth)
    report(pairs, unmatched, used_nearest, args.estimate)

    if args.out and not pairs.empty:
        pairs.to_csv(args.out, index=False)
        print(f"\nPer-instance errors written to {args.out}")


if __name__ == "__main__":
    main()
