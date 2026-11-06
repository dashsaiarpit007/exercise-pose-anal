"""Reference-based movement comparison for uploaded videos or a webcam.

Uses the existing YOLO + MediaPipe ExerciseEngine. Scores describe similarity to
the supplied reference, NOT clinical correctness or injury risk.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from pose_exercise_analyzer import ExerciseEngine, calculate_angle, select_side

ANGLE_KEYS = ("elbow", "hip", "knee", "body")
WEIGHTS = np.array([0.35, 0.30, 0.20, 0.15], dtype=float)


def angles_from_joints(j):
    return np.array([
        calculate_angle(j["shoulder"], j["elbow"], j["wrist"]),
        calculate_angle(j["shoulder"], j["hip"], j["knee"]),
        calculate_angle(j["hip"], j["knee"], j["ankle"]),
        calculate_angle(j["shoulder"], j["hip"], j["ankle"]),
    ], dtype=float)


def extract(engine, frame):
    """Return angle vector or None when no reliable pose is available."""
    box = engine.detector.detect(frame)
    landmarks = engine.estimator.estimate(frame, box) if box else None
    if not landmarks:
        return None
    side = max(("left", "right"), key=lambda s: sum(
        landmarks[i][2] for i in (11, 13, 15, 23, 25, 27) if s == "left"
    ) if s == "left" else sum(landmarks[i][2] for i in (12, 14, 16, 24, 26, 28)))
    indices = (11, 13, 15, 23, 25, 27) if side == "left" else (12, 14, 16, 24, 26, 28)
    if min(landmarks[i][2] for i in indices) < 0.45:
        return None
    return angles_from_joints(select_side(landmarks))


def read_reference(path, engine, sample_fps=8, max_seconds=180):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open reference video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    stride = max(1, round(fps / sample_fps))
    vectors, timestamps = [], []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or i / fps > max_seconds:
                break
            if i % stride == 0:
                a = extract(engine, frame)
                if a is not None:
                    vectors.append(a)
                    timestamps.append(i / fps)
            i += 1
    finally:
        cap.release()
    if len(vectors) < 5:
        raise ValueError("Reference needs at least five frames with clearly visible body joints.")
    return np.stack(vectors), timestamps


def difference(a, b):
    # A 45-degree weighted error corresponds to zero similarity; clamp to 0..100.
    delta = np.abs(a - b)
    error = float(np.dot(WEIGHTS, delta))
    return max(0.0, 100.0 * (1.0 - error / 45.0)), delta


def dtw(reference, candidate):
    """Monotonic dynamic-time-warping alignment; O(N*M) time and memory."""
    n, m = len(reference), len(candidate)
    costs = np.full((n + 1, m + 1), np.inf)
    costs[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score, _ = difference(reference[i - 1], candidate[j - 1])
            costs[i, j] = (100 - score) + min(
                costs[i - 1, j], costs[i, j - 1], costs[i - 1, j - 1]
            )
    i, j = n, m
    pairs = []
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        options = [costs[i - 1, j - 1], costs[i - 1, j], costs[i, j - 1]]
        move = int(np.argmin(options))
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def feedback(delta, score):
    if score >= 85:
        return "Movement closely matches reference"
    idx = int(np.argmax(WEIGHTS * delta))
    return f"Review {ANGLE_KEYS[idx]} alignment ({delta[idx]:.0f} deg difference)"


def annotate(frame, score, note, progress):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 20), -1)
    cv2.putText(frame, f"Reference similarity: {score:.1f}%" if score is not None
                else "Pose not detected", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.putText(frame, note[:65], (12, 61), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 220, 255), 2)
    cv2.putText(frame, f"Reference progress: {progress:.0f}%", (12, 87),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)


def report(rows, reference, output_dir, exercise, mode):
    valid = [r for r in rows if r["similarity"] is not None]
    if not valid:
        raise ValueError("No valid poses found in candidate input; no score can be computed.")
    scores = [r["similarity"] for r in valid]
    deviations = {key: round(float(np.mean([r["deviations_deg"][key] for r in valid])), 2)
                  for key in ANGLE_KEYS}
    summary = {
        "exercise": exercise, "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_frames": len(reference),
        "total_sampled_frames": len(rows), "valid_pose_frames": len(valid),
        "pose_coverage_percent": round(100 * len(valid) / len(rows), 1),
        "mean_similarity_percent": round(float(np.mean(scores)), 2),
        "minimum_similarity_percent": round(float(np.min(scores)), 2),
        "maximum_similarity_percent": round(float(np.max(scores)), 2),
        "mean_joint_deviation_degrees": deviations,
        "assessment": feedback(np.array(list(deviations.values())), float(np.mean(scores))),
        "disclaimer": "Similarity to one reference only. Not a clinical safety, accuracy or injury assessment.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seconds", "similarity_percent", "reference_frame", "feedback"] +
                        [f"{key}_deviation_deg" for key in ANGLE_KEYS])
        for r in rows:
            writer.writerow([r["seconds"], r["similarity"], r["reference_frame"],
                             r["feedback"]] + [r["deviations_deg"].get(k) for k in ANGLE_KEYS])
    return summary


def run_video(engine, reference, source, output_dir, exercise, sample_fps=8):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot open candidate video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, round(fps / sample_fps))
    samples, times, frames = [], [], []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                frames.append(frame)
                samples.append(extract(engine, frame))
                times.append(round(i / fps, 3))
            i += 1
    finally:
        cap.release()
    valid_indices = [k for k, x in enumerate(samples) if x is not None]
    if not valid_indices:
        raise ValueError("No valid poses in candidate video.")
    pairs = dtw(reference, np.stack([samples[k] for k in valid_indices]))
    mapping = {}
    for ref_i, candidate_j in pairs:
        mapping.setdefault(valid_indices[candidate_j], []).append(ref_i)
    rows = []
    for k, (sample, seconds) in enumerate(zip(samples, times)):
        if sample is None:
            rows.append({"seconds": seconds, "similarity": None, "reference_frame": None,
                         "feedback": "Pose not detected", "deviations_deg": {}})
            continue
        ref_i = int(np.median(mapping[k]))
        score, delta = difference(reference[ref_i], sample)
        rows.append({"seconds": seconds, "similarity": round(score, 2),
                     "reference_frame": ref_i, "feedback": feedback(delta, score),
                     "deviations_deg": dict(zip(ANGLE_KEYS, np.round(delta, 2).tolist()))})
    # Annotated sampled frames: compact output suitable for demonstrations.
    h, w = frames[0].shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_dir / "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), min(fps, sample_fps), (w, h))
    try:
        for frame, row in zip(frames, rows):
            progress = (100 * row["reference_frame"] / max(1, len(reference) - 1)
                        if row["reference_frame"] is not None else 0)
            annotate(frame, row["similarity"], row["feedback"], progress)
            writer.write(frame)
    finally:
        writer.release()
    return report(rows, reference, output_dir, exercise, "video")


def run_webcam(engine, reference, camera, output_dir, exercise):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise ValueError(f"Cannot open webcam: {camera}")
    rows, ref_i = [], 0
    start = cv2.getTickCount() / cv2.getTickFrequency()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            a = extract(engine, frame)
            now = cv2.getTickCount() / cv2.getTickFrequency() - start
            if a is None:
                row = {"seconds": round(now, 3), "similarity": None, "reference_frame": None,
                       "feedback": "Pose not detected", "deviations_deg": {}}
            else:
                # Local monotonic matching; allow a small hold/backtrack to accommodate pace.
                lo, hi = max(0, ref_i - 2), min(len(reference), ref_i + 7)
                candidates = [difference(reference[j], a)[0] for j in range(lo, hi)]
                best = lo + int(np.argmax(candidates))
                ref_i = max(ref_i, best)
                score, delta = difference(reference[best], a)
                row = {"seconds": round(now, 3), "similarity": round(score, 2),
                       "reference_frame": best, "feedback": feedback(delta, score),
                       "deviations_deg": dict(zip(ANGLE_KEYS, np.round(delta, 2).tolist()))}
            rows.append(row)
            progress = 100 * ref_i / max(1, len(reference) - 1)
            annotate(frame, row["similarity"], row["feedback"], progress)
            cv2.imshow("Reference exercise comparison (Q to finish)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return report(rows, reference, output_dir, exercise, "webcam")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Path to reference exercise video")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate", help="Path to video to compare")
    source.add_argument("--camera", type=int, help="Webcam index, usually 0")
    parser.add_argument("--exercise", choices=["pushup", "situp"], required=True)
    parser.add_argument("--output", default="accuracy_results")
    args = parser.parse_args()
    engine = ExerciseEngine(args.exercise)
    reference, _ = read_reference(args.reference, engine)
    # Reset the pose tracker after analyzing the reference video.
    engine.estimator.pose.close()
    from pose_exercise_analyzer import PoseEstimator
    engine.estimator = PoseEstimator()
    output = Path(args.output)
    if args.candidate:
        summary = run_video(engine, reference, args.candidate, output, args.exercise)
    else:
        summary = run_webcam(engine, reference, args.camera, output, args.exercise)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
