# Reference-based exercise comparison (accuracy branch)

This branch adds `reference_accuracy.py` to the existing YOLOv8 + MediaPipe exercise analyzer. It supports **push-ups and sit-ups** with a reference video and either a second video or a webcam.

## Setup

Use Python 3.10–3.12 with a MediaPipe version that includes `mp.solutions.pose`:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install opencv-python numpy "mediapipe==0.10.21" ultralytics
```

Ultralytics may download YOLOv8n weights on first use. Make sure your entire body is visible and both videos show similar camera angles. Avoid long videos for this initial implementation (reference capped at 180 seconds).

## Compare two uploaded/local video files

```bash
python reference_accuracy.py --exercise pushup --reference reference.mp4 --candidate attempt.mp4 --output results
```

## Compare against a live camera

```bash
python reference_accuracy.py --exercise situp --reference reference.mp4 --camera 0 --output results
```

Press **Q** to finish webcam capture. Outputs are `report.json` and `frames.csv`; video-to-video mode also writes `annotated.mp4`. Input videos are selected by their local paths; there is no browser upload interface in this branch.

The tool compares elbow, hip, knee, and body angles, using dynamic time warping to align different speeds in offline mode and a rolling alignment window in live mode. Scores measure similarity to the chosen reference, **not objective technique correctness, exercise safety, or medical risk**. Camera viewpoint, occlusion, individual anatomy, and reference quality affect scores. The live alignment is heuristic and can stall at the end of a reference sequence; it does not automatically loop reps.

No automated end-to-end video tests have been run yet. Test with representative footage before relying on the results.
