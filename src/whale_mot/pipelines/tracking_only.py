from pathlib import Path
import csv
import cv2
import numpy as np

from whale_mot.common.io import save_tracks_csv
from whale_mot.trackers.registry import build_tracker
from whale_mot.common.types import Detection


def run_tracking_only(cfg: dict) -> None:
    tracker = build_tracker(cfg["tracker"])

    video_path = Path(cfg["video_path"])
    detection_csv = Path(cfg["detection_csv"])
    output_csv = Path(cfg["output_track_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Load detections from CSV into Detection objects grouped by frame
    detections_by_frame: dict[int, list[Detection]] = {}

    with open(detection_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # support either "frame" or "frame_idx"
            frame_idx = int(row["frame"])

            det = Detection(
                xyxy=np.array(
                    [
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["x2"]),
                        float(row["y2"]),
                    ],
                    dtype=float,
                ),
                confidence=float(row["confidence"]),
                class_id=int(row["class_id"]),
                frame_idx=frame_idx,
            )
            detections_by_frame.setdefault(frame_idx, []).append(det)

    all_tracks = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detections_by_frame.get(frame_idx, [])
        tracks = tracker.update(detections, frame)
        all_tracks.extend(tracks)
        frame_idx += 1

    cap.release()

    save_tracks_csv(all_tracks, output_csv)

    print(f"Saved tracks to: {output_csv}")