from pathlib import Path
import cv2

from whale_mot.detectors.registry import build_detector
from whale_mot.trackers.registry import build_tracker
from whale_mot.common.io import save_detections_csv, save_tracks_csv


def run_experiment(cfg: dict) -> None:
    detector = build_detector(cfg["detector"])
    tracker = build_tracker(cfg["tracker"])

    video_path = Path(cfg["video_path"])
    detection_csv = Path(cfg["output_detection_csv"])
    track_csv = Path(cfg["output_track_csv"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    all_detections = []
    all_tracks = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.predict_frame(frame, frame_idx)
        tracks = tracker.update(detections, frame)

        all_detections.extend(detections)
        all_tracks.extend(tracks)

        frame_idx += 1

    cap.release()

    save_detections_csv(all_detections, detection_csv)
    save_tracks_csv(all_tracks, track_csv)

    print(f"Saved detections to: {detection_csv}")
    print(f"Saved tracks to: {track_csv}")