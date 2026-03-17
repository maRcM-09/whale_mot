# In this script, we can try a detector on its own and save its output in a csv file. 
# This is useful for debugging and for comparing different detectors 
# without the influence of the tracker. Or comparing different trackers without the influence of the detector.

from pathlib import Path
import cv2
import csv
from dataclasses import asdict
from whale_mot.common.io import save_detections_csv

from whale_mot.detectors.registry import build_detector
def run_detection_only(cfg: dict) -> None:
    detector = build_detector(cfg["detector"])

    video_path = Path(cfg["video_path"])
    output_csv = Path(cfg["output_detection_csv"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    all_detections = []
    frame_idx = 0
    keys = ["frame_idx", "x1", "y1", "x2", "y2", "confidence", "class_id"]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        detections = detector.predict_frame(frame, frame_idx)
        all_detections.extend([d for d in detections])

        frame_idx += 1

    cap.release()

    save_detections_csv(all_detections, output_csv)    

    print(f"Saved detections to: {output_csv}")