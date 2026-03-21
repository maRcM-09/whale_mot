import csv
from collections import defaultdict
from pathlib import Path
import argparse
import cv2
import matplotlib.pyplot as plt

def load_detections(csv_path):
    detections_by_frame = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            detections_by_frame[frame].append({
                "confidence": float(row["confidence"]),
                "x1": int(float(row["x1"])),
                "y1": int(float(row["y1"])),
                "x2": int(float(row["x2"])),
                "y2": int(float(row["y2"])),
            })
    return detections_by_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--detection", required=True)
    args = parser.parse_args()

    detections_by_frame = load_detections(args.detection)

    cap = cv2.VideoCapture(args.video)
    confidence_list = []
    # Create resizable window first
    cv2.namedWindow("YOLO11 Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO11 Detection", 800, 600)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for det in detections_by_frame.get(frame_idx, []):
            cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'Whale {det["confidence"]:.2f}',
                (det["x1"], max(20, det["y1"] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            confidence_list.append(det["confidence"])
        cv2.imshow("YOLO11 Detection", frame)
        #slow down framrate for visualization
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    plt.figure(figsize=(8, 4))
    plt.hist(confidence_list, bins=20, range=(0, 1), alpha=0.7, color='blue')
    plt.title("Detection Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()


if __name__ == "__main__":
    main()