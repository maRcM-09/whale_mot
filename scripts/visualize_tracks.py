import csv
from collections import defaultdict
from pathlib import Path
import argparse
import cv2


def load_tracks(csv_path):
    tracks_by_frame = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            tracks_by_frame[frame].append({
                "id": int(row["id"]),
                "x1": int(float(row["x1"])),
                "y1": int(float(row["y1"])),
                "x2": int(float(row["x2"])),
                "y2": int(float(row["y2"])),
            })
    return tracks_by_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tracks_by_frame = load_tracks(args.tracks)

    cap = cv2.VideoCapture(args.video)
    # Create resizable window first
    cv2.namedWindow("YOLO11 Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO11 Tracking", 800, 600)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    # saving for now at this size
    fps = 30
    w = int(cap.get(3))
    h = int(cap.get(4))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        f"{output_path}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for tr in tracks_by_frame.get(frame_idx, []):
            cv2.rectangle(frame, (tr["x1"], tr["y1"]), (tr["x2"], tr["y2"]), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'ID {tr["id"]}',
                (tr["x1"], max(20, tr["y1"] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.imshow("YOLO11 Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()