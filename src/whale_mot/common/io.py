import csv
from pathlib import Path
from whale_mot.common.types import Detection, Track

def save_detections_csv(
        detections:list[Detection], 
        path: str | Path) -> None:
    
    if isinstance(path, str):
        path = Path(path)
    
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['frame', 
             'x1',
             'y1', 
             'x2',
             'y2', 
             'confidence', 
             'class_id'])
        for det in detections:
            writer.writerow(
                [det.frame_idx, 
                 *det.xyxy.tolist(), 
                 det.confidence, 
                 det.class_id])
def save_tracks_csv(
        tracks: list[Track], 
        path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True, 
        exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame", 
             "id", 
             "x1", 
             "y1", 
             "x2", 
             "y2", 
             "confidence", 
             "class_id"])
        for t in tracks:
            writer.writerow(
                [t.frame_idx, 
                 t.track_id, 
                 *t.xyxy.tolist(), 
                 t.confidence, 
                 t.class_id])