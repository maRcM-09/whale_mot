import numpy as np
from norfair import Detection as NorfairDetection
from norfair import Tracker as NorfairCoreTracker

from whale_mot.common.types import Detection, Track
from whale_mot.trackers.base import BaseTracker


def centroid_distance(
        detection,
        tracked_object):
    return np.linalg.norm(
        detection.points - tracked_object.estimate
        )

class NorfairTracker(BaseTracker):
    def __init__(self, 
                 distance_threshold: float = 30.0):
        self.distance_threshold = distance_threshold
        self.tracker = NorfairCoreTracker(
            distance_function=centroid_distance,
            distance_threshold=distance_threshold
        )

    def update(self,
                detections: list[Detection],
                image=None) -> list[Track]:
        norfair_detections = []
        for d in detections:
            x1, y1, x2, y2 = d.xyxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            norfair_detections.append(
                NorfairDetection(
                    points=np.array([[cx, cy]], dtype=float),
                    scores=np.array([d.confidence], dtype=float),
                    data={"class_id": d.class_id, 
                            "xyxy": d.xyxy, 
                            "frame_idx": d.frame_idx},
                ))
        
        tracked_objects = self.tracker.update(norfair_detections)
        tracks : list[Track] = []
        frame_idx = detections[0].frame_idx if detections else -1

        for obj in tracked_objects:
            cx, cy = obj.estimate[0]

            if getattr(obj, "last_detection", None) is not None and obj.last_detection.data.get("xyxy") is not None:
                xyxy = np.array(obj.last_detection.data["xyxy"], dtype=float)
                class_id = int(obj.last_detection.data["class_id"])
            else:
                xyxy = np.array(
                    [cx - 40, 
                        cy - 20, 
                        cx + 40, 
                        cy + 20], dtype=float)
                class_id = 0

            tracks.append(
                Track(
                    track_id=int(obj.id),
                    xyxy=xyxy,
                    confidence=1.0,
                    class_id=class_id,
                    frame_idx=frame_idx,
                )
            )

        return tracks
    
    def reset(self) -> None:
        self.tracker = NorfairCoreTracker(
            distance_function=centroid_distance,
            distance_threshold=self.distance_threshold,
        )
