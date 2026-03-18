import numpy as np
from boxmot import ByteTrack

from whale_mot.common.types import Detection, Track
from whale_mot.trackers.base import BaseTracker

class ByteTrackTracker(BaseTracker):
    def __init__(self, 
                 min_confidence: float = 0.25,
                 track_threshold: float = 0.5,
                 match_threshold: float = 0.5,
                 track_buffer: int = 30,
                 frame_rate: int = 30)-> None:
        
        self.track_parameters = [min_confidence, 
                                 track_threshold, 
                                 match_threshold, 
                                 track_buffer, 
                                 frame_rate]
        
        self.tracker = self._build_tracker()
        
        
        
    def _build_tracker(self):
        return ByteTrack(
            min_conf=self.track_parameters[0],
            track_thresh=self.track_parameters[1],
            match_thresh=self.track_parameters[2],
            track_buffer=self.track_parameters[3],
            frame_rate=self.track_parameters[4],
        )
        
    def _detections_to_array(self, 
                             detections: list[Detection]) -> np.ndarray:
        if not detections:
            return np.empty((0, 6), dtype=np.float32)

        return np.array(
            [
                [
                    float(d.xyxy[0]),
                    float(d.xyxy[1]),
                    float(d.xyxy[2]),
                    float(d.xyxy[3]),
                    float(d.confidence),
                    float(d.class_id),
                ]
                for d in detections
            ],
            dtype=np.float32,
        )
    
    def _outputs_to_tracks(self, outputs, frame_idx: int) -> list[Track]:
        if outputs is None:
            return []

        outputs = np.asarray(outputs)
        if outputs.size == 0:
            return []

        if outputs.ndim == 1:
            outputs = outputs[None, :]

        tracks: list[Track] = []

        for row in outputs:
            if len(row) < 5:
                continue

            x1, y1, x2, y2 = map(float, row[:4])
            track_id = int(row[4])
            conf = float(row[5]) if len(row) > 5 else 1.0
            class_id = int(row[6]) if len(row) > 6 else 0
            tracks.append(
                Track(
                    track_id=track_id,
                    xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                    confidence=conf,
                    class_id=class_id,
                    frame_idx=frame_idx,
                )
            )
        return tracks

    def update(self, detections: list[Detection], 
               image: np.ndarray | None = None) -> list[Track]:

        frame_idx = detections[0].frame_idx if detections else -1
        dets = self._detections_to_array(detections)
        outputs = self.tracker.update(dets, image)

        return self._outputs_to_tracks(outputs, frame_idx)

    def reset(self) -> None:
        self.tracker = self._build_tracker()