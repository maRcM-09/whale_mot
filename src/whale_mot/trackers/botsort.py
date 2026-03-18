from __future__ import annotations

from pathlib import Path
import numpy as np
from boxmot import BotSort

from whale_mot.common.types import Detection, Track
from whale_mot.trackers.base import BaseTracker


class BoTSORTTracker(BaseTracker):
    def __init__(
        self,
        reid_weights: str | None,
        device: str = "cpu",
        half: bool = False,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        cmc_method: str = "ecc",
        frame_rate: int = 30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
        **kwargs,
    ) -> None:
        self.reid_weights = Path(reid_weights) if reid_weights is not None else None
        self.device = device
        self.half = half
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.cmc_method = cmc_method
        self.frame_rate = frame_rate
        self.fuse_first_associate = fuse_first_associate
        self.with_reid = with_reid
        self.kwargs = kwargs

        self.tracker = self._build_tracker()

    def _build_tracker(self):

        return BotSort(
            reid_weights=self.reid_weights,
            device=self.device,
            half=self.half,
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            proximity_thresh=self.proximity_thresh,
            appearance_thresh=self.appearance_thresh,
            cmc_method=self.cmc_method,
            frame_rate=self.frame_rate,
            fuse_first_associate=self.fuse_first_associate,
            with_reid=self.with_reid,
            **self.kwargs,
        )

    def _detections_to_array(self, detections: list[Detection]) -> np.ndarray:
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

        tracks = []
        for row in outputs:
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

    def update(self, detections: list[Detection], image=None) -> list[Track]:
        frame_idx = detections[0].frame_idx if detections else -1
        dets = self._detections_to_array(detections)
        outputs = self.tracker.update(dets, image)
        return self._outputs_to_tracks(outputs, frame_idx)

    def reset(self) -> None:
        self.tracker = self._build_tracker()