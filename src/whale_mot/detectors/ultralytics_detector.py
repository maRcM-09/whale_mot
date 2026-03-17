import numpy as np
from ultralytics import YOLO

from whale_mot.common.types import Detection
from whale_mot.detectors.base import BaseDetector

class UltralyticsDetector(BaseDetector):
    def __init__(self, 
                 model_name: str,
                 device: str = 'cpu',
                 conf: float = 0.25):
        self.model = YOLO(model_name)
        self.device = device
        self.conf_threshold = conf
        self.model.to(device)

    def predict_frame(self,
                      image: np.ndarray,
                      frame_idx: int) -> list[Detection]:
        
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        results = results[0]
        detections : list[Detection] = []

        if results.boxes is None:
            return detections
        
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, scores, classes):
            detections.append(
                Detection(
                    xyxy = box.astype(float),
                    confidence = float(conf),
                    class_id = int(cls_id),
                    frame_idx = frame_idx
                ))
        return detections