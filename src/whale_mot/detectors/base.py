from abc import ABC, abstractmethod
import numpy as np
from whale_mot.common.types import Detection

class BaseDetector(ABC):
    @abstractmethod
    def predict_frame(self,
                      imgage: np.ndarray,
                      frame_idx: int) -> list[Detection]:
        raise NotImplementedError()