from abc import ABC, abstractmethod
import numpy as np
from whale_mot.common.types import Detection, Track

class BaseTracker(ABC):
    @abstractmethod
    def update(self,
               detections: list[Detection],
               image: np.ndarray | None = None) -> list[Track]:
        """
        Update the tracker with new detections.
        """
        raise NotImplementedError
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the tracker to its initial state.
        """
        raise NotImplementedError