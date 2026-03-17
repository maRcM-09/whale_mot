from dataclasses import dataclass, field
from typing import DefaultDict
from typing import Any
import numpy as np

@dataclass
class Detection:
    """
    Represents a single detection in a video frame.

    Parameters:
    - xyxy: A numpy array of shape (4,) representing the bounding box coordinates in the format (x_min, y_min, x_max, y_max).
    - confidence: A float representing the confidence score of the detection.
    - class_id: An integer representing the class ID of the detected object.
    - frame_idx: An integer representing the index of the frame in which the detection was made.
    - metadata: A dictionary containing any additional information related to the detection.
    """
    xyxy: np.ndarray
    confidence: float
    class_id: int
    frame_idx: int
    metadata: DefaultDict = field(default_factory=DefaultDict)

@dataclass
class Track:
    """
    Represents a single track of an object across multiple video frames.
    Parameters:
        - track_id: An integer representing the unique ID of the track.
        - xyxy: A numpy array of shape (4,) representing the current bounding box coordinates of the tracked object in the format (x_min, y_min, x_max, y_max).
        - confidence: A float representing the confidence score of the current detection associated with the track.
        - class_id: An integer representing the class ID of the tracked object.
        - frame_idx: An integer representing the index of the current frame in which the track is being updated.
        - is_confirmed: A boolean indicating whether the track has been confirmed (i.e., it has been detected in multiple frames).
        - age: An integer representing the age of the track, which can be used to determine how long the track has been active.
        - hits: An integer representing the number of times the track has been updated with a detection.
        - metadata: A dictionary containing any additional information related to the track.
    """
    track_id: int
    xyxy: np.ndarray
    confidence: float
    class_id: int
    frame_idx: int
    is_confirmed: bool = True
    age: int = 0
    hits: int = 0
    metadata: DefaultDict = field(default_factory=DefaultDict)
