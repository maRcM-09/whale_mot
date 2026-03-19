"""
In this script, we implement the tracking metrics for evaluating 
the performance of multi-object tracking algorithms. 

The metrics include:

1. MOTA (Multiple Object Tracking Accuracy): 
    This metric combines false positives, false negatives, 
    and identity switches to provide an overall accuracy 
    measure of the tracking performance.
2. MOTP (Multiple Object Tracking Precision): 
    This metric measures the precision of the
3. IDF1 (ID F1 Score): 
    This metric evaluates the accuracy of the identity assignment in tracking, 
    considering both precision and recall of correctly identified tracks.
4. IDP (ID Precision): 
    This metric measures the precision of correctly identified tracks.
5. IDR (ID Recall): 
    This metric measures the recall of correctly identified tracks.
6. IDSW (Identity Switches): 
    This metric counts the number of times a tracked object is incorrectly assigned a new identity.
7. HOTA (Higher Order Tracking Accuracy): 
    This metric evaluates the tracking performance by considering both 
    detection and association errors, providing a more comprehensive 
    assessment of tracking accuracy.

We will use the `motmetrics` library to compute these metrics. 
The library provides a convenient interface for evaluating 
tracking performance based on the ground truth and predicted tracks.

"""
from __future__ import annotations


from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import motmetrics as mm

@dataclass
class TrackingMetricResult:
    mota: float
    motp: float
    idf1: float
    idp: float
    idr: float
    idsw: int

def _validate_columns(
        df: pd.DataFrame,
        required: list[str],
        name: str) -> None:
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")

def _load_csv(path: str | Path , name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, ['image_id', 'x1', 'y1', 'x2', 'y2', 'score'], name)
    return df

def compute_tracking_metrics(
        gt_csv: str | Path,
        pred_csv: str | Path,
        iou_threshold: float = 0.5) -> TrackingMetricResult:
    
    gt_df = _load_csv(gt_csv, "ground truth")
    pred_df = _load_csv(pred_csv, "predictions")

    # Create a MOTAccumulator to accumulate tracking results
    acc = mm.MOTAccumulator(auto_id=True)

    # Process each frame and update the accumulator
    for frame_id in sorted(gt_df['image_id'].unique()):
        gt_frame = gt_df[gt_df['image_id'] == frame_id]
        pred_frame = pred_df[pred_df['image_id'] == frame_id]

        gt_ids = gt_frame['track_id'].tolist()
        pred_ids = pred_frame['track_id'].tolist()

        gt_boxes = gt_frame[['x1', 'y1', 'x2', 'y2']].values
        pred_boxes = pred_frame[['x1', 'y1', 'x2', 'y2']].values

        # Compute IoU and update the accumulator
        acc.update(
            gt_ids, 
            pred_ids, 
            mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=iou_threshold)
        )


    # Compute metrics using the accumulator
    mh = mm.metrics.create()
    summary = mh.compute(acc, 
                         metrics=['mota', 'motp', 'idf1', 'idp', 'idr', 'idsw'], 
                         name='overall')

    return TrackingMetricResult(
        mota=summary.loc['overall', 'mota'],
        motp=summary.loc['overall', 'motp'],
        idf1=summary.loc['overall', 'idf1'],
        idp=summary.loc['overall', 'idp'],
        idr=summary.loc['overall', 'idr'],
        idsw=int(summary.loc['overall', 'idsw'])
    )