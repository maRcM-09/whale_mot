from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision 

@dataclass
class DetectionMetricResult:
    map: float
    map_50: float
    map_75: float
    map_1: float
    map_10: float
    map_100: float
    recall_at_iou: float
    precision_at_iou: float
    f1_at_iou: float
    tp: int
    fp: int
    fn: int

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

def compute_detection_metrics(
        gt_csv: str | Path,
        pred_csv: str | Path,
        iou_threshold: float = 0.5) -> DetectionMetricResult:
    
    gt_df = _load_csv(gt_csv, "ground truth")
    pred_df = _load_csv(pred_csv, "predictions")

    metric = MeanAveragePrecision(iou_threshold=iou_threshold)

    gt_boxes = []
    gt_labels = []
    for _, row in gt_df.iterrows():
        gt_boxes.append([row['x1'], row['y1'], row['x2'], row['y2']])
        gt_labels.append(row['class_id'])  

    pred_boxes = []
    pred_labels = []
    pred_scores = []
    
    for _, row in pred_df.iterrows():
        pred_boxes.append([row['x1'], row['y1'], row['x2'], row['y2']])
        pred_labels.append(row['class_id'])  
        pred_scores.append(row['score'])

    gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
    gt_labels_tensor = torch.tensor(gt_labels, dtype=torch.int64)
    pred_boxes_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
    pred_labels_tensor = torch.tensor(pred_labels, dtype=torch.int64)
    pred_scores_tensor = torch.tensor(pred_scores, dtype=torch.float32)

    metric.update(pred_boxes_tensor, pred_labels_tensor, pred_scores_tensor, gt_boxes_tensor, gt_labels_tensor)
    
    results = metric.compute()
    
    return DetectionMetricResult(
        map=results['map'],
        map_50=results['map_50'],
        map_75=results['map_75'],
        map_1=results['map_1'],
        map_10=results['map_10'],
        map_100=results['map_100'],
        recall_at_iou=results['recall_at_iou'],
        precision_at_iou=results['precision_at_iou'],
        f1_at_iou=results['f1_at_iou'],
        tp=results['tp'],
        fp=results['fp'],
        fn=results['fn']
    )

