from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


@dataclass
class DetectionMetricResult:
    map: float
    map_50: float
    map_75: float
    mar_1: float
    mar_10: float
    mar_100: float


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


def _load_gt_csv(path: str | Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, ["frame", "x1", "y1", "x2", "y2", "class_id"], name)
    return df


def _load_pred_csv(path: str | Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, ["frame", "x1", "y1", "x2", "y2", "confidence", "class_id"], name)
    return df


def _frame_to_target(frame_df: pd.DataFrame) -> dict:
    if len(frame_df) == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    boxes = torch.tensor(frame_df[["x1", "y1", "x2", "y2"]].to_numpy(), dtype=torch.float32)
    labels = torch.tensor(frame_df["class_id"].to_numpy(), dtype=torch.int64)

    return {
        "boxes": boxes,
        "labels": labels,
    }


def _frame_to_preds(frame_df: pd.DataFrame) -> dict:
    if len(frame_df) == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    boxes = torch.tensor(frame_df[["x1", "y1", "x2", "y2"]].to_numpy(), dtype=torch.float32)
    scores = torch.tensor(frame_df["confidence"].to_numpy(), dtype=torch.float32)
    labels = torch.tensor(frame_df["class_id"].to_numpy(), dtype=torch.int64)

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }


def compute_detection_metrics(
    gt_csv: str | Path,
    pred_csv: str | Path,
) -> DetectionMetricResult:
    gt_df = _load_gt_csv(gt_csv, "ground truth")
    pred_df = _load_pred_csv(pred_csv, "predictions")

    metric = MeanAveragePrecision()

    all_frames = sorted(set(gt_df["frame"].unique()).union(set(pred_df["frame"].unique())))

    preds = []
    targets = []

    for frame in all_frames:
        gt_frame = gt_df[gt_df["frame"] == frame]
        pred_frame = pred_df[pred_df["frame"] == frame]

        targets.append(_frame_to_target(gt_frame))
        preds.append(_frame_to_preds(pred_frame))

    metric.update(preds, targets)
    results = metric.compute()

    return DetectionMetricResult(
        map=float(results["map"]),
        map_50=float(results["map_50"]),
        map_75=float(results["map_75"]),
        mar_1=float(results["mar_1"]),
        mar_10=float(results["mar_10"]),
        mar_100=float(results["mar_100"]),
    )


if __name__ == "__main__":
    gt_csv = "/home/marcm/Documents/EPFL/MA4/whale_project/whale_mot/outputs/detections/sample_detections_finetuned.csv"
    pred_csv = "/home/marcm/Documents/EPFL/MA4/whale_project/whale_mot/outputs/detections/sample_detections_finetuned.csv"

    metrics = compute_detection_metrics(gt_csv, pred_csv)
    print(metrics)