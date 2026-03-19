from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MatchMode = Literal["id", "iou"]


@dataclass
class ContinuitySegment:
    gt_id: int
    pred_id: int
    start_frame: int
    end_frame: int
    length: int


@dataclass
class ContinuitySummary:
    num_segments: int
    avg_segment_length: float
    max_segment_length: int
    num_id_switches: int
    avg_coverage_ratio: float


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _load_tracking_csv(path: str | Path, is_gt: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["frame", "id", "x1", "y1", "x2", "y2"]
    if not is_gt and "conf" in df.columns:
        pass
    _validate_columns(df, required, "GT CSV" if is_gt else "Prediction CSV")
    return df.sort_values(["frame", "id"]).reset_index(drop=True)


def _xyxy_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def _match_by_id(gt_row: pd.Series, pred_frame: pd.DataFrame) -> Optional[int]:
    gt_id = int(gt_row["id"])
    match = pred_frame[pred_frame["id"] == gt_id]
    if len(match) == 0:
        return None
    return gt_id


def _best_match_for_gt(
    gt_row: pd.Series,
    pred_frame: pd.DataFrame,
    iou_threshold: float,
) -> Optional[int]:
    if len(pred_frame) == 0:
        return None

    gt_box = gt_row[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    best_iou = -1.0
    best_pred_id = None

    for _, pred in pred_frame.iterrows():
        pred_box = pred[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        iou = _xyxy_iou(gt_box, pred_box)
        if iou > best_iou:
            best_iou = iou
            best_pred_id = int(pred["id"])

    if best_iou >= iou_threshold:
        return best_pred_id
    return None


def compute_continuity_segments(
    gt_csv: str | Path,
    pred_csv: str | Path,
    match_mode: MatchMode = "id",
    iou_threshold: float = 0.5,
) -> list[ContinuitySegment]:
    """
    Compute continuous frame ranges where each GT track is matched correctly.

    match_mode:
        - "id": match GT and prediction by identical track ID
        - "iou": match GT box to best-overlapping predicted box
    """
    gt_df = _load_tracking_csv(gt_csv, is_gt=True)
    pred_df = _load_tracking_csv(pred_csv, is_gt=False)

    if match_mode not in {"id", "iou"}:
        raise ValueError("match_mode must be 'id' or 'iou'")

    segments: list[ContinuitySegment] = []

    for gt_id in sorted(gt_df["id"].unique()):
        gt_track = gt_df[gt_df["id"] == gt_id].sort_values("frame")

        current_pred_id: Optional[int] = None
        segment_start: Optional[int] = None
        previous_frame: Optional[int] = None

        for _, gt_row in gt_track.iterrows():
            frame = int(gt_row["frame"])
            pred_frame = pred_df[pred_df["frame"] == frame]

            if match_mode == "id":
                matched_pred_id = _match_by_id(gt_row, pred_frame)
            else:
                matched_pred_id = _best_match_for_gt(gt_row, pred_frame, iou_threshold)

            is_continuing = (
                matched_pred_id is not None
                and current_pred_id is not None
                and matched_pred_id == current_pred_id
                and previous_frame is not None
                and frame == previous_frame + 1
            )

            starts_new_segment = (
                matched_pred_id is not None
                and (
                    current_pred_id is None
                    or matched_pred_id != current_pred_id
                    or previous_frame is None
                    or frame != previous_frame + 1
                )
            )

            if is_continuing:
                previous_frame = frame
                continue

            if current_pred_id is not None and segment_start is not None and previous_frame is not None:
                segments.append(
                    ContinuitySegment(
                        gt_id=int(gt_id),
                        pred_id=int(current_pred_id),
                        start_frame=int(segment_start),
                        end_frame=int(previous_frame),
                        length=int(previous_frame - segment_start + 1),
                    )
                )

            if starts_new_segment:
                current_pred_id = matched_pred_id
                segment_start = frame
                previous_frame = frame
            else:
                current_pred_id = None
                segment_start = None
                previous_frame = None

        if current_pred_id is not None and segment_start is not None and previous_frame is not None:
            segments.append(
                ContinuitySegment(
                    gt_id=int(gt_id),
                    pred_id=int(current_pred_id),
                    start_frame=int(segment_start),
                    end_frame=int(previous_frame),
                    length=int(previous_frame - segment_start + 1),
                )
            )

    return segments


def summarize_continuity(
    gt_csv: str | Path,
    pred_csv: str | Path,
    segments: list[ContinuitySegment],
) -> ContinuitySummary:
    gt_df = _load_tracking_csv(gt_csv, is_gt=True)

    if not segments:
        return ContinuitySummary(
            num_segments=0,
            avg_segment_length=0.0,
            max_segment_length=0,
            num_id_switches=0,
            avg_coverage_ratio=0.0,
        )

    lengths = [s.length for s in segments]
    num_switches = 0
    coverage_ratios = []

    for gt_id in sorted(gt_df["id"].unique()):
        gt_track = gt_df[gt_df["id"] == gt_id]
        total_frames = len(gt_track)

        gt_segments = sorted(
            [s for s in segments if s.gt_id == gt_id],
            key=lambda s: s.start_frame,
        )

        covered = sum(s.length for s in gt_segments)
        coverage_ratios.append(covered / total_frames if total_frames > 0 else 0.0)

        for i in range(1, len(gt_segments)):
            if gt_segments[i].pred_id != gt_segments[i - 1].pred_id:
                num_switches += 1

    return ContinuitySummary(
        num_segments=len(segments),
        avg_segment_length=float(np.mean(lengths)),
        max_segment_length=int(np.max(lengths)),
        num_id_switches=num_switches,
        avg_coverage_ratio=float(np.mean(coverage_ratios)) if coverage_ratios else 0.0,
    )


def plot_continuity_segments(
    gt_csv: str | Path,
    segments: list[ContinuitySegment],
    save_path: str | Path | None = None,
    show: bool = True,
    min_label_length: int = 20,
) -> None:
    """
    Plot:
      - orange ticks = frames where GT exists
      - blue bars    = continuous correct tracking segments, aligned to frame bins
    """
    gt_df = _load_tracking_csv(gt_csv, is_gt=True)

    gt_ids = sorted(gt_df["id"].unique())
    if len(gt_ids) == 0:
        print("No GT tracks to plot.")
        return

    y_positions = {gt_id: i for i, gt_id in enumerate(gt_ids)}

    plt.figure(figsize=(13, max(4, len(gt_ids) * 0.7)))
    ax = plt.gca()

    # --------------------------------------------------------
    # 1) draw GT frames as orange ticks
    # --------------------------------------------------------
    for i, gt_id in enumerate(gt_ids):
        gt_track = gt_df[gt_df["id"] == gt_id].sort_values("frame")
        frames = gt_track["frame"].to_numpy()
        y = np.full_like(frames, fill_value=y_positions[gt_id], dtype=float)

        plt.plot(
            frames,
            y,
            linestyle="None",
            marker="|",
            markersize=10,
            markeredgewidth=1.5,
            color="orange",
            alpha=0.5,
            label="GT frames" if i == 0 else None,
            zorder=3,
        )

    # --------------------------------------------------------
    # 2) draw continuity segments as frame-aligned rectangles
    # --------------------------------------------------------
    bar_height = 0.22

    for i, seg in enumerate(segments):
        y_center = y_positions[seg.gt_id]

        # rectangle aligned to frame bins:
        # frame k occupies [k-0.5, k+0.5]
        x = seg.start_frame - 0.5
        width = seg.length
        y = y_center - bar_height / 2

        rect = plt.Rectangle(
            (x, y),
            width,
            bar_height,
            facecolor="blue",
            edgecolor="blue",
            alpha=0.9,
            label="Continuous match" if i == 0 else None,
            zorder=2,
        )
        ax.add_patch(rect)

        if seg.length >= min_label_length:
            plt.text(
                x=seg.start_frame + seg.length / 2 - 0.5,
                y=y_center + 0.12,
                s=f"pred {seg.pred_id}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.yticks(list(y_positions.values()), [f"GT {gt_id}" for gt_id in gt_ids])
    plt.xlabel("Frame")
    plt.ylabel("Ground-truth track")
    plt.title("Track continuity vs GT frame presence")
    plt.grid(True, axis="x", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def plot_track_id_over_time(
    pred_csv: str | Path,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Useful for single-object or mostly-single-object sequences:
    plots predicted track ID as a function of frame.
    """
    pred_df = _load_tracking_csv(pred_csv, is_gt=False)

    # if multiple detections per frame exist, take the first by sorted id
    series = pred_df.sort_values(["frame", "id"]).groupby("frame", as_index=False).first()

    plt.figure(figsize=(12, 4))
    plt.plot(series["frame"], series["id"], marker=".", linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("Predicted track ID")
    plt.title("Predicted track ID over time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    gt_csv = "/home/marcm/Documents/EPFL/MA4/whale_project/whale_mot/outputs/tracks/sample_tracks_initial.csv"
    pred_csv = "/home/marcm/Documents/EPFL/MA4/whale_project/whale_mot/outputs/tracks/sample_tracks_botsort_finetuned.csv"


    segments = compute_continuity_segments(
        gt_csv=gt_csv,
        pred_csv=pred_csv,
        match_mode="iou",   # use "id" for same-tracker sanity check
        iou_threshold=0.1,
    )

    summary = summarize_continuity(gt_csv, pred_csv, segments)

    print("Continuity summary:")
    print(summary)

    print("\nSegments:")
    for seg in segments[:20]:
        print(seg)

    plot_continuity_segments(
        gt_csv=gt_csv,
        segments=segments,
        save_path="outputs/eval/continuity_segments.png",
        show=True,
        min_label_length=25,
    )

    plot_track_id_over_time(
        pred_csv,
        save_path="outputs/eval/track_id_over_time.png",
        show=True,
    )