from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import (
    centered_moving_average_trajectory,
    iter_valid_trajectory_segments,
    load_trajectory,
    smooth_loaded_trajectory,
    smooth_trajectory_corners,
)


def measure_anchor_offset(trajectory_path: Path) -> tuple[float, float, int]:
    raw_trajectory = load_trajectory(trajectory_path)
    segments = iter_valid_trajectory_segments(raw_trajectory)
    if not segments:
        raise RuntimeError(f"{trajectory_path} 没有连续有效轨迹段")

    longest_segment = max(segments, key=len)
    smoothed_trajectory = smooth_loaded_trajectory(
        centered_moving_average_trajectory(raw_trajectory)
    )

    previous_corners: np.ndarray | None = None
    offsets: list[float] = []
    for frame_idx in longest_segment:
        raw_corners = raw_trajectory[frame_idx]
        smoothed_corners = smoothed_trajectory[frame_idx]
        if raw_corners is None or smoothed_corners is None:
            continue

        final_corners = smooth_trajectory_corners(previous_corners, smoothed_corners)
        previous_corners = final_corners
        raw_center = raw_corners.mean(axis=0)
        final_center = final_corners.mean(axis=0)
        offsets.append(float(np.linalg.norm(final_center - raw_center)))

    if not offsets:
        raise RuntimeError(f"{trajectory_path} 最长有效段没有可测量帧")

    return float(np.mean(offsets)), float(np.max(offsets)), len(offsets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure rendered text anchor offset against raw AE trajectory center."
    )
    parser.add_argument(
        "trajectory",
        nargs="?",
        default="trajectory.json",
        help="Path to trajectory JSON, defaults to trajectory.json.",
    )
    args = parser.parse_args()

    mean_offset, max_offset, frames = measure_anchor_offset(Path(args.trajectory))
    print(f"frames={frames}")
    print(f"mean={mean_offset:.2f}px")
    print(f"max={max_offset:.2f}px")


if __name__ == "__main__":
    main()
