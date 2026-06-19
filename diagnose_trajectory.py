"""
Export trajectory diagnostics for the prayer video template.

The script mirrors the coordinate-processing path used by app.add_text_to_video,
then writes per-frame CSV data and lightweight SVG charts. It does not render
video frames or call TTS.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

import app


DEFAULT_OUTPUT_DIR = Path("outputs") / "trajectory_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trajectory jitter diagnostics")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(app.DEFAULT_VIDEO_TEMPLATE_PATH),
        help="Template video path",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path(app.DEFAULT_TRAJECTORY_PATH),
        help="trajectory.json path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and SVG outputs",
    )
    return parser.parse_args()


def quad_stats(corners: np.ndarray) -> dict[str, float]:
    top = app.safe_length(corners[1] - corners[0])
    right = app.safe_length(corners[2] - corners[1])
    bottom = app.safe_length(corners[2] - corners[3])
    left = app.safe_length(corners[3] - corners[0])
    diag_a = app.safe_length(corners[2] - corners[0])
    diag_b = app.safe_length(corners[3] - corners[1])
    avg_edge = max((top + right + bottom + left) / 4.0, 1.0)
    return {
        "top": top,
        "right": right,
        "bottom": bottom,
        "left": left,
        "diag_a": diag_a,
        "diag_b": diag_b,
        "width_ratio": max(top, bottom) / max(min(top, bottom), 1.0),
        "height_ratio": max(left, right) / max(min(left, right), 1.0),
        "diagonal_ratio": max(diag_a, diag_b) / max(min(diag_a, diag_b), 1.0),
        "avg_edge": avg_edge,
    }


def perspective_matrix(corners: np.ndarray) -> np.ndarray:
    canvas_w, canvas_h = app.TEXT_CANVAS_SIZE
    src_points = np.array(
        [[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src_points, corners.astype(np.float32))


def normalized_matrix_delta(previous: np.ndarray | None, current: np.ndarray, avg_edge: float) -> float:
    if previous is None:
        return 0.0
    delta = float(np.linalg.norm(current - previous))
    return delta / max(avg_edge, 1.0)


def center_motion(previous: np.ndarray | None, current: np.ndarray | None) -> float:
    if previous is None or current is None:
        return 0.0
    return app.safe_length(current.mean(axis=0) - previous.mean(axis=0))


def avg_edge(corners: np.ndarray | None) -> float:
    if corners is None:
        return 0.0
    stats = quad_stats(corners)
    return (stats["top"] + stats["right"] + stats["bottom"] + stats["left"]) / 4.0


def normalized_edge_error(target: np.ndarray | None, final: np.ndarray | None) -> float:
    if target is None or final is None:
        return 0.0
    target_edge = avg_edge(target)
    if target_edge <= 1.0:
        return 0.0
    return abs(avg_edge(final) - target_edge) / target_edge


def normalized_shape_error(target: np.ndarray | None, final: np.ndarray | None) -> float:
    if target is None or final is None:
        return 0.0
    target_center = target.mean(axis=0)
    final_center = final.mean(axis=0)
    target_shape = target - target_center
    final_shape = final - final_center
    scale = max(avg_edge(target), 1.0)
    return float(np.mean(np.linalg.norm(final_shape - target_shape, axis=1)) / scale)


def normalized_area_error(target: np.ndarray | None, final: np.ndarray | None) -> float:
    if target is None or final is None:
        return 0.0
    target_area = app.quad_area(target)
    if target_area <= 1.0:
        return 0.0
    return abs(app.quad_area(final) - target_area) / target_area


def scale_or_none(corners: np.ndarray | None, width: int, height: int) -> np.ndarray | None:
    if corners is None:
        return None
    return app.scale_trajectory_corners(corners, width, height)


def build_rows(video_path: Path, trajectory_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    width, height, fps, frame_count = app.get_video_info(video_path)

    raw_trajectory = app.load_trajectory(trajectory_path)
    moving_avg_trajectory = app.centered_moving_average_trajectory(raw_trajectory)
    offline_trajectory = app.smooth_loaded_trajectory(moving_avg_trajectory)

    hold_start_frame = int(round(app.HOLD_VISIBLE_START_SECONDS * fps))
    hold_end_frame = int(round(app.HOLD_VISIBLE_END_SECONDS * fps))
    hold_reference_frame = max(0, int(round(app.HOLD_REFERENCE_SECONDS * fps)) - 1)
    max_extrapolate_frames = max(1, int(round(app.MAX_EXTRAPOLATE_SECONDS * fps)))
    hold_reference_corners = app.find_valid_corners_at_or_before(offline_trajectory, hold_reference_frame)
    reference_corners = app.find_first_valid_corners(offline_trajectory)
    scaled_reference = scale_or_none(reference_corners, width, height)

    rows: list[dict[str, object]] = []
    smoothed_corners: np.ndarray | None = None
    previous_valid_raw_corners: np.ndarray | None = None
    last_valid_raw_corners: np.ndarray | None = None
    previous_valid_frame_idx: int | None = None
    last_valid_frame_idx: int | None = None

    previous_raw_scaled: np.ndarray | None = None
    previous_offline_scaled: np.ndarray | None = None
    previous_final_scaled: np.ndarray | None = None
    previous_final_in_segment: np.ndarray | None = None
    previous_matrix: np.ndarray | None = None
    segment_id = 0

    for frame_idx in range(frame_count):
        raw_scaled = scale_or_none(raw_trajectory.get(frame_idx), width, height)
        offline_corners = offline_trajectory.get(frame_idx)
        offline_scaled = scale_or_none(offline_corners, width, height)

        corners = offline_corners
        is_predicted = False
        if corners is None and hold_start_frame <= frame_idx <= hold_end_frame:
            corners = app.extrapolate_corners(
                previous_valid=previous_valid_raw_corners,
                last_valid=last_valid_raw_corners,
                previous_frame=previous_valid_frame_idx,
                last_frame=last_valid_frame_idx,
                current_frame=frame_idx,
                max_frames=max_extrapolate_frames,
            )
            if corners is None:
                corners = hold_reference_corners
            is_predicted = corners is not None

        final_scaled: np.ndarray | None = None
        status = "none"
        visible_ratio = 0.0
        opacity_factor = 0.0
        matrix_delta = 0.0
        in_segment_center_motion = 0.0
        target_scaled: np.ndarray | None = None
        is_segment_start = False
        stats = {
            "top": 0.0,
            "right": 0.0,
            "bottom": 0.0,
            "left": 0.0,
            "diag_a": 0.0,
            "diag_b": 0.0,
            "width_ratio": 0.0,
            "height_ratio": 0.0,
            "diagonal_ratio": 0.0,
            "avg_edge": 1.0,
        }

        if corners is not None:
            dst_points = app.scale_trajectory_corners(corners, width, height)
            if app.is_valid_quad(dst_points):
                if not is_predicted:
                    previous_valid_raw_corners = last_valid_raw_corners
                    previous_valid_frame_idx = last_valid_frame_idx
                    last_valid_raw_corners = corners.copy()
                    last_valid_frame_idx = frame_idx

                if (
                    frame_idx >= int(round(app.TAIL_CONSTRAINT_START_SECONDS * fps))
                    and scaled_reference is not None
                ):
                    dst_points = app.constrain_tail_distortion(dst_points, scaled_reference)
                target_scaled = dst_points.copy()

                final_scaled = app.smooth_trajectory_corners(smoothed_corners, dst_points)
                if previous_final_in_segment is None:
                    is_segment_start = True
                    segment_id += 1
                else:
                    in_segment_center_motion = center_motion(previous_final_in_segment, final_scaled)
                smoothed_corners = final_scaled

                visible_ratio = app.visible_area_ratio(final_scaled, width, height)
                opacity_factor = app.opacity_from_visible_ratio(visible_ratio)
                stats = quad_stats(final_scaled)
                matrix = perspective_matrix(final_scaled)
                matrix_delta = normalized_matrix_delta(previous_matrix, matrix, stats["avg_edge"])
                previous_matrix = matrix
                status = "predicted" if is_predicted else "valid"
            else:
                smoothed_corners = None
                previous_matrix = None
                status = "invalid"
        else:
            smoothed_corners = None
            previous_matrix = None
            previous_final_in_segment = None

        rows.append(
            {
                "frame": frame_idx,
                "time_seconds": round(frame_idx / fps, 6),
                "status": status,
                "segment_id": segment_id if final_scaled is not None else "",
                "is_segment_start": is_segment_start,
                "raw_center_motion_px": center_motion(previous_raw_scaled, raw_scaled),
                "offline_center_motion_px": center_motion(previous_offline_scaled, offline_scaled),
                "final_center_motion_px": center_motion(previous_final_scaled, final_scaled),
                "final_in_segment_center_motion_px": in_segment_center_motion,
                "raw_avg_edge_px": avg_edge(raw_scaled),
                "offline_avg_edge_px": avg_edge(offline_scaled),
                "target_avg_edge_px": avg_edge(target_scaled),
                "final_avg_edge_px": avg_edge(final_scaled),
                "top_len_px": stats["top"],
                "right_len_px": stats["right"],
                "bottom_len_px": stats["bottom"],
                "left_len_px": stats["left"],
                "diag_a_len_px": stats["diag_a"],
                "diag_b_len_px": stats["diag_b"],
                "width_ratio": stats["width_ratio"],
                "height_ratio": stats["height_ratio"],
                "diagonal_ratio": stats["diagonal_ratio"],
                "visible_ratio": visible_ratio,
                "opacity_factor": opacity_factor,
                "perspective_matrix_delta": matrix_delta,
                "target_final_edge_error": normalized_edge_error(target_scaled, final_scaled),
                "target_final_shape_error": normalized_shape_error(target_scaled, final_scaled),
                "target_final_area_error": normalized_area_error(target_scaled, final_scaled),
                "raw_has_corners": raw_scaled is not None,
                "offline_has_corners": offline_scaled is not None,
                "final_has_corners": final_scaled is not None,
            }
        )

        previous_raw_scaled = raw_scaled if raw_scaled is not None else previous_raw_scaled
        previous_offline_scaled = offline_scaled if offline_scaled is not None else previous_offline_scaled
        previous_final_scaled = final_scaled if final_scaled is not None else previous_final_scaled
        previous_final_in_segment = final_scaled if final_scaled is not None else previous_final_in_segment

    metadata = {
        "video": str(video_path),
        "trajectory": str(trajectory_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
    }
    return rows, metadata


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_series(rows: list[dict[str, object]], key: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        value = float(row[key])
        if math.isfinite(value) and value > 0:
            points.append((float(row["time_seconds"]), value))
    return points


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[idx]


def polyline(points: list[tuple[float, float]], width: int, height: int, y_max: float) -> str:
    if not points:
        return ""
    x_min = min(p[0] for p in points)
    x_max = max(p[0] for p in points)
    x_span = max(x_max - x_min, 1e-6)
    plot_w = width - 90
    plot_h = height - 70
    coords = []
    for x, y in points:
        px = 60 + (x - x_min) / x_span * plot_w
        py = 20 + (1.0 - min(y / y_max, 1.0)) * plot_h
        coords.append(f"{px:.1f},{py:.1f}")
    return " ".join(coords)


def write_svg_chart(
    path: Path,
    title: str,
    series: list[tuple[str, str, list[tuple[float, float]]]],
    y_label: str,
) -> None:
    width = 1200
    height = 430
    all_values = [value for _, _, points in series for _, value in points]
    y_max = max(percentile(all_values, 0.98) * 1.15, 1.0)

    legend_items = []
    lines = []
    for idx, (label, color, points) in enumerate(series):
        line_points = polyline(points, width, height, y_max)
        if line_points:
            lines.append(
                f'<polyline points="{line_points}" fill="none" stroke="{color}" '
                'stroke-width="2" stroke-linejoin="round" stroke-linecap="round" />'
            )
        legend_y = 30 + idx * 24
        legend_items.append(
            f'<rect x="940" y="{legend_y - 12}" width="18" height="4" fill="{color}" />'
            f'<text x="966" y="{legend_y - 6}" font-size="14">{escape(label)}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbf7ef" />
  <text x="60" y="32" font-size="22" font-family="Microsoft YaHei, Arial" fill="#2f251f">{escape(title)}</text>
  <line x1="60" y1="360" x2="1170" y2="360" stroke="#8c7a68" stroke-width="1" />
  <line x1="60" y1="20" x2="60" y2="360" stroke="#8c7a68" stroke-width="1" />
  <text x="62" y="384" font-size="13" fill="#6d5d50">time (s)</text>
  <text x="12" y="24" font-size="13" fill="#6d5d50">{escape(y_label)}</text>
  <text x="12" y="363" font-size="12" fill="#6d5d50">0</text>
  <text x="12" y="72" font-size="12" fill="#6d5d50">98% <= {y_max:.3f}</text>
  {''.join(lines)}
  {''.join(legend_items)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def summarize(rows: list[dict[str, object]], metadata: dict[str, object]) -> str:
    final_rows = [r for r in rows if r["final_has_corners"]]
    predicted = [r for r in rows if r["status"] == "predicted"]

    def max_row(key: str) -> dict[str, object] | None:
        if not final_rows:
            return None
        return max(final_rows, key=lambda row: float(row[key]))

    summary_lines = [
        "# Trajectory Diagnostics",
        "",
        f"- video: {metadata['video']}",
        f"- trajectory: {metadata['trajectory']}",
        f"- size/fps/frames: {metadata['width']}x{metadata['height']} / {metadata['fps']:.3f} / {metadata['frame_count']}",
        f"- drawable frames after runtime processing: {len(final_rows)}",
        f"- predicted frames: {len(predicted)}",
        "",
        "## Max Spikes",
    ]

    for key, label in [
        ("raw_center_motion_px", "raw center motion"),
        ("offline_center_motion_px", "offline-smoothed center motion"),
        ("final_center_motion_px", "runtime final center motion across visible segments"),
        ("final_in_segment_center_motion_px", "runtime final center motion inside a visible segment"),
        ("diagonal_ratio", "diagonal ratio"),
        ("perspective_matrix_delta", "perspective matrix delta"),
    ]:
        row = max_row(key)
        if row is None:
            summary_lines.append(f"- {label}: no drawable frames")
        else:
            summary_lines.append(
                f"- {label}: {float(row[key]):.6f} at frame {row['frame']} ({float(row['time_seconds']):.3f}s)"
            )

    summary_lines.extend(
        [
            "",
            "## How To Read",
            "- If raw spikes are high but offline/final spikes drop, AE jitter is being filtered successfully.",
            "- If final spikes are higher than offline spikes, runtime EMA, prediction, or tail constraint may be introducing visible jumps.",
            "- If center motion is smooth but ratios or matrix delta spike, the shake is mostly perspective/shape jitter rather than whole-paper movement.",
        ]
    )
    return "\n".join(summary_lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, metadata = build_rows(args.video, args.trajectory)
    write_csv(output_dir / "trajectory_metrics.csv", rows)
    (output_dir / "summary.md").write_text(summarize(rows, metadata), encoding="utf-8")

    write_svg_chart(
        output_dir / "center_motion.svg",
        "Center Motion: raw vs offline vs final in visible segment",
        [
            ("raw center", "#9b2f1f", clean_series(rows, "raw_center_motion_px")),
            ("offline center", "#1f6f78", clean_series(rows, "offline_center_motion_px")),
            ("final in-segment center", "#d58a1f", clean_series(rows, "final_in_segment_center_motion_px")),
        ],
        "px/frame",
    )
    write_svg_chart(
        output_dir / "shape_ratios.svg",
        "Shape Ratios",
        [
            ("width ratio", "#9b2f1f", clean_series(rows, "width_ratio")),
            ("height ratio", "#1f6f78", clean_series(rows, "height_ratio")),
            ("diagonal ratio", "#d58a1f", clean_series(rows, "diagonal_ratio")),
        ],
        "ratio",
    )
    write_svg_chart(
        output_dir / "perspective_delta.svg",
        "Perspective Matrix Delta",
        [
            ("matrix delta", "#9b2f1f", clean_series(rows, "perspective_matrix_delta")),
        ],
        "normalized delta",
    )

    print(f"Wrote diagnostics to {output_dir.resolve()}")
    print(f"CSV: {(output_dir / 'trajectory_metrics.csv').resolve()}")
    print(f"Summary: {(output_dir / 'summary.md').resolve()}")


if __name__ == "__main__":
    main()
