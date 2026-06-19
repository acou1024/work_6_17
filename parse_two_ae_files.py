"""
把两个 AE Corner Pin 文本文件缝合成服务器使用的 trajectory.json。

运行示例：
    python parse_two_ae_files.py

也可以指定路径：
    python parse_two_ae_files.py \
        --first-ae "/path/to/front.txt" \
        --second-ae "/path/to/back.txt" \
        --video "/path/to/template.mp4" \
        --output "./trajectory.json"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_FIRST_AE_PATH = Path(
    "/Users/tt/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
    "xwechat_files/tengteng8124_edab/temp/drag/Adobe After Effects 6.0 Keyframe Da.txt"
)
DEFAULT_SECOND_AE_PATH = Path(
    "/Users/tt/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
    "xwechat_files/tengteng8124_edab/temp/drag/Adobe After Effects 6.0 Keyframe Da1.txt"
)
DEFAULT_VIDEO_PATH = Path("/Users/tt/Downloads/11954.MP4")
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "trajectory.json"

# 业务规则：7-10 秒红纸仍在画面内，不应误填 None；约 10 秒后才进入出画隐身段。
DEFAULT_GAP_START_SECONDS = 10.0
DEFAULT_GAP_END_SECONDS = 20.0

# 如果红纸只剩很小一条边露在画面里，继续贴字会穿帮，直接视为出画。
DEFAULT_MIN_VISIBLE_AREA_RATIO = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse two AE Corner Pin exports into trajectory.json")
    parser.add_argument("--first-ae", type=Path, default=DEFAULT_FIRST_AE_PATH, help="前半段 AE 文本")
    parser.add_argument("--second-ae", type=Path, default=DEFAULT_SECOND_AE_PATH, help="后半段 AE 文本")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH, help="无字模板视频，用于读取总帧数")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="输出 trajectory.json 路径")
    parser.add_argument("--gap-start", type=float, default=DEFAULT_GAP_START_SECONDS, help="红纸出画开始秒数")
    parser.add_argument("--gap-end", type=float, default=DEFAULT_GAP_END_SECONDS, help="红纸重新入画秒数")
    parser.add_argument(
        "--min-visible-area-ratio",
        type=float,
        default=DEFAULT_MIN_VISIBLE_AREA_RATIO,
        help="四边形裁剪到画面内后的最小有效面积比例",
    )
    return parser.parse_args()


def ensure_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在：{path}")
    return path


def read_video_info(video_path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法读取视频：{video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError("视频元信息异常，请检查模板视频")
    return width, height, fps, frame_count


def parse_ae_corner_pin(path: Path) -> tuple[dict[int, list[list[float]]], dict[str, Any]]:
    """
    解析 AE Corner Pin 文本。

    AE Corner Pin 四段属性通常顺序为：
        0001 左上、0002 右上、0003 左下、0004 右下

    OpenCV 透视目标点需要：
        左上、右上、右下、左下
    因此这里显式把第 3、4 个点重排。
    """
    ensure_path(path, "AE 文本文件")
    text = path.read_text(encoding="utf-8-sig")

    metadata: dict[str, Any] = {}
    for key, pattern in {
        "units_per_second": r"Units Per Second\s+([\d.]+)",
        "source_width": r"Source Width\s+([\d.]+)",
        "source_height": r"Source Height\s+([\d.]+)",
    }.items():
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            metadata[key] = int(value) if value.is_integer() else value

    sections: list[dict[int, list[float]]] = []
    current_section: dict[int, list[float]] | None = None
    keyframe_pattern = re.compile(r"^\s*(\d+)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*$")

    for line in text.splitlines():
        if line.startswith("Effects"):
            current_section = {}
            sections.append(current_section)
            continue

        if current_section is None:
            continue

        match = keyframe_pattern.match(line)
        if not match:
            continue

        frame = int(match.group(1))
        x = float(match.group(2))
        y = float(match.group(3))
        current_section[frame] = [round(x, 3), round(y, 3)]

    if len(sections) != 4:
        raise ValueError(f"{path.name} 应包含 4 个 Corner Pin 点段，实际解析到 {len(sections)} 个")

    common_frames = sorted(set.intersection(*(set(section.keys()) for section in sections)))
    if not common_frames:
        raise ValueError(f"{path.name} 没有解析到完整四角帧数据")

    trajectory: dict[int, list[list[float]]] = {}
    for frame in common_frames:
        upper_left = sections[0][frame]
        upper_right = sections[1][frame]
        lower_left = sections[2][frame]
        lower_right = sections[3][frame]
        trajectory[frame] = [upper_left, upper_right, lower_right, lower_left]

    metadata["frames"] = len(trajectory)
    metadata["min_frame"] = min(common_frames)
    metadata["max_frame"] = max(common_frames)
    return trajectory, metadata


def visible_area_in_frame(points: list[list[float]], width: int, height: int) -> float:
    """计算四边形与视频画面的相交面积。"""
    quad = np.array(points, dtype=np.float32)
    frame_rect = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    try:
        area, _ = cv2.intersectConvexConvex(quad, frame_rect)
    except cv2.error:
        return 0.0
    return float(max(area, 0.0))


def points_or_none(
    points: list[list[float]] | None,
    width: int,
    height: int,
    min_visible_area_ratio: float,
) -> list[list[float]] | None:
    if points is None:
        return None
    min_area = width * height * min_visible_area_ratio
    if visible_area_in_frame(points, width, height) < min_area:
        return None
    return points


def build_trajectory(
    first: dict[int, list[list[float]]],
    second: dict[int, list[list[float]]],
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    gap_start_seconds: float,
    gap_end_seconds: float,
    min_visible_area_ratio: float,
) -> dict[str, list[list[float]] | None]:
    gap_start_frame = int(round(gap_start_seconds * fps))
    gap_end_frame = int(round(gap_end_seconds * fps))
    merged: dict[str, list[list[float]] | None] = {}

    for frame in range(frame_count):
        if gap_start_frame <= frame < gap_end_frame:
            merged[str(frame)] = None
            continue

        source = first if frame < gap_start_frame else second
        merged[str(frame)] = points_or_none(
            source.get(frame),
            width=width,
            height=height,
            min_visible_area_ratio=min_visible_area_ratio,
        )

    return merged


def summarize(trajectory: dict[str, list[list[float]] | None], fps: float) -> str:
    valid_frames = [int(frame) for frame, value in trajectory.items() if value is not None]
    none_count = len(trajectory) - len(valid_frames)
    if not valid_frames:
        return f"总帧数 {len(trajectory)}，有效贴图帧 0，None 帧 {none_count}"

    segments: list[tuple[int, int]] = []
    start = previous = valid_frames[0]
    for frame in valid_frames[1:]:
        if frame == previous + 1:
            previous = frame
        else:
            segments.append((start, previous))
            start = previous = frame
    segments.append((start, previous))

    segment_text = ", ".join(
        f"{start}-{end}帧({start / fps:.2f}s-{end / fps:.2f}s)" for start, end in segments
    )
    return f"总帧数 {len(trajectory)}，有效贴图帧 {len(valid_frames)}，None 帧 {none_count}，有效段：{segment_text}"


def main() -> None:
    args = parse_args()
    first_path = ensure_path(args.first_ae, "前半段 AE 文本")
    second_path = ensure_path(args.second_ae, "后半段 AE 文本")
    video_path = ensure_path(args.video, "模板视频")

    width, height, fps, frame_count = read_video_info(video_path)
    first, first_meta = parse_ae_corner_pin(first_path)
    second, second_meta = parse_ae_corner_pin(second_path)

    trajectory = build_trajectory(
        first=first,
        second=second,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        gap_start_seconds=args.gap_start,
        gap_end_seconds=args.gap_end,
        min_visible_area_ratio=args.min_visible_area_ratio,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"前半段：{first_path}")
    print(f"后半段：{second_path}")
    print(f"模板视频：{video_path} | {width}x{height} | {fps:.3f}fps | {frame_count}帧")
    print(f"AE 元信息：first={first_meta}, second={second_meta}")
    print(summarize(trajectory, fps))
    print(f"已生成：{args.output.resolve()}")


if __name__ == "__main__":
    main()
