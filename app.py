"""
自动化祈愿视频生成系统

运行方式：
    streamlit run app.py

部署前需要准备：
1. 无字红纸模板视频，例如：/path/to/template.mp4
2. AE 缝合后的轨迹文件 trajectory.json
3. 纸面字体文件，例如：/path/to/kaiti.ttf
4. 可用的 TTS endpoint / api_key / voice_id
5. 服务器已安装 ffmpeg，并可在命令行直接调用
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import ffmpeg
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import dashscope


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_BIN = PROJECT_ROOT / "bin"
if LOCAL_BIN.exists():
    os.environ["PATH"] = f"{LOCAL_BIN}{os.pathsep}{os.environ.get('PATH', '')}"

load_dotenv(PROJECT_ROOT / ".env")


# =========================
# 服务器配置
# =========================

# 模板视频配置：支持多个模板，每个模板对应一组视频 + 轨迹坐标
TEMPLATE_CONFIGS = [
    {
        "label": "模板一（25秒·红纸两段）",
        "video": os.getenv("VIDEO_TEMPLATE_PATH_1", str(PROJECT_ROOT / "模板视频1.mp4")),
        "trajectory": os.getenv("TRAJECTORY_JSON_PATH_1", str(PROJECT_ROOT / "trajectory.json")),
    },
    {
        "label": "模板二（23秒）",
        "video": os.getenv("VIDEO_TEMPLATE_PATH_2", str(PROJECT_ROOT / "模板视频2.mp4")),
        "trajectory": os.getenv("TRAJECTORY_JSON_PATH_2", str(PROJECT_ROOT / "trajectory2.json")),
    },
]
# 默认使用模板一
DEFAULT_TEMPLATE_INDEX = 0
DEFAULT_VIDEO_TEMPLATE_PATH = TEMPLATE_CONFIGS[0]["video"]
DEFAULT_TRAJECTORY_PATH = TEMPLATE_CONFIGS[0]["trajectory"]

WINDOWS_FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
DEFAULT_PAPER_FONT_CANDIDATES = [
    WINDOWS_FONT_DIR / "simkai.ttf",
    WINDOWS_FONT_DIR / "SIMKAI.TTF",
    WINDOWS_FONT_DIR / "STKAITI.TTF",
    WINDOWS_FONT_DIR / "楷体.ttf",
    WINDOWS_FONT_DIR / "FZSTK.TTF",
    WINDOWS_FONT_DIR / "STXINGKA.TTF",
    PROJECT_ROOT / "public/fonts/Xingkai.ttf",
]


def resolve_default_font_path() -> str:
    configured = os.getenv("PAPER_FONT_PATH") or os.getenv("XINGSHU_FONT_PATH")
    if configured:
        return configured
    for candidate in DEFAULT_PAPER_FONT_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return str(PROJECT_ROOT / "public/fonts/Xingkai.ttf")


DEFAULT_FONT_PATH = resolve_default_font_path()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
TTS_VOICE = os.getenv("TTS_VOICE", "Eldric Sage")  # 沧明子：低沉苍老智慧老者音色
TTS_MODEL = os.getenv("TTS_MODEL", "qwen3-tts-flash")

# AE 导出的坐标源尺寸。当前两个 AE 文件元信息为 720x1280。
TRAJECTORY_SOURCE_WIDTH = int(os.getenv("TRAJECTORY_SOURCE_WIDTH", "720"))
TRAJECTORY_SOURCE_HEIGHT = int(os.getenv("TRAJECTORY_SOURCE_HEIGHT", "1280"))

# 文字先绘制到一个标准纸面画布，再整体透视映射到 AE 轨迹四角。
TEXT_CANVAS_SIZE = (900, 620)  # width, height

# 局部修补参数：7-10 秒禁止硬消失；缺坐标时根据最近轨迹预测红纸继续出画。
HOLD_VISIBLE_START_SECONDS = float(os.getenv("HOLD_VISIBLE_START_SECONDS", "7.0"))
HOLD_VISIBLE_END_SECONDS = float(os.getenv("HOLD_VISIBLE_END_SECONDS", "10.0"))
HOLD_REFERENCE_SECONDS = float(os.getenv("HOLD_REFERENCE_SECONDS", "6.0"))

# AE 轨迹中心和四角形状都逐帧跟随原始纸面抖动，避免放大文字边缘悬浮。
COORDINATE_SMOOTH_ALPHA = float(os.getenv("COORDINATE_SMOOTH_ALPHA", "1.0"))
CENTER_SOFT_FOLLOW_PIXELS = float(os.getenv("CENTER_SOFT_FOLLOW_PIXELS", "6.0"))
CENTER_HARD_FOLLOW_PIXELS = float(os.getenv("CENTER_HARD_FOLLOW_PIXELS", "22.0"))
COORDINATE_DEAD_ZONE_PIXELS = float(os.getenv("COORDINATE_DEAD_ZONE_PIXELS", "0.0"))
SHAPE_SOFT_FOLLOW_PIXELS = float(os.getenv("SHAPE_SOFT_FOLLOW_PIXELS", "2.5"))
SHAPE_HARD_FOLLOW_PIXELS = float(os.getenv("SHAPE_HARD_FOLLOW_PIXELS", "12.0"))
SHAPE_FAST_SMOOTH_ALPHA = float(os.getenv("SHAPE_FAST_SMOOTH_ALPHA", "1.0"))

# 离线轨迹处理：只保留 3 帧中值去单帧尖刺，其余中心/形状直通原始 AE 值。
TRAJECTORY_MOVING_AVERAGE_WINDOW = int(os.getenv("TRAJECTORY_MOVING_AVERAGE_WINDOW", "1"))
TRAJECTORY_MEDIAN_RADIUS = int(os.getenv("TRAJECTORY_MEDIAN_RADIUS", "1"))
TRAJECTORY_CENTER_SMOOTH_RADIUS = int(os.getenv("TRAJECTORY_CENTER_SMOOTH_RADIUS", "9"))
TRAJECTORY_SHAPE_SMOOTH_RADIUS = int(os.getenv("TRAJECTORY_SHAPE_SMOOTH_RADIUS", "6"))
TRAJECTORY_SMOOTH_SIGMA = float(os.getenv("TRAJECTORY_SMOOTH_SIGMA", "4.0"))
TRAJECTORY_CENTER_BLEND = float(os.getenv("TRAJECTORY_CENTER_BLEND", "0.0"))
TRAJECTORY_SHAPE_BLEND = float(os.getenv("TRAJECTORY_SHAPE_BLEND", "0.0"))
TRAJECTORY_MOTION_SOFT_PIXELS = float(os.getenv("TRAJECTORY_MOTION_SOFT_PIXELS", "6.0"))
TRAJECTORY_MOTION_HARD_PIXELS = float(os.getenv("TRAJECTORY_MOTION_HARD_PIXELS", "22.0"))
TRAJECTORY_FAST_CENTER_BLEND = float(os.getenv("TRAJECTORY_FAST_CENTER_BLEND", "0.0"))
TRAJECTORY_FAST_SHAPE_BLEND = float(os.getenv("TRAJECTORY_FAST_SHAPE_BLEND", "0.0"))

# 红纸半出画时按可见面积淡出，避免文字漂在香炉/背景上。
FADE_MIN_VISIBLE_RATIO = float(os.getenv("FADE_MIN_VISIBLE_RATIO", "0.16"))
FADE_FULL_VISIBLE_RATIO = float(os.getenv("FADE_FULL_VISIBLE_RATIO", "0.82"))
MAX_EXTRAPOLATE_SECONDS = float(os.getenv("MAX_EXTRAPOLATE_SECONDS", "1.5"))

# 镜头运动时整张纸已经在视频里模糊；文字 alpha 同步跟随纸面四角运动量拖影。
TEXT_MOTION_BLUR_SCALE = float(os.getenv("TEXT_MOTION_BLUR_SCALE", "0.7"))
TEXT_MOTION_BLUR_MAX_KERNEL = int(os.getenv("TEXT_MOTION_BLUR_MAX_KERNEL", "35"))
TEXT_MOTION_BLUR_GAUSSIAN_SCALE = float(os.getenv("TEXT_MOTION_BLUR_GAUSSIAN_SCALE", "0.12"))
TEXT_FAST_MOTION_THRESHOLD = float(os.getenv("TEXT_FAST_MOTION_THRESHOLD", "18.0"))
TEXT_FAST_OPACITY_MIN = float(os.getenv("TEXT_FAST_OPACITY_MIN", "0.72"))
TEXT_BASE_DARKEN = float(os.getenv("TEXT_BASE_DARKEN", "0.96"))
TEXT_EDGE_SOFTEN_SIGMA = float(os.getenv("TEXT_EDGE_SOFTEN_SIGMA", "0.60"))
TEXT_PAPER_TEXTURE_GAIN = float(os.getenv("TEXT_PAPER_TEXTURE_GAIN", "0.60"))
TEXT_HALO_WEIGHT = float(os.getenv("TEXT_HALO_WEIGHT", "0.06"))
TEXT_INK_GRAIN_GAIN = float(os.getenv("TEXT_INK_GRAIN_GAIN", "0.30"))
TEXT_INK_SINK_STRENGTH = float(os.getenv("TEXT_INK_SINK_STRENGTH", "0.22"))

# 真实写在纸上的字应该保持固定纸面占比；透视 warp 已经负责远近缩放。
# 这里默认只做固定版式补偿，避免逐帧“呼吸”造成漂浮感。
TEXT_LAYOUT_CONTENT_SCALE = float(os.getenv("TEXT_LAYOUT_CONTENT_SCALE", "0.82"))
TEXT_LAYOUT_Y_OFFSET_RATIO = float(os.getenv("TEXT_LAYOUT_Y_OFFSET_RATIO", "0.0"))
TEXT_LAYER_SCALE_GAIN = float(os.getenv("TEXT_LAYER_SCALE_GAIN", "0.0"))
TEXT_LAYER_SCALE_MIN = float(os.getenv("TEXT_LAYER_SCALE_MIN", "1.0"))
TEXT_LAYER_SCALE_MAX = float(os.getenv("TEXT_LAYER_SCALE_MAX", "1.0"))
TEXT_SCALE_SMOOTH_ALPHA = float(os.getenv("TEXT_SCALE_SMOOTH_ALPHA", "0.35"))
TEXT_LAYER_ROTATION_GAIN = float(os.getenv("TEXT_LAYER_ROTATION_GAIN", "1.0"))
TEXT_LAYER_ROTATION_MAX_DEGREES = float(os.getenv("TEXT_LAYER_ROTATION_MAX_DEGREES", "4.0"))

# 红纸检测只作为保守辅助：候选框必须和 AE 轨迹足够接近，才允许小幅修正。
PAPER_QUAD_CORRECTION_BLEND = float(os.getenv("PAPER_QUAD_CORRECTION_BLEND", "0.0"))
PAPER_QUAD_MAX_CENTER_ERROR = float(os.getenv("PAPER_QUAD_MAX_CENTER_ERROR", "28.0"))
PAPER_QUAD_MIN_AREA_RATIO = float(os.getenv("PAPER_QUAD_MIN_AREA_RATIO", "0.78"))
PAPER_QUAD_MAX_AREA_RATIO = float(os.getenv("PAPER_QUAD_MAX_AREA_RATIO", "1.24"))

# 第二段尾部刚性约束默认关闭，避免 20s 后把文字从原始纸面轨迹拽开。
TAIL_CONSTRAINT_START_SECONDS = float(os.getenv("TAIL_CONSTRAINT_START_SECONDS", "20.0"))
TAIL_SOFT_RIGID_BLEND = float(os.getenv("TAIL_SOFT_RIGID_BLEND", "0.0"))
TAIL_STRONG_RIGID_BLEND = float(os.getenv("TAIL_STRONG_RIGID_BLEND", "0.0"))
MAX_ASPECT_DRIFT = float(os.getenv("MAX_ASPECT_DRIFT", "0.18"))
MAX_EDGE_RATIO_DRIFT = float(os.getenv("MAX_EDGE_RATIO_DRIFT", "0.22"))
MAX_DIAGONAL_RATIO_DRIFT = float(os.getenv("MAX_DIAGONAL_RATIO_DRIFT", "0.18"))


# =========================
# 通用工具
# =========================


def ensure_file_exists(path: str | Path, label: str) -> Path:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"{label}不存在：{file_path}")
    return file_path


def get_video_info(video_path: str | Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法读取视频：{video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError("视频元信息异常，请检查模板视频是否损坏")
    return width, height, fps, frame_count


def fit_font(
    font_path: Path,
    text: str,
    max_size: int,
    min_size: int,
    max_width: int,
) -> ImageFont.FreeTypeFont:
    """根据文本宽度自动选择字号，避免姓名和生辰过长时越界。"""
    probe_image = Image.new("RGBA", (max_width, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe_image)

    for size in range(max_size, min_size - 1, -2):
        font = ImageFont.truetype(str(font_path), size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
    return ImageFont.truetype(str(font_path), min_size)


def apply_ink_grain(layer: Image.Image) -> Image.Image:
    """在纸面坐标里打散字边和笔画，让墨迹纹理跟随透视一起运动。"""
    if TEXT_INK_GRAIN_GAIN <= 0:
        return layer

    rgba = np.array(layer).astype(np.float32)
    alpha = rgba[:, :, 3] / 255.0
    if float(alpha.max()) <= 0:
        return layer

    h, w = alpha.shape
    rng = np.random.default_rng(20260618)
    coarse = rng.normal(0.0, 1.0, (max(2, h // 30), max(2, w // 30))).astype(np.float32)
    grain = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    grain = cv2.GaussianBlur(grain, (0, 0), 1.6)
    grain -= float(grain.min())
    peak = float(grain.max())
    if peak > 0:
        grain /= peak

    fine = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (3, 3), 0.6)
    fine -= float(fine.min())
    fine_peak = float(fine.max())
    if fine_peak > 0:
        fine /= fine_peak

    sparse = rng.random((h, w), dtype=np.float32)
    sparse = cv2.GaussianBlur(sparse, (0, 0), 0.7)
    sparse_mask = np.clip((sparse - 0.36) / 0.64, 0.0, 1.0)

    paper_fiber = 0.70 * grain + 0.30 * fine
    dry_brush = 1.0 - TEXT_INK_GRAIN_GAIN * (0.85 * paper_fiber + 0.35 * sparse_mask)
    dark_pools = 1.0 + TEXT_INK_GRAIN_GAIN * 0.38 * (1.0 - grain)
    edge = cv2.GaussianBlur(alpha, (3, 3), 0.42)
    alpha = np.clip(edge * dry_brush * dark_pools, 0.0, 1.0)
    rgba[:, :, 3] = alpha * 255.0
    return Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), mode="RGBA")


def text_seed(name: str, birthdate: str) -> int:
    """为同一组输入生成稳定的微扰种子，避免每次渲染都不一致。"""
    payload = f"{name.strip()}|{birthdate.strip()}"
    seed = 2166136261
    for ch in payload.encode("utf-8"):
        seed ^= ch
        seed = (seed * 16777619) & 0xFFFFFFFF
    return seed


PAPER_TEXT_TRANSLATION = str.maketrans(
    {
        "0": "〇",
        "1": "一",
        "2": "二",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "七",
        "8": "八",
        "9": "九",
        ":": " ",
        "：": " ",
        ",": " ",
        "，": " ",
    }
)


def paper_display_text(text: str) -> str:
    """纸面显示用更接近手写记录的中文数字和少标点形式。"""
    return " ".join(text.translate(PAPER_TEXT_TRANSLATION).split())


def _draw_text_stamp(
    base: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    center_x: float,
    center_y: float,
    angle: float,
    opacity: int,
    warm_tint: tuple[int, int, int],
    stroke_alpha: int = 58,
) -> None:
    """在独立小图上盖一枚略带扫描感的文字印记，再贴回大画布。"""
    probe = Image.new("RGBA", base.size, (0, 0, 0, 0))
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = max(18, int(round(text_w * 0.18)))
    pad_y = max(12, int(round(text_h * 0.22)))
    stamp_w = max(8, text_w + pad_x * 2)
    stamp_h = max(8, text_h + pad_y * 2)

    stamp = Image.new("RGBA", (stamp_w, stamp_h), (0, 0, 0, 0))
    stamp_draw = ImageDraw.Draw(stamp)
    text_x = pad_x - bbox[0]
    text_y = pad_y - bbox[1]

    # 先盖轻微底影，再盖主体，模拟纸上墨压不均。
    stamp_draw.text(
        (text_x + 1, text_y + 1),
        text,
        font=font,
        fill=(*warm_tint, max(12, stroke_alpha)),
    )
    stamp_draw.text(
        (text_x, text_y),
        text,
        font=font,
        fill=(*warm_tint, opacity),
        stroke_width=1,
        stroke_fill=(18, 8, 6, max(18, stroke_alpha)),
    )
    stamp_draw.text(
        (text_x - 1, text_y),
        text,
        font=font,
        fill=(*warm_tint, max(18, opacity - 70)),
    )

    if abs(angle) > 0.01:
        stamp = stamp.rotate(angle, resample=Image.BICUBIC, expand=True)

    dest_x = int(round(center_x - stamp.width / 2.0))
    dest_y = int(round(center_y - stamp.height / 2.0))
    base.alpha_composite(stamp, dest=(dest_x, dest_y))


def _draw_handwritten_line(
    base: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    center_x: float,
    center_y: float,
    angle: float,
    opacity: int,
    warm_tint: tuple[int, int, int],
    rng: np.random.Generator,
) -> None:
    """逐字盖印，给每个字稳定的轻微角度、字号和基线差异。"""
    probe = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    widths: list[int] = []
    max_h = 0
    for char in text:
        bbox = draw.textbbox((0, 0), char, font=font)
        char_w = max(1, bbox[2] - bbox[0])
        char_h = max(1, bbox[3] - bbox[1])
        widths.append(char_w)
        max_h = max(max_h, char_h)

    tracking = max(2, int(round(font.size * 0.05)))
    total_w = sum(widths) + tracking * max(0, len(widths) - 1)
    cursor_x = center_x - total_w / 2.0
    scale_ratio = font.size / max(1, 70 * 2)
    baseline_jitter = max(2.0, 3.0 * scale_ratio)

    for char, char_w in zip(text, widths):
        char_center_x = cursor_x + char_w / 2.0 + float(rng.uniform(-1.2, 1.2))
        char_center_y = center_y + float(rng.uniform(-baseline_jitter, baseline_jitter))
        size_jitter = 1.0 + float(rng.uniform(-0.03, 0.03))
        char_font = font
        if abs(size_jitter - 1.0) > 0.005:
            char_font = font.font_variant(size=max(8, int(round(font.size * size_jitter))))
        char_angle = angle + float(rng.uniform(-1.5, 1.5))
        char_opacity = int(np.clip(opacity + rng.integers(-16, 11), 160, 230))
        _draw_text_stamp(
            base,
            char,
            char_font,
            center_x=char_center_x,
            center_y=char_center_y,
            angle=char_angle,
            opacity=char_opacity,
            warm_tint=warm_tint,
            stroke_alpha=44,
        )
        cursor_x += char_w + tracking


# =========================
# 第一步：动态文字图像生成
# =========================


def create_text_layer(
    name: str,
    birthdate: str,
    font_path: str | Path,
    canvas_size: tuple[int, int] = TEXT_CANVAS_SIZE,
) -> Image.Image:
    """
    生成纸面坐标里的“原生墨迹”图层。

    不是把四行字规整地排成电脑字幕，而是：
    1. 先在高分辨率画布上生成一个轻微的纸面洗色块；
    2. 再把每行字当作独立墨迹印记盖上去；
    3. 最后整体降采样并打散边缘，让它更像扫描/印刷后的纸面内容。
    """
    font_file = ensure_file_exists(font_path, "行书字体文件")
    width, height = canvas_size
    work_scale = 2
    work_size = (width * work_scale, height * work_scale)
    layer = Image.new("RGBA", work_size, (0, 0, 0, 0))
    seed = text_seed(name, birthdate)
    rng = np.random.default_rng(seed)

    title_text = "上香祈愿"
    identity_text = paper_display_text(f"{name} {birthdate}")
    wish_text = "身体康泰 万事顺遂"

    title_font = fit_font(font_file, title_text, 118 * work_scale, 92 * work_scale, int(work_size[0] * 0.90))
    identity_font = fit_font(font_file, identity_text, 70 * work_scale, 52 * work_scale, int(work_size[0] * 0.92))
    wish_font = fit_font(font_file, wish_text, 70 * work_scale, 52 * work_scale, int(work_size[0] * 0.90))

    content_left = int(work_size[0] * 0.18)
    content_right = int(work_size[0] * 0.82)
    content_top = int(work_size[1] * 0.18)
    content_bottom = int(work_size[1] * 0.70)

    # 先放一个很轻的纸面洗色块，模拟墨迹浸入/印刷后在纸上形成的材料感。
    mask = np.zeros((work_size[1], work_size[0]), dtype=np.uint8)
    poly = np.array(
        [
            [content_left - 18, content_top - 10],
            [content_right + 16, content_top + 6],
            [content_right + 6, content_bottom + 14],
            [content_left - 12, content_bottom],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, poly, 255)
    mask = cv2.GaussianBlur(mask, (0, 0), 31.0)
    stain = np.zeros((work_size[1], work_size[0], 4), dtype=np.uint8)
    stain[:, :, 0] = 26
    stain[:, :, 1] = 10
    stain[:, :, 2] = 8
    stain[:, :, 3] = np.clip(mask.astype(np.float32) * 0.018, 0, 255).astype(np.uint8)
    layer = Image.alpha_composite(layer, Image.fromarray(stain, mode="RGBA"))

    warm_tint = (28, 18, 14)
    lines = [
        (title_text, title_font, work_size[0] * 0.51 + rng.integers(-3, 4), work_size[1] * 0.32, -0.12 + rng.uniform(-0.05, 0.05), 225),
        (identity_text, identity_font, work_size[0] * 0.50 + rng.integers(-4, 5), work_size[1] * 0.52, 0.04 + rng.uniform(-0.04, 0.04), 222),
        (wish_text, wish_font, work_size[0] * 0.51 + rng.integers(-3, 4), work_size[1] * 0.71, 0.04 + rng.uniform(-0.04, 0.04), 218),
    ]

    for text, font, center_x, center_y, angle, opacity in lines:
        _draw_handwritten_line(
            layer,
            text,
            font,
            center_x=float(center_x),
            center_y=float(center_y),
            angle=float(angle),
            opacity=int(opacity),
            warm_tint=warm_tint,
            rng=rng,
        )

    # 先把高分辨率图层轻微柔化，再缩回目标画布，形成更像扫描件的边缘。
    layer = layer.filter(ImageFilter.GaussianBlur(0.35))
    layer = layer.resize(canvas_size, Image.LANCZOS)
    return apply_ink_grain(layer)


# =========================
# 第二步：读取 trajectory.json 并逐帧贴图
# =========================


Trajectory = dict[int, Optional[np.ndarray]]


def load_trajectory(trajectory_path: str | Path) -> Trajectory:
    """
    读取 parse_two_ae_files.py 生成的 trajectory.json。

    JSON 格式：
        {
          "0": [[左上x, 左上y], [右上x, 右上y], [右下x, 右下y], [左下x, 左下y]],
          "180": null
        }
    """
    trajectory_file = ensure_file_exists(trajectory_path, "trajectory.json")
    raw_data = json.loads(trajectory_file.read_text(encoding="utf-8"))

    trajectory: Trajectory = {}
    for key, value in raw_data.items():
        frame_idx = int(key)
        if value is None:
            trajectory[frame_idx] = None
            continue

        corners = np.array(value, dtype=np.float32)
        if corners.shape != (4, 2):
            raise ValueError(f"trajectory.json 第 {frame_idx} 帧坐标格式错误，应为 4x2")
        trajectory[frame_idx] = corners

    return trajectory


def iter_valid_trajectory_segments(trajectory: Trajectory) -> list[list[int]]:
    """把连续存在四角坐标的帧切成片段，None 出画段不参与平滑。"""
    segments: list[list[int]] = []
    current: list[int] = []
    previous_frame: int | None = None

    for frame_idx in sorted(trajectory):
        corners = trajectory[frame_idx]
        if corners is None or not is_valid_quad(corners):
            if current:
                segments.append(current)
                current = []
            previous_frame = None
            continue

        if previous_frame is not None and frame_idx != previous_frame + 1 and current:
            segments.append(current)
            current = []

        current.append(frame_idx)
        previous_frame = frame_idx

    if current:
        segments.append(current)
    return segments


def centered_median_filter(values: np.ndarray, radius: int) -> np.ndarray:
    """对称中值滤波，先去掉 AE 点位里偶发的一帧尖刺。"""
    if radius <= 0 or len(values) <= 2:
        return values.astype(np.float32)

    filtered = np.empty_like(values, dtype=np.float32)
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        filtered[idx] = np.median(values[start:end], axis=0)
    return filtered


def centered_moving_average_trajectory(
    trajectory: Trajectory,
    window_size: int = TRAJECTORY_MOVING_AVERAGE_WINDOW,
) -> Trajectory:
    """
    可选居中滑动平均；默认关闭以保留纸面原始逐帧晃动。

    当 window_size > 1 时，对第 i 帧前后窗口内的有效坐标求平均；
    如果窗口里遇到 None，说明红纸出画或无数据，直接跳过不计入平均。
    """
    if window_size <= 1:
        return {
            frame_idx: None if corners is None else corners.copy()
            for frame_idx, corners in trajectory.items()
        }

    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2

    averaged_trajectory: Trajectory = {}
    for frame_idx in sorted(trajectory):
        current = trajectory[frame_idx]
        if current is None or not is_valid_quad(current):
            averaged_trajectory[frame_idx] = None
            continue

        samples = []
        for sample_idx in range(frame_idx - half_window, frame_idx + half_window + 1):
            sample = trajectory.get(sample_idx)
            if sample is not None and is_valid_quad(sample):
                samples.append(sample)

        if not samples:
            averaged_trajectory[frame_idx] = current.copy()
            continue

        averaged_trajectory[frame_idx] = np.mean(np.stack(samples), axis=0).astype(np.float32)

    return averaged_trajectory


def centered_gaussian_smooth(values: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    """
    对称高斯低通滤波。

    使用当前帧前后窗口一起算，所以不会把文字整体拖在纸张后面。
    """
    if radius <= 0 or len(values) <= 2:
        return values.astype(np.float32)

    sigma = max(float(sigma), 0.1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    base_weights = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma))

    smoothed = np.empty_like(values, dtype=np.float32)
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        weight_start = radius - (idx - start)
        weight_end = weight_start + (end - start)
        weights = base_weights[weight_start:weight_end].astype(np.float32)
        weights /= weights.sum()
        smoothed[idx] = np.tensordot(weights, values[start:end], axes=(0, 0))
    return smoothed


def smooth_loaded_trajectory(trajectory: Trajectory) -> Trajectory:
    """
    对 AE 轨迹做离线轻量处理，中心逐帧锁定原始纸面位置。

    处理思路：
    1. 只处理连续有效坐标段，None 出画段原样保留。
    2. 中心使用原始 AE 值，避免文字相对未防抖画面产生滞后。
    3. 四角相对中心的形状保留轻量去尖刺和平滑，减少透视扭动。
    """
    smoothed_trajectory: Trajectory = {
        frame_idx: None if corners is None else corners.copy()
        for frame_idx, corners in trajectory.items()
    }

    for segment in iter_valid_trajectory_segments(trajectory):
        if len(segment) < 3:
            continue

        raw = np.stack([trajectory[frame_idx] for frame_idx in segment]).astype(np.float32)
        despiked = centered_median_filter(raw, TRAJECTORY_MEDIAN_RADIUS)

        despiked_centers = despiked.mean(axis=1)
        centers = despiked_centers
        shapes = despiked - despiked_centers[:, None, :]

        center_motion_samples = np.zeros(len(segment), dtype=np.float32)
        shape_motion_samples = np.zeros(len(segment), dtype=np.float32)
        if len(segment) > 1:
            center_deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            shape_deltas = np.mean(np.linalg.norm(np.diff(shapes, axis=0), axis=2), axis=1)
            center_motion_samples[1:] = center_deltas
            center_motion_samples[0] = center_deltas[0]
            shape_motion_samples[1:] = shape_deltas
            shape_motion_samples[0] = shape_deltas[0]

        smooth_centers = centered_gaussian_smooth(
            centers,
            TRAJECTORY_CENTER_SMOOTH_RADIUS,
            TRAJECTORY_SMOOTH_SIGMA,
        )
        smooth_shapes = centered_gaussian_smooth(
            shapes,
            TRAJECTORY_SHAPE_SMOOTH_RADIUS,
            TRAJECTORY_SMOOTH_SIGMA,
        )

        # 片段边缘通常是红纸刚入画/快出画，少动原始位置，避免提前出现或延后消失。
        edge_radius = max(TRAJECTORY_CENTER_SMOOTH_RADIUS, TRAJECTORY_SHAPE_SMOOTH_RADIUS, 1)
        for local_idx, frame_idx in enumerate(segment):
            edge_distance = min(local_idx, len(segment) - 1 - local_idx)
            edge_ramp = min(1.0, edge_distance / float(edge_radius))
            motion = max(center_motion_samples[local_idx], shape_motion_samples[local_idx])
            motion_t = (motion - TRAJECTORY_MOTION_SOFT_PIXELS) / max(
                TRAJECTORY_MOTION_HARD_PIXELS - TRAJECTORY_MOTION_SOFT_PIXELS,
                1e-3,
            )
            motion_t = float(np.clip(motion_t, 0.0, 1.0))
            motion_t = motion_t * motion_t * (3.0 - 2.0 * motion_t)

            center_blend = (
                TRAJECTORY_CENTER_BLEND * (1.0 - motion_t) + TRAJECTORY_FAST_CENTER_BLEND * motion_t
            ) * edge_ramp
            shape_blend = (
                TRAJECTORY_SHAPE_BLEND * (1.0 - motion_t) + TRAJECTORY_FAST_SHAPE_BLEND * motion_t
            ) * edge_ramp

            center = centers[local_idx] * (1.0 - center_blend) + smooth_centers[local_idx] * center_blend
            shape = shapes[local_idx] * (1.0 - shape_blend) + smooth_shapes[local_idx] * shape_blend
            smoothed_trajectory[frame_idx] = (center + shape).astype(np.float32)

    return smoothed_trajectory


def scale_trajectory_corners(
    corners: np.ndarray,
    video_width: int,
    video_height: int,
) -> np.ndarray:
    """如果模板视频尺寸与 AE 坐标源尺寸不同，按比例缩放四角坐标。"""
    scale_x = video_width / TRAJECTORY_SOURCE_WIDTH
    scale_y = video_height / TRAJECTORY_SOURCE_HEIGHT
    scale = np.array([scale_x, scale_y], dtype=np.float32)
    return corners.astype(np.float32) * scale


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """把四点统一成 左上、右上、右下、左下 顺序。"""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("quad points must be 4x2")

    summed = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[int(np.argmin(summed))],
            pts[int(np.argmin(diff))],
            pts[int(np.argmax(summed))],
            pts[int(np.argmax(diff))],
        ],
        dtype=np.float32,
    )


def detect_paper_quad(frame_bgr: np.ndarray, seed_quad: np.ndarray) -> np.ndarray | None:
    """
    在当前帧里按红色纸张区域做轻量检测，返回外层纸面四边形。

    这一步只在文字渲染前做校正，不参与最终导出，失败时直接回退到原轨迹。
    """
    if seed_quad.shape != (4, 2):
        return None

    height, width = frame_bgr.shape[:2]
    seed_center = seed_quad.mean(axis=0)
    expanded = seed_center + (seed_quad - seed_center) * 1.8

    x0 = max(0, int(np.floor(expanded[:, 0].min() - 50)))
    y0 = max(0, int(np.floor(expanded[:, 1].min() - 50)))
    x1 = min(width, int(np.ceil(expanded[:, 0].max() + 50)))
    y1 = min(height, int(np.ceil(expanded[:, 1].max() + 50)))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    roi = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 35, 45]), np.array([14, 255, 255]))
    mask |= cv2.inRange(hsv, np.array([156, 35, 45]), np.array([179, 255, 255]))
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    seed_poly = (seed_quad - np.array([x0, y0], dtype=np.float32)).astype(np.float32)
    seed_center_roi = seed_poly.mean(axis=0)
    seed_area = max(abs(cv2.contourArea(seed_poly)), 1.0)

    best_box: np.ndarray | None = None
    best_score = -1e9
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 2000:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        box_center = box.mean(axis=0)
        distance = float(np.linalg.norm(box_center - seed_center_roi))
        area_ratio = area / seed_area
        score = math.log(max(area, 1.0)) - 0.012 * distance - abs(math.log(max(area_ratio, 1e-3) / 2.6)) * 0.25
        if score > best_score:
            best_score = score
            best_box = box

    if best_box is None:
        return None
    return order_quad_points(best_box + np.array([x0, y0], dtype=np.float32))


def build_soft_paper_mask(frame_bgr: np.ndarray, seed_quad: np.ndarray) -> np.ndarray | None:
    """为文字 alpha 生成红纸软遮罩，避免文字边缘漂到纸外背景上。"""
    if seed_quad.shape != (4, 2):
        return None

    height, width = frame_bgr.shape[:2]
    seed_center = seed_quad.mean(axis=0)
    expanded = seed_center + (seed_quad - seed_center) * 1.65

    x0 = max(0, int(np.floor(expanded[:, 0].min() - 35)))
    y0 = max(0, int(np.floor(expanded[:, 1].min() - 35)))
    x1 = min(width, int(np.ceil(expanded[:, 0].max() + 35)))
    y1 = min(height, int(np.ceil(expanded[:, 1].max() + 35)))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    roi = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 32, 42]), np.array([16, 255, 255]))
    mask |= cv2.inRange(hsv, np.array([154, 32, 42]), np.array([179, 255, 255]))
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    seed_center_roi = seed_center - np.array([x0, y0], dtype=np.float32)
    seed_area = quad_area(seed_quad)
    best_contour = None
    best_score = -1e9
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(seed_area * 0.6, 1500.0):
            continue

        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            continue
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)
        distance = float(np.linalg.norm(center - seed_center_roi))
        score = math.log(max(area, 1.0)) - distance * 0.01
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return None

    roi_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(roi_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    roi_mask = cv2.GaussianBlur(roi_mask, (21, 21), 0)
    full_mask = np.zeros((height, width), dtype=np.float32)
    full_mask[y0:y1, x0:x1] = roi_mask.astype(np.float32) / 255.0
    return np.clip(full_mask * 1.05, 0.0, 1.0)


def quad_to_unit_space(quad: np.ndarray, paper_quad: np.ndarray) -> np.ndarray:
    """把纸面上的四点投到单位纸面坐标里。"""
    unit_quad = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    to_unit = cv2.getPerspectiveTransform(paper_quad.astype(np.float32), unit_quad)
    return cv2.perspectiveTransform(quad.astype(np.float32)[None, :, :], to_unit)[0]


def unit_space_to_quad(unit_quad: np.ndarray, paper_quad: np.ndarray) -> np.ndarray:
    """把单位纸面坐标里的四点映射回当前帧纸面。"""
    unit_quad = np.asarray(unit_quad, dtype=np.float32)
    paper_quad = np.asarray(paper_quad, dtype=np.float32)
    from_unit = cv2.getPerspectiveTransform(
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        paper_quad.astype(np.float32),
    )
    return cv2.perspectiveTransform(unit_quad[None, :, :], from_unit)[0]


def correct_quad_with_detected_paper(
    current_quad: np.ndarray,
    paper_quad: np.ndarray | None,
) -> np.ndarray:
    """用检测到的红纸外框做保守的中心/尺度辅助校正，不接管透视形状。"""
    if paper_quad is None:
        return current_quad

    current_area = quad_area(current_quad)
    paper_area = quad_area(paper_quad)
    if current_area <= 1.0 or paper_area <= 1.0:
        return current_quad

    center_error = float(np.linalg.norm(paper_quad.mean(axis=0) - current_quad.mean(axis=0)))
    area_ratio = current_area / paper_area
    if (
        center_error > PAPER_QUAD_MAX_CENTER_ERROR
        or area_ratio < PAPER_QUAD_MIN_AREA_RATIO
        or area_ratio > PAPER_QUAD_MAX_AREA_RATIO
    ):
        return current_quad

    paper_center = paper_quad.mean(axis=0)
    current_center = current_quad.mean(axis=0)
    current_shape = current_quad - current_center

    # current_quad 是纸面内排版区域，不是纸外框；只跟随检测到的整体中心和轻微尺度。
    target_center = current_center * (1.0 - PAPER_QUAD_CORRECTION_BLEND) + paper_center * PAPER_QUAD_CORRECTION_BLEND
    return (target_center + current_shape).astype(np.float32)


def is_valid_quad(corners: np.ndarray) -> bool:
    """防御性检查：四边形面积过小或异常时跳过绘制。"""
    if corners.shape != (4, 2):
        return False
    area = abs(cv2.contourArea(corners.astype(np.float32)))
    return area > 20


def find_valid_corners_at_or_before(trajectory: Trajectory, frame_idx: int) -> np.ndarray | None:
    """向前寻找某一帧之前最近的有效四角坐标。"""
    for idx in range(frame_idx, -1, -1):
        corners = trajectory.get(idx)
        if corners is not None and is_valid_quad(corners):
            return corners.copy()
    return None


def find_first_valid_corners(trajectory: Trajectory) -> np.ndarray | None:
    """寻找轨迹文件里的第一组有效四角坐标，作为形状参考。"""
    for frame_idx in sorted(trajectory):
        corners = trajectory[frame_idx]
        if corners is not None and is_valid_quad(corners):
            return corners.copy()
    return None


def extrapolate_corners(
    previous_valid: np.ndarray | None,
    last_valid: np.ndarray | None,
    previous_frame: int | None,
    last_frame: int | None,
    current_frame: int,
    max_frames: int,
) -> np.ndarray | None:
    """
    缺坐标时，根据最后两组有效 AE 坐标预测纸张继续移动。

    这比固定沿用最后一帧更自然：红纸出画时，文字会继续随纸滑出并被裁切/淡出。
    """
    if last_valid is None or last_frame is None:
        return None
    if current_frame - last_frame > max_frames:
        return None
    if previous_valid is None or previous_frame is None or last_frame <= previous_frame:
        return last_valid.copy()

    velocity = (last_valid - previous_valid) / float(last_frame - previous_frame)
    predicted = last_valid + velocity * float(current_frame - last_frame)
    return predicted.astype(np.float32)


def safe_length(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def smooth_trajectory_corners(
    previous: np.ndarray | None,
    current: np.ndarray,
    alpha: float = COORDINATE_SMOOTH_ALPHA,
) -> np.ndarray:
    """
    自适应 EMA 低通滤波。

    中心点代表纸张的真实位移：大幅运动必须立即跟随，否则文字会滞后打滑。
    四角相对中心的形状代表透视和纸面形变：低速段低通压抖，快运动段提高跟随。
    如果快运动时仍使用低 alpha，文字区域的宽高会慢半拍，看起来像漂在纸面上。
    """
    if previous is None:
        return current.astype(np.float32)

    previous_center = previous.mean(axis=0)
    current_center = current.mean(axis=0)
    center_motion = safe_length(current_center - previous_center)

    previous_shape = previous - previous_center
    current_shape = current - current_center
    shape_motion = float(np.mean(np.linalg.norm(current_shape - previous_shape, axis=1)))

    # 中心点已经经过离线零相位平滑；运行时中心直接跟随，避免相位差。
    smoothed_center = current_center

    motion = max(center_motion, shape_motion)
    soft = max(SHAPE_SOFT_FOLLOW_PIXELS, COORDINATE_DEAD_ZONE_PIXELS)
    hard = max(SHAPE_HARD_FOLLOW_PIXELS, soft + 1e-3)
    follow_t = float(np.clip((motion - soft) / (hard - soft), 0.0, 1.0))
    follow_t = follow_t * follow_t * (3.0 - 2.0 * follow_t)

    slow_alpha = float(np.clip(alpha, 0.0, 1.0))
    fast_alpha = float(np.clip(SHAPE_FAST_SMOOTH_ALPHA, slow_alpha, 1.0))
    adaptive_alpha = slow_alpha * (1.0 - follow_t) + fast_alpha * follow_t
    if center_motion >= CENTER_HARD_FOLLOW_PIXELS:
        adaptive_alpha = max(adaptive_alpha, 0.98)

    smoothed_shape = current_shape * adaptive_alpha + previous_shape * (1.0 - adaptive_alpha)
    return (smoothed_center + smoothed_shape).astype(np.float32)


def quad_shape_stats(corners: np.ndarray) -> dict[str, float]:
    """计算四边形形状比例，用于判断尾段是否有拉扯变形。"""
    top = safe_length(corners[1] - corners[0])
    right = safe_length(corners[2] - corners[1])
    bottom = safe_length(corners[2] - corners[3])
    left = safe_length(corners[3] - corners[0])
    diag_a = safe_length(corners[2] - corners[0])
    diag_b = safe_length(corners[3] - corners[1])

    avg_width = max((top + bottom) / 2.0, 1.0)
    avg_height = max((left + right) / 2.0, 1.0)
    return {
        "aspect": avg_width / avg_height,
        "width_ratio": max(top, bottom) / max(min(top, bottom), 1.0),
        "height_ratio": max(left, right) / max(min(left, right), 1.0),
        "diagonal_ratio": max(diag_a, diag_b) / max(min(diag_a, diag_b), 1.0),
        "avg_width": avg_width,
        "avg_height": avg_height,
    }


def build_rigid_reference_quad(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    用当前中心点和当前宽度构建一个更接近第一帧比例的刚性矩形。

    这样保留纸张在画面里的大致位置和尺寸，但限制尾段倒梯形、
    对角线不均衡等容易穿帮的透视拉扯。
    """
    ref_stats = quad_shape_stats(reference)
    cur_stats = quad_shape_stats(current)
    center = current.mean(axis=0)

    top_vec = current[1] - current[0]
    bottom_vec = current[2] - current[3]
    x_axis = top_vec + bottom_vec
    if safe_length(x_axis) < 1:
        x_axis = reference[1] - reference[0]
    x_axis = x_axis / max(safe_length(x_axis), 1.0)
    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)

    width = cur_stats["avg_width"]
    height = width / max(ref_stats["aspect"], 0.1)
    half_w = width / 2.0
    half_h = height / 2.0

    rigid = np.array(
        [
            center - x_axis * half_w - y_axis * half_h,
            center + x_axis * half_w - y_axis * half_h,
            center + x_axis * half_w + y_axis * half_h,
            center - x_axis * half_w + y_axis * half_h,
        ],
        dtype=np.float32,
    )
    return rigid


def constrain_tail_distortion(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """尾段防畸变：形状超标时向第一帧刚性矩形比例收敛。"""
    cur_stats = quad_shape_stats(current)
    ref_stats = quad_shape_stats(reference)

    aspect_drift = abs(cur_stats["aspect"] / max(ref_stats["aspect"], 0.1) - 1.0)
    edge_drift = max(cur_stats["width_ratio"] - 1.0, cur_stats["height_ratio"] - 1.0)
    diag_drift = abs(cur_stats["diagonal_ratio"] / max(ref_stats["diagonal_ratio"], 1.0) - 1.0)

    rigid = build_rigid_reference_quad(current, reference)
    over_limit = (
        aspect_drift > MAX_ASPECT_DRIFT
        or edge_drift > MAX_EDGE_RATIO_DRIFT
        or diag_drift > MAX_DIAGONAL_RATIO_DRIFT
    )
    blend = TAIL_STRONG_RIGID_BLEND if over_limit else TAIL_SOFT_RIGID_BLEND
    return (current * (1.0 - blend) + rigid * blend).astype(np.float32)


def visible_area_ratio(corners: np.ndarray, width: int, height: int) -> float:
    """计算纸面四边形在画面中的可见比例。"""
    quad_area = abs(cv2.contourArea(corners.astype(np.float32)))
    if quad_area <= 1:
        return 0.0

    frame_rect = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    try:
        visible_area, _ = cv2.intersectConvexConvex(corners.astype(np.float32), frame_rect)
    except cv2.error:
        return 0.0
    return float(np.clip(visible_area / quad_area, 0.0, 1.0))


def opacity_from_visible_ratio(ratio: float) -> float:
    """红纸半出画时透明度平滑渐变，避免文字硬贴在背景上。"""
    if ratio <= FADE_MIN_VISIBLE_RATIO:
        return 0.0
    if ratio >= FADE_FULL_VISIBLE_RATIO:
        return 1.0

    t = (ratio - FADE_MIN_VISIBLE_RATIO) / (FADE_FULL_VISIBLE_RATIO - FADE_MIN_VISIBLE_RATIO)
    return float(t * t * (3.0 - 2.0 * t))


def apply_motion_blur_to_alpha(
    alpha: np.ndarray,
    motion_vector: np.ndarray | None,
    motion_amount: float | None = None,
) -> np.ndarray:
    """按纸面四角运动量给文字 alpha 加方向拖影和近似缩放模糊。"""
    if motion_vector is None:
        return alpha

    motion = np.array(motion_vector, dtype=np.float32)
    distance = safe_length(motion)
    blur_distance = float(motion_amount) if motion_amount is not None else distance
    if blur_distance < 3.0 or TEXT_MOTION_BLUR_SCALE <= 0:
        return alpha

    kernel_size = int(round(blur_distance * TEXT_MOTION_BLUR_SCALE))
    kernel_size = max(3, min(TEXT_MOTION_BLUR_MAX_KERNEL, kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = alpha
    if distance >= 1.0:
        direction = motion / max(distance, 1e-6)
        center = (kernel_size - 1) / 2.0
        half = center
        start = (
            int(round(center - direction[0] * half)),
            int(round(center - direction[1] * half)),
        )
        end = (
            int(round(center + direction[0] * half)),
            int(round(center + direction[1] * half)),
        )

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        cv2.line(kernel, start, end, 1.0, 1)
        kernel_sum = float(kernel.sum())
        if kernel_sum > 0:
            kernel /= kernel_sum
            blurred = cv2.filter2D(alpha, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    gaussian_sigma = min(
        TEXT_MOTION_BLUR_MAX_KERNEL / 3.0,
        blur_distance * max(0.0, TEXT_MOTION_BLUR_GAUSSIAN_SCALE),
    )
    if gaussian_sigma >= 0.35:
        gaussian = cv2.GaussianBlur(alpha, (0, 0), gaussian_sigma)
        mix = float(np.clip((blur_distance - 8.0) / 24.0, 0.0, 0.70))
        blurred = blurred * (1.0 - mix) + gaussian * mix

    return np.clip(blurred, 0.0, 1.0)


def quad_area(corners: np.ndarray | None) -> float:
    if corners is None or corners.shape != (4, 2):
        return 0.0
    return float(abs(cv2.contourArea(corners.astype(np.float32))))


def estimate_text_layer_scale(current_corners: np.ndarray, reference_corners: np.ndarray | None) -> float:
    """用纸面面积变化给文字层做轻微补偿缩放。"""
    ref_area = quad_area(reference_corners)
    cur_area = quad_area(current_corners)
    if ref_area <= 1.0 or cur_area <= 1.0:
        return 1.0

    paper_scale = math.sqrt(cur_area / ref_area)
    compensated = 1.0 + (paper_scale - 1.0) * TEXT_LAYER_SCALE_GAIN
    return float(np.clip(compensated, TEXT_LAYER_SCALE_MIN, TEXT_LAYER_SCALE_MAX))


def estimate_quad_rotation_degrees(current_corners: np.ndarray, reference_corners: np.ndarray | None) -> float:
    """用最小二乘相似变换估计纸面面内小角度旋转。"""
    if reference_corners is None or current_corners.shape != (4, 2) or reference_corners.shape != (4, 2):
        return 0.0
    matrix, _ = cv2.estimateAffinePartial2D(
        reference_corners.astype(np.float32),
        current_corners.astype(np.float32),
        method=cv2.LMEDS,
    )
    if matrix is None:
        return 0.0
    angle = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    angle *= TEXT_LAYER_ROTATION_GAIN
    return float(np.clip(angle, -TEXT_LAYER_ROTATION_MAX_DEGREES, TEXT_LAYER_ROTATION_MAX_DEGREES))


def scale_text_layer_rgba(layer: np.ndarray, scale: float, rotation_degrees: float = 0.0) -> np.ndarray:
    """围绕中心缩放/轻微旋转整层文字，不改变画布尺寸。"""
    if abs(scale - 1.0) < 0.01 and abs(rotation_degrees) < 0.01:
        return layer

    h, w = layer.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rotation_degrees, scale)
    matrix[1, 2] += h * TEXT_LAYOUT_Y_OFFSET_RATIO
    return cv2.warpAffine(
        layer,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def multiply_like_blend(
    frame_bgr: np.ndarray,
    warped_rgba: np.ndarray,
    opacity: float = 0.85,
    motion_vector: np.ndarray | None = None,
    motion_amount: float | None = None,
    paper_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    模拟墨迹渗入红纸的效果：文字不仅覆盖纸面，还会让周围区域微微变暗。

    1. Alpha 边缘做高斯模糊，消除数字贴图的硬边。
    2. 文字区域用更柔和的暗化曲线，避免纯黑硬压。
    3. 文字周围保留少量纸纹透出，让它更像印在纸上。
    """
    if warped_rgba.shape[2] != 4:
        raise ValueError("warped_rgba 必须是 RGBA 四通道图像")

    alpha = warped_rgba[:, :, 3].astype(np.float32) / 255.0
    if paper_mask is not None and paper_mask.shape == alpha.shape:
        alpha *= paper_mask.astype(np.float32)
    alpha = apply_motion_blur_to_alpha(alpha, motion_vector, motion_amount)
    warped_rgb = cv2.GaussianBlur(warped_rgba[:, :, :3], (3, 3), 0.7).astype(np.float32)

    # 文字主体 alpha：轻微模糊消除像素感
    alpha_core = cv2.GaussianBlur(alpha, (3, 3), max(0.2, TEXT_EDGE_SOFTEN_SIGMA))

    # 墨水扩散晕染：更大半径、更弱的模糊，模拟墨迹在纸纤维上轻微扩散
    alpha_halo = cv2.GaussianBlur(alpha, (7, 7), 2.0)

    # 混合：主体占主导，晕染只提供极其微弱的边缘过渡
    alpha_blended = alpha_core * (1.0 - TEXT_HALO_WEIGHT) + alpha_halo * TEXT_HALO_WEIGHT
    alpha_blended = np.clip(alpha_blended * opacity, 0, 1)

    frame_float = frame_bgr.astype(np.float32)
    paper_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    paper_gray = cv2.GaussianBlur(paper_gray, (3, 3), 0.8)

    # 更柔和的纸面暗化：把硬黑压暗改成暗红棕叠加感，保留纸纹。
    darken = 1.0 - alpha_blended[:, :, None] * TEXT_BASE_DARKEN
    paper_texture = 1.0 - (paper_gray[:, :, None] - 0.5) * TEXT_PAPER_TEXTURE_GAIN * alpha_blended[:, :, None]
    blended = frame_float * darken * paper_texture
    if TEXT_INK_SINK_STRENGTH > 0:
        sink = np.clip(alpha_blended[:, :, None] * TEXT_INK_SINK_STRENGTH, 0.0, 1.0)
        ink_tint_bgr = np.array([14.0, 18.0, 28.0], dtype=np.float32)
        warped_bgr = warped_rgb[:, :, ::-1]
        layer_tint = np.where(warped_bgr > 0.5, warped_bgr, ink_tint_bgr)
        blended = blended * (1.0 - sink) + layer_tint * sink
    return np.clip(blended, 0, 255).astype(np.uint8)


def add_text_to_video(
    video_path: str | Path,
    trajectory_path: str | Path,
    name: str,
    birthdate: str,
    font_path: str | Path = DEFAULT_FONT_PATH,
    output_path: str | Path | None = None,
    text_opacity: float = 0.85,
    max_duration_seconds: float | None = None,
) -> Path:
    """
    基于预载 trajectory.json 的无脑帧对齐贴图。

    每一帧逻辑：
    1. frame_idx 查 trajectory.json
    2. 值为 None：红纸出画，不绘制文字
    3. 值为四角坐标：直接 getPerspectiveTransform + warpPerspective
    """
    video_file = ensure_file_exists(video_path, "视频模板")
    trajectory = centered_moving_average_trajectory(load_trajectory(trajectory_path))
    trajectory = smooth_loaded_trajectory(trajectory)
    width, height, fps, _ = get_video_info(video_file)
    hold_start_frame = int(round(HOLD_VISIBLE_START_SECONDS * fps))
    hold_end_frame = int(round(HOLD_VISIBLE_END_SECONDS * fps))
    hold_reference_frame = max(0, int(round(HOLD_REFERENCE_SECONDS * fps)) - 1)
    max_extrapolate_frames = max(1, int(round(MAX_EXTRAPOLATE_SECONDS * fps)))
    hold_reference_corners = find_valid_corners_at_or_before(trajectory, hold_reference_frame)
    reference_corners = find_first_valid_corners(trajectory)
    if reference_corners is not None:
        reference_corners = scale_trajectory_corners(reference_corners, width, height)

    max_frames = None
    if max_duration_seconds and max_duration_seconds > 0:
        max_frames = max(1, int(max_duration_seconds * fps))

    if output_path is None:
        output_path = Path(tempfile.mkdtemp()) / "silent_with_text.mp4"
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    text_layer = create_text_layer(name, birthdate, font_path)
    text_rgba = np.array(text_layer)
    canvas_w, canvas_h = text_layer.size

    src_points = np.array(
        [[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]],
        dtype=np.float32,
    )

    cap = cv2.VideoCapture(str(video_file))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频：{output_file}")

    try:
        frame_idx = 0
        smoothed_corners: np.ndarray | None = None
        previous_final_corners_for_blur: np.ndarray | None = None
        smoothed_text_scale: float | None = None
        smoothed_paper_quad: np.ndarray | None = None
        previous_valid_raw_corners: np.ndarray | None = None
        last_valid_raw_corners: np.ndarray | None = None
        previous_valid_frame_idx: int | None = None
        last_valid_frame_idx: int | None = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            corners = trajectory.get(frame_idx)
            is_predicted_corners = False
            if corners is None and hold_start_frame <= frame_idx <= hold_end_frame:
                corners = extrapolate_corners(
                    previous_valid=previous_valid_raw_corners,
                    last_valid=last_valid_raw_corners,
                    previous_frame=previous_valid_frame_idx,
                    last_frame=last_valid_frame_idx,
                    current_frame=frame_idx,
                    max_frames=max_extrapolate_frames,
                )
                if corners is None:
                    corners = hold_reference_corners
                is_predicted_corners = corners is not None

            if corners is None:
                smoothed_corners = None
                previous_final_corners_for_blur = None
                smoothed_text_scale = None
                writer.write(frame)
                frame_idx += 1
                continue

            dst_points = scale_trajectory_corners(corners, width, height)
            if not is_valid_quad(dst_points):
                smoothed_corners = None
                previous_final_corners_for_blur = None
                smoothed_text_scale = None
                writer.write(frame)
                frame_idx += 1
                continue

            if not is_predicted_corners:
                previous_valid_raw_corners = last_valid_raw_corners
                previous_valid_frame_idx = last_valid_frame_idx
                last_valid_raw_corners = corners.copy()
                last_valid_frame_idx = frame_idx

            paper_quad = detect_paper_quad(frame, dst_points)
            if paper_quad is not None:
                if smoothed_paper_quad is None:
                    smoothed_paper_quad = paper_quad
                else:
                    # 外框检测会受香/手/运动模糊干扰；轻量平滑只稳定检测结果，不拖慢 AE 中心。
                    smoothed_paper_quad = (paper_quad * 0.35 + smoothed_paper_quad * 0.65).astype(np.float32)
                dst_points = correct_quad_with_detected_paper(dst_points, smoothed_paper_quad)

            if frame_idx >= int(round(TAIL_CONSTRAINT_START_SECONDS * fps)) and reference_corners is not None:
                dst_points = constrain_tail_distortion(dst_points, reference_corners)

            dst_points = smooth_trajectory_corners(smoothed_corners, dst_points)
            smoothed_corners = dst_points

            visible_ratio = visible_area_ratio(dst_points, width, height)
            frame_opacity = text_opacity * opacity_from_visible_ratio(visible_ratio)
            if frame_opacity <= 0.01:
                writer.write(frame)
                frame_idx += 1
                continue

            motion_vector = None
            corner_motion_amount = 0.0
            if previous_final_corners_for_blur is not None:
                corner_deltas = dst_points - previous_final_corners_for_blur
                motion_vector = corner_deltas.mean(axis=0)
                corner_motion_amount = float(np.linalg.norm(corner_deltas, axis=1).mean())
                if corner_motion_amount > TEXT_FAST_MOTION_THRESHOLD:
                    speed_t = float(np.clip((corner_motion_amount - TEXT_FAST_MOTION_THRESHOLD) / 17.0, 0.0, 1.0))
                    frame_opacity *= (0.80 * (1.0 - speed_t) + TEXT_FAST_OPACITY_MIN * speed_t)

            dynamic_text_scale = estimate_text_layer_scale(dst_points, reference_corners)
            if smoothed_text_scale is None:
                smoothed_text_scale = dynamic_text_scale
            else:
                smoothed_text_scale = (
                    smoothed_text_scale * (1.0 - TEXT_SCALE_SMOOTH_ALPHA)
                    + dynamic_text_scale * TEXT_SCALE_SMOOTH_ALPHA
                )
            text_scale = TEXT_LAYOUT_CONTENT_SCALE * smoothed_text_scale
            text_rotation = estimate_quad_rotation_degrees(dst_points, reference_corners)
            frame_text_rgba = scale_text_layer_rgba(text_rgba, text_scale, text_rotation)
            frame_canvas_h, frame_canvas_w = frame_text_rgba.shape[:2]
            frame_src_points = np.array(
                [[0, 0], [frame_canvas_w, 0], [frame_canvas_w, frame_canvas_h], [0, frame_canvas_h]],
                dtype=np.float32,
            )

            perspective_matrix = cv2.getPerspectiveTransform(frame_src_points, dst_points)
            warped = cv2.warpPerspective(
                frame_text_rgba,
                perspective_matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            paper_mask = build_soft_paper_mask(frame, dst_points)
            result_frame = multiply_like_blend(
                frame,
                warped,
                opacity=frame_opacity,
                motion_vector=motion_vector,
                motion_amount=corner_motion_amount,
                paper_mask=paper_mask,
            )
            writer.write(result_frame)
            previous_final_corners_for_blur = dst_points
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return output_file


# =========================
# 第三步：TTS API 调用
# =========================


def build_tts_text(name: str, birthdate: str) -> str:
    return (
        f"上香祈愿。弟子 {name}，生辰 {birthdate}。"
        "祈福：身体健康，工作顺利，万事顺遂，所愿皆成。"
    )


def generate_tts_audio(
    name: str,
    birthdate: str,
    output_path: str | Path,
    api_key: str = DASHSCOPE_API_KEY,
    voice: str = TTS_VOICE,
    model: str = TTS_MODEL,
) -> Path:
    """
    调用阿里云百炼 Qwen3-TTS 生成祈福语音。

    默认使用「沧明子」(Eldric Sage) 音色 —— 低沉苍老的智慧老者，
    语速偏慢、语调沉稳，适合庙宇祈福场景。
    """
    if not api_key:
        raise RuntimeError("请先在 .env 中配置 DASHSCOPE_API_KEY")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tts_text = build_tts_text(name, birthdate)

    response = dashscope.MultiModalConversation.call(
        model=model,
        api_key=api_key,
        text=tts_text,
        voice=voice,
        language_type="Chinese",
        stream=False,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"TTS API 调用失败 (HTTP {response.status_code}): {response.message}"
        )

    if not response.output or not response.output.audio:
        raise RuntimeError("TTS API 未返回音频数据，请检查 API Key 和音色名称是否正确")

    audio_url = response.output.audio.url
    audio_response = requests.get(audio_url, timeout=60)
    audio_response.raise_for_status()

    output_file.write_bytes(audio_response.content)
    if output_file.stat().st_size < 1024:
        raise RuntimeError("TTS API 返回的音频文件过小，请检查接口响应")
    return output_file


# =========================
# 第四步：音视频无缝封装
# =========================


def merge_video_and_audio(
    silent_video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    使用 FFmpeg 合并视频与音频。

    对齐规则：
    1. 最终长度以原视频画面长度为准。
    2. 音频短于视频：apad 自动补静音。
    3. 音频长于视频：-t 截断到视频长度。
    """
    silent_video = ensure_file_exists(silent_video_path, "带字无声视频")
    audio_file = ensure_file_exists(audio_path, "TTS 音频")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    _, _, fps, frame_count = get_video_info(silent_video)
    duration = frame_count / fps

    video_stream = ffmpeg.input(str(silent_video)).video
    audio_stream = ffmpeg.input(str(audio_file)).audio.filter("apad", pad_dur=3600)

    try:
        (
            ffmpeg.output(
                video_stream,
                audio_stream,
                str(output_file),
                t=f"{duration:.3f}",
                vcodec="libx264",
                pix_fmt="yuv420p",
                acodec="aac",
                audio_bitrate="192k",
                movflags="+faststart",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"FFmpeg 执行失败：\n{stderr}") from exc

    return output_file


# =========================
# 第五步：端到端生成流程
# =========================


def generate_prayer_video(
    name: str,
    birthdate: str,
    video_path: str | None = None,
    trajectory_path: str | None = None,
) -> tuple[Path, Path]:
    temp_dir = Path(tempfile.mkdtemp(prefix="prayer_video_"))
    silent_video = temp_dir / "01_silent_with_text.mp4"
    tts_audio = temp_dir / "02_tts_audio.wav"
    final_video = temp_dir / f"prayer_{int(time.time())}.mp4"

    video = video_path or DEFAULT_VIDEO_TEMPLATE_PATH
    traj = trajectory_path or DEFAULT_TRAJECTORY_PATH

    add_text_to_video(
        video_path=video,
        trajectory_path=traj,
        name=name,
        birthdate=birthdate,
        font_path=DEFAULT_FONT_PATH,
        output_path=silent_video,
        text_opacity=0.98,
    )
    generate_tts_audio(name=name, birthdate=birthdate, output_path=tts_audio)
    merge_video_and_audio(
        silent_video_path=silent_video,
        audio_path=tts_audio,
        output_path=final_video,
    )
    return final_video, temp_dir


# =========================
# Streamlit Web 前端
# =========================


def main() -> None:
    st.set_page_config(page_title="祈愿视频生成系统", layout="centered")
    st.title("🙏 祈愿视频生成系统")

    # 模板选择
    template_labels = [cfg["label"] for cfg in TEMPLATE_CONFIGS]
    template_idx = st.selectbox(
        "选择视频模板",
        range(len(template_labels)),
        format_func=lambda i: template_labels[i],
        index=DEFAULT_TEMPLATE_INDEX,
    )
    selected_video = TEMPLATE_CONFIGS[template_idx]["video"]
    selected_trajectory = TEMPLATE_CONFIGS[template_idx]["trajectory"]

    # 检查轨迹文件是否存在
    if not Path(selected_trajectory).exists():
        st.warning(f"⚠ 模板「{template_labels[template_idx]}」的轨迹坐标文件尚未准备好，请先选择其他模板。")
        return

    name = st.text_input("姓名", placeholder="如：张三")
    birthdate = st.text_input("生辰", placeholder="如：1990年正月初一")

    submitted = st.button("一键生成祈愿视频", type="primary", use_container_width=True)
    if not submitted:
        return

    if not name.strip():
        st.error("请填写姓名")
        return
    if not birthdate.strip():
        st.error("请填写生辰")
        return

    temp_dir: Path | None = None
    try:
        with st.spinner("正在生成祈愿视频，请稍候..."):
            final_video, temp_dir = generate_prayer_video(
                name=name.strip(),
                birthdate=birthdate.strip(),
                video_path=selected_video,
                trajectory_path=selected_trajectory,
            )
            video_bytes = final_video.read_bytes()

        st.success("生成成功")
        st.video(video_bytes)
        st.download_button(
            "下载 MP4",
            data=video_bytes,
            file_name=f"祈愿视频_{name.strip()}_{int(time.time())}.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
    except Exception as exc:
        st.error(str(exc))
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
