import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app

# 字体兜底：优先用 .env 里的 simkai.ttf，没有就退回仓库字体
font = os.getenv("PAPER_FONT_PATH") or ""
if not font or not Path(font).exists():
    font = str(Path(app.PROJECT_ROOT) / "public/fonts/Xingkai.ttf")

out = app.add_text_to_video(
    video_path="模板视频1.mp4",
    trajectory_path="trajectory.json",
    name="张三",
    birthdate="一九九零年五月初一",
    font_path=font,
    output_path="outputs/preview_after.mp4",
)
print("rendered:", out)
