"""
祈愿视频生成系统 - 桌面启动器

双击运行后自动打开浏览器，员工输入姓名生辰即可生成祈愿视频。
关闭命令行窗口即可退出。
"""
import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 加入本地 ffmpeg
FFMPEG_DIR = PROJECT_ROOT / "ffmpeg" / "bin"
if FFMPEG_DIR.exists():
    os.environ["PATH"] = f"{FFMPEG_DIR}{os.pathsep}{os.environ.get('PATH', '')}"

def main():
    print("=" * 50)
    print("  祈愿视频生成系统 v1.0")
    print("=" * 50)
    print()
    print("正在启动，请稍候...")
    print()

    port = 8501
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(PROJECT_ROOT / "app.py"),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.serverAddress", "localhost",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(2)
    url = f"http://localhost:{port}"
    print(f"浏览器将自动打开: {url}")
    print("如果没有自动打开，请手动在浏览器中打开上面的地址")
    print()
    print("按 Ctrl+C 或关闭此窗口退出")
    print()

    webbrowser.open(url)

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("正在退出...")
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
