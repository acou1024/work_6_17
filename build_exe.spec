# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置：祈愿视频生成系统

打包命令：
    pyinstaller build_exe.spec

打包后会在 dist/ 目录生成可运行的文件夹。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()

a = Analysis(
    ['launcher.py'],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        # Python 源码
        ('app.py', '.'),
        ('parse_two_ae_files.py', '.'),
        # 资产文件
        ('trajectory.json', '.'),
        ('模板视频1.mp4', '.'),
        ('模板视频2.mp4', '.'),
        ('语言素材.m4a', '.'),
        ('.env', '.'),
        # 字体
        ('public/fonts/Xingkai.ttf', 'public/fonts'),
        # Streamlit 静态文件（打包后需要）
        ('requirements.txt', '.'),
    ],
    hiddenimports=[
        'streamlit',
        'streamlit.runtime',
        'streamlit.web',
        'streamlit.web.server',
        'streamlit.web.server.starlette',
        'streamlit.commands',
        'cv2',
        'numpy',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        'ffmpeg',
        'requests',
        'dotenv',
        'dashscope',
        'altair',
        'pandas',
        'pyarrow',
        'watchdog',
        'rich',
        'pydeck',
        'toml',
        'blinker',
        'gitpython',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'tensorflow',
        'torch',
    ],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='祈愿视频生成系统',
    debug=False,
    bootloader_ignore_signals=False,
    icon=None,
    console=True,
    # 控制台窗口保持中文不乱码
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='祈愿视频生成系统',
)
