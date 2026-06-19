# ChangeLog

## 2026-06-19 16:31 四角形状逐帧锁定

- 主题：消除纸面文字放大后边缘相对红纸的轻微悬浮/游动。
- 关键文件：`app.py`。
- 改动：将中心和四角形状的运行时跟随默认值改为逐帧直通原始 AE 轨迹；离线轨迹处理仅保留 3 帧中值去单帧尖刺，关闭形状平滑混合。
- 验证：`python -m py_compile app.py` 通过；`python tools/render_preview.py` 已重新生成 `outputs/preview_after.mp4`；视频为 30fps/752 帧，并抽取 `outputs/preview_after_shape_lock_review/frame_001.jpg`、`frame_002.jpg`、`frame_003.jpg` 对应第 250/252/255 帧，目检确认剧烈晃动段大字贴纸且边缘未见明显相对游动。

## 2026-06-19 16:09 纸面文字放大

- 主题：进一步放大纸面文字，使内容宽度接近红纸宽度的 85%。
- 关键文件：`app.py`。
- 改动：提高 `create_text_layer` 中标题、身份行和祝福行的 `fit_font` 字号上限，并扩大最大适配宽度，保持原生文字渲染避免拉伸变糊。
- 验证：`python -m py_compile app.py` 通过；`python tools/render_preview.py` 已重新生成 `outputs/preview_after.mp4`，并抽取 `outputs/preview_after_enlarged_review/frame_001s.jpg`、`frame_004s.jpg`、`frame_008s.jpg` 目检确认文字明显变大、接近纸宽 85% 且晃动段未见明显漂移。

## 2026-06-19 15:22 马善政毛笔手写字体

- 主题：将纸面文字切换为随仓库携带的 MaShanZheng 毛笔手写字体，并恢复版式占纸比例。
- 关键文件：`.env`、`public/fonts/MaShanZheng-Regular.ttf`、`public/fonts/MaShanZheng-OFL.txt`。
- 改动：新增 MaShanZheng 字体与 OFL 许可证；`PAPER_FONT_PATH` 指向仓库字体；`TEXT_LAYOUT_CONTENT_SCALE` 调整为 `1.0`。
- 验证：`python -m py_compile tools/render_preview.py` 通过；`python tools/render_preview.py` 已生成 `outputs/preview_after.mp4`，并抽取 `outputs/preview_after_review/frame_001s.jpg`、`frame_004s.jpg`、`frame_008s.jpg` 目检确认毛笔手写字体、版式更饱满且晃动段未见明显漂移。

## 2026-06-19 13:17 纸面文字锚点锁定

- 主题：修复镜头晃动时文字相对红纸悬浮/打滑的问题。
- 关键文件：`app.py`、`.env`、`tools/measure_anchor.py`。
- 改动：关闭轨迹滑动平均和运行时死区冻结，纸面中心逐帧直通原始 AE 轨迹；四角形状仅保留轻量去尖刺/平滑，并开启文字运动模糊匹配纸面快晃。
- 验证：`tools/measure_anchor.py` 修复前 `mean=4.26px, max=82.22px`，修复后 `mean=0.00px, max=0.00px`；`python -m py_compile app.py tools/measure_anchor.py` 通过；已生成 `outputs/anchor_before_old_smoothing.mp4`、`outputs/anchor_after_locked_to_paper.mp4` 和左右对比 `outputs/anchor_lock_before_after_compare.mp4`。

## 2026-06-19 12:58 正式楷体祝福帖观感

- 主题：替换视频纸面字体，降低祝福帖文字的不正式和突兀感。
- 关键文件：`app.py`、`.env`。
- 改动：默认纸面字体改为系统楷体 `simkai.ttf`，将手写字体降为兜底候选；降低文字随机倾斜，收敛版式并加深墨迹压纸效果，减少视频中花字/贴层感。
- 验证：`python -m py_compile app.py diagnose_trajectory.py` 通过；已生成 `outputs/candidate_formal_simkai_native.mp4` 并替换 `outputs/final_current_best.mp4`；旧版备份为 `outputs/final_current_best.bak_before_formal_simkai_20260619_125800.mp4`；关键帧和片段在 `outputs/candidate_formal_simkai_native_review_20260619_125000/`。

## 2026-06-19 11:16 文字运动锁定降抖

- 主题：降低镜头抖动时纸面文字自身晃动感。
- 关键文件：`app.py`。
- 改动：关闭红纸检测结果对文字中心的逐帧校正，避免检测噪声拉动文字；降低运行时四角形状快速跟随强度，减少轨迹角点高频抖动传导到文字透视。
- 验证：`python -m py_compile app.py diagnose_trajectory.py` 通过；已生成 `outputs/candidate_motion_locked.mp4`，对比帧和慢速片段在 `outputs/candidate_motion_locked_review_20260619_111552/`。

## 2026-06-19 02:22 清晰稳定版纸面文字

- 主题：修复纸面字看不清、镜头运动时字体变化幅度偏大的问题。
- 关键文件：`app.py`。
- 改动：关闭文字 alpha 独立运动模糊，增强墨迹压纸强度；四行小字改为三行更大的中心签名区，缩小整体映射区域以降低边缘透视形变带来的视觉变化；Web 生成入口同步使用新版不透明度。
- 验证：`python -m py_compile app.py diagnose_trajectory.py` 通过；已生成 `outputs/candidate_stable_readable_native.mp4` 并替换 `outputs/final_current_best.mp4`；旧版备份为 `outputs/final_current_best.bak_before_stable_readable_native_20260619_022214.mp4`；对比帧和慢速片段在 `outputs/candidate_stable_readable_native_review_20260619_021216/`。

## 2026-06-19 00:10 纸面原生墨迹母版

- 主题：放弃继续微调透明字体贴层，改用纸面坐标里的扫描感墨迹母版。
- 关键文件：`app.py`。
- 改动：文字层改为高分辨率纸面母版渲染，加入稳定随机微扰、中文数字显示、少标点签名式排版、轻量纸面洗色和基于图层颜色的墨迹融合；Web 生成入口同步使用新版压纸强度。
- 验证：`python -m py_compile app.py diagnose_trajectory.py` 通过；已生成 `outputs/candidate_paper_native_stronger.mp4` 并替换 `outputs/final_current_best.mp4`；旧版备份为 `outputs/final_current_best.bak_before_paper_native_20260619_001021.mp4`；对比帧在 `outputs/candidate_paper_native_stronger_review_20260619_000223/`。

## 2026-06-18 18:18 视频文字观感优化

- 主题：降低纸面文字突兀感，改善连续播放时的贴片感。
- 关键文件：`app.py`、`.env`。
- 改动：默认纸面字体优先使用楷体，降低文字硬黑强度，增强边缘柔化、纸纹融合和高速运动模糊，版式略缩小并下移。
- 验证：`python -m py_compile app.py diagnose_trajectory.py` 通过；已生成 `outputs/candidate_font_balanced.mp4` 并替换 `outputs/final_current_best.mp4`；旧版备份为 `outputs/final_current_best.bak_before_font_balance_20260618_184605.mp4`；对比帧在 `outputs/candidate_font_balanced_review/`。

## 2026-06-18 18:52 纸面字缩小加深

- 主题：按“更小、更深、带纸纹”方向再调一版。
- 关键文件：`app.py`、`.env`。
- 改动：进一步降低标题/正文字号，收紧整体版式，提深墨色并增加纸纹参与；后续改为优先使用仿手写 `FZSTK.TTF` 小号版式，并加入纸面坐标固定的墨迹颗粒。
- 验证：`python -m py_compile app.py diagnose_trajectory.py` 通过；已生成 `outputs/candidate_handwritten_small.mp4` 并替换 `outputs/final_current_best.mp4`；旧版备份为 `outputs/final_current_best.bak_before_handwritten_small_20260618_215534.mp4`；对比帧在 `outputs/candidate_handwritten_small_review/`。
