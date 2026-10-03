"""傅里叶旋转圆盘：SVG/图片 → 等弧长采样 → FFT → H.264 MP4。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

try:
    import cv2
    import imageio_ffmpeg
    import numpy as np
    from svgelements import SVG, Path as SVGPath, Shape, Move, Rect
except ImportError as exc:
    raise SystemExit(f"缺少依赖 {exc.name}。请执行：python -m pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parent


def resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    """第一步：闭合折线等弧长采样，不重复末端点。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if count < 16 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("采样数至少 16，轮廓至少含 3 个有限坐标点。")
    # 去除相邻重复点，防止插值自变量重复。
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10]]
    if len(points) > 1 and np.linalg.norm(points[-1] - points[0]) < 1e-10:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("轮廓退化。")
    closed = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(lengths.sum())
    if total < 1e-10:
        raise ValueError("轮廓长度为零。")
    arc = np.r_[0.0, np.cumsum(lengths)]
    targets = np.linspace(0, total, count, endpoint=False)
    return np.column_stack([np.interp(targets, arc, closed[:, j]) for j in (0, 1)])


def perimeter(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())


def svg_background_rect(element, svg) -> bool:
    """只标记覆盖画布的实心无描边矩形；是否忽略还取决于后续有没有目标轮廓。"""
    if not isinstance(element, Rect) or element.fill.value is None or element.stroke.value is not None:
        return False
    if element.rx or element.ry or svg.width <= 0 or svg.height <= 0:
        return False
    canvas = np.array([0.0, 0.0, float(svg.width), float(svg.height)])
    tolerance = max(svg.width, svg.height) * 1e-6
    if not np.allclose(element.bbox(), canvas, atol=tolerance, rtol=0):
        return False
    # bbox 相同的旋转矩形仍可能是图形，检查实际顶点，避免把菱形误当背景。
    corners = np.array([(seg.end.x, seg.end.y) for seg in SVGPath(element) if seg.end is not None])
    on_x = np.isclose(corners[:, 0], 0, atol=tolerance) | np.isclose(corners[:, 0], svg.width, atol=tolerance)
    on_y = np.isclose(corners[:, 1], 0, atol=tolerance) | np.isclose(corners[:, 1], svg.height, atol=tolerance)
    return bool(np.all(on_x & on_y))


def extract_svg(file: Path, samples: int, close_open: bool) -> list[np.ndarray]:
    """第二步（矢量）：解析变换后的路径，将曲线高密度离散化。"""
    svg = SVG.parse(str(file), reify=True)
    records, open_count = [], 0
    # 支持 viewBox、组变换、基本形状；SVG 文字需先转换为路径。
    for order, element in enumerate(svg.elements()):
        if not isinstance(element, Shape) or element.values.get("visibility") == "hidden":
            continue
        background = svg_background_rect(element, svg)
        for subpath in SVGPath(element).as_subpaths():
            segments = [seg for seg in subpath if not isinstance(seg, Move)]
            if not segments:
                continue
            start, end = segments[0].start, segments[-1].end
            closed = abs(complex(end.x - start.x, end.y - start.y)) < 1e-7
            if not closed:
                open_count += 1
                if not close_open:
                    continue
            lengths = np.array([seg.length(error=1e-5) for seg in segments])
            total = float(lengths.sum())
            if total <= 1e-9:
                continue
            points = []
            for seg, length in zip(segments, lengths):
                steps = max(8, math.ceil(samples * 8 * length / total))
                for u in np.linspace(0, 1, steps, endpoint=False):
                    p = seg.point(float(u))
                    points.append((p.x, p.y))
            points.append((end.x, end.y))
            if len(points) >= 3:
                records.append((np.asarray(points, dtype=np.float64), background, order))
    # 仅忽略位于有效目标下方的画布背景；只有一个矩形的 SVG 仍可正常导入。
    first_foreground = min((order for _, background, order in records if not background), default=-1)
    candidates = [points for points, background, order in records
                  if not (background and order < first_foreground)]
    ignored = len(records) - len(candidates)
    if ignored:
        print(f"SVG 已忽略 {ignored} 个覆盖画布的背景矩形。")
    if open_count:
        action = "以直线闭合" if close_open else "跳过（可用 --close-open 补直线闭合）"
        print(f"SVG 有 {open_count} 条开放路径，已{action}。")
    if not candidates:
        raise ValueError("SVG 无可用闭合路径，请将文字转为路径或使用 --close-open。")
    return candidates


def extract_raster(file: Path, threshold: int | None, polarity: str,
                   edges: bool) -> list[np.ndarray]:
    """第二步（位图）：灰度、二值化、可选 Canny 检测、提取闭合轮廓。"""
    # imdecode + fromfile 支持中文文件路径。
    image = cv2.imdecode(np.fromfile(str(file), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法解码图片：{file}")
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        # 透明 PNG 合成到白色背景。
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        image = (image[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    dark = polarity == "dark" or (polarity == "auto" and np.median(border) >= 127)
    flag = cv2.THRESH_BINARY_INV if dark else cv2.THRESH_BINARY
    if threshold is None:
        _, binary = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, threshold, 255, flag)
    if edges:
        binary = cv2.Canny(binary, 60, 180)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = [c[:, 0, :].astype(np.float64) for c in contours
                  if len(c) >= 8 and cv2.arcLength(c, True) >= 12]
    if not candidates:
        raise ValueError("未检测到轮廓，请检查图形或调整 --threshold / --polarity。")
    return candidates


def load_contour(file: Path, samples: int, contour_index: int = 0,
                 close_open: bool = False, threshold: int | None = None,
                 polarity: str = "auto", edges: bool = False) -> np.ndarray:
    """选取一个连续闭合轮廓，多个不相连轮廓不强行拼接。"""
    if not file.is_file():
        raise ValueError(f"输入文件不存在：{file}")
    if file.suffix.lower() == ".svg":
        candidates = extract_svg(file, samples, close_open)
    elif file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        candidates = extract_raster(file, threshold, polarity, edges)
    else:
        raise ValueError("支持 SVG、PNG、JPG、BMP、WebP。")
    candidates.sort(key=perimeter, reverse=True)
    if not 0 <= contour_index < len(candidates):
        raise ValueError(f"轮廓编号超出范围，可选 0～{len(candidates) - 1}。")
    print(f"检测到 {len(candidates)} 个轮廓，按周长降序选择第 {contour_index} 个。")
    points = resample_closed(candidates[contour_index], samples)
    # 图像 y 向下，数学 y 向上；移至原点并保持纵横比。
    z = points[:, 0] - 1j * points[:, 1]
    z -= z.mean()
    extent = max(float(np.ptp(z.real)), float(np.ptp(z.imag)))
    if extent < 1e-9:
        raise ValueError("轮廓尺寸为零。")
    return z * (2 / extent)


DEFAULT_FONTS = [
    Path("C:/Windows/Fonts/STXINGKA.TTF"),
    Path("C:/Windows/Fonts/simkai.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
    Path("C:/Windows/Fonts/msyh.ttc"),
]


def load_stroke_trajectory(text: str, samples: int, close_loop: bool = True,
                           speed_ratio: float = 45.0, style: str = "xuejun") -> np.ndarray:
    """旧版笔顺轨迹及几何变形，保留命令行兼容；不是商业字体。"""
    import urllib.request, urllib.parse, json

    cache_dir = ROOT / "assets" / "strokes"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def get_char_medians(ch: str):
        c_file = cache_dir / f"{ch}.json"
        if c_file.is_file():
            return json.loads(c_file.read_text(encoding="utf-8"))["medians"]
        url = f"https://cdn.jsdelivr.net/npm/hanzi-writer-data@2.0/{urllib.parse.quote(ch)}.json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            content = resp.read().decode("utf-8")
            c_file.write_text(content, encoding="utf-8")
            return json.loads(content)["medians"]

    def catmull_rom_smooth(pts: np.ndarray, density: float = 3.5) -> np.ndarray:
        if len(pts) < 2:
            return np.array(pts, dtype=np.float64)
        if len(pts) == 2:
            n = max(4, int(round(np.linalg.norm(pts[1] - pts[0]) / 18.0 * density)))
            return np.linspace(pts[0], pts[1], max(4, n))
        p0 = 2 * pts[0] - pts[1]
        pn = 2 * pts[-1] - pts[-2]
        extended = np.vstack([[p0], pts, [pn]])
        res = []
        for i in range(len(extended) - 3):
            p_0, p_1, p_2, p_3 = extended[i:i+4]
            seg_dist = float(np.linalg.norm(p_2 - p_1))
            n = max(3, int(round(seg_dist / 16.0 * density)))
            for t in np.linspace(0, 1, n, endpoint=False):
                t2 = t * t
                t3 = t2 * t
                v = 0.5 * ((2*p_1) + (-p_0 + p_2)*t + (2*p_0 - 5*p_1 + 4*p_2 - p_3)*t2 + (-p_0 + 3*p_1 - 3*p_2 + p_3)*t3)
                res.append(v)
        res.append(pts[-1])
        return np.array(res, dtype=np.float64)

    def get_tangent(pts: np.ndarray, at_start: bool = True) -> np.ndarray:
        if len(pts) < 2:
            return np.array([1.0, 0.0])
        idx = min(3, len(pts)-1) if at_start else max(0, len(pts)-4)
        vec = (pts[idx] - pts[0]) if at_start else (pts[-1] - pts[idx])
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-6 else np.array([1.0, 0.0])

    def make_calligraphic_arc(p_tail: np.ndarray, d_out: np.ndarray,
                              q_head: np.ndarray, d_in: np.ndarray, num_pts: int = 24) -> np.ndarray:
        dist = float(np.linalg.norm(q_head - p_tail))
        scale = min(dist * 0.38, 140.0)
        c1 = p_tail + d_out * scale
        c2 = q_head - d_in * scale
        lift = -min(dist * 0.15, 60.0)
        c1[1] += lift
        c2[1] += lift
        t = np.linspace(0, 1, num_pts)[:, None]
        return (1 - t)**3 * p_tail + 3 * (1 - t)**2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * q_head

    chars = [c for c in text.strip() if c.strip()]
    if not chars:
        chars = ["刘", "鸣", "昊"]

    char_width = 1000
    char_spacing = 130
    all_char_strokes = []

    for idx, ch in enumerate(chars):
        medians = get_char_medians(ch)
        x_offset = idx * (char_width + char_spacing)
        char_strokes = []
        cx = x_offset + char_width / 2.0
        cy = 480.0
        for stroke in medians:
            raw_pts = np.array([[pt[0] + x_offset, 960 - pt[1]] for pt in stroke], dtype=np.float64)

            if style == "xuejun":
                # 旧版几何变形：
                # 1. 瘦劲挺拔、纵向修长（高重心，硬笔行书现代清秀感）
                raw_pts[:, 0] = (raw_pts[:, 0] - cx) * 0.86 + cx
                raw_pts[:, 1] = (raw_pts[:, 1] - cy) * 1.15 + cy

                # 2. 标志性大仰角（横画向右上显著昂起约 9.5°）
                raw_pts[:, 1] -= (raw_pts[:, 0] - cx) * 0.165

                # 3. 整字右倾前冲动势（约 5.5°）
                raw_pts[:, 0] += (cy - raw_pts[:, 1]) * 0.09

                # 4. 撇捺开张大舒展（长撇大捺长甩，破体取势）
                is_left_sweep = (raw_pts[-1, 0] < raw_pts[0, 0] - 50) and (raw_pts[-1, 1] > raw_pts[0, 1] + 70)
                if is_left_sweep:
                    raw_pts[:, 0] -= np.linspace(0, 45, len(raw_pts))
                    raw_pts[:, 1] += np.linspace(0, 25, len(raw_pts))
            else:
                # 传统行楷：温和仰角
                raw_pts[:, 0] += (cy - raw_pts[:, 1]) * 0.05
                raw_pts[:, 1] -= (raw_pts[:, 0] - cx) * 0.04

            # 5. 样条平滑：圆折代替方折，体现硬笔圆润韧劲
            smoothed = catmull_rom_smooth(raw_pts, density=3.5)
            char_strokes.append(smoothed)
        all_char_strokes.append(char_strokes)

    segments = []
    for c_idx, char_strokes in enumerate(all_char_strokes):
        for s_idx, stroke_pts in enumerate(char_strokes):
            segments.append((stroke_pts, "stroke"))  # 实笔手写
            if s_idx < len(char_strokes) - 1:
                d_out = get_tangent(stroke_pts, at_start=False)
                next_stroke = char_strokes[s_idx + 1]
                d_in = get_tangent(next_stroke, at_start=True)
                arc = make_calligraphic_arc(stroke_pts[-1], d_out, next_stroke[0], d_in, num_pts=22)
                segments.append((arc, "stroke_air"))

        if c_idx < len(all_char_strokes) - 1:
            last_s = char_strokes[-1]
            next_first_s = all_char_strokes[c_idx + 1][0]
            d_out = get_tangent(last_s, at_start=False)
            d_in = get_tangent(next_first_s, at_start=True)
            arc = make_calligraphic_arc(last_s[-1], d_out, next_first_s[0], d_in, num_pts=36)
            segments.append((arc, "char_air"))  # 字间空中跃迁

    if close_loop:
        p_tail = all_char_strokes[-1][-1][-1]
        p_head = all_char_strokes[0][0][0]
        total_w = len(chars) * (char_width + char_spacing) - char_spacing
        bottom_y = 1080.0
        ctrl_pts = np.array([
            p_tail,
            [p_tail[0] + 130, p_tail[1] + 70],
            [total_w + 30, bottom_y + 10],
            [total_w * 0.55, bottom_y - 25],
            [total_w * 0.15, bottom_y + 20],
            [-70.0, bottom_y - 80],
            [-100.0, (p_head[1] + bottom_y) * 0.4],
            [p_head[0] - 50, p_head[1] - 40],
            p_head
        ], dtype=np.float64)
        bottom_flourish = catmull_rom_smooth(ctrl_pts, density=3.5)
        segments.append((bottom_flourish, "flourish"))

    # 笔速调控：字间超高速非匀速弹射，实笔有起笔/折笔顿笔、行笔疾驰
    seg_times = []
    total_time = 0.0
    for pts, seg_type in segments:
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        L = float(dists.sum())
        if seg_type == "char_air":
            air_speed = speed_ratio if style == "xuejun" else 20.0
            dt = max(L / air_speed, 3.0)  # 字间超高速飞渡（时间占比极小，近乎瞬间掠过）
        elif seg_type == "stroke_air":
            dt = max(L / 22.0, 4.0)  # 笔画间轻快飞渡
        elif seg_type == "flourish":
            dt = max(L / 4.0, 25.0)  # 艺术签名底托
        else:
            dt = max(L * 1.05, 20.0)  # 实笔手写
        seg_times.append(dt)
        total_time += dt

    resampled = []
    for (pts, seg_type), dt in zip(segments, seg_times):
        n_pts = max(2, int(round(samples * (dt / total_time))))
        dists = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        total_L = dists[-1]
        if total_L < 1e-6:
            interp_pts = np.tile(pts[0], (n_pts, 1))
        else:
            u = np.linspace(0, 1, n_pts, endpoint=False)
            if seg_type == "char_air":
                # 鞭梢弹射动力学：中间段以近 9 倍速超高速掠过（非匀速）
                s_norm = (1 - 0.88) * u + 0.88 * (6 * u**5 - 15 * u**4 + 10 * u**3)
            elif seg_type == "stroke_air":
                # 笔画间平滑弹射
                s_norm = (1 - 0.75) * u + 0.75 * (3 * u**2 - 2 * u**3)
            elif seg_type == "stroke":
                # 手写提按律动：起笔微顿、行笔疾驰、收笔留白
                s_norm = u - 0.0637 * np.sin(2 * np.pi * u)
            else:
                s_norm = (1 - 0.6) * u + 0.6 * (3 * u**2 - 2 * u**3)
            targets = np.clip(s_norm * total_L, 0.0, total_L)
            interp_x = np.interp(targets, dists, pts[:, 0])
            interp_y = np.interp(targets, dists, pts[:, 1])
            interp_pts = np.column_stack([interp_x, interp_y])
        resampled.append(interp_pts)

    continuous_pts = np.vstack(resampled)
    continuous_pts = resample_closed(continuous_pts, samples)
    z = continuous_pts[:, 0] - 1j * continuous_pts[:, 1]
    z -= z.mean()
    extent = max(float(np.ptp(z.real)), float(np.ptp(z.imag)))
    if extent < 1e-9:
        raise ValueError("笔画轨迹尺寸为零。")
    style_label = "旧版变形笔顺" if style == "xuejun" else "旧版单线笔顺"
    print(f"文字 '{''.join(chars)}' {style_label}体态笔顺轨迹加载成功：共 {len(segments)} 段笔画与飞渡，重采样至 {samples} 点。")
    return z * (2 / extent)


def load_text_contour(text: str, samples: int, font_path: str | Path | None = None,
                      close_loop: bool = True, text_style: str = "ordered") -> np.ndarray:
    """从中文文字生成一条连续的行书签名、笔顺或字体轮廓轨迹。"""
    if text_style in ("ordered", "signature"):
        from signature import build_ordered_signature, build_signature

        candidates = ([Path(font_path)] if font_path else []) + DEFAULT_FONTS
        if text_style=="signature" and font_path and not Path(font_path).is_file():
            raise ValueError("所选字体文件不存在，请重新选择字体。")
        chosen_font = next((candidate for candidate in candidates if candidate.is_file()), None)
        if text_style=="signature" and chosen_font is None:
            raise ValueError("未找到可用中文行楷字体，请在参数中指定本地字体文件。")
        path = (build_ordered_signature(text, close_loop) if text_style=="ordered"
                else build_signature(text, chosen_font, close_loop))
        points = resample_closed(path.points, samples)
        z = points[:, 0] - 1j * points[:, 1]
        z -= z.mean()
        extent = max(float(np.ptp(z.real)), float(np.ptp(z.imag)))
        if extent < 1e-9:
            raise ValueError("签名轨迹尺寸为零。")
        label = "标准笔顺单线字形" if text_style=="ordered" else f"{chosen_font.name} 实验映射"
        print(f"文字 '{text}' | {label} | 采样 {samples} 点。")
        return z * (2 / extent)
    if text_style in ("xuejun", "stroke"):
        try:
            return load_stroke_trajectory(text, samples, close_loop=close_loop, style=text_style)
        except Exception as e:
            print(f"单线笔画获取异常（{e}），回退到字体轮廓模式。")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as err:
        raise SystemExit("文字功能需要 Pillow 库，请执行：python -m pip install pillow") from err

    text = text.strip() or "傅里叶"
    font_file = None
    if font_path and Path(font_path).is_file():
        font_file = str(font_path)
    else:
        for candidate in DEFAULT_FONTS:
            if candidate.is_file():
                font_file = str(candidate)
                break
    if not font_file:
        raise ValueError("未找到可用中文字体文件，请在参数中指定字体路径。")

    font_size = 260
    font = ImageFont.truetype(font_file, font_size)
    dummy = Image.new("L", (1, 1), 0)
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    tw, th = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])

    pad_x, pad_y = 120, 90
    extra_bottom = 120 if close_loop else 20
    canvas_w = tw + pad_x * 2
    canvas_h = th + pad_y * 2 + extra_bottom

    im = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(im)
    start_x = pad_x - bbox[0]
    start_y = pad_y - bbox[1]
    draw.text((start_x, start_y), text, font=font, fill=255)

    arr = np.array(im)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(arr)
    comp_ids = list(range(1, num_labels))
    if not comp_ids:
        raise ValueError(f"文字 '{text}' 渲染为空。")
    comp_ids.sort(key=lambda idx: centroids[idx][0])

    # 1. 笔画部件间前向牵丝桥接 (从左至右)
    for i in range(len(comp_ids) - 1):
        c1, c2 = comp_ids[i], comp_ids[i + 1]
        m1 = (labels == c1).astype(np.uint8)
        m2 = (labels == c2).astype(np.uint8)
        b1 = cv2.morphologyEx(m1, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        b2 = cv2.morphologyEx(m2, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

        pts1 = np.argwhere(b1 > 0)
        pts2 = np.argwhere(b2 > 0)
        if len(pts1) > 200:
            pts1 = pts1[np.random.choice(len(pts1), 200, replace=False)]
        if len(pts2) > 200:
            pts2 = pts2[np.random.choice(len(pts2), 200, replace=False)]

        dists = np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=2)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        p1, p2 = pts1[min_idx[0]], pts2[min_idx[1]]
        pt1 = (int(p1[1]), int(p1[0]))
        pt2 = (int(p2[1]), int(p2[0]))
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2 + 6)

        curve_pts = []
        for t in np.linspace(0, 1, 20):
            x = (1 - t)**2 * pt1[0] + 2 * (1 - t) * t * mid[0] + t**2 * pt2[0]
            y = (1 - t)**2 * pt1[1] + 2 * (1 - t) * t * mid[1] + t**2 * pt2[1]
            curve_pts.append((int(x), int(y)))
        for k in range(len(curve_pts) - 1):
            draw.line([curve_pts[k], curve_pts[k + 1]], fill=255, width=7)

    # 2. 首尾相连闭合底托线 (末字尾部 -> 底部长弧 -> 首字起笔)
    if close_loop and len(comp_ids) >= 1:
        last_pts = np.argwhere(labels == comp_ids[-1])
        tail_idx = np.argmax(last_pts[:, 1] * 0.8 + last_pts[:, 0] * 1.0)
        p_tail = (int(last_pts[tail_idx, 1]), int(last_pts[tail_idx, 0]))

        first_pts = np.argwhere(labels == comp_ids[0])
        head_idx = np.argmin(first_pts[:, 1] * 1.0 + first_pts[:, 0] * 0.8)
        p_head = (int(first_pts[head_idx, 1]), int(first_pts[head_idx, 0]))

        c1 = (p_tail[0] + 35, p_tail[1] + 60)
        c2 = (canvas_w - pad_x + 20, canvas_h - 45)
        c3 = (canvas_w // 2, canvas_h - 35)
        c4 = (pad_x - 30, canvas_h - 45)
        c5 = (pad_x // 2 - 10, (p_head[1] + canvas_h) // 2)

        flourish = []
        for t in np.linspace(0, 1, 60):
            x = (1 - t)**3 * p_tail[0] + 3 * (1 - t)**2 * t * c1[0] + 3 * (1 - t) * t**2 * c2[0] + t**3 * c3[0]
            y = (1 - t)**3 * p_tail[1] + 3 * (1 - t)**2 * t * c1[1] + 3 * (1 - t) * t**2 * c2[1] + t**3 * c3[1]
            flourish.append((int(x), int(y)))
        for t in np.linspace(0, 1, 60):
            x = (1 - t)**3 * c3[0] + 3 * (1 - t)**2 * t * c4[0] + 3 * (1 - t) * t**2 * c5[0] + t**3 * p_head[0]
            y = (1 - t)**3 * c3[1] + 3 * (1 - t)**2 * t * c4[1] + 3 * (1 - t) * t**2 * c5[1] + t**3 * p_head[1]
            flourish.append((int(x), int(y)))

        for k in range(len(flourish) - 1):
            prog = k / len(flourish)
            w_k = int(6 + 6 * np.sin(prog * np.pi))
            draw.line([flourish[k], flourish[k + 1]], fill=255, width=w_k)

    # 3. 提取轮廓 (外边界 + 内部孔洞) 并进行欧拉双向缝合
    arr2 = np.array(im)
    blurred = cv2.GaussianBlur(arr2, (3, 3), 0.8)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    outers = []
    holes = []
    for idx, (c, h) in enumerate(zip(contours, hier[0])):
        if h[3] == -1:
            outers.append((idx, c))
        else:
            holes.append((idx, c, h[3]))

    if not outers:
        raise ValueError("未能提取到有效轮廓。")

    outers.sort(key=lambda item: cv2.arcLength(item[1], True), reverse=True)
    main_outer = outers[0][1]
    main_outer_idx = outers[0][0]
    my_holes = [c for (idx, c, parent) in holes if parent == main_outer_idx]

    main_pts = cv2.approxPolyDP(main_outer, 0.6, True)[:, 0, :]
    simplified_holes = []
    for h in my_holes:
        pts = cv2.approxPolyDP(h, 0.6, True)[:, 0, :]
        if len(pts) >= 3:
            simplified_holes.append(pts)

    # 将所有内部孔洞通过微缝缝合进入主闭合回路
    stitched_loop = list(main_pts)
    for hole_pts in simplified_holes:
        loop_arr = np.array(stitched_loop)
        dists = np.linalg.norm(loop_arr[:, None, :] - hole_pts[None, :, :], axis=2)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        idx_loop, idx_hole = min_idx[0], min_idx[1]
        hole_rotated = list(np.roll(hole_pts, -idx_hole, axis=0))
        splice = hole_rotated + [hole_rotated[0], loop_arr[idx_loop]]
        stitched_loop = stitched_loop[:idx_loop + 1] + splice + stitched_loop[idx_loop + 1:]

    stitched_arr = np.array(stitched_loop)
    points = resample_closed(stitched_arr, samples)
    z = points[:, 0] - 1j * points[:, 1]
    z -= z.mean()
    extent = max(float(np.ptp(z.real)), float(np.ptp(z.imag)))
    if extent < 1e-9:
        raise ValueError("轮廓尺寸为零。")
    loop_desc = "已添加首尾相连底托线" if close_loop else "未加首尾底线"
    print(f"文字 '{text}' 载入成功：融合 {len(simplified_holes)} 个内孔及全笔画（{loop_desc}），等弧长重采样至 {samples} 点。")
    return z * (2 / extent)


@dataclass(frozen=True)
class Epicycle:
    """第三步：圆盘参数（半径，整数频率，初相位）。"""
    radius: float
    frequency: int
    phase: float


class FourierModel:
    def __init__(self, z: np.ndarray, count: int = 200, sort: str = "radius"):
        if not 1 <= count <= len(z):
            raise ValueError("圆盘数必须在 1～采样点数之间。")
        # 正向 FFT 默认未缩放，除以 N 才是旋转向量的幅度。
        coefficients = np.fft.fft(z) / len(z)
        frequencies = np.rint(np.fft.fftfreq(len(z), d=1 / len(z))).astype(int)
        if sort == "radius":
            indices = np.argsort(-np.abs(coefficients), kind="stable")[:count]
        else:
            indices = np.lexsort((frequencies < 0, np.abs(frequencies)))[:count]
        self.coefficients, self.frequencies = coefficients[indices], frequencies[indices]
        self.radii, self.phases = np.abs(self.coefficients), np.angle(self.coefficients)
        self.epicycles = [Epicycle(float(r), int(f), float(p))
                          for r, f, p in zip(self.radii, self.frequencies, self.phases)]

    def chain(self, t: float) -> np.ndarray:
        """第四步：累加旋转向量，前一个末端成为后一个圆心。"""
        vectors = self.coefficients * np.exp(2j * np.pi * self.frequencies * (t % 1))
        return np.r_[0j, np.cumsum(vectors)]

    def curve(self, times: np.ndarray) -> np.ndarray:
        # 分块计算，避免巨大的复数矩阵占用内存。
        times = np.asarray(times, dtype=np.float64)
        result = np.empty(len(times), dtype=np.complex128)
        for start in range(0, len(times), 512):
            t = times[start:start + 512]
            result[start:start + len(t)] = np.exp(
                2j * np.pi * t[:, None] * self.frequencies
            ) @ self.coefficients
        return result

    def bounds(self) -> tuple[float, float, float, float]:
        """覆盖整个周期的圆盘范围，保持固定镜头且防止裁切。"""
        n = max(1024, 4 * int(np.abs(self.frequencies).max()) + 4)
        low_x = low_y = math.inf
        high_x = high_y = -math.inf
        # 用速度上界补偿时间采样网格间的范围误差。
        speed = np.r_[0.0, np.cumsum(2 * np.pi * self.radii * np.abs(self.frequencies))]
        pad = self.radii + speed[:-1] / (2 * n)
        for start in range(0, n, 256):
            t = np.arange(start, min(start + 256, n)) / n
            vectors = self.coefficients * np.exp(2j * np.pi * t[:, None] * self.frequencies)
            centers = np.column_stack([np.zeros(len(t)), np.cumsum(vectors, axis=1)])[:, :-1]
            low_x = min(low_x, float(np.min(centers.real - pad)))
            high_x = max(high_x, float(np.max(centers.real + pad)))
            low_y = min(low_y, float(np.min(centers.imag - pad)))
            high_y = max(high_y, float(np.max(centers.imag + pad)))
        return low_x, high_x, low_y, high_y


class Renderer:
    """第五步：白色轨迹、柔和暖色圆盘、半径箭头和钟面刻度。"""
    def __init__(self, model: FourierModel, width: int, height: int,
                 trail_mode: str = "loop", trace_samples: int = 6000,
                 ring_width: float = 2.4, ring_opacity: float = 0.90,
                 ticks: int = 12, fading=None, render_fps: float = 60):
        self.model, self.width, self.height, self.trail_mode = model, width, height, trail_mode
        self.ring_opacity, self.ticks = ring_opacity, ticks
        self.fading = fading
        self.render_fps = render_fps
        self.ring_cache = {}
        x0, x1, y0, y1 = model.bounds()
        self.center = complex((x0 + x1) / 2, (y0 + y1) / 2)
        self.scale = min(width * 0.84 / max(x1 - x0, 1e-9), height * 0.84 / max(y1 - y0, 1e-9))
        self.line_width = max(1, round(height / 360))
        self.ring_width = max(1, round(ring_width * height / 1080))
        self.vector_width = max(1, round(height / 540))
        self.trace_count = max(trace_samples, 8 * int(np.abs(model.frequencies).max()) + 8)
        self.trace = (self.pixels(model.curve(np.linspace(0, 1, self.trace_count + 1)))
                      if fading is None else np.empty((0, 2)))
        self.trace_fixed = np.rint(self.trace * 16).astype(np.int32)
        times = np.linspace(0, 1, self.trace_count+1)
        visible = (model.visible_between(times[:-1], times[1:])
                   if hasattr(model, "visible_between") else np.ones(self.trace_count, bool))
        changes = np.diff(np.r_[False, visible, False].astype(int))
        self.trace_runs = list(zip(np.flatnonzero(changes==1), np.flatnonzero(changes==-1)))
        # 目标轮廓使用纯白色；整条折线一次绘制，避免逐段出现亮度不均。
        self.trace_color = (255, 255, 255)
        self.background = np.zeros((height, width, 3), dtype=np.uint8)
        self.static = self.make_trace(self.trace_count) if fading is None else self.background.copy()
        # OpenCV 是 BGR 顺序：珊瑚红、鼠尾草绿、柔和紫、暖金色。
        palette = [(102, 124, 225), (149, 184, 145),
                   (166, 126, 192), (121, 193, 220)]
        self.circle_colors = [palette[i % len(palette)] for i in range(len(model.radii))]
        self.pointer_colors = [tuple(round(0.25 * channel + 0.75 * 255) for channel in color)
                               for color in self.circle_colors]
        # 末端光晕也使用中性白色，不给白色曲线染色。
        self.halo_radius = max(8, round(height * 0.014))
        r = self.halo_radius
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        strength = np.exp(-(xx * xx + yy * yy) / (2 * (r / 3) ** 2))
        peak = 128 if fading is not None else 48
        self.halo = np.repeat((strength * peak).astype(np.uint8)[:, :, None], 3, axis=2)

    def pixels(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values) - self.center
        return np.column_stack([self.width / 2 + values.real * self.scale,
                                self.height / 2 - values.imag * self.scale])

    @staticmethod
    def fixed_point(point: np.ndarray) -> tuple[int, int]:
        # 亚像素坐标，shift=4 对应 1/16 像素精度。
        return tuple(int(v) for v in np.rint(np.asarray(point) * 16))

    def draw_trace(self, image: np.ndarray, count: int):
        if count > 0:
            for start, end in self.trace_runs:
                if start >= count:
                    break
                cv2.polylines(image, [self.trace_fixed[start:min(end,count)+1]], False,
                              self.trace_color, self.line_width, cv2.LINE_AA, shift=4)

    def make_trace(self, count: int) -> np.ndarray:
        trace = np.zeros_like(self.background)
        self.draw_trace(trace, count)
        # 收敛为柔和的白色光晕，纯白线条核心仍保持清楚。
        glow = cv2.GaussianBlur(trace, (0, 0), max(1.0, self.height / 540))
        outer = cv2.GaussianBlur(trace, (0, 0), max(2.0, self.height / 180))
        base = cv2.addWeighted(self.background, 1, outer, 0.18, 0)
        base = cv2.addWeighted(base, 1, glow, 0.35, 0)
        return cv2.add(base, trace)

    @staticmethod
    def smoothstep(value):
        value = np.clip(value, 0, 1)
        return value*value*(3-2*value)

    def draw_dial(self, rings: np.ndarray, center: np.ndarray, radius: float, index: int,
                  angle: float = 0., angular_speed: float = 0.):
        visibility = float(self.smoothstep((radius-.45)/2.5))
        if visibility <= 1e-6:
            return
        color = np.asarray(self.circle_colors[index], dtype=float)
        thickness = 1 + (self.ring_width-1)*float(self.smoothstep((radius-6)/10))
        key = (round(radius*2), round(thickness*4))
        if key not in self.ring_cache:
            r = key[0]/2
            extent = math.ceil(r+thickness/2+3)
            y, x = np.mgrid[-extent:extent+1, -extent:extent+1]
            distance = np.hypot(x, y)
            band = np.abs(distance-r) < thickness/2+2.6
            if len(self.ring_cache) > 512:
                self.ring_cache.clear()
            self.ring_cache[key] = (x[band], y[band])
        dx, dy = self.ring_cache[key]
        origin = np.floor(center).astype(int)
        fraction = center-origin
        distance = np.hypot(dx-fraction[0], dy-fraction[1])
        radial = np.clip(thickness/2+.5-np.abs(distance-radius), 0, 1)
        azimuth = np.arctan2(dy-fraction[1], dx-fraction[0])
        # High-speed markings blur toward a uniform ring rather than aliasing
        # into intermittent/reversed teeth. The ring itself stays visible.
        shutter = 1 / (1+(angular_speed/max(1., self.render_fps*1.2))**2)
        light = .64 + .34*shutter*np.cos(azimuth-angle-.8)
        shade = light[:, None]*color
        xx, yy = origin[0]+dx, origin[1]+dy
        valid = (xx>=0)&(xx<self.width)&(yy>=0)&(yy<self.height)
        alpha = (radial*visibility)[valid, None]
        target = rings[yy[valid], xx[valid]].astype(float)
        rings[yy[valid], xx[valid]] = np.clip(target*(1-alpha)+shade[valid]*alpha, 0, 255).astype(np.uint8)
        tick_count = self.ticks
        tooth_shutter = 1 / (1+(angular_speed*max(1, tick_count)
                                / max(1., np.pi*self.render_fps*.8))**2)
        tick_visibility = visibility*tooth_shutter*float(self.smoothstep((radius-10)/12))
        if tick_visibility <= .002 or tick_count == 0:
            return
        tick_color = tuple(round(v*tick_visibility) for v in color*.7+255*.3)
        for tick in range(tick_count):
            tick_angle = angle + 2 * np.pi * tick / tick_count
            direction = np.array([np.cos(tick_angle), np.sin(tick_angle)])
            major = tick % max(1, tick_count // 4) == 0
            length = min(radius * (0.075 if major else 0.045),
                         self.height * (0.013 if major else 0.008))
            length = max(2.0, length)
            outer = center + direction * (radius - thickness / 2)
            inner = center + direction * (radius - thickness / 2 - length)
            cv2.line(rings, self.fixed_point(inner), self.fixed_point(outer),
                     tick_color, max(1, thickness - 1), cv2.LINE_AA, shift=4)

    def draw_pointer(self, image: np.ndarray, start: np.ndarray, end: np.ndarray, index: int,
                     opacity: float = 1.):
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 0.35:
            return
        visibility = float(self.smoothstep((length-.45)/2.5))*opacity
        if visibility <= .002:
            return
        color = tuple(round(v*visibility) for v in self.pointer_colors[index])
        if self.fading is not None:
            # 漂移模式下指针与圆心只是辅助线，不能抢占末端光点的亮度。
            color = tuple(round(channel * .58) for channel in color)
        # 半径对应傅里叶向量，箭头从圆心指向下一圆心。
        cv2.line(image, self.fixed_point(start), self.fixed_point(end), color,
                 self.vector_width if length >= 12 else 1, cv2.LINE_AA, shift=4)
        if length >= 6:
            direction = vector / length
            normal = np.array([-direction[1], direction[0]])
            head_length = min(length * 0.24, max(5.0, self.height * 0.014))
            half_width = head_length * 0.42
            base = end - direction * head_length
            triangle = np.rint(np.array([end, base + normal * half_width,
                                        base - normal * half_width]) * 16).astype(np.int32)
            cv2.fillConvexPoly(image, triangle, color, cv2.LINE_AA, shift=4)
        if length >= 12:
            cv2.circle(image, self.fixed_point(start), max(16, round(self.height / 480 * 16)),
                       color, -1, cv2.LINE_AA, shift=4)

    def frame(self, t: float) -> np.ndarray:
        phase = t % 1
        count = self.trace_count if self.trail_mode == "loop" else int(phase * self.trace_count)
        # loop 预填充完整历史；draw 周期末清空，原有循环规则保持一致。
        if self.fading is not None:
            image = self.background.copy()
            radii = self.model.parameters(phase)[0]
        else:
            image = self.static.copy() if self.trail_mode == "loop" else self.make_trace(count)
            radii = self.model.radii
        chain = self.pixels(self.model.chain(t))
        angles = np.arctan2(np.diff(chain[:, 1]), np.diff(chain[:, 0]))
        speeds = (np.abs(self.model.parameters(phase)[2])*2*np.pi
                  if hasattr(self.model, "parameters") else np.abs(self.model.frequencies)*2*np.pi/60)
        rings, fills = np.zeros_like(image), np.zeros_like(image)
        for i, radius in enumerate(radii):
            radius_pixels = float(radius * self.scale)
            self.draw_dial(rings, chain[i], radius_pixels, i, angles[i], speeds[i])
            # 大圆盘带极淡色底，接近参考图；小圆不填充，避免细节堆成色块。
            if radius_pixels >= 18:
                cv2.circle(fills, self.fixed_point(chain[i]), max(1, round(radius_pixels * 16)),
                           self.circle_colors[i], -1, cv2.LINE_AA, shift=4)
        image = cv2.addWeighted(image, 1, fills, 0.055, 0)
        image = cv2.addWeighted(image, 1, rings, self.ring_opacity, 0)
        for i in range(len(self.model.radii)):
            # Tiny fast vectors are mechanical detail, not additional pen tips.
            softness = .45+.55/(1+(speeds[i]/max(1., self.render_fps*1.2))**2)
            self.draw_pointer(image, chain[i], chain[i + 1], i, softness)
        # 新模式只显示按年龄渐隐的动态尾迹，原 loop/draw 模式仍使用纯白曲线。
        if self.fading is not None:
            # 避免旧轨迹与圆环/指针相交时叠加出新的亮斑。
            image = cv2.max(image, self.fading.layer(self, phase))
        else:
            self.draw_trace(image, count)
        x, y = (int(v) for v in np.rint(chain[-1]))
        r = self.halo_radius
        x0, x1 = max(0, x - r), min(self.width, x + r + 1)
        y0, y1 = max(0, y - r), min(self.height, y + r + 1)
        if x1 > x0 and y1 > y0:
            halo = self.halo[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)]
            strength = self.fading.glow if self.fading is not None else 1.0
            image[y0:y1, x0:x1] = cv2.addWeighted(image[y0:y1, x0:x1], 1, halo, strength, 0)
        # 只有当前真实末端是纯白高亮点，尾迹中不额外绘制任何光点。
        head_radius = (max(3, round(self.height / 300)) if self.fading is not None
                       else max(2, round(self.height / 360)))
        cv2.circle(image, self.fixed_point(chain[-1]), head_radius * 16,
                   (255, 255, 255), -1, cv2.LINE_AA, shift=4)
        return image



class ExportWatchdog:
    """编码管道写入/收尾卡住时终止子进程，取消操作也能打断阻塞写入。"""
    def __init__(self, process, cancel_event, timeout=120):
        self.process, self.cancel_event, self.timeout = process, cancel_event, timeout
        self.last_activity = time.monotonic()
        self.stage, self.reason = "生成帧", None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def beat(self, stage):
        self.stage = stage
        self.last_activity = time.monotonic()
        self.check()

    def check(self):
        if self.reason:
            raise RuntimeError(self.reason)
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RuntimeError("导出已取消。")

    def _run(self):
        while not self.stop_event.wait(.05):
            if self.cancel_event is not None and self.cancel_event.is_set():
                self.reason = "导出已取消。"
            elif self.timeout and time.monotonic() - self.last_activity > self.timeout:
                self.reason = f"导出无进展：{self.stage}超过 {self.timeout:g} 秒。已终止编码并清理临时文件。"
            if self.reason:
                try:
                    if self.process.poll() is None:
                        self.process.kill()
                except OSError:
                    pass
                return

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1)


def encode_video(renderer: Renderer, output: Path, fps: int, duration: float,
                 crf: int = 18, preset: str = "medium", encoder: str = "libx264",
                 ffmpeg: str | None = None, overwrite: bool = False,
                 progress=None, cancel_event=None, stall_timeout=120) -> int:
    """第六步：逐帧写入 FFmpeg 管道，无需保存整个帧序列。"""
    if not math.isfinite(stall_timeout) or stall_timeout < 0:
        raise ValueError("stall_timeout 必须为非负有限数；0 表示关闭超时检查。")
    if output.suffix.lower() != ".mp4":
        raise ValueError("输出必须为 .mp4。")
    if output.exists() and not overwrite:
        raise ValueError(f"文件已存在：{output}。使用 --overwrite 允许替换。")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = round(duration * fps)
    if frames < 2:
        raise ValueError("视频至少需要 2 帧。")
    repeat_count = 1
    # 无漂移时整个镜头、指针和历史尾迹都按基础旋转周期重复。
    # 只渲染一段再无损复制编码片段，不缓存数 GB 的原始图像。
    model = renderer.model
    cycles = getattr(model, "cycles", 1)
    fading = getattr(renderer, "fading", None)
    cache_aligned = fading is None or fading.cache_count % cycles == 0
    if getattr(model, "strength", None) == 0 and cycles > 1 and frames % cycles == 0 and frames//cycles >= 2 and cache_aligned:
        repeat_count = cycles
    render_frames = frames // repeat_count
    binary = ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()
    # 同目录临时文件，成功后才替换正式视频。
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as tmp:
        temporary = Path(tmp.name)
    command = [binary, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-s",
               f"{renderer.width}x{renderer.height}", "-r", str(fps), "-i", "-",
               "-an", "-c:v", encoder]
    if encoder == "libx264":
        command += ["-crf", str(crf), "-preset", preset]
    elif encoder == "h264_nvenc":
        command += ["-preset", "p4", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    else:
        command += ["-b:v", "20M"]
    command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    process, watchdog, looped = None, None, None
    last_report = 0.0
    start = time.perf_counter()
    print(f"输出 {frames} 帧 | {renderer.width}×{renderer.height} | {fps} FPS | {encoder}")
    if repeat_count > 1:
        print(f"固定运动复用：渲染 {render_frames} 帧 / {render_frames/fps:g} 秒周期，无损重复 {repeat_count} 次。")
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            watchdog = ExportWatchdog(process, cancel_event, stall_timeout)
            try:
                for i in range(render_frames):
                    # [0,1)，不额外编码重复的 t=1；循环边界保持一个正常帧间隔。
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("导出已取消。")
                    watchdog.beat("生成视频帧")
                    frame = renderer.frame(i / frames)
                    watchdog.beat("写入 FFmpeg 编码管道")
                    process.stdin.write(np.ascontiguousarray(frame).tobytes())
                    watchdog.check()
                    now = time.perf_counter()
                    if progress is not None and (i % max(1, render_frames // 100) == 0 or i == render_frames - 1 or now-last_report >= .5):
                        progress((i + 1) * repeat_count, frames)
                        last_report = now
                    if i % max(1, render_frames // 20) == 0 or i == render_frames - 1:
                        elapsed = time.perf_counter() - start
                        print(f"\r进度 {(i + 1) / render_frames:6.1%} | 已用 {elapsed:.1f} 秒", end="", flush=True)
            except (BrokenPipeError, OSError) as exc:
                watchdog.check()
                try:
                    process.stdin.close()
                except OSError:
                    pass
                process.wait()
                errors.seek(0)
                raise RuntimeError("FFmpeg 失败：" + errors.read().decode("utf-8", "replace")) from exc
            watchdog.beat("收尾封装 MP4")
            process.stdin.close()
            returncode = process.wait()
            watchdog.check()
            if returncode:
                errors.seek(0)
                raise RuntimeError("FFmpeg 失败：" + errors.read().decode("utf-8", "replace"))
            if repeat_count > 1:
                watchdog.close()
                with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as tmp:
                    looped = Path(tmp.name)
                # stream copy 不二次压缩，逐次循环的帧数、FPS 与原片段相同。
                mux = [binary, "-hide_banner", "-loglevel", "error", "-y", "-stream_loop",
                       str(repeat_count-1), "-i", str(temporary), "-t", str(frames/fps),
                       "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart", str(looped)]
                process = subprocess.Popen(mux, stdin=subprocess.DEVNULL, stderr=errors)
                watchdog = ExportWatchdog(process, cancel_event, stall_timeout)
                watchdog.beat("无损重复并封装 MP4")
                returncode = process.wait()
                watchdog.check()
                if returncode:
                    errors.seek(0)
                    raise RuntimeError("FFmpeg 循环封装失败：" + errors.read().decode("utf-8", "replace"))
        (looped or temporary).replace(output)
        print(f"\n输出完成：{output.resolve()}（{output.stat().st_size / 1024**2:.2f} MB）")
        print(f"实际时长：{frames / fps:.6f} 秒。")
        return frames
    finally:
        if watchdog is not None:
            watchdog.close()
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="傅里叶旋转圆盘高清动态壁纸生成器")
    p.add_argument("--input", "-i", type=Path, default=ROOT / "assets" / "demo.svg", help="SVG / PNG / JPG")
    p.add_argument("--output", "-o", type=Path, default=Path("Fourier_Wallpaper.mp4"), help="输出 MP4")
    p.add_argument("--circles", "-c", type=int, default=200, help="圆盘数，默认 200")
    p.add_argument("--samples", type=int, default=2000, help="等弧长点数，默认 2000")
    p.add_argument("--duration", type=float, default=60, help="完整循环时长/秒，默认 60")
    p.add_argument("--fps", type=int, default=60, help="帧率，默认 60")
    p.add_argument("--width", type=int, default=1920, help="宽度，默认 1920")
    p.add_argument("--height", type=int, default=1080, help="高度，默认 1080")
    p.add_argument("--sort", choices=["radius", "frequency"], default="radius")
    p.add_argument("--trail-mode", choices=["drift", "loop", "draw"], default="drift", help="漂移渐隐 / 固定白色闭合曲线 / 从零绘制")
    p.add_argument("--contour-index", type=int, default=0, help="轮廓按周长降序编号，从 0 开始")
    p.add_argument("--close-open", action="store_true", help="以直线闭合 SVG 开放路径")
    p.add_argument("--threshold", type=int, help="位图阈值 0～255，默认 Otsu")
    p.add_argument("--polarity", choices=["auto", "dark", "light"], default="auto")
    p.add_argument("--edges", action="store_true", help="位图使用 Canny 边缘检测")
    p.add_argument("--crf", type=int, default=18, help="质量 0～51，越小越清晰")
    p.add_argument("--preset", choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"], default="medium")
    p.add_argument("--encoder", choices=["libx264", "h264_nvenc", "h264_qsv", "h264_amf"], default="libx264")
    p.add_argument("--ffmpeg", help="自定义 FFmpeg 路径，默认使用依赖自带版本")
    p.add_argument("--ring-width", type=float, default=2.4, help="1080p 圆环线宽，默认 2.4")
    p.add_argument("--ring-opacity", type=float, default=0.90, help="圆环亮度/透明度 0～1，默认 0.90")
    p.add_argument("--ticks", type=int, default=12, help="大圆刻度数量，0 关闭，默认 12")
    p.add_argument("--preview", type=Path, help="同时保存中周期 PNG 预览")
    p.add_argument("--gui", action="store_true", help="打开交互窗口")
    p.add_argument("--headless", action="store_true", help="命令行直接渲染视频")
    p.add_argument("--source", choices=["image", "random", "text"], default="text", help="运动来源：text (汉字名字，默认) / image (本地图片) / random (随机运动)")
    p.add_argument("--text", type=str, default="傅里叶", help="输入的中文名字或文字")
    p.add_argument("--text-style", choices=["ordered", "signature", "xuejun", "stroke", "contour"], default="ordered", help="文字样式：ordered (标准笔顺单线，默认) / signature (行书字体实验映射) / contour (字体轮廓) / xuejun、stroke (旧版)")
    p.add_argument("--font", type=str, default="C:/Windows/Fonts/STXINGKA.TTF", help="本地字体文件路径 (signature 和 contour 模式生效，默认华文行楷)")
    p.add_argument("--no-close-loop", dest="close_loop", action="store_false", default=True, help="使用直线返回起点，关闭字外回线")
    p.add_argument("--show-transfers", action="store_true", help="显示字内换笔连接线；默认抬笔时不留墨")
    p.add_argument("--seed", type=int, default=42, help="随机运动种子")
    p.add_argument("--max-frequency", type=int, default=64, help="随机初始频谱的最大频率")
    p.add_argument("--spectral-decay", type=float, default=1.3, help="随机高频衰减指数")
    p.add_argument("--drift-strength", "--drift_strength", type=float, default=.18, help="漂移强度 0～0.75")
    p.add_argument("--drift-period", "--drift_period", type=float, default=30, help="目标漂移周期/秒，会对齐完整循环")
    p.add_argument("--rotation-period", type=float, default=10, help="基础旋转周期/秒，会对齐完整循环")
    p.add_argument("--trail-length", "--trail_length", type=float, default=8, help="尾迹保留时间/秒")
    p.add_argument("--fade-power", "--fade_power", type=float, default=2, help="渐隐指数 0.2～8")
    p.add_argument("--glow-strength", "--glow_strength", type=float, default=.8, help="尾迹光晕强度 0～3")
    p.add_argument("--trail-color", choices=["cyan", "blue", "purple"], default="cyan")
    p.add_argument("--overwrite", action="store_true", help="允许覆盖已有视频")
    return p


def make_renderer(a, width=None, height=None):
    """GUI 和命令行共用同一个构建流程，预览与导出具有一致的参数和随机种子。"""
    from dynamics import DriftingModel, FadingTrail, random_model
    if a.source == "random":
        base = random_model(a.samples, a.circles, a.seed, a.max_frequency, a.spectral_decay)
        print(f"随机种子 {a.seed} | 实际圆盘 {len(base.radii)} 个")
    elif a.source == "text":
        font_path = getattr(a, "font", None)
        close_loop = getattr(a, "close_loop", True)
        text_str = getattr(a, "text", "傅里叶")
        text_style = getattr(a, "text_style", "ordered")
        signature_path = None
        if text_style in ("ordered", "signature"):
            from signature import build_ordered_signature, build_signature, SignatureMotion
            if text_style=="signature" and font_path and not Path(font_path).is_file():
                raise ValueError("所选字体文件不存在，请重新选择字体。")
            chosen_font = next((p for p in ([Path(font_path)] if font_path else [])+DEFAULT_FONTS if p.is_file()),None)
            if text_style=="signature" and chosen_font is None:
                raise ValueError("未找到可用中文字体，请选择字体文件。")
            signature_path = (build_ordered_signature(text_str, close_loop) if text_style=="ordered"
                              else build_signature(text_str,chosen_font,close_loop))
            points = resample_closed(signature_path.points,a.samples)
            z = points[:,0]-1j*points[:,1]
            signature_origin = z.mean()
            signature_scale = 2/max(float(np.ptp(z.real)),float(np.ptp(z.imag)))
            z = (z-signature_origin)*signature_scale
            count = sum(s.kind=="stroke" for s in signature_path.segments)
            description = f"校准主笔画 {count} 笔" if signature_path.calibrated else f"主笔画映射 {count} 段（近似）"
            label = "标准笔顺单线字形" if text_style=="ordered" else f"实验字体映射 {chosen_font.name}"
            print(f"{label} | {description if text_style=='signature' else f'{count} 笔'} | 等弧长采样")
        else:
            z = load_text_contour(text_str, a.samples, font_path, close_loop, text_style=text_style)
        base = FourierModel(z, a.circles, a.sort)
        error = np.sqrt(np.mean(np.abs(base.curve(np.arange(len(z)) / len(z)) - z) ** 2))
        print(f"文字采样 {len(z)} 点 | 圆盘 {len(base.epicycles)} 个 | 归一化重建 RMSE {error:.6g}")
    else:
        z = load_contour(a.input, a.samples, a.contour_index, a.close_open, a.threshold, a.polarity, a.edges)
        base = FourierModel(z, a.circles, a.sort)
        error = np.sqrt(np.mean(np.abs(base.curve(np.arange(len(z)) / len(z)) - z) ** 2))
        print(f"采样 {len(z)} 点 | 圆盘 {len(base.epicycles)} 个 | 归一化重建 RMSE {error:.6g}")
    model, fading = base, None
    if a.source == "text" and signature_path is not None:
        actual_duration = round(a.duration*a.fps)/a.fps
        cycles = max(1,round(actual_duration/a.rotation_period)) if a.trail_mode=="drift" else 1
        model = SignatureMotion(base, signature_path, actual_duration, cycles,
                                origin=signature_origin, scale=signature_scale,
                                show_transfers=getattr(a, "show_transfers", False))
        if a.trail_mode == "drift":
            fading = FadingTrail(model,a.trail_length,a.fade_power,a.glow_strength,
                                 sample_rate=max(120,2*a.fps),color=a.trail_color,constant_width=True)
        print("字迹完全由傅里叶向量之和绘制；各圆盘匀速、线宽固定，盘数决定还原程度。")
    elif a.trail_mode == "drift":
        actual_duration = round(a.duration * a.fps) / a.fps
        model = DriftingModel(base, actual_duration, a.drift_strength, a.drift_period, a.rotation_period)
        fading = FadingTrail(model, a.trail_length, a.fade_power, a.glow_strength,
                             sample_rate=max(120, 2 * a.fps), color=a.trail_color)
        periods = ", ".join(f"{period:.3g}" for period in model.actual_periods)
        print(f"完整循环 {actual_duration:.3g}s | 基础旋转 {actual_duration/model.cycles:.3g}s | 漂移周期 {periods}s")
    if a.trail_mode == "draw":
        print("draw 模式周期末会清空轨迹，不承诺无缝。")
    return Renderer(model, width or a.width, height or a.height, a.trail_mode,
                    ring_width=a.ring_width, ring_opacity=a.ring_opacity, ticks=a.ticks, fading=fading,
                    render_fps=min(a.fps, 30) if width else a.fps)


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if a.gui or (argv is None and len(sys.argv) == 1):
        from gui import run_gui
        return run_gui(a)
    if a.samples < 16 or not 1 <= a.circles <= a.samples:
        p.error("samples 至少 16；circles 在 1～samples 之间。")
    if not math.isfinite(a.duration) or a.duration <= 0 or a.fps <= 0:
        p.error("duration、fps 必须为正数。")
    if a.width < 64 or a.height < 64 or a.width % 2 or a.height % 2:
        p.error("H.264 yuv420p 要求宽高为至少 64 的偶数。")
    if not 0 <= a.crf <= 51 or (a.threshold is not None and not 0 <= a.threshold <= 255):
        p.error("crf 范围 0～51；threshold 范围 0～255。")
    if not math.isfinite(a.ring_width) or not 0 < a.ring_width <= 12:
        p.error("ring-width 必须大于 0 且不超过 12。")
    if not math.isfinite(a.ring_opacity) or not 0 <= a.ring_opacity <= 1 or not 0 <= a.ticks <= 60:
        p.error("ring-opacity 范围 0～1，ticks 范围 0～60。")
    if a.output.resolve() == a.input.resolve():
        p.error("输出不能覆盖输入。")
    if a.preview and (a.preview.suffix.lower() != ".png" or a.preview.resolve() in {a.input.resolve(), a.output.resolve()}):
        p.error("preview 必须为 PNG，且不能与输入或视频路径相同。")
    try:
        renderer = make_renderer(a)
        if a.preview:
            a.preview.parent.mkdir(parents=True, exist_ok=True)
            ok, data = cv2.imencode(".png", renderer.frame(0.375))
            if not ok:
                raise RuntimeError("PNG 编码失败。")
            data.tofile(str(a.preview))
        encode_video(renderer, a.output, a.fps, a.duration, a.crf, a.preset, a.encoder, a.ffmpeg, a.overwrite)
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已取消渲染，未完成视频已清理。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
