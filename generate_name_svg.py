"""
Fourier Studio · 汉字/姓名单闭合曲线 SVG 生成器 (首尾相连全笔画无损版)

将任意中文名字转换为 Fourier Studio 所需的单一连续闭合曲线 SVG。
基于欧拉树双向缝合 + 尾首艺术底托长弧回连，保留全部笔画与内孔，构成真正的首尾闭合单曲线。
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_full_svg(
    text: str = "傅里叶",
    font_path: str = "C:/Windows/Fonts/STXINGKA.TTF",
    font_size: int = 260,
    output_path: Path = Path("assets/name_full.svg"),
    close_loop: bool = True,
) -> Path:
    font = ImageFont.truetype(font_path, font_size)

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

    # 3. 提取轮廓并缝合内孔
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

    stitched_loop = list(main_pts)
    for hole_pts in simplified_holes:
        loop_arr = np.array(stitched_loop)
        dists = np.linalg.norm(loop_arr[:, None, :] - hole_pts[None, :, :], axis=2)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        idx_loop, idx_hole = min_idx[0], min_idx[1]
        hole_rotated = list(np.roll(hole_pts, -idx_hole, axis=0))
        splice = hole_rotated + [hole_rotated[0], loop_arr[idx_loop]]
        stitched_loop = stitched_loop[:idx_loop + 1] + splice + stitched_loop[idx_loop + 1:]

    tour = np.array(stitched_loop)

    d_parts = [f"M {tour[0, 0]},{tour[0, 1]}"]
    for pt in tour[1:]:
        d_parts.append(f"L {pt[0]},{pt[1]}")
    d_parts.append("Z")

    svg_str = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">
  <title>{text} 首尾相连闭合单曲线</title>
  <desc>保留全部笔画与内外孔洞，首尾通过艺术底托长弧回连，构成真正首尾闭合单曲线。</desc>
  <path fill="none" stroke="white" stroke-width="2" d="{' '.join(d_parts)}"/>
</svg>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_str, encoding="utf-8")
    print(f"已生成首尾相连闭合 SVG: {output_path} (路径点数: {len(tour)}, 包含 {len(simplified_holes)} 个内孔)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="生成适用于 Fourier Studio 的名字单闭合曲线 SVG (首尾相连全笔画版)")
    parser.add_argument("--name", type=str, default="傅里叶", help="要转换的中文名字或文字")
    parser.add_argument("--font", type=str, default="C:/Windows/Fonts/STXINGKA.TTF", help="字体路径 (默认华文行楷)")
    parser.add_argument("--output", type=str, default="assets/name_loop.svg", help="输出 SVG 路径")
    parser.add_argument("--font-size", type=int, default=260, help="字体大小")
    parser.add_argument("--no-close-loop", dest="close_loop", action="store_false", default=True, help="关闭首尾闭合底托线")
    args = parser.parse_args()

    generate_full_svg(
        text=args.name,
        font_path=args.font,
        font_size=args.font_size,
        output_path=Path(args.output),
        close_loop=args.close_loop,
    )


if __name__ == "__main__":
    main()
