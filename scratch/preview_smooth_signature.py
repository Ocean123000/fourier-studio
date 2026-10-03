"""Compare genuine Fourier reconstruction with 2, 18 and 180 disks at 60 FPS."""
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import ROOT, build_parser, encode_video, make_renderer, resample_closed


def main():
    common = ["--headless", "--text-style", "signature", "--samples", "2000",
              "--duration", "12", "--rotation-period", "12", "--fps", "60",
              "--trail-mode", "draw", "--show-transfers", "--width", "480", "--height", "360"]
    renderers = [make_renderer(build_parser().parse_args(common+["--circles", str(n)]))
                 for n in (2, 18, 180)]
    # Same camera, target, duration and geometric pen width across all panels.
    bounds = np.array([r.model.bounds() for r in renderers])
    x0, x1 = bounds[:, 0].min(), bounds[:, 1].max()
    y0, y1 = bounds[:, 2].min(), bounds[:, 3].max()
    center = complex((x0+x1)/2, (y0+y1)/2)
    scale = min(480*.84/(x1-x0), 360*.84/(y1-y0))
    errors = []
    for r in renderers:
        r.center, r.scale = center, scale
        r.trace = r.pixels(r.model.curve(np.linspace(0, 1, r.trace_count+1)))
        r.trace_fixed = np.rint(r.trace*16).astype(np.int32)
        r.static = r.make_trace(r.trace_count)
        p = resample_closed(r.model.path.points, 2000)
        target = (p[:, 0]-1j*p[:, 1]-r.model.origin)*r.model.scale
        errors.append(np.sqrt(np.mean(np.abs(r.model.curve(np.arange(2000)/2000)-target)**2)))
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)

    def frame(time):
        image = np.zeros((440, 1440, 3), np.uint8)
        for i, r in enumerate(renderers):
            image[42:402, i*480:(i+1)*480] = r.frame(time)
        view = Image.fromarray(image[:, :, ::-1])
        label = ImageDraw.Draw(view)
        for i, r in enumerate(renderers):
            label.text((i*480+18, 10), f"{len(r.model.radii)} 个圆盘 · RMSE {errors[i]:.4f}",
                       font=font, fill="#d9e5f0")
        label.text((18, 410), "同一目标：傅里叶 / 华文行楷    各盘匀速 · 固定线宽 · 字迹 = 圆盘末端轨迹",
                   font=font, fill="#93bdc9")
        return np.asarray(view)[:, :, ::-1].copy()

    video = SimpleNamespace(width=1440, height=440, model=renderers[-1].model, fading=None, frame=frame)
    encode_video(video, ROOT/"fourier_disk_comparison.mp4", 60, 12,
                 crf=16, preset="veryfast", overwrite=True)
    Image.fromarray(frame(.995)[:, :, ::-1]).save(ROOT/"fourier_disk_comparison.png")


if __name__ == "__main__":
    main()
