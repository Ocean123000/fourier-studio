"""Show several names using the same direct stroke-data pipeline."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import ROOT, build_parser, make_renderer


def main():
    names = ("傅里叶", "张三", "李四", "周八")
    poster = Image.new("RGB", (1280, 820), "#090d12")
    label_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    label = ImageDraw.Draw(poster)
    for index, name in enumerate(names):
        args = build_parser().parse_args(["--headless", "--text", name,
                                         "--text-style", "ordered", "--trail-mode", "loop",
                                         "--samples", "2000", "--circles", "180"])
        renderer = make_renderer(args, 640, 360)
        frame = renderer.frame(.35)
        x, y = index%2*640, index//2*410
        poster.paste(Image.fromarray(frame[:, :, ::-1]), (x, y+42))
        label.text((x+20, y+10), f"示例：{name} · 标准笔顺单线 · 180 个圆盘", font=label_font, fill="#dce7ee")
    poster.save(ROOT/"ordered_names_preview.png")


if __name__=="__main__":
    main()
