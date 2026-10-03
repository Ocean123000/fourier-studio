"""数值正确性、轮廓输入、循环渲染和 MP4 导出的集成测试。"""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from main import FourierModel, Renderer, encode_video, extract_svg, load_contour, resample_closed


class FourierTests(unittest.TestCase):
    def test_equal_arc_square(self):
        p = resample_closed(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), 16)
        np.testing.assert_allclose(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1), 0.25)
        self.assertFalse(np.array_equal(p[0], p[-1]))

    def test_degenerate_input(self):
        with self.assertRaises(ValueError):
            resample_closed(np.zeros((4, 2)), 16)

    def test_fft_full_reconstruction(self):
        rng = np.random.default_rng(7)
        z = rng.normal(size=63) + 1j * rng.normal(size=63)
        model = FourierModel(z, len(z))
        np.testing.assert_allclose(model.curve(np.arange(len(z)) / len(z)), z, atol=1e-12)

    def test_negative_frequency(self):
        t = np.arange(64) / 64
        model = FourierModel(2 * np.exp(-2j * np.pi * 3 * t), 1)
        self.assertEqual(model.epicycles[0].frequency, -3)
        self.assertAlmostEqual(model.epicycles[0].radius, 2)
        np.testing.assert_allclose(model.chain(0.125)[-1], 2 * np.exp(-2j * np.pi * 3 * 0.125), atol=1e-12)

    def test_sort_and_period(self):
        t = np.arange(64) / 64
        model = FourierModel(np.exp(2j * np.pi * t) + 0.2 * np.exp(-4j * np.pi * t), 8)
        self.assertTrue(np.all(np.diff(model.radii) <= 0))
        np.testing.assert_array_equal(model.chain(0), model.chain(1))
        self.assertEqual(FourierModel(np.exp(2j * np.pi * t), 5, "frequency").frequencies.tolist(), [0, 1, -1, 2, -2])

    def test_svg_transforms_subpaths_and_open(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "shape.svg"
            p.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">'
                         '<g transform="translate(10,20) scale(2)">'
                         '<path d="M0 0L10 0L10 10L0 10Z M30 0L32 0L32 2L30 2Z"/>'
                         '</g><path d="M100 100L110 110L120 100"/></svg>', encoding="utf-8")
            paths = extract_svg(p, 100, False)
            self.assertEqual(len(paths), 2)
            np.testing.assert_allclose(paths[0].min(axis=0), [10, 20])
            np.testing.assert_allclose(paths[0].max(axis=0), [30, 40])
            self.assertEqual(len(extract_svg(p, 100, True)), 3)
            self.assertEqual(load_contour(p, 100, 1).shape, (100,))

    def test_svg_canvas_background_is_not_target(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "lemniscate.svg"
            p.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000" viewBox="0 0 1000 1000">'
                         '<rect width="100%" height="100%" fill="white"/>'
                         '<path d="M500 500 C420 250 120 250 120 500 C120 750 420 750 500 500 '
                         'C580 250 880 250 880 500 C880 750 580 750 500 500" '
                         'fill="none" stroke="blue"/></svg>', encoding="utf-8")
            candidates = extract_svg(p, 100, False)
            self.assertEqual(len(candidates), 1)
            np.testing.assert_allclose(candidates[0].min(axis=0), [120, 312.5], atol=.02)
            np.testing.assert_allclose(candidates[0].max(axis=0), [880, 687.5], atol=.02)
            z = load_contour(p, 1000)
            self.assertAlmostEqual(np.ptp(z.imag)/np.ptp(z.real), 375/760, places=4)

    def test_svg_background_viewbox_and_rect_only(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "rect.svg"
            header = '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="160" viewBox="10 20 100 80">'
            rect = '<rect x="10" y="20" width="100" height="80" fill="white"/>'
            p.write_text(header + rect + '</svg>', encoding="utf-8")
            self.assertEqual(len(extract_svg(p, 100, False)), 1)
            p.write_text(header + rect + '<circle cx="60" cy="60" r="20"/>' + '</svg>', encoding="utf-8")
            candidates = extract_svg(p, 100, False)
            self.assertEqual(len(candidates), 1)
            np.testing.assert_allclose(candidates[0].min(axis=0), [60, 40], atol=.01)
            # 有描边的矩形以及画在图形上方的矩形保留为可选轮廓。
            p.write_text(header + rect.replace('/>', ' stroke="black"/>') + '<circle cx="60" cy="60" r="20"/>' + '</svg>', encoding="utf-8")
            self.assertEqual(len(extract_svg(p, 100, False)), 2)
            p.write_text(header + '<circle cx="60" cy="60" r="20"/>' + rect + '</svg>', encoding="utf-8")
            self.assertEqual(len(extract_svg(p, 100, False)), 2)

    def test_bernoulli_asset_equation_and_crossing(self):
        p = Path(__file__).resolve().parents[1] / "assets" / "bernoulli_lemniscate.svg"
        self.assertEqual(len(extract_svg(p, 256, False)), 1)
        z = load_contour(p, 2000)
        x, y = z.real, z.imag
        # x 半宽已归一化为 1，数学双纽线满足 (x²+y²)²=x²-y²。
        residual = (x*x+y*y)**2 - (x*x-y*y)
        self.assertLess(np.abs(residual).max(), 1e-5)
        self.assertLess(abs(z[500]), 1e-6)
        self.assertLess(abs(z[1500]), 1e-6)
        self.assertAlmostEqual(np.ptp(y)/np.ptp(x), 1/(2*np.sqrt(2)), places=4)
        model = FourierModel(z, 60)
        self.assertLess(np.sqrt(np.mean(abs(model.curve(np.arange(2000)/2000)-z)**2)), 1e-4)

    def test_raster_formats_and_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            for extension, transparent in [(".png", False), (".jpg", False), (".png", True)]:
                if transparent:
                    image = np.zeros((100, 120, 4), dtype=np.uint8)
                    image[20:80, 20:100, 3] = 255
                else:
                    image = np.full((100, 120), 255, dtype=np.uint8)
                    image[20:80, 20:100] = 0
                p = Path(directory) / ("轮廓" + extension)
                ok, encoded = cv2.imencode(extension, image)
                self.assertTrue(ok)
                encoded.tofile(str(p))
                z = load_contour(p, 128)
                self.assertEqual(len(z), 128)
                self.assertAlmostEqual(float(np.ptp(z.real)), 2)
                self.assertAlmostEqual(float(abs(z.mean())), 0, places=12)
                self.assertEqual(len(load_contour(p, 128, edges=True)), 128)

    def test_periodic_frames_and_motion(self):
        t = np.arange(64) / 64
        model = FourierModel(np.exp(2j * np.pi * t) + .3 * np.exp(-4j * np.pi * t), 8)
        renderer = Renderer(model, 320, 180, trace_samples=256)
        np.testing.assert_array_equal(renderer.frame(0), renderer.frame(1))
        self.assertFalse(np.array_equal(renderer.frame(0), renderer.frame(.25)))
        self.assertGreater(int(renderer.frame(.25).max()), 230)
        self.assertGreater(np.count_nonzero(renderer.frame(.25)[:, :, 0] > 100), 100)

    def test_video_roundtrip_and_failure_cleanup(self):
        t = np.arange(32) / 32
        renderer = Renderer(FourierModel(np.exp(2j * np.pi * t), 4), 160, 96, trace_samples=128)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "small.mp4"
            self.assertEqual(encode_video(renderer, output, 10, .4, preset="ultrafast"), 4)
            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened())
            self.assertEqual(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            self.assertEqual(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 160)
            self.assertEqual(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 96)
            decoded = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self.assertEqual(frame.shape, (96, 160, 3))
                decoded += 1
            capture.release()
            self.assertEqual(decoded, 4)
            before = output.read_bytes()
            with self.assertRaises(ValueError):
                encode_video(renderer, output, 10, .4)
            with patch("main.subprocess.Popen", side_effect=OSError("expected test failure")):
                with self.assertRaises(OSError):
                    encode_video(renderer, output, 10, .4, overwrite=True)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(len(list(Path(directory).glob("*.mp4"))), 1)

    def test_text_contour_and_renderer(self):
        from main import load_text_contour, make_renderer, build_parser
        from signature import build_ordered_signature

        defaults = build_parser().parse_args(["--headless"])
        self.assertEqual(defaults.text_style, "ordered")
        with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected download")):
            z = load_text_contour("傅里叶", 256, close_loop=True)
            other_name = load_text_contour("李四", 256, close_loop=True)
        self.assertFalse(np.array_equal(z, other_name))
        pen = build_ordered_signature("傅里叶").points
        np.testing.assert_allclose(pen[0], pen[-1], atol=1e-9)
        z = load_text_contour("傅里叶", 256, close_loop=True)
        self.assertEqual(len(z), 256)
        self.assertTrue(np.isfinite(z).all())
        # Test make_renderer with source="text"
        args = build_parser().parse_args([
            "--headless", "--source", "text", "--text", "傅里叶",
            "--circles", "30", "--samples", "256", "--duration", "2",
            "--trail-length", "1",
            "--fps", "10", "--width", "160", "--height", "96"
        ])
        renderer = make_renderer(args, 160, 96)
        frame = renderer.frame(0.0)
        self.assertEqual(frame.shape, (96, 160, 3))


if __name__ == "__main__":
    unittest.main()

