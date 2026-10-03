"""验证漂移积分、周期边界、尾迹历史和导出取消。"""
from pathlib import Path
import tempfile
import time
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import cv2
from main import FourierModel, Renderer, build_parser, encode_video, make_renderer
from dynamics import DriftingModel, FadingTrail, random_model


class DynamicTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(128) / 128
        self.base = FourierModel(np.exp(2j*np.pi*t) + .3*np.exp(-4j*np.pi*t) +
                                 .1*np.exp(6j*np.pi*t), 8)
        self.motion = DriftingModel(self.base, 60, .2, 30, 10)

    def test_disabled_drift_preserves_model(self):
        motion = DriftingModel(self.base, 60, 0, 30, 10)
        for u in (.0, .13, .27, .51):
            np.testing.assert_allclose(motion.chain(u), self.base.chain(u*6), atol=1e-12)

    def test_parameters_and_position_repeat(self):
        for start, end in zip(self.motion.parameters(0), self.motion.parameters(1)):
            np.testing.assert_array_equal(start, end)
        np.testing.assert_array_equal(self.motion.chain(0), self.motion.chain(1))
        for u in np.linspace(0, 1, 40):
            radii, _, _ = self.motion.parameters(u)
            self.assertTrue(np.all(radii >= 0))
            self.assertTrue(np.isfinite(self.motion.chain(u)).all())
        self.assertGreater(abs(self.motion.chain(0)[-1] - self.motion.chain(1/6)[-1]), .005)

    def test_frequency_is_integrated(self):
        u, h = .23, 1e-6
        numerical = (self.motion.angles(u+h)-self.motion.angles(u-h))/(2*h*self.motion.loop_seconds)
        phases_plus = self.motion.parameters(u+h)[1]
        phases_minus = self.motion.parameters(u-h)[1]
        phase_rate = (phases_plus-phases_minus)/(2*h*self.motion.loop_seconds)
        expected = 2*np.pi*self.motion.parameters(u)[2] + phase_rate
        np.testing.assert_allclose(numerical, expected, atol=1e-8, rtol=1e-8)

    def test_boundary_velocity_matches(self):
        h = 1e-7
        at_zero = self.motion.chain(0)
        before = (at_zero-self.motion.chain(1-h))/h
        after = (self.motion.chain(h)-at_zero)/h
        np.testing.assert_allclose(before, after, atol=.002, rtol=1e-4)

    def test_trail_age_and_history_repeat(self):
        trail = FadingTrail(self.motion, 8, 2, .8)
        np.testing.assert_array_equal(trail.points(0), trail.points(1))
        self.assertEqual(float(trail.opacity(1)), 0)
        self.assertEqual(float(trail.opacity(0)), 1)
        self.assertTrue(np.all(np.diff(trail.alpha) >= 0))
        expected = self.motion.curve(np.mod(-trail.ages/60, 1))
        np.testing.assert_allclose(trail.points(0), expected, atol=.002)
        self.assertGreater(np.ptp(trail.points(0).real), .1)
        # 精确帧时间可能与尾迹年龄相消到负的浮点微量，不能在周期尾端越界。
        short = FadingTrail(DriftingModel(self.base, 2, .2, 30, 10), 1, 2, 0)
        for u in np.linspace(0, 1, 81):
            self.assertTrue(np.isfinite(short.points(u)).all())

    def test_frames_repeat_with_fade(self):
        trail = FadingTrail(self.motion, 8, 2, .8)
        renderer = Renderer(self.motion, 320, 180, "drift", fading=trail)
        np.testing.assert_array_equal(renderer.frame(0), renderer.frame(1))
        self.assertFalse(np.array_equal(renderer.frame(0), renderer.frame(.25)))
        self.assertGreater(renderer.frame(.25).max(), 220)
        # 重复渲染同一时间不依赖之前的帧顺序或状态。
        renderer.frame(.77)
        np.testing.assert_array_equal(renderer.frame(.25), renderer.frame(.25))

    def straight_trail(self, line_width=3, power=2, glow=0):
        trail = FadingTrail(self.motion, 8, power, glow)
        renderer = SimpleNamespace(background=np.zeros((120, 1000, 3), np.uint8),
                                   width=1000, height=120, line_width=line_width,
                                   pixels=lambda p: np.column_stack([p.real, p.imag]))
        trail.points = lambda u: np.linspace(30, 970, len(trail.ages)) + 60j
        return trail, renderer

    def test_line_width_has_no_secondary_brightness_bands(self):
        # 旧实现在线宽跨过 2/3 像素时两次回落；用直线隔离运动和圆环影响。
        for width in (1, 2, 3, 6):
            for power in (.2, 2, 8):
                for glow in (0, .8, 3):
                    trail, renderer = self.straight_trail(width, power, glow)
                    row = trail.layer(renderer, 0)[60, 40:968].astype(float)
                    self.assertTrue(np.all(np.diff(row, axis=0) >= -1))
                    self.assertGreater(row[-1, 0], 240)
                    self.assertGreater(row[-1].mean(), row[400].mean())

    def test_glow_increases_halo_without_relighting_old_core(self):
        trail, renderer = self.straight_trail(glow=0)
        plain = trail.layer(renderer, 0)
        trail.glow = 3
        glowing = trail.layer(renderer, 0)
        # 新发光参数对可见光晕确实生效；中心的年龄渐变不会被光晕抬平。
        self.assertGreater(glowing[55, 750, 0], plain[55, 750, 0] + 30)
        np.testing.assert_array_equal(glowing[60, 100:950], plain[60, 100:950])
        self.assertGreater(int(glowing[60, 950, 0]), 245)

    def test_trail_rasterization_is_sampling_density_independent(self):
        trail, renderer = self.straight_trail(glow=3)
        before = trail.layer(renderer, 0)
        # 在每个旧时间段中再插一个点，直线的发光场应保持相同。
        trail.ages = np.linspace(trail.length, 0, 2 * len(trail.ages) - 1)
        after = trail.layer(renderer, 0)
        self.assertLessEqual(np.abs(before.astype(int)-after.astype(int)).max(), 2)

    def test_tail_head_is_exact_endpoint(self):
        for color in ("cyan", "blue", "purple"):
            trail = FadingTrail(self.motion, 8, 2, 3, color=color)
            renderer = Renderer(self.motion, 320, 180, "drift", fading=trail)
            for u in (0, .1, .375, .8, 1):
                self.assertEqual(trail.points(u)[-1], self.motion.chain(u)[-1])
                head = np.rint(renderer.pixels(self.motion.chain(u))[-1]).astype(int)
                np.testing.assert_array_equal(renderer.frame(u)[head[1], head[0]], [255]*3)

    def test_random_reproducibility(self):
        a = random_model(256, 20, 7, 16, 1.3)
        b = random_model(256, 20, 7, 16, 1.3)
        c = random_model(256, 20, 8, 16, 1.3)
        np.testing.assert_array_equal(a.coefficients, b.coefficients)
        self.assertFalse(np.allclose(a.coefficients, c.coefficients))
        self.assertTrue(np.all(np.abs(a.frequencies) <= 16))

    def test_drift_camera_contains_disks(self):
        x0, x1, y0, y1 = self.motion.bounds()
        for u in np.linspace(0, 1, 80):
            center = self.motion.chain(u)[:-1]
            radii = self.motion.parameters(u)[0]
            self.assertTrue(np.all(center.real-radii >= x0))
            self.assertTrue(np.all(center.real+radii <= x1))
            self.assertTrue(np.all(center.imag-radii >= y0))
            self.assertTrue(np.all(center.imag+radii <= y1))

    def test_cli_aliases_and_drifting_video(self):
        args = build_parser().parse_args(["--headless", "--source", "random", "--samples", "128",
                                        "--circles", "8", "--max-frequency", "8", "--duration", "2",
                                        "--fps", "10", "--trail_length", "1", "--drift_strength", ".2",
                                        "--width", "160", "--height", "96"])
        renderer = make_renderer(args)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"drift.mp4"
            self.assertEqual(encode_video(renderer, output, 10, .4, preset="ultrafast"), 4)
            cap = cv2.VideoCapture(str(output))
            self.assertTrue(cap.isOpened())
            self.assertEqual(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            self.assertTrue(cap.read()[0])
            cap.release()

    def test_fixed_motion_reuses_period_without_changing_video(self):
        motion = DriftingModel(self.base, 2, 0, 2, .5)
        trail = FadingTrail(motion, 1, 2, .8)
        renderer = Renderer(motion, 160, 96, "drift", fading=trail)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"repeated.mp4"
            with patch.object(renderer, "frame", wraps=renderer.frame) as render:
                self.assertEqual(encode_video(renderer, output, 20, 2, preset="ultrafast"), 40)
                self.assertEqual(render.call_count, 10)
            cap = cv2.VideoCapture(str(output))
            decoded = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                decoded.append(frame)
            self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 20)
            cap.release()
            self.assertEqual(len(decoded), 40)
            for i in range(10, 40):
                np.testing.assert_array_equal(decoded[i], decoded[i%10])
            self.assertEqual(len(list(Path(directory).glob("*.mp4"))), 1)

    def blocked_encoder(self):
        # 模拟活着但不消费 stdin 的编码器，验证取消可以真正打断管道阻塞。
        killed = threading.Event()
        class Pipe:
            closed = False
            def write(self, data):
                if not killed.wait(3):
                    raise AssertionError("阻塞的编码器没有被终止")
                raise BrokenPipeError("encoder terminated")
            def close(self):
                self.closed = True
        return SimpleNamespace(stdin=Pipe(), poll=lambda: -9 if killed.is_set() else None,
                               kill=killed.set, wait=lambda: -9), killed

    def test_stalled_encoder_is_stopped_and_existing_video_preserved(self):
        renderer = Renderer(self.base, 160, 96)
        process, killed = self.blocked_encoder()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"existing.mp4"
            output.write_bytes(b"original")
            started = time.perf_counter()
            with patch("main.subprocess.Popen", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "导出无进展.*写入"):
                    encode_video(renderer, output, 10, .4, overwrite=True, stall_timeout=.1, ffmpeg="mock-ffmpeg")
            self.assertLess(time.perf_counter()-started, 1)
            self.assertTrue(killed.is_set())
            self.assertTrue(process.stdin.closed)
            self.assertEqual(output.read_bytes(), b"original")
            self.assertEqual(len(list(Path(directory).glob("*.mp4"))), 1)

    def test_cancel_interrupts_blocked_encoding_pipe(self):
        renderer = Renderer(self.base, 160, 96)
        process, killed = self.blocked_encoder()
        cancel = threading.Event()
        timer = threading.Timer(.1, cancel.set)
        with tempfile.TemporaryDirectory() as directory:
            timer.start()
            try:
                with patch("main.subprocess.Popen", return_value=process):
                    with self.assertRaisesRegex(RuntimeError, "取消"):
                        encode_video(renderer, Path(directory)/"cancel.mp4", 10, .4,
                                     cancel_event=cancel, stall_timeout=5, ffmpeg="mock-ffmpeg")
                self.assertTrue(killed.is_set())
                self.assertFalse(list(Path(directory).glob("*.mp4")))
            finally:
                timer.cancel()

    def test_cancel_preserves_existing_output(self):
        trail = FadingTrail(self.motion, 8, 2, 0)
        renderer = Renderer(self.motion, 160, 96, "drift", fading=trail)
        cancel = threading.Event()
        cancel.set()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"existing.mp4"
            output.write_bytes(b"original file")
            with self.assertRaisesRegex(RuntimeError, "取消"):
                encode_video(renderer, output, 10, .4, overwrite=True, cancel_event=cancel)
            self.assertEqual(output.read_bytes(), b"original file")
            self.assertEqual(len(list(Path(directory).glob("*.mp4"))), 1)


if __name__ == "__main__":
    unittest.main()
