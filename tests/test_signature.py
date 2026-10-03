"""Font fidelity, ordered strokes, transfers and collision-free returns."""
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from main import FourierModel, build_parser, load_text_contour, make_renderer, resample_closed
from signature import (FONT_PRESETS, SignatureMotion, _occupied, _thin,
                       build_signature)


class SignatureTests(unittest.TestCase):
    def test_bad_font_reports_error_instead_of_silent_substitution(self):
        with self.assertRaisesRegex(ValueError, "字体文件不存在"):
            load_text_contour("傅里叶", 256, font_path="assets/fonts/missing.ttf", text_style="signature")
        with self.assertRaisesRegex(ValueError, "字体不包含"):
            build_signature(chr(0x10FFFF), FONT_PRESETS["华文行楷（清晰）"])

    def test_all_fonts_keep_ink_and_order_with_cached_names(self):
        for name, font in FONT_PRESETS.items():
            if not font.is_file():
                continue
            with self.subTest(font=name), patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                for text in ("傅里叶", "李四"):
                    path = build_signature(text, font)
                    strokes = [s for s in path.segments if s.kind == "stroke"]
                    order = [(s.char, s.stroke) for s in strokes]
                    self.assertEqual(order, sorted(order))
                    self.assertTrue(_occupied(np.vstack([s.points for s in strokes]), path.mask).all())
                    # Only ordered main strokes belong to the target. Completing
                    # every skeleton spur creates additional strokes and loops.
                    if path.calibrated:
                        self.assertEqual(len(strokes), 22)
                    if text == "傅里叶":
                        for index, count in enumerate((6, 8, 8)):
                            self.assertEqual({s.stroke for s in strokes if s.char == index}, set(range(count)))

    def test_straight_transfers_and_exterior_return(self):
        for name, font in FONT_PRESETS.items():
            if not font.is_file():
                continue
            with self.subTest(font=name):
                path = build_signature("傅里叶", font)
                transfers = [s for s in path.segments if s.kind == "char"]
                self.assertEqual(len(transfers), 2)
                lengths = [np.linalg.norm(np.diff(s.points, axis=0), axis=1).sum()
                           for s in path.segments]
                arc = np.r_[0., np.cumsum(lengths)] / sum(lengths)
                for i, segment in enumerate(path.segments):
                    if segment.kind == "char":
                        self.assertEqual(segment.points.shape, (2, 2))
                closure = path.segments[-1]
                self.assertEqual(closure.kind, "return")
                np.testing.assert_array_equal(closure.points[0], path.segments[-2].points[-1])
                np.testing.assert_array_equal(closure.points[-1], path.segments[0].points[0])
                # Only the immediate start/end caps may overlap their own ink.
                away = ((np.linalg.norm(closure.points-closure.points[0], axis=1) > 24)
                        & (np.linalg.norm(closure.points-closure.points[-1], axis=1) > 24))
                self.assertFalse(_occupied(closure.points[away], path.mask).any())
                # Also check the finite Fourier reconstruction shown in the UI.
                samples = resample_closed(path.points, 2000)
                z = samples[:, 0]-1j*samples[:, 1]
                center = z.mean()
                scale = 2/max(np.ptp(z.real), np.ptp(z.imag))
                motion = SignatureMotion(FourierModel((z-center)*scale, 180), path)
                times = np.linspace(arc[-2], arc[-1], 3000, endpoint=False)
                reconstructed = motion.curve(times)/scale+center
                actual = np.column_stack([reconstructed.real, -reconstructed.imag])
                away = ((np.linalg.norm(actual-closure.points[0], axis=1) > 24)
                        & (np.linalg.norm(actual-closure.points[-1], axis=1) > 24))
                self.assertFalse(_occupied(actual[away], path.mask).any())

    def test_transfers_use_same_linear_clock_as_strokes(self):
        path = build_signature("傅里叶", FONT_PRESETS["华文行楷（清晰）"])
        p = resample_closed(path.points, 2000)
        base = FourierModel(p[:, 0]-1j*p[:, 1], 180)
        motion = SignatureMotion(base, path, 60, 6)
        times = np.linspace(-.1, 1.1, 200)
        np.testing.assert_array_equal(motion.phase_rate(times), np.full(len(times), 6))
        np.testing.assert_allclose(motion.curve(times), base.curve(times*6), atol=1e-9)
        np.testing.assert_allclose(motion.chain(0), motion.chain(1), atol=1e-9)

    def test_font_signature_is_deterministic_and_text_dependent(self):
        font = FONT_PRESETS["华文行楷（清晰）"]
        first = build_signature("傅里叶", font)
        second = build_signature("傅里叶", font)
        np.testing.assert_allclose(first.points, second.points)
        other = build_signature("李四", font)
        self.assertFalse(np.array_equal(first.points, other.points))
        self.assertTrue(any(s.kind == "stroke" for s in first.segments))

    def test_skeleton_branch_is_not_promoted_to_an_additional_stroke(self):
        from signature import _guided_strokes
        skeleton = np.zeros((30, 30), bool)
        skeleton[15, 3:27] = True
        skeleton[3:15, 15] = True
        strokes = _guided_strokes(skeleton, [np.array([[3., 15.], [26., 15.]])])
        self.assertEqual(len(strokes), 1)
        self.assertEqual(len(strokes[0]), 1)
        self.assertTrue((strokes[0][0][:, 1]==15).all())

    def test_standard_font_does_not_drift_and_no_close_lifts_home(self):
        args = build_parser().parse_args(["--headless", "--samples", "512", "--circles", "80"])
        args.drift_strength = .75
        renderer = make_renderer(args, 160, 96)
        motion = renderer.model
        self.assertEqual(motion.strength, 0)
        np.testing.assert_allclose(motion.curve([.023, .023+1/motion.cycles]),
                                   np.repeat(motion.curve([.023]), 2), atol=1e-10)
        path = build_signature("傅里叶", args.font, close_loop=False)
        self.assertEqual(path.segments[-1].kind, "lift")
        self.assertNotIn("return", [s.kind for s in path.segments])


if __name__ == "__main__":
    unittest.main()
