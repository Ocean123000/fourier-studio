"""Regressions for arbitrary names, without font skeleton inference or patches."""
import unittest
from unittest.mock import patch

import numpy as np

from main import FourierModel, build_parser, make_renderer, resample_closed
from signature import _occupied, build_ordered_signature, stroke_medians


class OrderedTextTests(unittest.TestCase):
    names = ("张三", "王五", "赵六", "孙七", "李四",
             "周八", "吴九", "郑十", "傅里叶")

    def test_every_stroke_preserves_source_shape_and_order_for_different_names(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("cached data only")), \
             patch("signature._thin", side_effect=AssertionError("must not infer a font skeleton")), \
             patch("signature._guided_strokes", side_effect=AssertionError("must not remap strokes")):
            for name in self.names:
                with self.subTest(name=name):
                    path = build_ordered_signature(name)
                    self.assertEqual(path.source_kind, "ordered")
                    strokes = [s for s in path.segments if s.kind=="stroke"]
                    expected = [(i, j, p*240/1024+[i*260, 0])
                                for i, char in enumerate(name)
                                for j, p in enumerate(stroke_medians(char))]
                    self.assertEqual(len(strokes), len(expected))
                    shift = strokes[0].points[0]-expected[0][2][0]
                    for actual, (i, j, target) in zip(strokes, expected):
                        self.assertEqual((actual.char, actual.stroke), (i, j))
                        np.testing.assert_allclose(actual.points, target+shift, atol=1e-12)
                    # Transfers are separate segments; they cannot add strokes.
                    self.assertEqual(sum(s.kind=="char" for s in path.segments), len(name)-1)
                    np.testing.assert_array_equal(path.points[0], path.points[-1])
                    closure = path.segments[-1].points
                    away = ((np.linalg.norm(closure-closure[0], axis=1)>24)
                            & (np.linalg.norm(closure-closure[-1], axis=1)>24))
                    self.assertFalse(_occupied(closure[away], path.mask).any())

    def test_default_new_name_does_not_need_a_font_or_enter_experimental_mapping(self):
        args = build_parser().parse_args(["--headless", "--text", "张三", "--font", "missing.ttf"])
        with patch("signature.build_signature", side_effect=AssertionError("no experimental mapping")):
            renderer = make_renderer(args, 640, 360)
        self.assertEqual(renderer.model.path.source_kind, "ordered")
        self.assertEqual(sum(s.kind=="stroke" for s in renderer.model.path.segments), 26)
        self.assertEqual(renderer.frame(.31).shape, (360, 640, 3))

    def test_new_names_remain_actual_fourier_reconstructions_at_every_disk_count(self):
        for name in ("张三", "周八"):
            path = build_ordered_signature(name)
            points = resample_closed(path.points, 2000)
            target = points[:, 0]-1j*points[:, 1]
            target = (target-target.mean())*2/max(np.ptp(target.real), np.ptp(target.imag))
            times = np.arange(len(target))/len(target)
            errors = []
            for count in (2, 18, 180):
                model = FourierModel(target, count)
                exact = np.exp(2j*np.pi*np.outer(times, model.frequencies))@model.coefficients
                np.testing.assert_allclose(model.curve(times), exact, atol=1e-12)
                errors.append(np.sqrt(np.mean(np.abs(exact-target)**2)))
            self.assertGreater(errors[0], errors[1]*2)
            self.assertGreater(errors[1], errors[2]*3)

    def test_missing_character_data_is_an_error_not_a_font_fallback(self):
        args = build_parser().parse_args(["--headless", "--text", "张三"])
        with patch("signature.stroke_medians", side_effect=ValueError("缺少笔顺数据")), \
             patch("signature.build_signature", side_effect=AssertionError("no fallback")):
            with self.assertRaisesRegex(ValueError, "缺少笔顺数据"):
                make_renderer(args, 160, 96)


if __name__ == "__main__":
    unittest.main()
