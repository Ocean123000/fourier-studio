"""The rendered pen and ink must come from the actual fixed Fourier sum."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from main import FourierModel, Renderer, build_parser, make_renderer, resample_closed
from dynamics import FadingTrail
from signature import SignatureMotion


class SmoothSignatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = build_parser().parse_args(["--headless"])
        cls.renderer = make_renderer(args, 640, 360)
        cls.motion = cls.renderer.model

    def test_one_linear_clock_without_per_stroke_stops(self):
        m = self.motion
        times = np.linspace(-1, 2, 301)
        np.testing.assert_array_equal(m.phase(times), times*m.cycles)
        np.testing.assert_array_equal(m.phase_rate(times), np.full(len(times), m.cycles))
        np.testing.assert_allclose(m.angles(1)-m.angles(0), 2*np.pi*m.frequencies)
        np.testing.assert_allclose(m.chain(0), m.chain(1), atol=1e-12)

    def test_reported_speed_matches_actual_continuous_angles(self):
        m = self.motion
        expected = m.frequencies/m.loop_seconds
        for t in np.linspace(0, 1, 27):
            h = 1e-7
            actual = (m.angles(t+h)-m.angles(t-h))/(4*np.pi*h*m.loop_seconds)
            np.testing.assert_allclose(m.parameters(t)[2], actual, atol=1e-6, rtol=1e-6)
            np.testing.assert_array_equal(m.parameters(t)[2], expected)

    def test_pen_is_always_at_the_actual_vector_sum(self):
        r, m = self.renderer, self.motion
        blank = np.zeros_like(r.background)
        with patch.object(r, "draw_dial"), patch.object(r, "draw_pointer"), \
             patch.object(r.fading, "layer", return_value=blank), \
             patch.object(r, "circle_colors", [[0, 0, 0]]*len(m.radii)):
            for t in (.01, .13, .31, .59, .83):
                exact = np.sum(m.base.coefficients*np.exp(2j*np.pi*m.frequencies*t))
                self.assertAlmostEqual(abs(m.chain(t)[-1]-exact), 0, places=11)
                x, y = np.rint(r.pixels([exact])[0]).astype(int)
                np.testing.assert_array_equal(r.frame(t)[y, x], [255, 255, 255])

    def test_original_font_silhouette_and_ordered_main_strokes(self):
        # Font mapping remains explicitly experimental; it is not the default.
        # Stroke count now depends on the font skeleton, since no per-name
        # calibration file ships with the repository.
        args = build_parser().parse_args(["--headless", "--text-style", "signature"])
        m = make_renderer(args, 640, 360).model
        self.assertGreater(sum(s.kind == "stroke" for s in m.path.segments), 0)
        font = ImageFont.truetype("C:/Windows/Fonts/STXINGKA.TTF", 240)
        top = min(font.getbbox(c)[1] for c in "傅里叶")
        original = Image.new("L", (m.path.mask.shape[1], m.path.mask.shape[0]))
        draw, x = ImageDraw.Draw(original), 60.
        for char in "傅里叶":
            draw.text((round(x), 60-top), char, font=font, fill=255)
            x += font.getlength(char)+20
        np.testing.assert_array_equal(m.path.mask, np.asarray(original)>=128)

    def test_font_mask_cannot_bypass_the_reconstruction(self):
        r, m = self.renderer, self.motion
        expected = r.frame(.31)
        with patch.object(m.path, "mask", np.ones_like(m.path.mask)):
            np.testing.assert_array_equal(expected, r.frame(.31))
        r.frame(.73)
        np.testing.assert_array_equal(expected, r.frame(.31))

    def test_two_eighteen_and_180_disks_have_different_reconstruction_errors(self):
        m = self.motion
        points = resample_closed(m.path.points, 2000)
        target = (points[:, 0]-1j*points[:, 1]-m.origin)*m.scale
        times = np.arange(len(target))/len(target)
        errors = []
        for count in (2, 18, 180):
            base = FourierModel(target, count)
            motion = SignatureMotion(base, m.path, 60, 1)
            r = Renderer(motion, 640, 360, "draw")
            self.assertEqual(len(base.radii), count)
            expected = np.exp(2j*np.pi*np.outer(times, base.frequencies)) @ base.coefficients
            np.testing.assert_allclose(motion.curve(times), expected, atol=1e-12)
            np.testing.assert_allclose(r.trace, r.pixels(motion.curve(
                np.linspace(0, 1, r.trace_count+1))), atol=1e-12)
            errors.append(np.sqrt(np.mean(np.abs(expected-target)**2)))
        self.assertGreater(errors[0], errors[1]*3)
        self.assertGreater(errors[1], errors[2]*3)

    def test_pen_up_only_controls_ink_and_does_not_change_the_mechanism(self):
        m = self.motion
        joined = SignatureMotion(m.base, m.path, m.loop_seconds, m.cycles, show_transfers=True)
        times = np.linspace(-.1, 1.1, 400)
        np.testing.assert_array_equal(m.curve(times), joined.curve(times))
        for i, segment in enumerate(m.path.segments):
            start, end = m.arc[i:i+2]/m.cycles
            if end-start < 1e-12:
                continue  # A zero-length transfer has no time interval to mask.
            self.assertEqual(bool(m.visible((start+end)/2)), segment.kind!="lift")
            self.assertTrue(joined.visible_between(start, end))
            if segment.kind=="lift":
                self.assertFalse(m.visible_between(start-1e-6, end+1e-6))
            np.testing.assert_array_equal(m.chain(start), joined.chain(start))
            np.testing.assert_array_equal(m.parameters(start)[2], joined.parameters(start)[2])

    def test_ink_never_draws_a_line_across_a_pen_up_interval(self):
        m = self.motion
        r = Renderer(m, 640, 360, "loop")
        times = np.linspace(0, 1, r.trace_count+1)
        expected = m.visible_between(times[:-1], times[1:])
        actual = np.zeros(r.trace_count, bool)
        for start, end in r.trace_runs:
            actual[start:end] = True
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(actual.any())
        self.assertFalse(actual.all())

    def test_fading_changes_brightness_without_tapering_the_pen_width(self):
        r = self.renderer
        trail = FadingTrail(self.motion, 8, 2, 0, constant_width=True)
        # Isolated horizontal segments with different ages, identical geometry.
        with patch.object(r, "line_width", 6):
            points = np.array([[100., 180.], [400., 180.]])
            trail.ages = np.array([6., 6.])
            old = trail.emission(r, .3, points=points)[:, 250]
            trail.ages = np.array([2., 2.])
            new = trail.emission(r, .3, points=points)[:, 250]
            self.assertGreater(new.max(), old.max())
            np.testing.assert_allclose(old/old.max(), new/new.max(), atol=1e-6)

    def test_gradient_ring_rotates_and_small_rings_fade_continuously(self):
        r = self.renderer
        center = np.array([320., 180.])
        with patch.object(r, "ticks", 0):
            a, b = np.zeros_like(r.background), np.zeros_like(r.background)
            r.draw_dial(a, center, 80, 0, 0)
            r.draw_dial(b, center, 80, 0, np.pi/2)
            values = np.array([a[180, 400].sum(), a[260, 320].sum(), a[180, 240].sum(), a[100, 320].sum()])
            self.assertGreater(np.ptp(values), 40)
            self.assertGreater(cv2.norm(a, b), 50)
            c, d = np.zeros_like(a), np.zeros_like(a)
            r.draw_dial(c, center, 1.0, 0)
            r.draw_dial(d, center, 1.01, 0)
            self.assertLess(np.abs(c.astype(int)-d.astype(int)).max(), 5)

    def test_fast_teeth_lose_contrast_before_they_alias(self):
        r = self.renderer
        center = np.array([320., 180.])
        for speed, minimum, maximum in ((0, 40, 256), (200, 0, 10)):
            with_teeth, plain = np.zeros_like(r.background), np.zeros_like(r.background)
            with patch.object(r, "ticks", 12):
                r.draw_dial(with_teeth, center, 80, 0, .2, speed)
            with patch.object(r, "ticks", 0):
                r.draw_dial(plain, center, 80, 0, .2, speed)
            difference = np.abs(with_teeth.astype(int)-plain.astype(int)).max()
            self.assertGreaterEqual(difference, minimum)
            self.assertLess(difference, maximum)


if __name__ == "__main__":
    unittest.main()
