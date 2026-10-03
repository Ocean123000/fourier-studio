"""使用真实 Tk 控件和事件循环验证交互流程，不依赖人工点击。"""
from pathlib import Path
import tempfile
import gc
import threading
import time
import tkinter as tk
import unittest
from unittest.mock import patch
import cv2

from gui import FourierApp
from main import ROOT


class GuiTests(unittest.TestCase):
    def test_transfer_visibility_option_is_passed_to_preview_settings(self):
        self.app.open_controls()
        self.assertFalse(self.app.read_settings().show_transfers)
        self.app.show_transfers_var.set(True)
        self.assertTrue(self.app.read_settings().show_transfers)

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = FourierApp(self.root)
        self.errors = []
        self.error_patch = patch("gui.messagebox.showerror",
                                 side_effect=lambda *args, **kwargs: self.errors.append(args))
        self.error_patch.start()
        self.app.launch_button.invoke()
        self.app.controls.withdraw()
        self.root.update()

    def tearDown(self):
        self.app.shutdown()
        if self.app.export_thread:
            self.app.export_thread.join(timeout=10)
        self.error_patch.stop()
        del self.app
        gc.collect()  # Destroy Tk reference cycles on their owning thread.

    def pump_until(self, predicate, timeout=20):
        end = time.perf_counter()+timeout
        while not predicate() and time.perf_counter()<end:
            self.root.update()
            time.sleep(.015)
        self.root.update()
        self.assertTrue(predicate(), self.app.status.get())
        self.assertFalse(self.errors)

    def small_settings(self):
        for name, value in {"samples":128, "circles":8, "max_frequency":8,
                            "duration":2, "fps":10, "trail_length":1}.items():
            self.app.values[name].set(value)

    def test_controls_wait_for_preview_click(self):
        end = time.perf_counter() + .25
        while time.perf_counter() < end:
            self.root.update()
            time.sleep(.01)
        self.assertFalse(self.app.building)
        self.assertIsNone(self.app.renderer)
        self.assertNotIn("disabled", self.app.apply_button.state())
        self.assertEqual(self.app.preview_frames, 0)

    def test_font_choices_apply_to_preview(self):
        self.small_settings()
        self.app.values["text_style"].set("signature (行书字体映射·实验)")
        self.app.style_combo.event_generate("<<ComboboxSelected>>")
        self.root.update()
        for choice in ("志莽行书（连笔）", "龙藏体（草书）", "刘建毛草（草书）"):
            self.assertIn(choice, self.app.font_combo["values"])
            self.app.font_preset.set(choice)
            self.app.font_combo.event_generate("<<ComboboxSelected>>")
            self.root.update()
            self.assertEqual(Path(self.app.read_settings().font), self.app.available_fonts[choice])
            previous = self.app.build_id
            self.app.apply_button.invoke()
            self.pump_until(lambda: not self.app.building and self.app.renderer is not None)
            self.assertEqual(self.app.build_id, previous + 1)
            self.assertEqual(self.app.active_settings.font, str(self.app.available_fonts[choice]))
            self.assertNotIn("disabled", self.app.apply_button.state())

    def test_default_uses_ordered_data_and_font_controls_follow_style(self):
        self.assertEqual(self.app.read_settings().text_style, "ordered")
        for control in (self.app.font_combo, self.app.font_entry, self.app.font_browse_button):
            self.assertIn("disabled", control.state())
        self.app.values["text_style"].set("contour (字体外轮廓)")
        self.app.style_combo.event_generate("<<ComboboxSelected>>")
        self.root.update()
        for control in (self.app.font_combo, self.app.font_entry, self.app.font_browse_button):
            self.assertNotIn("disabled", control.state())

    def test_default_preview_click_and_reapply(self):
        self.app.apply_button.invoke()
        self.pump_until(lambda: self.app.preview_frames >= 3)
        self.assertEqual(self.app.active_settings.samples, 2000)
        self.assertEqual(self.app.active_settings.circles, 180)
        self.assertNotIn("disabled", self.app.apply_button.state())
        first = self.app.last_frame.copy()
        frames = self.app.preview_frames
        self.pump_until(lambda: self.app.preview_frames >= frames + 3)
        self.assertGreater(cv2.norm(first, self.app.last_frame), 0)
        self.app.pause_button.invoke()
        self.assertFalse(self.app.playing)
        build_id = self.app.build_id
        self.app.apply_button.invoke()
        self.pump_until(lambda: not self.app.building and self.app.playing)
        self.assertEqual(self.app.build_id, build_id + 1)
        self.assertNotIn("disabled", self.app.apply_button.state())

    def test_slow_render_keeps_buttons_responsive(self):
        self.small_settings()
        self.app.apply_button.invoke()
        self.pump_until(lambda: self.app.preview_frames > 0)
        started, release = threading.Event(), threading.Event()
        renderer = self.app.renderer
        original_frame = renderer.frame
        def slow_frame(phase):
            started.set()
            release.wait(3)
            return original_frame(phase)
        with patch.object(renderer, "frame", side_effect=slow_frame):
            try:
                self.pump_until(started.is_set)
                self.assertTrue(self.app.preview_pending)
                clicked = threading.Event()
                def pause():
                    self.app.pause_button.invoke()
                    clicked.set()
                self.root.after(20, pause)
                self.pump_until(clicked.is_set, timeout=.5)
                self.assertFalse(self.app.playing)
                self.assertFalse(release.is_set())
                self.assertNotIn("disabled", self.app.apply_button.state())
            finally:
                release.set()
            self.pump_until(lambda: not self.app.preview_pending)

    def test_preparation_timeout_allows_retry_and_discards_late_result(self):
        self.small_settings()
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        def stalled_build(*args):
            started.set()
            release.wait(3)
            finished.set()
            return object()
        with patch("gui.make_renderer", side_effect=stalled_build):
            try:
                self.app.PREVIEW_BUILD_TIMEOUT = .1
                self.app.apply_button.invoke()
                self.pump_until(started.is_set)
                self.pump_until(lambda: not self.app.building, timeout=1)
                self.assertNotIn("disabled", self.app.apply_button.state())
                self.assertIsNone(self.app.renderer)
            finally:
                release.set()
            self.pump_until(finished.is_set)
            self.root.update()
            self.assertIsNone(self.app.renderer)
        self.app.PREVIEW_BUILD_TIMEOUT = 30
        self.app.apply_button.invoke()
        self.pump_until(lambda: self.app.preview_frames > 0)

    def test_worker_exit_restores_preview_button(self):
        with patch("gui.make_renderer", side_effect=SystemExit("missing test dependency")):
            with patch("gui.traceback.print_exc"), patch("gui.messagebox.showerror") as error:
                self.app.apply_button.invoke()
                self.pump_until(lambda: not self.app.building)
                self.assertNotIn("disabled", self.app.apply_button.state())
                error.assert_called_once()
        self.small_settings()
        self.app.apply_button.invoke()
        self.pump_until(lambda: self.app.preview_frames > 0)

    def test_image_picker_and_preview_pause(self):
        self.assertIsNotNone(self.app.controls)
        with patch("gui.filedialog.askopenfilename", return_value=str(ROOT/"assets"/"demo.svg")):
            self.app.browse_button.invoke()
        self.assertEqual(Path(self.app.values["input"].get()), ROOT/"assets"/"demo.svg")
        self.small_settings()
        self.app.apply_button.invoke()
        self.pump_until(lambda: self.app.renderer is not None and self.app.preview_frames > 0)
        self.assertEqual(self.app.last_frame.shape, (450, 800, 3))
        self.assertIsNotNone(self.app.photo)
        self.app.pause_button.invoke()
        self.assertFalse(self.app.playing)
        self.root.update()
        before = self.app.scene_seconds
        time.sleep(.06)
        self.root.update()
        self.assertEqual(before, self.app.scene_seconds)
        self.app.pause_button.invoke()
        self.assertTrue(self.app.playing)

    def test_random_regenerate_and_export(self):
        self.small_settings()
        self.app.source.set("random")
        self.app._source_changed()
        seed = int(self.app.values["seed"].get())
        self.app.regenerate_button.invoke()
        self.pump_until(lambda: self.app.renderer is not None)
        self.assertEqual(int(self.app.values["seed"].get()), seed+1)
        self.app.active_settings.width, self.app.active_settings.height = 160, 96
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"gui.mp4"
            self.app.export_video(output)
            self.assertFalse(self.app.playing)
            self.pump_until(lambda: not self.app.exporting, timeout=30)
            self.assertTrue(self.app.playing)
            self.assertTrue(output.is_file(), self.app.status.get())
            self.assertEqual(self.app.progress.get(), 100)
            cap = cv2.VideoCapture(str(output))
            self.assertTrue(cap.isOpened())
            self.assertEqual(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 20)
            cap.release()

    def test_preview_is_suspended_and_eta_shown_during_export(self):
        self.small_settings()
        self.app.apply_button.invoke()
        self.pump_until(lambda: self.app.renderer is not None and self.app.preview_frames > 0)
        started, release = threading.Event(), threading.Event()
        def slow_export(*args, progress, **kwargs):
            progress(1, 20)
            started.set()
            release.wait(3)
        with tempfile.TemporaryDirectory() as directory:
            with patch("gui.encode_video", side_effect=slow_export):
                try:
                    frames = self.app.preview_frames
                    self.app.export_video(Path(directory)/"example.mp4")
                    self.pump_until(lambda: started.is_set() and "预计剩余" in self.app.status.get())
                    for _ in range(3):
                        time.sleep(.025)
                        self.root.update()
                    self.assertEqual(self.app.preview_frames, frames)
                    self.assertFalse(self.app.playing)
                    self.assertLess(self.app.progress.get(), 100)
                finally:
                    release.set()
                self.pump_until(lambda: not self.app.exporting)
                self.assertTrue(self.app.playing)


if __name__ == "__main__":
    unittest.main()
