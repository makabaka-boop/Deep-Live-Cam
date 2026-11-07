"""Qt tests for the desktop face swap range feature.

Covers the user-visible contract:
* the current preview frame can be marked as start/end (both included),
* marks can be cleared, restoring whole-video processing,
* preview only runs swap/enhance processors inside the marked range,
* the range controls only appear for eligible CFR single-track MP4 targets,
* choosing a different target clears the old marks,
* an invalid range blocks export (with a reason) before the save dialog.
"""
import os
import sys
import types
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import modules.globals  # noqa: E402
import modules.ui as ui  # noqa: E402

_APP = None


def setUpModule():
    global _APP
    _APP = QApplication.instance() or QApplication([])


class _RecordingProcessor:
    """Fills processed frames with a marker and logs each invocation."""

    def __init__(self, call_log):
        self._call_log = call_log

    def process_frame(self, source_face, frame, target_face=None):
        self._call_log.append("processed")
        frame[:] = 9
        return frame


def _install_frame_core_stub(call_log):
    """refresh_frame() lazily imports modules.processors.frame.core."""
    fake = types.ModuleType("modules.processors.frame.core")
    fake.get_frame_processors_modules = lambda _names: [
        _RecordingProcessor(call_log)
    ]
    return mock.patch.dict(
        sys.modules, {"modules.processors.frame.core": fake}, clear=False
    )


class PreviewRangeControlTests(unittest.TestCase):
    def setUp(self):
        modules.globals.source_path = "/tmp/source.jpg"
        modules.globals.target_path = "/tmp/clip.mp4"
        modules.globals.map_faces = False
        modules.globals.nsfw_filter = False
        modules.globals.frame_processors = ["face_swapper"]
        ui.clear_face_swap_range()

        self.frame_reads = []

        def fake_get_video_frame(_path, position):
            self.frame_reads.append(position)
            return np.zeros((16, 16, 3), dtype=np.uint8)

        patches = [
            mock.patch.object(ui, "is_face_swap_range_supported",
                              return_value=True),
            mock.patch.object(ui, "get_video_frame_total", return_value=100),
            mock.patch.object(ui, "get_video_frame", fake_get_video_frame),
            mock.patch.object(ui, "imread_unicode",
                              lambda _path: np.zeros((16, 16, 3), np.uint8)),
            mock.patch.object(ui, "get_one_face", lambda _frame: object()),
            # The synthetic target path does not exist on disk.
            mock.patch.object(ui, "is_image", return_value=False),
            mock.patch.object(ui, "is_video", return_value=True),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

        self.call_log = []
        core_patch = _install_frame_core_stub(self.call_log)
        core_patch.start()
        self.addCleanup(core_patch.stop)

        self.window = ui.PreviewWindow()
        self.addCleanup(self.window.deleteLater)
        self.window.show()
        self.addCleanup(self.window.hide)
        self.window.init_for_target()

    def test_range_controls_show_for_eligible_target(self):
        self.assertTrue(self.window._range_supported)
        # show() was called even though the top-level window is not shown.
        self.assertFalse(self.window._range_bar.isHidden())

    def test_marking_current_frame_sets_inclusive_bounds(self):
        # Slider position 5 -> 0-based decode index 4.
        self.window._slider.setValue(5)
        self.window._on_mark_start()
        self.assertEqual(modules.globals.face_swap_start_frame, 4)
        self.assertEqual(modules.globals.face_swap_range_target,
                         "/tmp/clip.mp4")

        # Slider position 8 -> 0-based decode index 7.
        self.window._slider.setValue(8)
        self.window._on_mark_end()
        self.assertEqual(modules.globals.face_swap_end_frame, 7)

        label = self.window._range_label.text()
        self.assertIn("5", label)
        self.assertIn("8", label)

    def test_preview_processes_only_frames_inside_range(self):
        self.window._slider.setValue(5)
        self.window._on_mark_start()  # index 4
        self.window._slider.setValue(8)
        self.window._on_mark_end()    # index 7
        self.call_log.clear()

        # Position 6 -> decode index 5 (inside 4..7): processed.
        self.window.refresh_frame(6)
        self.assertEqual(self.call_log, ["processed"])

        # Position 9 -> decode index 8 (outside): shown as original.
        self.call_log.clear()
        self.window.refresh_frame(9)
        self.assertEqual(self.call_log, [])

        # Position 5 -> decode index 4 (start boundary, included).
        self.call_log.clear()
        self.window.refresh_frame(5)
        self.assertEqual(self.call_log, ["processed"])

    def test_clear_restores_whole_video_preview(self):
        self.window._slider.setValue(5)
        self.window._on_mark_start()
        self.window._slider.setValue(8)
        self.window._on_mark_end()
        self.assertTrue(modules.globals.face_swap_start_frame is not None)

        self.window._on_clear_range()
        self.assertIsNone(modules.globals.face_swap_start_frame)
        self.assertIsNone(modules.globals.face_swap_end_frame)
        self.assertIsNone(modules.globals.face_swap_range_target)

        self.call_log.clear()
        self.window.refresh_frame(9)
        self.assertEqual(self.call_log, ["processed"])


class IneligibleTargetTests(unittest.TestCase):
    def test_range_controls_hidden_for_unsupported_target(self):
        modules.globals.source_path = "/tmp/source.jpg"
        modules.globals.target_path = "/tmp/clip.mkv"
        modules.globals.map_faces = False
        modules.globals.nsfw_filter = False
        ui.clear_face_swap_range()

        with mock.patch.object(ui, "is_face_swap_range_supported",
                               return_value=False), \
             mock.patch.object(ui, "is_image", return_value=False), \
             mock.patch.object(ui, "is_video", return_value=True), \
             mock.patch.object(ui, "get_video_frame_total", return_value=10):
            window = ui.PreviewWindow()
            self.addCleanup(window.deleteLater)
            window.show()
            self.addCleanup(window.hide)
            window.init_for_target()
            self.assertFalse(window._range_supported)
            self.assertTrue(window._range_bar.isHidden())


class TargetChangeClearsMarksTests(unittest.TestCase):
    def test_selecting_new_target_clears_range_marks(self):
        ui.clear_face_swap_range()
        modules.globals.target_path = "/tmp/old.mp4"
        modules.globals.face_swap_start_frame = 10
        modules.globals.face_swap_end_frame = 40
        modules.globals.face_swap_range_target = "/tmp/old.mp4"

        stub_self = types.SimpleNamespace(
            target_label=types.SimpleNamespace(
                setPixmap=lambda _pm: None,
                setText=lambda _text: None,
                clear=lambda: None,
            )
        )
        with mock.patch.object(ui.QFileDialog, "getOpenFileName",
                               return_value=("/tmp/new.mp4", "")), \
             mock.patch.object(ui, "is_image", return_value=False), \
             mock.patch.object(ui, "is_video", return_value=True), \
             mock.patch.object(ui, "render_video_preview",
                               return_value=None), \
             mock.patch.object(ui, "render_image_preview",
                               return_value=None):
            ui.MainWindow._on_select_target(stub_self)

        self.assertEqual(modules.globals.target_path, "/tmp/new.mp4")
        self.assertIsNone(modules.globals.face_swap_start_frame)
        self.assertIsNone(modules.globals.face_swap_end_frame)
        self.assertIsNone(modules.globals.face_swap_range_target)


class ExportGuardTests(unittest.TestCase):
    def setUp(self):
        ui.clear_face_swap_range()
        modules.globals.target_path = "/tmp/clip.mp4"
        modules.globals.map_faces = False
        self.started = []
        self.stub_self = types.SimpleNamespace(
            _start_cb=lambda: self.started.append(True)
        )
        # The synthetic target path does not exist on disk.
        for patcher in (
            mock.patch.object(ui, "is_image", return_value=False),
            mock.patch.object(ui, "is_video", return_value=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _start(self):
        ui.MainWindow._select_output_and_start(self.stub_self)

    def test_invalid_range_blocks_save_dialog_and_export(self):
        modules.globals.face_swap_start_frame = 40
        modules.globals.face_swap_end_frame = 10
        modules.globals.face_swap_range_target = "/tmp/clip.mp4"

        with mock.patch.object(ui, "validate_face_swap_range",
                               return_value="The start frame cannot be later "
                                            "than the end frame."), \
             mock.patch.object(ui.QFileDialog, "getSaveFileName",
                               side_effect=AssertionError("dialog must not open")), \
             mock.patch.object(ui.QMessageBox, "warning") as warning:
            self._start()

        self.assertEqual(self.started, [])
        self.assertEqual(warning.call_count, 1)

    def test_valid_range_opens_dialog_and_starts_export(self):
        modules.globals.face_swap_start_frame = 2
        modules.globals.face_swap_end_frame = 9
        modules.globals.face_swap_range_target = "/tmp/clip.mp4"

        with mock.patch.object(ui, "validate_face_swap_range",
                               return_value=None), \
             mock.patch.object(
                 ui.QFileDialog, "getSaveFileName",
                 return_value=("/tmp/out.mp4", "")):
            self._start()

        self.assertEqual(self.started, [True])
        self.assertEqual(modules.globals.output_path, "/tmp/out.mp4")

    def test_dialog_cancel_does_not_start(self):
        with mock.patch.object(ui, "validate_face_swap_range",
                               return_value=None), \
             mock.patch.object(ui.QFileDialog, "getSaveFileName",
                               return_value=("", "")):
            self._start()
        self.assertEqual(self.started, [])


if __name__ == "__main__":
    unittest.main()
