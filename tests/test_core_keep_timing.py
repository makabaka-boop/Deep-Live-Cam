import importlib
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch


class _Timing:
    def __init__(self, count=3, duration=0.1, has_audio=True):
        self.durations = [0.05, 0.03, 0.02][:count]
        while len(self.durations) < count:
            self.durations.append(0.02)
        self.pts = [0.0] * count
        self.duration = duration
        self.has_audio = has_audio


@contextmanager
def _patched_core(calls, errors, *, timing=None, probe_reason=None,
                  frame_count=3, verify_reason=None, audio_mux_ok=True):
    class Processor:
        NAME = "test_processor"

        def pre_start(self):
            return True

        def pre_check(self):
            return True

        def process_video(self, source_path, frame_paths):
            calls.append(("process_video", source_path, tuple(frame_paths)))

    def _probe(_path):
        calls.append(("probe",))
        if probe_reason is not None:
            return None, probe_reason
        return timing if timing is not None else _Timing(has_audio=False), None

    utilities_stub = types.SimpleNamespace(
        has_image_extension=lambda _path: False,
        is_image=lambda _path: False,
        is_video=lambda _path: True,
        detect_fps=lambda _path: 30.0,
        create_video=lambda *_a, **_k: True,
        create_video_vfr=lambda target, durations: calls.append(
            ("create_video_vfr", len(durations))
        ) or True,
        probe_video_timing=_probe,
        verify_video_timing=lambda *_a, **_k: calls.append(("verify",)) or verify_reason,
        restore_audio=lambda *_a, **_k: calls.append(("restore_audio",)),
        restore_audio_timed=lambda *_a, **_k: calls.append(("restore_audio_timed",)) or audio_mux_ok,
        extract_frames=lambda target: calls.append(("extract_frames",)),
        get_temp_frame_paths=lambda _target: [f"{index:04d}.png" for index in range(1, frame_count + 1)],
        get_temp_output_path=lambda target: os.path.join(target, "temp", "temp.mp4"),
        create_temp=lambda target: calls.append(("create_temp",)),
        move_temp=lambda target, output: calls.append(("move_temp", output)),
        clean_temp=lambda target: calls.append(("clean_temp",)),
        normalize_output_path=lambda _s, _t, output: output,
    )

    stubs = {
        "cv2": types.SimpleNamespace(
            IMREAD_COLOR=1,
            imdecode=lambda *_a, **_k: None,
            imencode=lambda *_a, **_k: (True, types.SimpleNamespace(tofile=lambda *_x: None)),
        ),
        "numpy": types.SimpleNamespace(uint8=object, fromfile=lambda *_a, **_k: b""),
        "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
        "onnxruntime": types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]),
        "tensorflow": types.SimpleNamespace(),
        "modules.metadata": types.SimpleNamespace(name="Deep-Live-Cam", version="test"),
        "modules.ui": types.SimpleNamespace(
            check_and_ignore_nsfw=lambda *_a, **_k: False,
            update_status=lambda *_a, **_k: calls.append(("status",)),
            show_error=lambda message, *_a, **_k: errors.append(message),
            init=lambda *_a, **_k: types.SimpleNamespace(mainloop=lambda: None),
        ),
        "modules.processors.frame.core": types.SimpleNamespace(
            get_frame_processors_modules=lambda _names: [Processor()],
            process_video_in_memory=lambda *_a, **_k: True,
        ),
        "modules.utilities": utilities_stub,
    }
    with patch.dict(sys.modules, stubs, clear=False):
        sys.modules.pop("modules.core", None)
        yield importlib.import_module("modules.core")
        sys.modules.pop("modules.core", None)


def _configure(core, **overrides):
    g = core.modules.globals
    g.source_path = "source.jpg"
    g.target_path = "target.mp4"
    g.output_path = "output.mp4"
    g.frame_processors = ["face_swapper"]
    g.headless = True
    g.keep_fps = False
    g.keep_timing = True
    g.keep_audio = True
    g.keep_frames = False
    g.map_faces = False
    g.nsfw_filter = False
    g.many_faces = False
    g.execution_threads = 1
    g.execution_providers = ["CPUExecutionProvider"]
    g.max_memory = None
    for key, value in overrides.items():
        setattr(g, key, value)


class KeepTimingExportTests(unittest.TestCase):
    def test_unreliable_timing_aborts_before_any_processing(self):
        calls, errors = [], []
        with _patched_core(calls, errors, probe_reason="boom") as core:
            _configure(core)
            core.start()

        self.assertTrue(errors)
        self.assertIn("boom", errors[0])
        self.assertNotIn(("create_temp",), calls)
        self.assertNotIn(("extract_frames",), calls)
        self.assertFalse(os.path.exists("output.mp4"))

    def test_only_supports_mp4(self):
        calls, errors = [], []
        with _patched_core(calls, errors) as core:
            _configure(core, target_path="target.mkv")
            with patch.object(core.os.path, "splitext", return_value=("target", ".mkv")):
                core.start()
        self.assertTrue(errors)
        self.assertNotIn(("probe",), calls)

    def test_map_faces_is_rejected(self):
        calls, errors = [], []
        with _patched_core(calls, errors) as core:
            _configure(core, map_faces=True)
            core.start()
        self.assertTrue(errors)
        self.assertNotIn(("probe",), calls)

    def test_frame_count_mismatch_aborts_without_output(self):
        calls, errors = [], []
        with _patched_core(calls, errors, timing=_Timing(count=3), frame_count=2) as core:
            _configure(core)
            core.start()

        self.assertTrue(errors)
        self.assertIn(("clean_temp",), calls)
        self.assertNotIn(("create_video_vfr", 3), calls)
        self.assertNotIn(("move_temp", "output.mp4"), calls)

    def test_encoded_timing_mismatch_aborts(self):
        calls, errors = [], []
        with _patched_core(calls, errors, timing=_Timing(has_audio=False),
                           verify_reason="bad duration") as core:
            _configure(core)
            core.start()
        self.assertTrue(any("bad duration" in e for e in errors))
        self.assertIn(("clean_temp",), calls)
        self.assertNotIn(("move_temp", "output.mp4"), calls)

    def test_happy_path_keeps_audio_in_sync(self):
        calls, errors = [], []
        with _patched_core(calls, errors, timing=_Timing(has_audio=True)) as core:
            _configure(core, keep_audio=True)
            with patch.object(core.os.path, "isfile", return_value=True):
                core.start()

        self.assertEqual(errors, [])
        step = {call[0]: index for index, call in enumerate(calls)}
        self.assertLess(step["create_temp"], step["extract_frames"])
        self.assertLess(step["extract_frames"], step["process_video"])
        self.assertLess(step["process_video"], step["create_video_vfr"])
        self.assertLess(step["create_video_vfr"], step["verify"])
        self.assertLess(step["verify"], step["restore_audio_timed"])
        self.assertIn(("create_video_vfr", 3), calls)
        self.assertNotIn(("move_temp", "output.mp4"), calls)

    def test_unchecked_audio_produces_silent_video(self):
        calls, errors = [], []
        with _patched_core(calls, errors, timing=_Timing(has_audio=True)) as core:
            _configure(core, keep_audio=False)
            with patch.object(core.os.path, "isfile", return_value=True):
                core.start()

        self.assertEqual(errors, [])
        self.assertIn(("move_temp", "output.mp4"), calls)
        self.assertNotIn(("restore_audio_timed",), calls)

    def test_missing_source_audio_produces_silent_video(self):
        calls, errors = [], []
        with _patched_core(calls, errors, timing=_Timing(has_audio=False)) as core:
            _configure(core, keep_audio=True)
            with patch.object(core.os.path, "isfile", return_value=True):
                core.start()

        self.assertIn(("move_temp", "output.mp4"), calls)
        self.assertNotIn(("restore_audio_timed",), calls)

    def test_audio_mux_failure_removes_partial_output(self):
        calls, errors = [], []
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "output.mp4")
            with open(output, "wb") as handle:
                handle.write(b"partial")
            with _patched_core(calls, errors, timing=_Timing(has_audio=True),
                               audio_mux_ok=False) as core:
                _configure(core, output_path=output)
                with patch.object(core.os.path, "isfile", return_value=True):
                    core.start()
            self.assertTrue(errors)
            self.assertFalse(os.path.exists(output))


if __name__ == "__main__":
    unittest.main()
