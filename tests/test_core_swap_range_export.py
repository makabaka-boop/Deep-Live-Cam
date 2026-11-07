"""Tests for export-time face swap range handling in modules.core.start.

Covers the desktop contract:
* an incomplete/reversed range stops the export before any temp work,
* a valid range flows into the in-memory pipeline,
* on disk fallback only the marked frame files are processed while every
  extracted frame is still reassembled,
* map_faces exports ignore range marks entirely.
"""
import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock


def _fake_globals():
    globals_module = types.ModuleType("modules.globals")
    globals_module.source_path = "source.jpg"
    globals_module.target_path = "target.mp4"
    globals_module.output_path = "output.mp4"
    globals_module.frame_processors = ["face_swapper"]
    # Non-headless so core.update_status forwards messages to the (stubbed)
    # UI layer, which is what the user-facing failure contract goes through.
    globals_module.headless = False
    globals_module.keep_fps = False
    globals_module.keep_audio = False
    globals_module.keep_frames = False
    globals_module.map_faces = False
    globals_module.nsfw_filter = False
    globals_module.execution_threads = 2
    globals_module.execution_providers = ["CPUExecutionProvider"]
    globals_module.max_memory = None
    globals_module.face_swap_start_frame = None
    globals_module.face_swap_end_frame = None
    globals_module.face_swap_range_target = "target.mp4"
    return globals_module


def _load_real_utilities(globals_module):
    """Load the real utilities module bound to the fake globals."""
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.imdecode = lambda *a, **k: None
    fake_cv2.imencode = lambda *a, **k: (
        True, types.SimpleNamespace(tofile=lambda: None)
    )
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.uint8 = int
    fake_numpy.fromfile = lambda *a, **k: b""
    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, *a, **k: iterable

    with mock.patch.dict(
        sys.modules,
        {
            "modules.globals": globals_module,
            "cv2": fake_cv2,
            "numpy": fake_numpy,
            "tqdm": fake_tqdm,
        },
        clear=False,
    ):
        sys.modules.pop("modules", None)
        sys.modules.pop("modules.utilities", None)
        module = importlib.import_module("modules.utilities")
        module.modules.globals = globals_module
        return module


@contextmanager
def _patched_core(calls, statuses, pipe_result):
    globals_module = _fake_globals()
    real_utils = _load_real_utilities(globals_module)

    class Processor:
        NAME = "face_swapper"

        def pre_start(self):
            return True

        def pre_check(self):
            return True

        def process_video(self, source_path, frame_paths):
            calls.append(
                ("process_video", source_path, tuple(frame_paths))
            )

    def ten_frame_paths(target_path):
        # Deliberately unsorted: range filtering must re-establish order.
        names = [
            "0010.png", "0002.png", "0007.png", "0001.png", "0004.png",
            "0009.png", "0003.png", "0005.png", "0008.png", "0006.png",
        ]
        return [f"{target_path}/{name}" for name in names]

    stub_utils = types.ModuleType("modules.utilities")
    # Real range logic (validation, mark binding, file selection)...
    for name in dir(real_utils):
        if not name.startswith("__"):
            setattr(stub_utils, name, getattr(real_utils, name))
    # ...with every FFmpeg/filesystem side effect replaced by a recorder.
    stub_utils.has_image_extension = lambda _path: False
    stub_utils.is_image = lambda _path: False
    stub_utils.is_video = lambda _path: True
    stub_utils.detect_fps = lambda _path: 24.0
    stub_utils.create_video = lambda target_path, fps: calls.append(
        ("create_video", target_path, fps)
    ) or True
    stub_utils.extract_frames = lambda target_path: calls.append(
        ("extract_frames", target_path)
    )
    stub_utils.get_temp_frame_paths = ten_frame_paths
    stub_utils.restore_audio = lambda *a, **k: calls.append(("restore_audio",))
    stub_utils.create_temp = lambda target_path: calls.append(
        ("create_temp", target_path)
    )
    stub_utils.move_temp = lambda target_path, output_path: calls.append(
        ("move_temp", target_path, output_path)
    )
    stub_utils.clean_temp = lambda target_path: calls.append(
        ("clean_temp", target_path)
    )
    stub_utils.normalize_output_path = lambda _s, _t, output: output

    stubs = {
        "cv2": types.SimpleNamespace(
            IMREAD_COLOR=1,
            imdecode=lambda *a, **k: None,
            imencode=lambda *a, **k: (
                True, types.SimpleNamespace(tofile=lambda: None)
            ),
        ),
        "numpy": types.SimpleNamespace(uint8=object, fromfile=lambda *a, **k: b""),
        "torch": types.SimpleNamespace(
            cuda=types.SimpleNamespace(empty_cache=lambda: None)
        ),
        "onnxruntime": types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"]
        ),
        "tensorflow": types.SimpleNamespace(),
        "modules.globals": globals_module,
        "modules.metadata": types.SimpleNamespace(
            name="Deep-Live-Cam", version="test"
        ),
        "modules.ui": types.SimpleNamespace(
            check_and_ignore_nsfw=lambda *a, **k: False,
            update_status=lambda message: statuses.append(message),
            init=lambda *a, **k: types.SimpleNamespace(mainloop=lambda: None),
        ),
        "modules.processors.frame.core": types.SimpleNamespace(
            get_frame_processors_modules=lambda _names: [Processor()],
            process_video_in_memory=lambda *a, **k: (
                calls.append(("pipe",)) or pipe_result
            ),
        ),
        "modules.utilities": stub_utils,
    }

    with mock.patch.dict(sys.modules, stubs, clear=False):
        sys.modules.pop("modules", None)
        sys.modules.pop("modules.core", None)
        # modules.core reads modules.globals while it is being imported, so
        # bind the fake onto the package before importing the submodule.
        package = importlib.import_module("modules")
        package.globals = globals_module
        module = importlib.import_module("modules.core")
        module.modules.globals = globals_module
        module._test_globals = globals_module
        module._test_utils = real_utils
        yield module
        sys.modules.pop("modules.core", None)


class SwapRangeExportTests(unittest.TestCase):
    def _run_start(self, core):
        with mock.patch.object(core.os.path, "isfile", return_value=True):
            core.start()

    def test_reversed_range_aborts_export_before_temp_work(self):
        calls, statuses = [], []
        with _patched_core(calls, statuses, pipe_result=True) as core:
            core._test_globals.face_swap_start_frame = 40
            core._test_globals.face_swap_end_frame = 10
            with mock.patch.object(core._test_utils,
                                   "is_face_swap_range_supported",
                                   return_value=True), \
                 mock.patch.object(core._test_utils,
                                   "get_video_frame_count",
                                   return_value=100):
                self._run_start(core)

        self.assertEqual(calls, [])
        self.assertIn(
            "The start frame cannot be later than the end frame.",
            statuses,
        )

    def test_half_marked_range_aborts_export(self):
        calls, statuses = [], []
        with _patched_core(calls, statuses, pipe_result=True) as core:
            core._test_globals.face_swap_start_frame = 10
            core._test_globals.face_swap_end_frame = None
            with mock.patch.object(core._test_utils,
                                   "is_face_swap_range_supported",
                                   return_value=True), \
                 mock.patch.object(core._test_utils,
                                   "get_video_frame_count",
                                   return_value=100):
                self._run_start(core)

        self.assertEqual(calls, [])
        self.assertTrue(
            any("both the start frame and the end frame" in s for s in statuses)
        )

    def test_mark_beyond_video_length_aborts_export(self):
        calls, statuses = [], []
        with _patched_core(calls, statuses, pipe_result=True) as core:
            core._test_globals.face_swap_start_frame = 0
            core._test_globals.face_swap_end_frame = 99
            with mock.patch.object(core._test_utils,
                                   "is_face_swap_range_supported",
                                   return_value=True), \
                 mock.patch.object(core._test_utils,
                                   "get_video_frame_count",
                                   return_value=50):
                self._run_start(core)

        self.assertEqual(calls, [])
        self.assertTrue(
            any("cannot be found in the target video" in s for s in statuses)
        )

    def test_valid_range_uses_in_memory_pipeline(self):
        calls, statuses = [], []
        with _patched_core(calls, statuses, pipe_result=True) as core:
            core._test_globals.face_swap_start_frame = 2
            core._test_globals.face_swap_end_frame = 4
            with mock.patch.object(core._test_utils,
                                   "is_face_swap_range_supported",
                                   return_value=True), \
                 mock.patch.object(core._test_utils,
                                   "get_video_frame_count",
                                   return_value=10):
                self._run_start(core)

        self.assertIn(("pipe",), calls)
        self.assertNotIn(("extract_frames", "target.mp4"), calls)
        self.assertIn(("move_temp", "target.mp4", "output.mp4"), calls)

    def test_disk_fallback_processes_only_range_frames(self):
        calls, statuses = [], []
        with _patched_core(calls, statuses, pipe_result=False) as core:
            core._test_globals.face_swap_start_frame = 2
            core._test_globals.face_swap_end_frame = 4
            with mock.patch.object(core._test_utils,
                                   "is_face_swap_range_supported",
                                   return_value=True), \
                 mock.patch.object(core._test_utils,
                                   "get_video_frame_count",
                                   return_value=10):
                self._run_start(core)

        process_calls = [c for c in calls if c[0] == "process_video"]
        self.assertEqual(len(process_calls), 1)
        _, source_path, frame_paths = process_calls[0]
        self.assertEqual(source_path, "source.jpg")
        # 0-based [2, 4] -> 1-based extraction names 0003..0005, sorted.
        self.assertEqual(
            [path.rsplit("/", 1)[-1] for path in frame_paths],
            ["0003.png", "0004.png", "0005.png"],
        )
        # Every extracted frame is still reassembled into the output video.
        self.assertIn(("create_video", "target.mp4", 30.0), calls)
        self.assertIn(("move_temp", "target.mp4", "output.mp4"), calls)

    def test_map_faces_ignores_range_marks_and_processes_all_frames(self):
        calls, statuses = [], []
        with _patched_core(calls, statuses, pipe_result=False) as core:
            core._test_globals.map_faces = True
            core._test_globals.face_swap_start_frame = 2
            core._test_globals.face_swap_end_frame = 4
            with mock.patch.object(core._test_utils,
                                   "is_face_swap_range_supported",
                                   side_effect=AssertionError("must not probe")), \
                 mock.patch.object(core._test_utils,
                                   "get_video_frame_count",
                                   side_effect=AssertionError("must not probe")):
                self._run_start(core)

        process_calls = [c for c in calls if c[0] == "process_video"]
        self.assertEqual(len(process_calls), 1)
        _, _, frame_paths = process_calls[0]
        self.assertEqual(len(frame_paths), 10)
        self.assertEqual(
            [path.rsplit("/", 1)[-1] for path in frame_paths],
            [f"{i:04d}.png" for i in range(1, 11)],
        )


if __name__ == "__main__":
    unittest.main()
