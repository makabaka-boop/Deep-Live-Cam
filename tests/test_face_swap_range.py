"""Unit tests for the face swap frame range helpers in modules.utilities.

The range feature marks a 0-based, inclusive [start, end] interval inside
the target video. These tests cover the pure logic (index mapping, temp frame
filtering, mark binding and export-time validation) without invoking ffprobe;
ffprobe-backed functions are replaced with stubs.
"""
import importlib
import os
import sys
import types
import unittest
from unittest import mock


def _load_utilities_module(globals_overrides=None):
    fake_globals = types.ModuleType("modules.globals")
    fake_globals.execution_threads = 0
    fake_globals.log_level = "error"
    fake_globals.execution_providers = []
    fake_globals.video_encoder = "libx264"
    fake_globals.video_quality = 23
    fake_globals.face_swap_start_frame = None
    fake_globals.face_swap_end_frame = None
    fake_globals.face_swap_range_target = None
    for key, value in (globals_overrides or {}).items():
        setattr(fake_globals, key, value)

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.imdecode = lambda *args, **kwargs: None
    fake_cv2.imencode = lambda *args, **kwargs: (
        True,
        types.SimpleNamespace(tofile=lambda *_: None),
    )

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.uint8 = int
    fake_numpy.fromfile = lambda *args, **kwargs: b""

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, *args, **kwargs: iterable

    with mock.patch.dict(
        sys.modules,
        {
            "modules.globals": fake_globals,
            "cv2": fake_cv2,
            "numpy": fake_numpy,
            "tqdm": fake_tqdm,
        },
        clear=False,
    ):
        sys.modules.pop("modules", None)
        sys.modules.pop("modules.utilities", None)
        module = importlib.import_module("modules.utilities")
        # The real module reaches globals through the package attribute; make
        # sure it resolves to the fake after the import context exits too.
        module.modules.globals = fake_globals
        module._test_globals = fake_globals
        return module


class FrameIndexPredicateTests(unittest.TestCase):
    def setUp(self):
        self.utilities = _load_utilities_module()

    def test_unset_bounds_means_whole_video(self):
        self.assertTrue(self.utilities.frame_index_in_face_swap_range(0, None, None))
        self.assertTrue(self.utilities.frame_index_in_face_swap_range(99, None, None))

    def test_range_is_inclusive_on_both_ends(self):
        predicate = self.utilities.frame_index_in_face_swap_range
        self.assertFalse(predicate(1, 2, 4))
        self.assertTrue(predicate(2, 2, 4))
        self.assertTrue(predicate(3, 2, 4))
        self.assertTrue(predicate(4, 2, 4))
        self.assertFalse(predicate(5, 2, 4))

    def test_single_frame_range(self):
        predicate = self.utilities.frame_index_in_face_swap_range
        self.assertTrue(predicate(7, 7, 7))
        self.assertFalse(predicate(6, 7, 7))
        self.assertFalse(predicate(8, 7, 7))


class PreviewPositionMappingTests(unittest.TestCase):
    def setUp(self):
        self.utilities = _load_utilities_module()

    def test_positions_zero_and_one_both_map_to_first_frame(self):
        mapping = self.utilities.preview_position_to_frame_index
        self.assertEqual(mapping(0), 0)
        self.assertEqual(mapping(1), 0)

    def test_later_positions_map_to_previous_decode_index(self):
        mapping = self.utilities.preview_position_to_frame_index
        self.assertEqual(mapping(2), 1)
        self.assertEqual(mapping(10), 9)
        self.assertEqual(mapping(300), 299)


class TempFrameSelectionTests(unittest.TestCase):
    def setUp(self):
        self.utilities = _load_utilities_module()

    def _names(self, paths):
        return [os.path.basename(path) for path in paths]

    def test_no_marks_keeps_every_frame_sorted(self):
        utilities = self.utilities
        paths = [f"/t/{i:04d}.png" for i in (10, 3, 1, 7)]
        selected = utilities.select_temp_frames_in_range(paths, None, None)
        self.assertEqual(
            self._names(selected),
            ["0001.png", "0003.png", "0007.png", "0010.png"],
        )

    def test_marks_select_inclusive_files(self):
        utilities = self.utilities
        # 0-based range [2, 4] <-> extraction names 0003.png .. 0005.png
        paths = [f"/t/{i:04d}.png" for i in range(1, 11)]
        selected = utilities.select_temp_frames_in_range(paths, 2, 4)
        self.assertEqual(
            self._names(selected), ["0003.png", "0004.png", "0005.png"]
        )

    def test_range_starting_at_first_frame(self):
        utilities = self.utilities
        paths = [f"/t/{i:04d}.png" for i in range(1, 6)]
        selected = utilities.select_temp_frames_in_range(paths, 0, 1)
        self.assertEqual(self._names(selected), ["0001.png", "0002.png"])

    def test_unexpected_names_are_processed_not_skipped(self):
        utilities = self.utilities
        paths = ["/t/0001.png", "/t/notes.txt~frame.png", "/t/0003.png"]
        selected = utilities.select_temp_frames_in_range(paths, 1, 1)
        # Only the oddly named file survives: 0001 (index 0) and 0003
        # (index 2) are both outside [1, 1], but non-sequence names are
        # processed rather than silently left unswapped.
        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0].endswith("notes.txt~frame.png"))


class ActiveRangeBindingTests(unittest.TestCase):
    def test_marks_bound_to_another_target_are_ignored(self):
        utilities = _load_utilities_module(
            {
                "face_swap_start_frame": 4,
                "face_swap_end_frame": 8,
                "face_swap_range_target": "/movies/a.mp4",
            }
        )
        self.assertEqual(
            utilities.get_active_face_swap_range("/movies/b.mp4"),
            (None, None),
        )
        self.assertEqual(
            utilities.get_active_face_swap_range("/movies/a.mp4"),
            (4, 8),
        )

    def test_no_marks_is_whole_video_regardless_of_binding(self):
        utilities = _load_utilities_module(
            {"face_swap_range_target": "/movies/a.mp4"}
        )
        self.assertEqual(
            utilities.get_active_face_swap_range("/movies/a.mp4"),
            (None, None),
        )


class ValidateRangeTests(unittest.TestCase):
    def setUp(self):
        self.utilities = _load_utilities_module()

    def _support(self, supported=True, frame_count=100):
        return mock.patch.multiple(
            self.utilities,
            is_face_swap_range_supported=lambda _path: supported,
            get_video_frame_count=lambda _path: frame_count,
        )

    def test_no_marks_always_passes_without_probing(self):
        utilities = self.utilities
        with mock.patch.object(
            utilities, "is_face_swap_range_supported",
            side_effect=AssertionError("must not probe"),
        ):
            self.assertIsNone(
                utilities.validate_face_swap_range("x.mp4", None, None)
            )

    def test_only_one_end_marked_is_rejected(self):
        utilities = self.utilities
        with self._support():
            self.assertEqual(
                utilities.validate_face_swap_range("x.mp4", 10, None),
                utilities.FACE_SWAP_RANGE_INCOMPLETE,
            )
            self.assertEqual(
                utilities.validate_face_swap_range("x.mp4", None, 10),
                utilities.FACE_SWAP_RANGE_INCOMPLETE,
            )

    def test_reversed_range_is_rejected(self):
        utilities = self.utilities
        with self._support():
            self.assertEqual(
                utilities.validate_face_swap_range("x.mp4", 40, 10),
                utilities.FACE_SWAP_RANGE_REVERSED,
            )

    def test_negative_index_is_rejected(self):
        utilities = self.utilities
        with self._support():
            self.assertEqual(
                utilities.validate_face_swap_range("x.mp4", -1, 10),
                utilities.FACE_SWAP_RANGE_REVERSED,
            )

    def test_end_beyond_video_length_is_not_found(self):
        utilities = self.utilities
        with self._support(frame_count=100):
            self.assertEqual(
                utilities.validate_face_swap_range("x.mp4", 90, 100),
                utilities.FACE_SWAP_RANGE_FRAME_NOT_FOUND,
            )
            self.assertEqual(
                utilities.validate_face_swap_range("x.mp4", 100, 100),
                utilities.FACE_SWAP_RANGE_FRAME_NOT_FOUND,
            )

    def test_unknown_frame_count_is_not_found(self):
        utilities = self.utilities
        with self._support(frame_count=None):
            self.assertEqual(
                utilities.validate_face_swap_range("x.mp4", 0, 10),
                utilities.FACE_SWAP_RANGE_FRAME_NOT_FOUND,
            )

    def test_ineligible_target_is_rejected(self):
        utilities = self.utilities
        with self._support(supported=False):
            self.assertEqual(
                utilities.validate_face_swap_range("movie.mkv", 0, 10),
                utilities.FACE_SWAP_RANGE_INELIGIBLE,
            )

    def test_valid_range_passes_including_last_frame(self):
        utilities = self.utilities
        with self._support(frame_count=100):
            self.assertIsNone(
                utilities.validate_face_swap_range("x.mp4", 0, 99)
            )
            self.assertIsNone(
                utilities.validate_face_swap_range("x.mp4", 30, 30)
            )


class EligibilityProbeParsingTests(unittest.TestCase):
    def setUp(self):
        self.utilities = _load_utilities_module()

    def test_non_mp4_extension_rejected_without_probe(self):
        utilities = self.utilities
        with mock.patch.object(
            utilities, "probe_video_streams",
            side_effect=AssertionError("must not probe"),
        ):
            self.assertFalse(
                utilities.is_face_swap_range_supported("movie.mkv")
            )
            self.assertFalse(
                utilities.is_face_swap_range_supported("")
            )

    def test_single_cfr_video_stream_is_supported(self):
        utilities = self.utilities
        # probe_video_streams already strips non-video streams; an audio
        # track next to the one video track is perfectly fine.
        video_stream = {
            "codec_type": "video",
            "r_frame_rate": "30000/1001",
            "avg_frame_rate": "30000/1001",
        }
        with mock.patch.object(utilities, "probe_video_streams",
                               return_value=[video_stream]):
            self.assertTrue(
                utilities.is_face_swap_range_supported("movie.mp4")
            )

    def test_two_video_tracks_are_rejected(self):
        utilities = self.utilities
        one = {"codec_type": "video", "r_frame_rate": "30/1",
               "avg_frame_rate": "30/1"}
        with mock.patch.object(utilities, "probe_video_streams",
                               return_value=[one, dict(one)]):
            self.assertFalse(
                utilities.is_face_swap_range_supported("movie.mp4")
            )

    def test_variable_frame_rate_is_rejected(self):
        utilities = self.utilities
        streams = [
            {"codec_type": "video", "r_frame_rate": "30/1",
             "avg_frame_rate": "25/1"},
        ]
        with mock.patch.object(utilities, "probe_video_streams",
                               return_value=streams):
            self.assertFalse(
                utilities.is_face_swap_range_supported("movie.mp4")
            )

    def test_rate_parser(self):
        parse = self.utilities._parse_frame_rate
        self.assertAlmostEqual(parse("30000/1001"), 30000 / 1001)
        self.assertEqual(parse("24"), 24.0)
        self.assertIsNone(parse("0/0"))
        self.assertIsNone(parse(""))
        self.assertIsNone(parse(None))


if __name__ == "__main__":
    unittest.main()
