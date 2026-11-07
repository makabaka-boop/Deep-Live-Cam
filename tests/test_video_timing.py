from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


def _load_utilities_module():
    fake_globals = types.ModuleType("modules.globals")
    fake_globals.execution_threads = 0
    fake_globals.log_level = "error"
    fake_globals.execution_providers = []
    fake_globals.video_encoder = "libx264"
    fake_globals.video_quality = 23

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.imdecode = lambda *args, **kwargs: None
    fake_cv2.imencode = lambda *args, **kwargs: (True, types.SimpleNamespace(tofile=lambda *_: None))

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
        return importlib.import_module("modules.utilities")


def _probe_payload(pts_durations, stream_duration=None, nb_frames=None,
                   audio=True, include_duration=True, start_time="0.000000"):
    streams = [{
        "codec_type": "video",
        "start_time": start_time,
        "duration": f"{stream_duration:.6f}" if stream_duration is not None else "N/A",
        "nb_frames": str(nb_frames) if nb_frames is not None else "N/A",
    }]
    if audio:
        streams.append({"codec_type": "audio"})
    meta = {"streams": streams, "format": {"duration": f"{stream_duration:.6f}" if stream_duration else "N/A"}}

    frames = []
    for index, (pts, duration) in enumerate(pts_durations):
        frame = {"pts_time": f"{pts:.6f}"}
        if include_duration and duration is not None:
            frame["duration_time"] = f"{duration:.6f}"
        frames.append(frame)
    return json.dumps(meta), json.dumps({"frames": frames})


class ProbeVideoTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.utilities = _load_utilities_module()

    def _patch_probe(self, meta_json, frames_json):
        def _check_output(cmd, encoding="utf-8"):
            if "frame=pts_time" in " ".join(cmd):
                return frames_json
            return meta_json
        return mock.patch.object(self.utilities.subprocess, "check_output", side_effect=_check_output)

    def test_rejects_non_mp4(self) -> None:
        timing, reason = self.utilities.probe_video_timing("demo.mkv")
        self.assertIsNone(timing)
        self.assertIn(".mp4", reason)

    def test_reads_variable_frame_timing_with_container_durations(self) -> None:
        # Phone-style VFR: 0.05s, 0.03s, then a 0.02s hold on the last frame.
        meta, frames = _probe_payload(
            [(0.0, 0.05), (0.05, 0.03), (0.08, 0.02)],
            stream_duration=0.1,
            nb_frames=3,
        )
        with self._patch_probe(meta, frames):
            timing, reason = self.utilities.probe_video_timing("clip.mp4")
        self.assertIsNone(reason)
        self.assertEqual(timing.durations, [0.05, 0.03, 0.02])
        self.assertAlmostEqual(timing.duration, 0.1, places=6)
        self.assertTrue(timing.has_audio)

    def test_derives_durations_and_last_hold_from_pts(self) -> None:
        # No per-frame duration_time fields: gaps plus stream duration give
        # the last frame's hold time.
        meta, frames = _probe_payload(
            [(0.0, None), (0.05, None), (0.08, None)],
            stream_duration=0.1,
            nb_frames=3,
            include_duration=False,
            audio=False,
        )
        with self._patch_probe(meta, frames):
            timing, reason = self.utilities.probe_video_timing("clip.mp4")
        self.assertIsNone(reason)
        self.assertEqual([round(d, 6) for d in timing.durations], [0.05, 0.03, 0.02])
        self.assertFalse(timing.has_audio)

    def test_rejects_multiple_video_streams(self) -> None:
        meta = json.dumps({
            "streams": [{"codec_type": "video"}, {"codec_type": "video"}],
            "format": {},
        })
        frames = json.dumps({"frames": [{"pts_time": "0.0", "duration_time": "0.05"}]})
        with self._patch_probe(meta, frames):
            timing, reason = self.utilities.probe_video_timing("clip.mp4")
        self.assertIsNone(timing)
        self.assertIn("one video stream", reason)

    def test_rejects_declared_frame_count_mismatch(self) -> None:
        meta, frames = _probe_payload(
            [(0.0, 0.05), (0.05, 0.05)],
            stream_duration=0.1,
            nb_frames=3,  # claims 3 frames, only 2 timestamped frames
        )
        with self._patch_probe(meta, frames):
            timing, reason = self.utilities.probe_video_timing("clip.mp4")
        self.assertIsNone(timing)
        self.assertIn("frame count", reason.lower())

    def test_rejects_non_monotonic_timestamps(self) -> None:
        # Two frames stamped at the same instant: no trustworthy per-frame gap.
        meta, frames = _probe_payload(
            [(0.0, None), (0.0, None)],
            stream_duration=0.1,
            include_duration=False,
        )
        with self._patch_probe(meta, frames):
            timing, reason = self.utilities.probe_video_timing("clip.mp4")
        self.assertIsNone(timing)
        self.assertIn("timestamp", reason.lower())

    def test_falls_back_to_best_effort_timestamp(self) -> None:
        meta = json.dumps({
            "streams": [{"codec_type": "video", "duration": "0.050000", "nb_frames": "1"}],
            "format": {"duration": "0.050000"},
        })
        frames = json.dumps({"frames": [
            {"pts_time": "N/A", "best_effort_timestamp_time": "0.000000", "duration_time": "0.050000"},
        ]})
        with self._patch_probe(meta, frames):
            timing, reason = self.utilities.probe_video_timing("clip.mp4")
        self.assertIsNone(reason)
        self.assertEqual(timing.pts, [0.0])

    def test_probe_failure_is_unreliable_not_an_exception(self) -> None:
        with mock.patch.object(
            self.utilities.subprocess, "check_output",
            side_effect=self.utilities.subprocess.CalledProcessError(1, ["ffprobe"]),
        ):
            timing, reason = self.utilities.probe_video_timing("clip.mp4")
        self.assertIsNone(timing)
        self.assertIn("ffprobe", reason)


class VerifyVideoTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.utilities = _load_utilities_module()

    def test_matching_output_passes(self) -> None:
        payload = json.dumps({
            "frames": [{"pts_time": "0.0"}, {"pts_time": "0.05"}, {"pts_time": "0.08"}],
            "format": {"duration": "0.100000"},
        })
        with mock.patch.object(self.utilities.subprocess, "check_output", return_value=payload):
            self.assertIsNone(self.utilities.verify_video_timing("out.mp4", 3, 0.1))

    def test_frame_count_mismatch_fails(self) -> None:
        payload = json.dumps({
            "frames": [{"pts_time": "0.0"}, {"pts_time": "0.05"}],
            "format": {"duration": "0.100000"},
        })
        with mock.patch.object(self.utilities.subprocess, "check_output", return_value=payload):
            reason = self.utilities.verify_video_timing("out.mp4", 3, 0.1)
        self.assertIn("frames", reason)

    def test_duration_mismatch_fails(self) -> None:
        payload = json.dumps({
            "frames": [{"pts_time": "0.0"}, {"pts_time": "0.05"}, {"pts_time": "0.08"}],
            "format": {"duration": "0.250000"},
        })
        with mock.patch.object(self.utilities.subprocess, "check_output", return_value=payload):
            reason = self.utilities.verify_video_timing("out.mp4", 3, 0.1)
        self.assertIn("duration", reason.lower())


class CreateVideoVfrTests(unittest.TestCase):
    def setUp(self) -> None:
        self.utilities = _load_utilities_module()
        # The test loads a stubbed modules.globals; importing it bound the
        # stub onto the real package object. Re-bind defensively in case the
        # patch context already unwound.
        stub = getattr(self.utilities.modules, "globals", None)
        if stub is None:
            stub = types.ModuleType("modules.globals")
            stub.execution_threads = 0
            stub.log_level = "error"
            stub.execution_providers = []
            stub.video_encoder = "libx264"
            stub.video_quality = 23
            setattr(self.utilities.modules, "globals", stub)

    def test_manifest_places_each_frame_at_its_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "clip.mp4")
            frame_dir = self.utilities.get_temp_directory_path(target)
            os.makedirs(frame_dir)
            for name in ("0001.png", "0002.png", "0003.png"):
                with open(os.path.join(frame_dir, name), "wb"):
                    pass

            captured = {}

            def fake_run_ffmpeg(args):
                captured["args"] = args
                # Encode a marker file so the isfile() check passes.
                with open(self.utilities.get_temp_output_path(target), "wb"):
                    pass
                return True

            with mock.patch.object(self.utilities, "run_ffmpeg", side_effect=fake_run_ffmpeg):
                ok = self.utilities.create_video_vfr(target, [0.05, 0.03, 0.02])

            self.assertTrue(ok)
            manifest_path = os.path.join(frame_dir, self.utilities.TIMING_MANIFEST_FILE)
            manifest = open(manifest_path, encoding="utf-8").read()
            self.assertIn("ffconcat version 1.0", manifest)
            self.assertIn("file 0001.png\nduration 0.050000000", manifest)
            self.assertIn("file 0003.png\nduration 0.020000000", manifest)
            # The last file must be repeated so its hold time takes effect.
            self.assertTrue(manifest.rstrip().endswith("file 0003.png"))

            args = captured["args"]
            self.assertIn("-f", args)
            self.assertEqual(args[args.index("-f") + 1], "concat")
            self.assertIn("-vsync", args)
            self.assertEqual(args[args.index("-vsync") + 1], "0")

    def test_frame_count_mismatch_aborts_without_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "clip.mp4")
            frame_dir = self.utilities.get_temp_directory_path(target)
            os.makedirs(frame_dir)
            with open(os.path.join(frame_dir, "0001.png"), "wb"):
                pass

            with mock.patch.object(self.utilities, "run_ffmpeg") as run_ffmpeg:
                self.assertFalse(self.utilities.create_video_vfr(target, [0.05, 0.03]))
            run_ffmpeg.assert_not_called()


if __name__ == "__main__":
    unittest.main()
