"""Tests for frame-range gating inside the in-memory export pipeline.

_run_pipe_pipeline reads raw frames from an FFmpeg decoder pipe, processes
them and writes them to an encoder pipe. These tests replace subprocess.Popen
with in-memory fakes (no FFmpeg binary required) and assert that:

* only frames in the marked range reach face detection / frame processors,
* frames outside the range pass through byte-for-byte,
* every frame is still written in the original order,
* detection pipelining resumes correctly when the range re-enters.
"""
import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import tqdm as _real_tqdm_pkg
import cv2 as _real_cv2


FRAME_COUNT = 5
WIDTH = 2
HEIGHT = 2
FRAME_SIZE = WIDTH * HEIGHT * 3
SWAP_MARKER = 7


class _FakeStdout:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read(self, _size):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _FakeStdin:
    def __init__(self):
        self.data = bytearray()
        self.closed = False

    def write(self, data):
        self.data.extend(data)
        return len(data)

    def close(self):
        self.closed = True


class _FakeProc:
    def __init__(self, stdout=None, stdin=None):
        self.stdout = stdout
        self.stdin = stdin
        self.stderr = types.SimpleNamespace(read=lambda: b"")
        self.returncode = 0
        self.killed = False

    def wait(self):
        return 0

    def poll(self):
        return None

    def kill(self):
        self.killed = True


class _MarkerProcessor:
    """Stands in for face_swapper/face_enhancer: paints every processed
    frame with SWAP_MARKER so out-of-range frames are recognizable."""

    def __init__(self, call_log):
        self._call_log = call_log

    def process_frame(self, source_face, frame, target_face=None):
        self._call_log.append(("process_frame", target_face))
        frame[:] = SWAP_MARKER
        return frame


class _PipeHarness:
    def __init__(self):
        self.detector_calls = []
        self.processor_calls = []
        self.frames_in = [
            bytes([0] * FRAME_SIZE) for _ in range(FRAME_COUNT)
        ]
        self.written = b""

    def detector(self, frame):
        # Record the mean pixel value so call order vs. processed frames
        # can be inspected.
        self.detector_calls.append(int(np.asarray(frame).flat[0]))
        return object()

    def run(self, swap_start, swap_end, many_faces=False):
        harness = self

        # The pipeline imports modules.utilities lazily (inside the
        # function), so the fake globals must stay installed for the whole
        # run, not just the module import.
        fake_globals = types.ModuleType("modules.globals")
        fake_globals.many_faces = many_faces
        fake_globals.execution_providers = []
        fake_globals.execution_threads = 1
        fake_face_analyser = types.ModuleType("modules.face_analyser")
        fake_face_analyser.get_one_face = harness.detector

        reader_proc = _FakeProc(stdout=_FakeStdout(self.frames_in))
        writer_stdin = _FakeStdin()
        writer_proc = _FakeProc(stdin=writer_stdin)

        def fake_popen(_cmd, stdout=None, stdin=None, stderr=None):
            if stdout is not None:
                return reader_proc
            return writer_proc

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as handle:
            output_path = handle.name
        with mock.patch.dict(
            sys.modules,
            {
                "modules.globals": fake_globals,
                "modules.face_analyser": fake_face_analyser,
                # Other test modules can leave stub numpy/tqdm/cv2 modules in
                # sys.modules on older Pythons; pin the real ones.
                "numpy": np,
                "tqdm": _real_tqdm_pkg,
                "cv2": _real_cv2,
            },
            clear=False,
        ):
            # Force a fresh frame.core/utilities import bound to the fake
            # globals and real heavy dependencies. The whole parent chain is
            # popped so the import machinery builds throwaway package objects;
            # rebinding .globals on them must never touch the real "modules"
            # package that other test modules (and their Qt threads) use.
            for module_name in (
                "modules",
                "modules.processors",
                "modules.processors.frame",
                "modules.utilities",
                "modules.processors.frame.core",
            ):
                sys.modules.pop(module_name, None)
            package = importlib.import_module("modules")
            package.globals = fake_globals
            frame_core = importlib.import_module("modules.processors.frame.core")
            try:
                with mock.patch.object(frame_core.subprocess, "Popen",
                                       side_effect=fake_popen):
                    success = frame_core._run_pipe_pipeline(
                        "target.mp4", output_path, 30.0,
                        object(),
                        [_MarkerProcessor(harness.processor_calls)],
                        WIDTH, HEIGHT, FRAME_SIZE, FRAME_COUNT,
                        "libx264", ["-crf", "23"],
                        swap_start, swap_end,
                    )
            finally:
                self.written = bytes(writer_stdin.data)
        os.unlink(output_path)
        return success

    def frame_values(self):
        chunks = [
            self.written[i:i + FRAME_SIZE]
            for i in range(0, len(self.written), FRAME_SIZE)
        ]
        return [set(chunk) for chunk in chunks]


class InMemoryRangeGatingTests(unittest.TestCase):
    def test_all_frames_written_in_order_with_originals_outside(self):
        harness = _PipeHarness()
        success = harness.run(2, 3)  # 0-based inclusive middle range

        self.assertTrue(success)
        # Every frame made it into the output, same count, same order.
        self.assertEqual(len(harness.written), FRAME_COUNT * FRAME_SIZE)
        values = harness.frame_values()
        self.assertEqual(
            values,
            [
                {0},               # frame 0: original
                {0},               # frame 1: original
                {SWAP_MARKER},     # frame 2: swapped
                {SWAP_MARKER},     # frame 3: swapped
                {0},               # frame 4: original
            ],
        )
        # Processor only ran twice; detection for three in-range-adjacent
        # frames (a sync detect on entering the range + two pipelined ones).
        self.assertEqual(len(harness.processor_calls), 2)
        self.assertEqual(len(harness.detector_calls), 3)

    def test_range_starting_at_first_frame(self):
        harness = _PipeHarness()
        success = harness.run(0, 1)

        self.assertTrue(success)
        values = harness.frame_values()
        self.assertEqual(values[0], {SWAP_MARKER})
        self.assertEqual(values[1], {SWAP_MARKER})
        self.assertEqual(values[2], {0})
        self.assertEqual(values[4], {0})

    def test_single_frame_range(self):
        harness = _PipeHarness()
        success = harness.run(4, 4)

        self.assertTrue(success)
        values = harness.frame_values()
        self.assertEqual(values[3], {0})
        self.assertEqual(values[4], {SWAP_MARKER})

    def test_no_marks_processes_every_frame(self):
        harness = _PipeHarness()
        success = harness.run(None, None)

        self.assertTrue(success)
        self.assertTrue(all(v == {SWAP_MARKER} for v in harness.frame_values()))
        self.assertEqual(len(harness.processor_calls), FRAME_COUNT)
        # Sync detection on frame 0 plus one pipelined detection per frame.
        self.assertEqual(len(harness.detector_calls), FRAME_COUNT + 1)

    def test_many_faces_mode_skips_detection_outside_range(self):
        harness = _PipeHarness()
        success = harness.run(1, 1, many_faces=True)

        self.assertTrue(success)
        values = harness.frame_values()
        self.assertEqual(values[0], {0})
        self.assertEqual(values[1], {SWAP_MARKER})
        self.assertEqual(values[2], {0})
        # many_faces never uses the pipelined detector.
        self.assertEqual(harness.detector_calls, [])

    def test_processor_receives_target_face_inside_range(self):
        harness = _PipeHarness()
        harness.run(2, 2)

        # The single processed frame got a non-None target_face from the
        # detector (synchronous detection at range entry).
        target_faces = [
            face for _name, face in harness.processor_calls
        ]
        self.assertEqual(len(target_faces), 1)
        self.assertIsNotNone(target_faces[0])


if __name__ == "__main__":
    unittest.main()
