from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_face_swapper():
    """Import the real face swapper with heavy UI/model deps stubbed.

    cv2/numpy are real so the warp/feather/compositing math is exercised;
    insightface (ONNX models) and modules.ui (PySide) are never needed for
    layer construction.
    """
    existing = sys.modules.get("insightface")
    if not hasattr(getattr(existing, "app", None), "common"):
        insightface = types.ModuleType("insightface")
        app = types.ModuleType("insightface.app")
        common = types.ModuleType("insightface.app.common")

        class Face:  # noqa: D401 - minimal stand-in
            pass

        common.Face = Face
        app.common = common
        insightface.app = app
        model_zoo = types.ModuleType("insightface.model_zoo")
        insightface.model_zoo = model_zoo
        utils = types.ModuleType("insightface.utils")
        face_align = types.ModuleType("insightface.utils.face_align")
        face_align.norm_crop = lambda *a, **k: None
        utils.face_align = face_align
        insightface.utils = utils
        for name, module in {
            "insightface": insightface,
            "insightface.app": app,
            "insightface.app.common": common,
            "insightface.model_zoo": model_zoo,
            "insightface.utils": utils,
            "insightface.utils.face_align": face_align,
        }.items():
            sys.modules[name] = module
    ui_stub = types.ModuleType("modules.ui")
    ui_stub.update_status = lambda *a, **k: None
    ui_stub.check_and_ignore_nsfw = lambda *a, **k: False
    ui_stub.init = lambda *a, **k: None
    # sklearn (pulled in by cluster_analysis) is binary-incompatible with the
    # numpy in this test env and irrelevant to the layer math.
    cluster_stub = types.ModuleType("modules.cluster_analysis")
    cluster_stub.find_cluster_centroids = lambda *a, **k: None
    cluster_stub.find_closest_centroid = lambda *a, **k: (0, 0.0)
    with mock.patch.dict(sys.modules, {
        "modules.ui": ui_stub,
        "modules.cluster_analysis": cluster_stub,
    }):
        # Discard modules sibling tests may have left stubbed (e.g. dummy
        # modules.utilities / modules.typing) so the import chain rebuilds on
        # the real modules with THIS loader's stubs.
        for name in (
            "modules",
            "modules.core",
            "modules.globals",
            "modules.metadata",
            "modules.typing",
            "modules.utilities",
            "modules.face_analyser",
            "modules.gpu_processing",
            "modules.platform_info",
            "modules.processors",
            "modules.processors.frame",
            "modules.processors.frame.core",
            "modules.processors.frame.face_swapper",
        ):
            sys.modules.pop(name, None)
        face_swapper = importlib.import_module("modules.processors.frame.face_swapper")
        # stdlib 3.9 patch.dict rolls back sys.modules keys added inside the
        # context, so keep direct references before it exits.
        core_module = sys.modules["modules.core"]
        globals_module = sys.modules["modules.globals"]
    return face_swapper, core_module, globals_module


# Loaded lazily in setUpModule (no import side effects at collection time).
fs = None
core = None
g = None
_PUBLISHED_KEYS = (
    "modules.core",
    "modules.globals",
    "modules.processors.frame.face_swapper",
)
_saved_modules: dict = {}
_MISSING = object()


def setUpModule() -> None:
    """Publish the loaded modules only while these tests run.

    stdlib 3.9 patch.dict rolls back sys.modules keys added inside the loader
    context, and core's lazy `from modules.processors.frame import face_swapper`
    needs the exact objects back in sys.modules. Keeping this out of module
    scope avoids polluting sibling test modules during collection.
    """
    global fs, core, g
    face_swapper, core_module, globals_module = _load_face_swapper()
    fs, core, g = face_swapper, core_module, globals_module
    for key in _PUBLISHED_KEYS:
        _saved_modules[key] = sys.modules.get(key, _MISSING)
        sys.modules[key] = {
            "modules.core": core,
            "modules.globals": g,
            "modules.processors.frame.face_swapper": fs,
        }[key]


def tearDownModule() -> None:
    for key in _PUBLISHED_KEYS:
        if _saved_modules.get(key) is _MISSING:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = _saved_modules[key]


class _DummySwapper:
    """Stand-in for insightface INSwapper: returns a preset aligned face."""

    input_size = (128, 128)

    def __init__(self, bgr_fake, M):
        self._bgr = bgr_fake
        self._M = M

    def get(self, frame, target_face, source_face, paste_back=False):
        return self._bgr.copy(), self._M.copy()


def _face():
    return SimpleNamespace(normed_embedding=np.zeros(4, dtype=np.float32))


class LayerMathTests(unittest.TestCase):
    def setUp(self) -> None:
        g.opacity = 1.0
        g.mouth_mask = False
        g.poisson_blend = False
        g.many_faces = False
        g.map_faces = False
        g.sharpness = 0.0
        g.fp_ui = {
            "face_enhancer": False,
            "face_enhancer_gpen256": False,
            "face_enhancer_gpen512": False,
        }
        g.execution_providers = []
        g.target_path = None
        g.headless = True

        rng = np.random.RandomState(42)
        self.h, self.w = 220, 300
        self.target = rng.randint(0, 256, (self.h, self.w, 3), dtype=np.uint8)
        self.bgr_fake = rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        # insightface's M maps original-frame coords into the aligned 128
        # space (norm_crop2 convention): the aligned face lands back at
        # scale 1.1 with its top-left near (60, 40) in the target.
        aligned_to_target = np.array([[1.1, 0.0, 60.0], [0.0, 1.1, 40.0]], dtype=np.float32)
        self.M = cv2.invertAffineTransform(aligned_to_target)
        self.aimg = np.empty((128, 128, 3), dtype=np.uint8)
        self.dummy = _DummySwapper(self.bgr_fake, self.M)
        patcher = mock.patch.object(fs, "get_face_swapper", lambda: self.dummy)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _layer(self, target=None):
        return fs.build_face_layer(_face(), _face(), target if target is not None else self.target)

    def _composite(self, layer, target):
        alpha = layer[..., 3:4].astype(np.float32) / 255.0
        comp = alpha * layer[..., :3].astype(np.float32) + (1.0 - alpha) * target.astype(np.float32)
        return np.rint(comp).clip(0, 255).astype(np.uint8)

    def test_layer_covers_only_face_at_full_strength(self) -> None:
        layer = self._layer()
        self.assertEqual(layer.shape, (self.h, self.w, 4))
        self.assertEqual(layer.dtype, np.uint8)
        # Far outside the face everything is transparent.
        self.assertEqual(int(layer[0, 0, 3]), 0)
        self.assertEqual(int(layer[-1, -1, 3]), 0)
        self.assertGreater((layer[..., 3] == 0).mean(), 0.5)
        # Aligned-face centre maps to (130.4, 110.4) → opaque core.
        self.assertGreaterEqual(int(layer[110, 130, 3]), 250)

    def test_composite_matches_normal_pasteback(self) -> None:
        normal = fs._fast_paste_back(self.target.copy(), self.bgr_fake, self.aimg, self.M)
        layer = self._layer()
        comp = self._composite(layer, self.target)
        diff = np.abs(comp.astype(np.int16) - normal.astype(np.int16))
        self.assertLessEqual(diff.max(), 4)
        self.assertLessEqual(diff.mean(), 1.0)

    def test_swap_strength_scales_alpha_and_composite(self) -> None:
        layer_full = self._layer()
        g.opacity = 0.4
        layer = self._layer()

        solid = layer_full[..., 3] > 80
        ratio = layer[..., 3][solid].astype(np.float32) / layer_full[..., 3][solid]
        np.testing.assert_allclose(ratio, np.full_like(ratio, 0.4), atol=0.02)

        swapped = fs._fast_paste_back(self.target.copy(), self.bgr_fake, self.aimg, self.M)
        normal = fs.gpu_add_weighted(self.target, 0.6, swapped, 0.4, 0)
        comp = self._composite(layer, self.target)
        diff = np.abs(comp.astype(np.int16) - normal.astype(np.int16))
        self.assertLessEqual(diff.max(), 5)
        self.assertLessEqual(diff.mean(), 1.5)

    def test_mouth_mask_helper_matches_apply_mouth_area(self) -> None:
        rng = np.random.RandomState(7)
        frame_t = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        swapped = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        min_x, min_y, max_x, max_y = 60, 70, 140, 150
        polygon = np.array([[72, 82], [128, 82], [128, 138], [72, 138]], dtype=np.int32)
        cutout = frame_t[min_y:max_y, min_x:max_x].copy()

        normal = fs.apply_mouth_area(
            swapped.copy(), cutout, (min_x, min_y, max_x, max_y),
            np.zeros((200, 200), dtype=np.uint8), polygon,
        )
        m = fs._build_mouth_feather_mask(frame_t.shape, (min_x, min_y, max_x, max_y), polygon)
        self.assertGreater(float(m[110, 100]), 0.99)
        self.assertEqual(float(m[0, 0]), 0.0)

        comp = m[..., None] * frame_t.astype(np.float32) + (1.0 - m[..., None]) * swapped.astype(np.float32)
        comp = np.rint(comp).clip(0, 255).astype(np.uint8)
        diff = np.abs(comp.astype(np.int16) - normal.astype(np.int16))
        self.assertLessEqual(diff.max(), 3)

    def test_kept_mouth_punches_alpha_and_matches_normal_export(self) -> None:
        min_x, min_y, max_x, max_y = 110, 80, 190, 160
        polygon = np.array([[120, 90], [180, 90], [180, 150], [120, 150]], dtype=np.int32)
        cutout = self.target[min_y:max_y, min_x:max_x].copy()

        def fake_mouth(face, frame):
            return np.zeros((self.h, self.w), np.uint8), cutout, (min_x, min_y, max_x, max_y), polygon

        layer_no_mouth = self._layer()
        with mock.patch.object(fs, "create_lower_mouth_mask", fake_mouth):
            g.mouth_mask = True
            layer = self._layer()

        cx, cy = 150, 120  # well inside both the face and the mouth polygon
        self.assertGreater(int(layer_no_mouth[cy, cx, 3]), 200)
        self.assertLess(int(layer[cy, cx, 3]), int(layer_no_mouth[cy, cx, 3]))

        swapped = fs._fast_paste_back(self.target.copy(), self.bgr_fake, self.aimg, self.M)
        normal = fs.apply_mouth_area(
            swapped.copy(), cutout, (min_x, min_y, max_x, max_y),
            np.zeros((self.h, self.w), dtype=np.uint8), polygon,
        )
        comp = self._composite(layer, self.target)
        diff = np.abs(comp.astype(np.int16) - normal.astype(np.int16))
        self.assertLessEqual(diff.max(), 5)
        self.assertLessEqual(diff.mean(), 1.5)


class AvailabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        g.many_faces = False
        g.map_faces = False
        g.poisson_blend = False
        g.sharpness = 0.0
        g.fp_ui = {
            "face_enhancer": False,
            "face_enhancer_gpen256": False,
            "face_enhancer_gpen512": False,
        }

    def test_available_for_single_face_jpeg_in_supported_setup(self) -> None:
        g.target_path = "/tmp/poster.JPG"
        ok, reason = fs.transparent_layer_available()
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "")

    def test_rejects_unsupported_configurations(self) -> None:
        g.target_path = "/tmp/poster.jpg"

        g.target_path = "/tmp/poster.png"
        self.assertFalse(fs.transparent_layer_available()[0])

        g.target_path = "/tmp/poster.jpg"
        g.many_faces = True
        self.assertIn("Many faces", fs.transparent_layer_available()[1])

        g.many_faces = False
        g.map_faces = True
        self.assertIn("Map faces", fs.transparent_layer_available()[1])

        g.map_faces = False
        g.fp_ui["face_enhancer"] = True
        self.assertIn("Face Enhancer", fs.transparent_layer_available()[1])

        g.fp_ui["face_enhancer"] = False
        g.poisson_blend = True
        self.assertIn("Poisson", fs.transparent_layer_available()[1])

        g.poisson_blend = False
        g.sharpness = 1.5
        self.assertIn("Sharpness", fs.transparent_layer_available()[1])

        g.sharpness = 0.0
        g.target_path = None
        self.assertFalse(fs.transparent_layer_available()[0])


class ProcessImageLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        g.opacity = 1.0
        g.mouth_mask = False
        g.execution_providers = []
        g.headless = True
        rng = np.random.RandomState(99)
        self.frame = rng.randint(0, 256, (160, 200, 3), dtype=np.uint8)
        self.bgr_fake = rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        # Face pastes back with top-left near (36, 16) of the target.
        self.M = cv2.invertAffineTransform(
            np.array([[1.0, 0.0, 36.0], [0.0, 1.0, 16.0]], dtype=np.float32)
        )
        self.dummy = _DummySwapper(self.bgr_fake, self.M)

    def _write_image(self, path, frame):
        ok = fs.imwrite_unicode(path, frame)
        self.assertTrue(ok)

    def test_failures_report_reason_and_leave_no_output(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            source_path = os.path.join(tmp, "source.jpg")
            target_path = os.path.join(tmp, "target.jpg")
            output_path = os.path.join(tmp, "layer.png")
            self._write_image(source_path, self.frame)
            self._write_image(target_path, self.frame)

            with mock.patch.object(fs, "get_one_face", lambda frame: None):
                self.assertFalse(fs.process_image_layer(source_path, target_path, output_path))
            self.assertFalse(os.path.exists(output_path))

            g.opacity = 0.0
            with mock.patch.object(fs, "get_one_face", lambda frame: _face()):
                self.assertFalse(fs.process_image_layer(source_path, target_path, output_path))
            self.assertFalse(os.path.exists(output_path))
            g.opacity = 1.0

            answers = iter([_face(), None])
            with mock.patch.object(fs, "get_one_face", lambda frame: next(answers)):
                self.assertFalse(fs.process_image_layer(source_path, target_path, output_path))
            self.assertFalse(os.path.exists(output_path))

    def test_save_failure_leaves_no_layer_or_temp(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            source_path = os.path.join(tmp, "source.jpg")
            target_path = os.path.join(tmp, "target.jpg")
            output_path = os.path.join(tmp, "layer.png")
            self._write_image(source_path, self.frame)
            self._write_image(target_path, self.frame)

            # A pre-existing file at the output path must survive a failed run.
            with open(output_path, "wb") as handle:
                handle.write(b"keep-me")

            with mock.patch.object(fs, "get_one_face", lambda frame: _face()), \
                    mock.patch.object(fs, "get_face_swapper", lambda: self.dummy), \
                    mock.patch.object(fs, "imwrite_unicode", lambda *a, **k: False):
                ok = fs.process_image_layer(source_path, target_path, output_path)
            self.assertFalse(ok)
            self.assertEqual(open(output_path, "rb").read(), b"keep-me")
            self.assertFalse(os.path.exists(os.path.splitext(output_path)[0] + ".part.png"))

    def test_success_writes_same_size_bgra_png(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            source_path = os.path.join(tmp, "source.jpg")
            target_path = os.path.join(tmp, "target.jpg")
            output_path = os.path.join(tmp, "layer.png")
            self._write_image(source_path, self.frame)
            self._write_image(target_path, self.frame)

            with mock.patch.object(fs, "get_one_face", lambda frame: _face()), \
                    mock.patch.object(fs, "get_face_swapper", lambda: self.dummy):
                ok = fs.process_image_layer(source_path, target_path, output_path)
            self.assertTrue(ok)
            written = fs.imread_unicode(output_path, cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(written)
            self.assertEqual(written.shape, (160, 200, 4))
            self.assertEqual(int(written[0, 0, 3]), 0)
            self.assertFalse(os.path.exists(os.path.splitext(output_path)[0] + ".part.png"))


class CoreExportBranchTests(unittest.TestCase):
    def setUp(self) -> None:
        g.headless = True
        g.target_path = None
        g.output_path = None
        g.source_path = None

    def test_unavailable_guard_writes_nothing(self) -> None:
        called = []
        with mock.patch.object(fs, "transparent_layer_available", lambda: (False, "blocked")), \
                mock.patch.object(fs, "process_image_layer",
                                  lambda *a: called.append(a) or True):
            core.export_transparent_face_layer()
        self.assertEqual(called, [])

    def test_success_forces_png_and_calls_layer_export(self) -> None:
        g.source_path = "/tmp/source.jpg"
        g.target_path = "/tmp/target.jpg"
        g.output_path = "/tmp/output.jpg"
        captured = {}

        def fake_process(source_path, target_path, output_path):
            captured["args"] = (source_path, target_path, output_path)
            return True

        with mock.patch.object(fs, "transparent_layer_available", lambda: (True, "")), \
                mock.patch.object(fs, "process_image_layer", fake_process):
            core.export_transparent_face_layer()
        self.assertEqual(captured["args"], ("/tmp/source.jpg", "/tmp/target.jpg", "/tmp/output.png"))
        self.assertEqual(g.output_path, "/tmp/output.png")


if __name__ == "__main__":
    unittest.main()
