"""Tests of the native Python module.

Point CUDA_TAGS_EXTENSION at a built module (build/cuda_tags*.so) or install
the package; skipped when neither is available. GPU detection checks run
only when the module was built with CUDA (on a Jetson).
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

GOLDEN = Path(__file__).resolve().parent / 'golden'


def load_native():
    """Loads the extension without leaving it in sys.modules, where it would
    shadow the pure-Python package that other tests import."""
    before = {k: v for k, v in sys.modules.items() if k.split('.')[0] == 'cuda_tags'}
    try:
        path = os.environ.get('CUDA_TAGS_EXTENSION')
        if path:
            spec = importlib.util.spec_from_file_location('cuda_tags', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        try:
            return importlib.import_module('cuda_tags.cuda_tags')
        except ImportError:
            return None
    finally:
        for name in [k for k in sys.modules if k.split('.')[0] == 'cuda_tags']:
            del sys.modules[name]
        sys.modules.update(before)


native = load_native() if np is not None else None


def read_pgm(path):
    data = path.read_bytes()
    magic, width, height, maximum, pixels = data.split(maxsplit=4)
    assert magic == b'P5' and maximum == b'255'
    return np.frombuffer(pixels, np.uint8).reshape(int(height), int(width))


def read_scene(path):
    scene = {'tags': {}}
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        kind, *values = line.split()
        if kind == 'camera':
            scene['camera'] = [float(v) for v in values]
        elif kind == 'tag_size':
            scene['tag_size'] = float(values[0])
        elif kind == 'tag':
            numbers = [float(v) for v in values[2:10]]
            translation = [float(v) for v in values[-3:]]
            scene['tags'][int(values[0])] = dict(
                corners=np.array(numbers).reshape(4, 2), translation=translation)
    return scene


@unittest.skipIf(native is None, 'native cuda_tags module not available')
class BindingTests(unittest.TestCase):
    def test_parameters_are_double_precision_and_default_to_zero(self):
        camera = native.CameraMatrix(600.123456789012, 320.5, 601.0, 240.25)
        self.assertEqual(camera.fx, 600.123456789012)
        empty = native.CameraMatrix()
        self.assertEqual((empty.fx, empty.cx, empty.fy, empty.cy), (0, 0, 0, 0))
        distortion = native.DistCoeffs(-0.123456789012345, 0, 0, 0, 1e-9)
        self.assertEqual(distortion.k1, -0.123456789012345)
        self.assertEqual(distortion.k3, 1e-9)
        blank = native.DistCoeffs()
        self.assertEqual((blank.k1, blank.k2, blank.p1, blank.p2, blank.k3), (0,) * 5)

    def test_to_gray_honours_strides(self):
        rng = np.random.default_rng(4765)
        image = rng.integers(0, 256, (24, 40, 3), dtype=np.uint8)
        expected = native.to_gray(image)
        self.assertEqual(expected.shape, (24, 40))
        # A view into a wider image: padded rows, no copy needed.
        wide = np.zeros((24, 64, 3), np.uint8)
        wide[:, :40] = image
        np.testing.assert_array_equal(native.to_gray(wide[:, :40]), expected)
        # Non-contiguous layouts (every other column, flipped rows,
        # transposed) are copied first, never read as garbage.
        doubled = np.repeat(image, 2, axis=1)
        np.testing.assert_array_equal(native.to_gray(doubled[:, ::2]), expected)
        np.testing.assert_array_equal(native.to_gray(image[::-1])[::-1], expected)
        transposed = np.ascontiguousarray(image.transpose(1, 0, 2)).transpose(1, 0, 2)
        np.testing.assert_array_equal(native.to_gray(transposed), expected)

    def test_to_gray_channel_orders(self):
        red_bgr = np.zeros((8, 8, 3), np.uint8)
        red_bgr[..., 2] = 255
        self.assertAlmostEqual(int(native.to_gray(red_bgr)[0, 0]), 76, delta=1)
        self.assertAlmostEqual(int(native.to_gray(red_bgr, 'rgb')[0, 0]), 29, delta=1)
        red_bgra = np.zeros((8, 8, 4), np.uint8)
        red_bgra[..., 2] = 255
        red_bgra[..., 3] = 255
        self.assertAlmostEqual(int(native.to_gray(red_bgra)[0, 0]), 76, delta=1)
        self.assertAlmostEqual(int(native.to_gray(red_bgra, 'rgb')[0, 0]), 29, delta=1)
        gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
        np.testing.assert_array_equal(native.to_gray(gray), gray)

    def test_rejects_bad_images(self):
        with self.assertRaises(TypeError):
            native.to_gray(np.zeros((8, 8), np.float32))
        with self.assertRaises(ValueError):
            native.to_gray(np.zeros((8, 8, 2), np.uint8))
        with self.assertRaises(ValueError):
            native.to_gray(np.zeros(8, np.uint8))
        with self.assertRaises(ValueError):
            native.to_gray(np.zeros((8, 8, 3), np.uint8), 'yuv')

    def test_rejects_bad_configuration(self):
        camera = native.CameraMatrix(600, 320, 600, 240)
        with self.assertRaises(ValueError):
            native.CudaTagsWrapper(native.tag36h11, camera, native.DistCoeffs(),
                                   1, 642, 480)
        with self.assertRaises(ValueError):
            native.CudaTagsWrapper(native.tag36h11, native.CameraMatrix(),
                                   native.DistCoeffs(), 1, 640, 480)

    @unittest.skipIf(native is not None and native.cuda_available(),
                     'module has CUDA')
    def test_without_cuda_construction_fails_cleanly(self):
        camera = native.CameraMatrix(600, 320, 600, 240)
        with self.assertRaisesRegex(RuntimeError, 'without CUDA'):
            native.CudaTagsWrapper(native.tag36h11, camera, native.DistCoeffs(),
                                   1, 640, 480)

    @unittest.skipIf(native is None or not native.cuda_available(), 'no CUDA')
    def test_golden_image_on_gpu(self):
        scene = read_scene(GOLDEN / 'tags_640x480.txt')
        image = read_pgm(GOLDEN / 'tags_640x480.pgm')
        fx, fy, cx, cy = scene['camera']
        detector = native.CudaTagsWrapper(
            native.tag36h11, native.CameraMatrix(fx, cx, fy, cy),
            native.DistCoeffs(), 2, 640, 480, tag_size_m=scene['tag_size'])
        for frame in (image, np.dstack([image] * 3)):
            found = {d.id: d for d in detector.process(frame)}
            self.assertEqual(set(found), set(scene['tags']))
            for tag_id, expected in scene['tags'].items():
                corners = np.array(found[tag_id].corners) - 0.5
                np.testing.assert_allclose(corners, expected['corners'], atol=0.3)
                self.assertIsNotNone(found[tag_id].pose)
                np.testing.assert_allclose(found[tag_id].pose.translation,
                                           expected['translation'], atol=0.03)
        with self.assertRaises(ValueError):
            detector.process(np.zeros((240, 320), np.uint8))


if __name__ == '__main__':
    sys.exit(unittest.main())
