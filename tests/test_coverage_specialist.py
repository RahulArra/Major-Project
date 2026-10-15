import unittest
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coverage_patch_utils import (
    binary_or,
    confusion_counts,
    coverage_group,
    label_counts,
    metrics_from_counts,
    patch_windows,
    georeferencing_matches,
    raster_pixel_dimensions_match,
    reassemble_patches,
    transform_aligned,
    validate_parent_splits,
    validate_patch_grid,
)


class CoverageSpecialistTests(unittest.TestCase):
    def test_coverage_groups_include_flood_free_and_exact_boundary(self):
        self.assertEqual(coverage_group(100, 0), "LOW")
        self.assertEqual(coverage_group(100, 29), "LOW")
        self.assertEqual(coverage_group(100, 30), "HIGH")
        self.assertEqual(coverage_group(0, 0), "IGNORE")

    def test_label_counts_exclude_ignore_from_coverage(self):
        counts = label_counts(np.array([[-1, 0], [1, 1]], dtype=np.int8))
        self.assertEqual(counts["valid_pixels"], 3)
        self.assertEqual(counts["ignored_pixels"], 1)
        self.assertEqual(counts["flood_pixels"], 2)
        self.assertAlmostEqual(counts["coverage"], 2 / 3)
        self.assertIsNone(label_counts(np.full((2, 2), -1))["coverage"])

    def test_patch_windows_and_bounds(self):
        windows = patch_windows()
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows, [(0, 0, 256, 256), (0, 256, 256, 256),
                                   (256, 0, 256, 256), (256, 256, 256, 256)])
        validate_patch_grid(256, 256, 256, 256)
        with self.assertRaises(ValueError):
            validate_patch_grid(256, 256, 256, 257)
        with self.assertRaises(ValueError):
            validate_patch_grid(255, 256, 0, 0)

    def test_pixel_dimensions_and_georeferencing_are_reported_separately(self):
        a = SimpleNamespace(width=512, height=512, crs="EPSG:4326", transform=(8.983152841195215e-05, 0, 0))
        b = SimpleNamespace(width=512, height=512, crs="EPSG:4326", transform=(8.983152841195184e-05, 0, 0))
        self.assertTrue(raster_pixel_dimensions_match(a, b))
        self.assertTrue(georeferencing_matches(a, b))
        b.transform = (8.983152841195215e-05, 0, 1e-6)
        self.assertTrue(raster_pixel_dimensions_match(a, b))
        self.assertFalse(georeferencing_matches(a, b))
        b.transform = a.transform
        b.width = 256
        self.assertFalse(raster_pixel_dimensions_match(a, b))

    def test_synchronized_spatial_augmentation_preserves_alignment(self):
        target = np.zeros((4, 4), dtype=np.float32)
        target[0, 1] = 1
        image = np.stack([target, target + 2])
        ignore = np.zeros((4, 4), dtype=bool)
        ignore[0, 1] = True
        img_t, target_t, ignore_t = transform_aligned(
            image, target, ignore, horizontal=True, rotations=1,
        )
        self.assertTrue(np.array_equal(img_t[0], target_t))
        self.assertTrue(np.array_equal(ignore_t, target_t.astype(bool)))

    def test_split_integrity(self):
        self.assertEqual(validate_parent_splits([("a", "train"), ("b", "validation")]),
                         {"a": "train", "b": "validation"})
        with self.assertRaises(ValueError):
            validate_parent_splits([("a", "train"), ("a", "test")])

    def test_patch_reassembly_exactly_covers_parent(self):
        image = np.arange(512 * 512).reshape(512, 512)
        patches = [(r, c, image[r:r + h, c:c + w])
                   for r, c, h, w in patch_windows()]
        self.assertTrue(np.array_equal(reassemble_patches(patches), image))
        with self.assertRaises(ValueError):
            reassemble_patches(patches[:-1])

    def test_metrics_exclude_ignored_pixels(self):
        target = np.array([[1, 0, 1, 0]], dtype=bool)
        prediction = np.array([[1, 1, 0, 1]], dtype=bool)
        ignore = np.array([[False, False, True, False]])
        counts = confusion_counts(prediction, target, ignore)
        self.assertEqual(counts, {"tp": 1, "tn": 0, "fp": 2, "fn": 0, "valid_pixels": 3})
        metrics = metrics_from_counts(counts)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        self.assertAlmostEqual(metrics["dice"], 1 / 2)

    def test_binary_or_fusion(self):
        a = np.array([[True, False], [False, False]])
        b = np.array([[False, False], [True, False]])
        self.assertTrue(np.array_equal(binary_or(a, b), [[True, False], [True, False]]))
        with self.assertRaises(ValueError):
            binary_or(a, np.zeros((1, 2), dtype=bool))


if __name__ == "__main__":
    unittest.main()
