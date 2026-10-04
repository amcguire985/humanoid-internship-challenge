"""Controlled perturbation contracts; no additional physical trials."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from libero_xy_trials import assert_xy_only, largest_valid_on_ray


class PerturbationTests(unittest.TestCase):
    def test_only_object_xy_may_change(self):
        before = np.arange(15, dtype=float)
        after = before.copy()
        after[5:7] += [.02, -.02]
        assert_xy_only(before, after, slice(5, 12), [.02, -.02])
        for index in [0, 7, 8, 12]:
            damaged = after.copy()
            damaged[index] += .001
            with self.assertRaises(ValueError):
                assert_xy_only(before, damaged, slice(5, 12), [.02, -.02])

    def test_valid_offset_is_unchanged(self):
        requested = np.array([-.02, .02])
        actual, _, adjustment = largest_valid_on_ray(requested, lambda p: (True, {}))
        np.testing.assert_array_equal(actual, requested)
        self.assertFalse(adjustment['reduced'])

    def test_reduction_preserves_direction_and_boundary(self):
        requested = np.array([-.04, .04])
        actual, _, adjustment = largest_valid_on_ray(
            requested, lambda p: (np.linalg.norm(p) <= .025, {}))
        self.assertTrue(adjustment['reduced'])
        self.assertLessEqual(np.linalg.norm(actual), .025)
        self.assertLess(.025 - np.linalg.norm(actual), 1e-6)
        self.assertAlmostEqual(actual[0] / requested[0], actual[1] / requested[1])

    def test_invalid_nominal_is_rejected(self):
        with self.assertRaises(ValueError):
            largest_valid_on_ray([0, 0], lambda p: (False, {'reason': 'collision'}))


if __name__ == '__main__':
    unittest.main()
