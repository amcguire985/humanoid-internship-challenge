"""Geometric checks independent of detector and video content."""
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from track_apriltags import estimate_pose, fit_intrinsics, marker_points, relative_transform

class GeometryTests(unittest.TestCase):
    def test_pose_and_size_scale(self):
        k = np.array([[1500., 0, 540], [0, 1500, 960], [0, 0, 1]])
        rotation = np.array([2.8, 0.2, 0.1])
        translation = np.array([0.1, -0.03, 0.8])
        image = cv2.projectPoints(marker_points(.1), rotation, translation, k, np.zeros(5))[0]
        pose = estimate_pose(image, .1, k, np.zeros(5))
        np.testing.assert_allclose(pose["tvec"].ravel(), translation, atol=1e-7)
        self.assertLess(pose["error"], 1e-6)
        scaled = estimate_pose(image, .2, k, np.zeros(5))
        np.testing.assert_allclose(scaled["tvec"].ravel(), translation * 2, atol=1e-7)

    def test_relative_pose_cancels_camera_motion(self):
        reference = np.eye(4)
        reference[:3, :3] = cv2.Rodrigues(np.array([.1, .2, 1.]))[0]
        reference[:3, 3] = [1, 2, 3]
        relative = np.eye(4)
        relative[:3, :3] = cv2.Rodrigues(np.array([.4, -.2, .3]))[0]
        relative[:3, 3] = [.2, .3, .4]
        moving = reference @ relative
        camera_motion = np.eye(4)
        camera_motion[:3, :3] = cv2.Rodrigues(np.array([-.3, .1, .5]))[0]
        camera_motion[:3, 3] = [-.3, .1, .6]
        np.testing.assert_allclose(relative_transform(reference, moving), relative, atol=1e-12)
        np.testing.assert_allclose(relative_transform(camera_motion @ reference, camera_motion @ moving), relative, atol=1e-12)

    def test_crop_pixel_coordinates(self):
        k = [[1000, 0, 768], [0, 1000, 1024], [0, 0, 1]]
        adjusted = fit_intrinsics(k, (1536, 2048), (1080, 1920), "center-crop")
        np.testing.assert_allclose(adjusted, [[937.5, 0, 540], [0, 937.5, 960], [0, 0, 1]])
        np.testing.assert_allclose(k, [[1000, 0, 768], [0, 1000, 1024], [0, 0, 1]])

    def test_strict_resolution_check(self):
        with self.assertRaises(ValueError):
            fit_intrinsics(np.eye(3), (1536, 2048), (1080, 1920), "strict")

if __name__ == "__main__":
    unittest.main()

