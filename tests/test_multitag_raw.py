"""Check stationary registration and current-frame reference selection."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from track_multitag_raw import register_world, camera_world, cube_center

class MultiTagTests(unittest.TestCase):
    def pose(self, xyz, error=1):
        t = np.eye(4)
        t[:3, 3] = xyz
        return {'transform': t, 'error': error}

    def test_cube_center_agrees_across_orthogonal_and_opposite_faces(self):
        import cv2
        centre = np.array([.1, .2, .3])
        for rvec in ([0., 0., 0.], [np.pi, 0., 0.], [np.pi / 2, 0., .0]):
            transform = np.eye(4)
            transform[:3, :3] = cv2.Rodrigues(np.array(rvec))[0]
            transform[:3, 3] = centre + transform[:3, 2] * .0225
            np.testing.assert_allclose(cube_center(transform, .045), centre)
        with self.assertRaises(ValueError):
            cube_center(np.eye(4), 0)

    def test_registration_chain_and_camera_motion(self):
        frames = [{0: [self.pose([0, 0, 1])], 1: [self.pose([.2, 0, 1])]},
                  {1: [self.pose([.5, 0, 2])], 2: [self.pose([.5, .3, 2])]}]
        layout = register_world(frames)
        np.testing.assert_allclose(layout[2][:3, 3], [.2, .3, 0])
        world, source = camera_world({2: [self.pose([.7, .8, 3])]}, layout)
        self.assertEqual(source, 2)
        np.testing.assert_allclose(world[:3, 3], [.5, .5, 3])
        self.assertEqual(camera_world({}, layout), (None, None))

    def test_duplicates_and_disconnected_tags_do_not_register(self):
        layout = register_world([{0: [self.pose([0, 0, 1])],
                                  1: [self.pose([1, 0, 1]), self.pose([2, 0, 1])]},
                                 {2: [self.pose([0, 0, 1])], 3: [self.pose([1, 0, 1])]}])
        self.assertEqual(set(layout), {0})
        self.assertEqual(camera_world({3: [self.pose([1, 0, 1])]}, layout), (None, None))

if __name__ == '__main__':
    unittest.main()
