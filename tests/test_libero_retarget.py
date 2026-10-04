"""Task geometry contracts, independent of MuJoCo."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from retarget_libero_object import retarget_transport, load_transport, smoothstep


class RetargetTests(unittest.TestCase):
    def setUp(self):
        self.settings = dict(axis_permutation=[0, 1, 2], axis_signs=[1, 1, 1],
                             horizontal_scale=2., vertical_scale=.5)

    def test_rotated_task_preserves_lateral_and_lift(self):
        # Human +x -> robot +y, human +y lateral -> robot -x.
        t = np.array([0., .5, 1.])
        human = np.array([[10, 20, 30], [10.5, 20.1, 30.2], [11, 20, 30]])
        path, progress, report = retarget_transport(t, human, [11, 20, 30], [2, 3, 4], [2, 5, 4], self.settings)
        np.testing.assert_allclose(path, [[2, 3, 4], [1.8, 4, 4.1], [2, 5, 4]])
        np.testing.assert_allclose(progress, [0, .5, 1])
        self.assertAlmostEqual(report['horizontal_scale_factor'], 2.)
        self.assertAlmostEqual(report['endpoint_correction_norm_m'], 0.)

    def test_partial_transport_correction_is_distributed(self):
        t = np.linspace(0, 1, 101)
        human = np.column_stack([.4*t, t*0, t*0])
        path, _, report = retarget_transport(t, human, [1, 0, 0], [0, 0, 0], [0, 2, 0], self.settings)
        np.testing.assert_allclose(path[:, 1], .8*t+1.2*smoothstep(t))
        np.testing.assert_allclose(path[[0, -1]], [[0, 0, 0], [0, 2, 0]])
        self.assertLess(np.linalg.norm(path[-1]-path[-2]), .01)
        self.assertAlmostEqual(report['endpoint_correction_norm_m'], 1.2)

    def test_axis_permutation_and_sign(self):
        self.settings.update(axis_permutation=[2, 0, 1], axis_signs=[-1, 1, 1])
        path, _, _ = retarget_transport([0, .5, 1], [[0, 0, 0], [0, .2, -.5], [0, 0, -1]],
                                        [0, 0, -1], [0, 0, 0], [1, 0, 0], self.settings)
        np.testing.assert_allclose(path, [[0, 0, 0], [.5, 0, .1], [1, 0, 0]])

    def test_degenerate_and_nonfinite_inputs_rejected(self):
        for goal in ([0, 0, 1], [float('nan'), 0, 0]):
            with self.assertRaises(ValueError):
                retarget_transport([0, 1], [[0, 0, 0], [1, 0, 0]], goal, [0, 0, 0], [1, 0, 0], self.settings)

    def test_loader_selects_transport_and_rejects_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'metadata.json').write_text(json.dumps(dict(transport_start=1, transport_end=1.1, goal_position=[1, 2, 3])))
            (path/'processed.csv').write_text('time,object_x,object_y,object_z,valid_processed,phase\n0,9,9,9,1,pickup\n1,0,0,0,1,transport\n1.1,1,0,0,1,transport\n2,9,9,9,1,lower\n')
            t, p, goal, phase, _ = load_transport(path/'processed.csv', path/'metadata.json')
            np.testing.assert_allclose(t, [0, .1]); np.testing.assert_allclose(p, [[0, 0, 0], [1, 0, 0]])
            self.assertEqual(phase, ['transport', 'transport'])
            with self.assertRaises(ValueError): load_transport(path/'processed.csv', path/'metadata.json', .05)
            (path/'processed.csv').write_text((path/'processed.csv').read_text().replace('1.1,1,0,0,1', '1.1,1,0,0,0'))
            with self.assertRaises(ValueError): load_transport(path/'processed.csv', path/'metadata.json')


if __name__ == '__main__':
    unittest.main()
