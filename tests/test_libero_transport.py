"""Numerical controller contract tests, without requiring MuJoCo."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from replay_libero_transport import (load_human_trajectory, map_to_libero_frame,
    build_desired_eef_trajectory, desired_pose_to_action, inspect_controller, evaluate_tracking)


class TransportTests(unittest.TestCase):
    def setUp(self):
        controller = SimpleNamespace(name='OSC_POSE', use_delta=True, impedance_mode='fixed',
            input_min=-1., input_max=1., output_min=np.array([-.05]*3+[-.5]*3),
            output_max=np.array([.05]*3+[.5]*3))
        self.env = SimpleNamespace(env=SimpleNamespace(robots=[SimpleNamespace(controller=controller)],
                                  action_spec=(-np.ones(7),np.ones(7))))
        self.config = inspect_controller(self.env)

    def test_normalization_stops_at_missing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.csv'
            path.write_text('time_s,x_m,y_m,z_m\n2,10,20,30\n2.05,11,22,33\n2.1,,,\n2.15,99,99,99\n')
            t,p,info = load_human_trajectory(path)
        np.testing.assert_allclose(t,[0,.05])
        np.testing.assert_allclose(p,[[0,0,0],[1,2,3]])
        self.assertEqual(info['source_stop_reason'],'missing_source_position')

    def test_permutation_signs_and_scale(self):
        p,m = map_to_libero_frame([[1,2,3]], [1,0,2], [-1,1,1], [.5,.5,.5])
        np.testing.assert_allclose(p,[[-1,.5,1.5]])

    def test_inverse_scaling_and_orientation_composition(self):
        current = Rotation.from_euler('xyz',[.3,-.2,.1]).as_matrix()
        target = Rotation.from_rotvec([.02,-.03,.04]).as_matrix()@current
        action,clipped = desired_pose_to_action(np.array([.01,-.005,0]),np.zeros(3),target,current,self.config)
        np.testing.assert_allclose(action,[.2,-.1,0,.04,-.06,.08,-1],atol=1e-12)
        recovered = Rotation.from_rotvec(action[3:6]*.5).as_matrix()@current
        np.testing.assert_allclose(recovered,target,atol=1e-12)
        self.assertFalse(clipped)

    def test_closed_loop_correction_and_saturation(self):
        action,clipped = desired_pose_to_action(np.zeros(3),np.array([.2,-.2,0]),np.eye(3),np.eye(3),self.config)
        np.testing.assert_allclose(action,[-.5,.5,0,0,0,0,-1])
        self.assertTrue(clipped)

    def test_resampling_preserves_endpoint_and_speed_bound(self):
        t,p,factor = build_desired_eef_trajectory(np.array([0.,1.]),np.array([[0,0,0],[1,0,0]]),np.array([2,3,4]),20,1,.1)
        np.testing.assert_allclose(p[0],[2,3,4])
        np.testing.assert_allclose(p[-1],[3,3,4])
        self.assertLessEqual(np.max(np.linalg.norm(np.diff(p,axis=0),axis=1)/np.diff(t)),.1000001)
        self.assertEqual(factor,10)

    def test_metrics_use_euclidean_error(self):
        rows=[dict(desired=np.zeros(3),actual=np.array([.003,.004,0]),clipped=False)]
        metrics,errors=evaluate_tracking(rows,True)
        for key in ['mean_error_m','rmse_m','maximum_error_m','final_position_error_m']:
            self.assertAlmostEqual(metrics[key],.005)

    def test_reject_absolute_controller(self):
        self.env.env.robots[0].controller.use_delta=False
        with self.assertRaises(ValueError): inspect_controller(self.env)

if __name__=='__main__': unittest.main()
