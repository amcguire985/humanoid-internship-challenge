import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bowl_phone_constraints import extract, MotionConstraints
from bowl_human_reference import Reference
from bowl_transport_guidance import Settings
from test_bowl_hybrid import controller

class ConstraintTests(unittest.TestCase):
    def limiter(self,initial=None):
        return MotionConstraints(dict(acceleration_limit_m_s2=.5,allowable_tilt_deg=10.),controller(),Settings(),np.zeros(7) if initial is None else initial,.05)

    def test_extraction_units_and_actual_timestamps(self):
        t=np.linspace(0,2,81)**1.1
        p=np.column_stack([.5*t*t,np.zeros(len(t)),np.zeros(len(t))])
        r=np.repeat(np.eye(3)[None],len(t),axis=0)
        limits=extract(Reference(t,p,r))
        self.assertAlmostEqual(limits['acceleration_limit_m_s2'],1.,places=9)
        self.assertEqual(limits['allowable_tilt_deg'],0.)
        self.assertTrue(np.isfinite(limits['acceleration_limit_m_s2']))

    def test_acceleration_bound_and_action_bounds(self):
        limiter=self.limiter()
        rng=np.random.default_rng(4)
        previous=np.zeros(3)
        for i in range(100):
            action,info=limiter.apply(rng.uniform(-1,1,7),Rotation.from_euler('x',.6).as_matrix(),.05,i*.05)
            velocity=limiter.decode(action[:3])/.05
            self.assertLessEqual(np.linalg.norm(velocity-previous)/.05,.5+1e-10)
            self.assertTrue(np.all(np.abs(action)<=1))
            self.assertLessEqual(np.max(np.abs(info['constraint_rotation_correction'])),.15+1e-10)
            previous=velocity

    def test_constant_velocity_and_yaw_only(self):
        proposed=np.array([.2,-.1,.1,0.,0.,0.,1.])
        limiter=self.limiter(proposed)
        result,info=limiter.apply(proposed,Rotation.from_euler('z',2.).as_matrix(),.05,1.)
        np.testing.assert_allclose(result,proposed,atol=1e-12)
        self.assertAlmostEqual(info['commanded_acceleration_m_s2'],0.)
        np.testing.assert_array_equal(info['constraint_rotation_correction'],[0,0,0])

    def test_no_tilt_correction_below_allowance(self):
        result,info=self.limiter().apply(np.zeros(7),Rotation.from_euler('x',.1).as_matrix(),.05,1.)
        np.testing.assert_array_equal(result[3:6],0)

if __name__=='__main__': unittest.main()
