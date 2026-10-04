import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from postprocess_trajectory import fill_gaps,smooth,interpolate_rotation

class PostprocessingTests(unittest.TestCase):
    def test_shortest_rotation_path_and_quaternion_sign(self):
        q=Rotation.from_euler('z',[[179],[-179]],degrees=True).as_quat()
        mid=interpolate_rotation(q[0],q[1],0.5)
        self.assertAlmostEqual(np.rad2deg(Rotation.from_quat(mid).magnitude()),180)
        np.testing.assert_allclose(Rotation.from_quat(interpolate_rotation(q[0],-q[0],0.5)).as_matrix(),Rotation.from_quat(q[0]).as_matrix(),atol=1e-12)

    def test_timestamp_interpolation_and_gap_limits(self):
        t=np.array([0,.02,.06,.1,.14,.18,.22,.26])
        p=np.column_stack([t,t*2,t*3])
        q=Rotation.from_euler('z',t[:,None]).as_quat()
        p[1]=np.nan; q[1]=np.nan
        p[3:7]=np.nan; q[3:7]=np.nan
        out,rot,labels=fill_gaps(t,p,q)
        np.testing.assert_allclose(out[1],[.02,.04,.06])
        self.assertEqual(labels[1],'interpolated')
        self.assertTrue(np.isnan(out[3:7]).all())
        self.assertAlmostEqual(Rotation.from_quat(rot[1]).as_rotvec()[2],.02)
        out,_,_=fill_gaps(np.array([0,.5,1]),p[:3],q[:3])
        self.assertTrue(np.isnan(out[1]).all())

    def test_ema_reset_and_constant_pose(self):
        t=np.arange(6)*.03
        p=np.array([[1.,0,0],[1,0,0],[np.nan]*3,[5,0,0],[6,0,0],[6,0,0]])
        q=np.tile([0.,0,0,1],(6,1)); q[2]=np.nan
        out,rot=smooth(t,p,q,.05)
        np.testing.assert_allclose(out[:2],p[:2])
        np.testing.assert_allclose(out[3],p[3])
        self.assertAlmostEqual(out[4,0],5+(-np.expm1(-.03/.05)))
        np.testing.assert_allclose(np.linalg.norm(rot[[0,1,3,4,5]],axis=1),1)

    def test_no_endpoint_extrapolation(self):
        t=np.arange(4)*.03
        p=np.ones((4,3)); q=np.tile([0.,0,0,1],(4,1))
        p[[0,3]]=np.nan; q[[0,3]]=np.nan
        out,_,labels=fill_gaps(t,p,q)
        self.assertTrue(np.isnan(out[[0,3]]).all())
        self.assertEqual(list(labels),['missing','observed','observed','missing'])

if __name__=='__main__': unittest.main()
