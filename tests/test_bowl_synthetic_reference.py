"""Synthetic reference checks without CUDA or LIBERO."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from bowl_synthetic_reference import generate
from bowl_human_reference import Reference
from bowl_transport_guidance import Guidance, Settings, pose
from test_bowl_hybrid import controller

class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.start=np.array([0.,0.,.95])
        self.goal=np.array([.3,.1,.97])
        self.r=Rotation.from_euler('xyz',[.2,.1,.7]).as_matrix()

    def test_endpoints_timing_orientation_and_continuity(self):
        ref=generate(self.start,self.goal,self.r)
        np.testing.assert_array_equal(ref.p[0],self.start)
        np.testing.assert_array_equal(ref.p[-1],self.goal)
        self.assertEqual(ref.t[-1],10.8)
        np.testing.assert_allclose(np.diff(ref.t),.05)
        np.testing.assert_allclose(ref.r,np.repeat(self.r[None],len(ref.t),axis=0))
        speed=np.linalg.norm(np.diff(ref.p,axis=0),axis=1)/np.diff(ref.t)
        self.assertLess(speed.max(),1.875*np.linalg.norm(self.goal-self.start)/10.8+1e-10)
        self.assertLess(max(speed[0],speed[-1]),1e-4)
        p,_=ref.sample([5.4-1e-8,5.4+1e-8])
        self.assertLess(np.linalg.norm(p[1]-p[0]),1e-8)

    def test_existing_loader_alignment_and_position_only_guidance(self):
        ref=generate(self.start,self.goal,self.r)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'reference.csv'; ref.save(path)
            loaded=Reference.load(path)
        bowl=pose(self.start,self.r); eef=pose(self.start+[.02,0,.05],self.r)
        guide=Guidance(loaded,bowl,eef,self.goal,controller(),Settings(orientation_enabled=False))
        np.testing.assert_allclose(guide.reference.p,ref.p,atol=1e-12)
        np.testing.assert_allclose(guide.reference.r,ref.r,atol=1e-12)
        policy=np.array([.1,.1,.1,.05,-.03,.02,1.])
        action,info=guide.apply(policy,bowl,eef,1.,.05)
        np.testing.assert_array_equal(action[3:6],policy[3:6])
        np.testing.assert_allclose(np.array(info['desired_eef_pose'])[:3,:3],eef[:3,:3])
        self.assertEqual(info['orientation_mode'],'position_only')

    def test_configurable_duration_and_invalid_input(self):
        self.assertEqual(generate(self.start,self.goal,self.r,duration=7.25).t[-1],7.25)
        for duration in (0,-1,float('nan')):
            with self.assertRaises(ValueError): generate(self.start,self.goal,self.r,duration=duration)
        with self.assertRaises(ValueError): generate(self.start,self.start,self.r)

if __name__=='__main__': unittest.main()
