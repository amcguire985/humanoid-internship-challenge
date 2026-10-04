import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from describe_object_trajectory import load_trajectory,normalize_trajectory,compute_velocity,segment_motion,compute_metrics

class TaskSpaceTests(unittest.TestCase):
    def test_known_pick_place_motion(self):
        t=np.arange(0,5.01,.05)
        p=np.zeros((len(t),3))
        p[:,2]=np.where(t<1,0,np.where(t<2,t-1,np.where(t<3,1,np.where(t<4,4-t,0))))
        p[:,0]=np.clip(t-2,0,1)
        breaks=np.r_[True,np.zeros(len(t)-1,dtype=bool)]
        relative=normalize_trajectory(p+[2,3,4])
        v,s=compute_velocity(t,p,breaks)
        labels,pickup,placement,_=segment_motion(t,v,s,breaks)
        self.assertTrue({'lift','horizontal_transport','lower','stationary_before_pickup','stationary_after_placement'}<=set(labels))
        self.assertAlmostEqual(pickup,1,delta=.1)
        self.assertAlmostEqual(placement,4,delta=.1)
        metrics=compute_metrics(t,p,relative,breaks,pickup,placement)
        self.assertAlmostEqual(metrics['total_3d_path_length_m'],3)
        self.assertAlmostEqual(metrics['total_horizontal_displacement_m'],1)
        self.assertAlmostEqual(metrics['maximum_vertical_lift_m'],1)

    def test_gap_does_not_create_velocity_or_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.csv'
            path.write_text('time,x,y,z\n0,0,0,0\n1,1,0,0\n2,,,\n3,100,0,0\n4,101,0,0\n')
            t,p,breaks,info=load_trajectory(path)
        self.assertEqual(info['skipped_rows'],1)
        v,s=compute_velocity(t,p,breaks)
        np.testing.assert_allclose(v[:,0],1)
        metrics=compute_metrics(t,p,normalize_trajectory(p),breaks,None,None)
        self.assertEqual(metrics['total_3d_path_length_m'],2)

    def test_stationary_has_no_invented_events(self):
        t=np.arange(10)/10
        p=np.zeros((10,3)); breaks=np.r_[True,np.zeros(9,dtype=bool)]
        v,s=compute_velocity(t,p,breaks)
        labels,pickup,placement,_=segment_motion(t,v,s,breaks)
        self.assertEqual(set(labels),{'stationary'})
        self.assertIsNone(pickup); self.assertIsNone(placement)

if __name__=='__main__': unittest.main()
