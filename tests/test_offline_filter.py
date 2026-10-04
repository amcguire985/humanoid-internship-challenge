import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from filter_trajectory import reject_spikes,filter_segments,evaluate,gap_records
from postprocess_trajectory import fill_gaps

class OfflineFilterTests(unittest.TestCase):
    def test_spike_rejection_preserves_steps_and_fast_motion(self):
        t=np.arange(9)/30;p=np.zeros((9,3));q=np.tile([0.,0,0,1],(9,1))
        p[4,0]=.5
        cp,cq,rejected,_=reject_spikes(t,p,q)
        self.assertEqual(np.flatnonzero(rejected).tolist(),[4])
        fp,_,labels=fill_gaps(t,cp,cq)
        np.testing.assert_allclose(fp,0)
        self.assertEqual(labels[4],'interpolated')
        p[4:,0]=.5
        self.assertFalse(reject_spikes(t,p,q)[2].any())
        p[:,0]=t*3
        self.assertFalse(reject_spikes(t,p,q)[2].any())

    def test_rotation_spike(self):
        t=np.arange(7)/30;p=np.zeros((7,3));rv=np.zeros((7,3));rv[3,2]=np.pi/2
        self.assertEqual(np.flatnonzero(reject_spikes(t,p,Rotation.from_rotvec(rv).as_quat())[2]).tolist(),[3])

    def test_sg_preserves_polynomial_and_rotation_wrap(self):
        t=np.linspace(0,1,31);p=np.column_stack([t,t*t,np.ones(len(t))])
        rv=np.column_stack([np.zeros(len(t)),np.zeros(len(t)),np.deg2rad(170+20*t)])
        q=Rotation.from_rotvec(rv).as_quat();q[::2]*=-1
        fp,fq,_=filter_segments(t,p,q)
        np.testing.assert_allclose(fp,p,atol=1e-10)
        np.testing.assert_allclose((Rotation.from_quat(q).inv()*Rotation.from_quat(fq)).magnitude(),0,atol=1e-10)

    def test_gaps_and_short_segments_are_not_bridged(self):
        t=np.arange(20)/30;p=np.zeros((20,3));q=np.tile([0.,0,0,1],(20,1))
        p[8:12]=np.nan;q[8:12]=np.nan;p[12:]=1
        for cutoff in [None,3.]:
            fp,fq,_=filter_segments(t,p,q,cutoff=cutoff)
            self.assertTrue(np.isnan(fp[8:12]).all())
            np.testing.assert_allclose(fp[:8],0,atol=1e-10)
            np.testing.assert_allclose(fp[12:],1,atol=1e-10)

    def test_butterworth_reduces_noise_with_unit_rotations(self):
        t=np.arange(121)/30;wave=np.sin(2*np.pi*9*t)*.01
        p=np.column_stack([wave,wave,wave]);rv=np.column_stack([wave,wave,wave]);q=Rotation.from_rotvec(rv).as_quat();q[::2]*=-1
        fp,fq,_=filter_segments(t,p,q,cutoff=3.)
        self.assertLess(fp[15:-15].std(),p[15:-15].std()/4)
        np.testing.assert_allclose(np.linalg.norm(fq,axis=1),1,atol=1e-12)
        with self.assertRaises(ValueError):filter_segments(t,p,q,cutoff=20.)

    def test_metrics_exclude_gaps_and_record_lengths(self):
        t=np.arange(6,dtype=float);p=np.column_stack([t,np.zeros(6),np.zeros(6)]);q=np.tile([0.,0,0,1],(6,1))
        p[2]=np.nan;q[2]=np.nan
        m=evaluate(t,p,q,p,q,np.isfinite(p).all(axis=1),[],1.1)
        self.assertEqual(m['path_length_m'],3.)
        self.assertEqual(m['peak_velocity_m_s'],1.)
        self.assertEqual(m['max_deviation_all_raw_mm'],0.)
        gaps=gap_records(t,np.array([False,True,True,False,False,False]),np.zeros(6,dtype=bool))
        self.assertEqual(gaps[0]['frames'],2)
        self.assertEqual(gaps[0]['endpoint_span_s'],3.)

if __name__=='__main__':unittest.main()
