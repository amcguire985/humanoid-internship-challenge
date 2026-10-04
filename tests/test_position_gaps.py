"""Boundary checks for interpolation-only centre trajectories."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from postprocess_trajectory import fill_position_gaps

class PositionGapTests(unittest.TestCase):
    def test_ten_missing_frames_fill_but_eleven_do_not(self):
        for missing in (10,11):
            times=np.arange(missing+2)/30
            positions=np.full((missing+2,3),np.nan)
            positions[0]=[0,1,2]
            positions[-1]=[3,4,5]
            filled,labels=fill_position_gaps(times,positions,10,.4)
            np.testing.assert_array_equal(filled[[0,-1]],positions[[0,-1]])
            if missing==10:
                self.assertTrue((labels[1:-1]=='interpolated').all())
                np.testing.assert_allclose(filled[:,0],np.linspace(0,3,missing+2))
            else:
                self.assertTrue(np.isnan(filled[1:-1]).all())

    def test_no_extrapolation_or_long_span_fill(self):
        times=np.array([0,.1,.2,.8,.9])
        positions=np.full((5,3),np.nan)
        positions[1]=0
        positions[3]=1
        filled,labels=fill_position_gaps(times,positions,10,.4)
        np.testing.assert_array_equal(labels,['missing','observed','missing','observed','missing'])
        self.assertTrue(np.isnan(filled[[0,2,4]]).all())

if __name__=='__main__':
    unittest.main()
