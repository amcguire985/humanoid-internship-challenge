import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gap_fill_centers import fill_position_gaps

class PositionGapTests(unittest.TestCase):
    def test_ten_frame_gap_fills_without_changing_observations(self):
        times=np.arange(14)/30
        p=np.full((14,3),np.nan)
        p[1]=[0,1,2]
        p[12]=[11,12,13]
        result,labels,gaps=fill_position_gaps(times,p,10)
        np.testing.assert_allclose(result[2:12],np.arange(1,11)[:,None]+np.array([0,1,2]))
        np.testing.assert_array_equal(result[[1,12]],p[[1,12]])
        self.assertEqual(list(labels[[0,13]]),['missing','missing'])
        self.assertEqual(np.sum(labels=='interpolated'),10)
        self.assertGreater(gaps[0]['endpoint_span_s'],.15)

    def test_long_gap_and_optional_time_cap_remain_missing(self):
        p=np.full((13,3),np.nan)
        p[0]=0
        p[12]=12
        result,labels,_=fill_position_gaps(np.arange(13),p,10)
        self.assertEqual(np.sum(labels=='interpolated'),0)
        p[11]=11
        result,labels,_=fill_position_gaps(np.arange(13),p,10,max_span=.15)
        self.assertEqual(np.sum(labels=='interpolated'),0)

    def test_irregular_timing(self):
        result,labels,_=fill_position_gaps([0,.1,.4],np.array([[0,0,0],[np.nan]*3,[4,8,12]]),10)
        np.testing.assert_allclose(result[1],[1,2,3])

if __name__=='__main__':
    unittest.main()
