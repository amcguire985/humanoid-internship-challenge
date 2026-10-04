import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from smooth_centers import smooth_positions

class GentleSmoothingTests(unittest.TestCase):
    def test_local_fit_preserves_quadratic_with_irregular_timestamps(self):
        t=np.array([0,.031,.067,.10,.131,.168,.20,.235,.268])
        p=np.column_stack((t,t*t,2+t+t*t))
        np.testing.assert_allclose(smooth_positions(t,p,'savgol',5),p,atol=1e-12)

    def test_gap_reset_and_short_segment_passthrough(self):
        t=np.arange(12)/30
        p=np.zeros((12,3))
        p[5]=np.nan
        p[6:]=10
        for method,strength in [('savgol',5),('savgol',7),('ema',.03)]:
            result=smooth_positions(t,p,method,strength)
            self.assertTrue(np.isnan(result[5]).all())
            np.testing.assert_allclose(result[:5],0,atol=1e-12)
            np.testing.assert_allclose(result[6:],10,atol=1e-12)

    def test_filters_reduce_small_high_frequency_noise(self):
        t=np.arange(120)/30
        p=np.tile((.001*(-1.)**np.arange(120))[:,None],(1,3))
        for method,strength in [('savgol',5),('savgol',7),('ema',.03),('ema',.05)]:
            result=smooth_positions(t,p,method,strength)
            self.assertLess(result[10:-10].std(),p[10:-10].std())

if __name__=='__main__':
    unittest.main()
