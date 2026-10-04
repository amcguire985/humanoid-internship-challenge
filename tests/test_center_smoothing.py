import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_center_smoothing import smooth_positions

class CentreSmoothingTests(unittest.TestCase):
    def test_quadratic_preserved_with_irregular_timing(self):
        times=np.cumsum(np.linspace(.025,.04,20))
        p=np.column_stack([times**2,times,2*times**2-times])
        np.testing.assert_allclose(smooth_positions(times,p,'savgol',window=7),p,atol=1e-12)

    def test_gaps_remain_and_ema_restarts(self):
        times=np.arange(20)/30
        p=np.zeros((20,3))
        p[8:11]=np.nan
        p[11:]=10
        for method in ('savgol','ema'):
            result=smooth_positions(times,p,method)
            self.assertTrue(np.isnan(result[8:11]).all())
            np.testing.assert_allclose(result[:8],0,atol=1e-12)
            np.testing.assert_allclose(result[11:],10,atol=1e-12)

if __name__=='__main__':
    unittest.main()
