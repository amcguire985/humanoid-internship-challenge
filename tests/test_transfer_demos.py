import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from extract_transfer_demos import detect
from process_task_demo import validate_config,clean_trajectory

class MultiTransferTests(unittest.TestCase):
 def test_cleaner_preserves_large_gap(self):
  t=np.arange(30)/30; p=np.column_stack((t,np.zeros(30),np.zeros(30))); p[8:20]=np.nan
  _,smoothed,provenance,_,_,_=clean_trajectory(t,p,np.ones(30),validate_config({}))
  self.assertTrue(np.isnan(smoothed[8:20]).all())
  self.assertTrue(np.all(provenance[8:20]=='missing'))
 def test_rest_detection_keeps_different_endpoints(self):
  t=np.arange(90)/30; p=np.zeros((90,3)); p[30:60,0]=np.linspace(0,.3,30); p[60:,0]=.3
  speed=np.zeros(90); speed[30:60]=.3
  rests=detect(t,p,speed,validate_config({}))
  self.assertEqual(len(rests),2)
  self.assertAlmostEqual(rests[1]['position'][0],.3)
 def test_rest_detection_does_not_merge_lift_and_return(self):
  t=np.arange(90)/30; p=np.zeros((90,3)); p[30:60,2]=.1
  speed=np.zeros(90); speed[30:60]=.3
  self.assertEqual(len(detect(t,p,speed,validate_config({}))),2)

if __name__=='__main__': unittest.main()
