import sys,tempfile,unittest,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from annotate_transfer_demos import resolve,initialize

class AnnotationTests(unittest.TestCase):
 def metadata(self):
  return dict(id='demo_002',direction='target -> object',source_video='video.MOV',parameters={},
   pickup_time=1.,transport_start=2.,transport_end=4.,release_time=5.,
   automatic_phase_estimates=dict(pickup_time=.5,transport_start=2.,transport_end=4.,release_time=5.5))
 def test_manual_overrides_and_automatic_preserved(self):
  m=self.metadata(); before=json.dumps(m); a=dict(demo_id='demo_002',grasp_time_seconds=1.2,release_time_seconds=5.2)
  b,s,_=resolve(np.arange(0,6.01,.1),m,a)
  self.assertAlmostEqual(b['pickup_time'],1.2); self.assertAlmostEqual(b['release_time'],5.2)
  self.assertEqual(s['pickup_time']['source'],'manual'); self.assertEqual(json.dumps(m),before)
 def test_optional_manual_transport_wins(self):
  b,_,_=resolve(np.arange(0,6.01,.1),self.metadata(),dict(demo_id='demo_002',transport_start_time_seconds=2.5,transport_end_time_seconds=3.5))
  self.assertAlmostEqual(b['transport_start'],2.5); self.assertAlmostEqual(b['transport_end'],3.5)
 def test_invalid_manual_order_not_guessed_over(self):
  with self.assertRaises(ValueError): resolve(np.arange(0,6.01,.1),self.metadata(),dict(demo_id='demo_002',grasp_time_seconds=4.5))
 def test_existing_file_never_overwritten(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'annotations.json'; content='{"demo_id":"demo_002","grasp_time_seconds":1.23}'
   path.write_text(content); initialize(Path(tmp),path); self.assertEqual(path.read_text(),content)
 def test_reverse_direction_is_not_a_timing_rejection(self):
  b,_,_=resolve(np.arange(0,6.01,.1),self.metadata(),dict(demo_id='demo_002',grasp_time_seconds=1,release_time_seconds=5))
  self.assertEqual(b['release_time'],5)

 def test_opposite_task_direction_maps_without_time_reversal(self):
  from retarget_libero_object import retarget_transport
  times=np.array([0.,1.,2.]); positions=np.array([[0.,0.,0.],[.5,.1,.2],[1.,0.,0.]])
  settings=dict(axis_permutation=[0,1,2],axis_signs=[1,1,1],horizontal_scale=1.,vertical_scale=1.)
  a,_,_=retarget_transport(times,positions,np.array([1.,0.,0.]),np.zeros(3),np.array([.3,0.,0.]),settings)
  mirrored=positions.copy(); mirrored[:,:2]*=-1
  b,_,_=retarget_transport(times,mirrored,np.array([-1.,0.,0.]),np.zeros(3),np.array([.3,0.,0.]),settings)
  np.testing.assert_allclose(a,b); self.assertGreater(b[1,2],0)

if __name__=='__main__': unittest.main()
