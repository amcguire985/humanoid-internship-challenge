from pathlib import Path
p=Path('tests/test_transfer_annotations.py'); s=p.read_text(encoding='utf-8-sig')
s=s.replace("if __name__=='__main__': unittest.main()", """ def test_opposite_task_direction_maps_without_time_reversal(self):
  from retarget_libero_object import retarget_transport
  times=np.array([0.,1.,2.]); positions=np.array([[0.,0.,0.],[.5,.1,.2],[1.,0.,0.]])
  settings=dict(axis_permutation=[0,1,2],axis_signs=[1,1,1],horizontal_scale=1.,vertical_scale=1.)
  a,_,_=retarget_transport(times,positions,np.array([1.,0.,0.]),np.zeros(3),np.array([.3,0.,0.]),settings)
  mirrored=positions.copy(); mirrored[:,:2]*=-1
  b,_,_=retarget_transport(times,mirrored,np.array([-1.,0.,0.]),np.zeros(3),np.array([.3,0.,0.]),settings)
  np.testing.assert_allclose(a,b); self.assertGreater(b[1,2],0)

if __name__=='__main__': unittest.main()""")
p.write_text(s)
