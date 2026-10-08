import sys, unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bowl_placement import Placement
from bowl_transport_guidance import pose

def controller():
    return dict(input_min=-np.ones(6),input_max=np.ones(6),output_min=-np.array([.05]*3+[.5]*3),output_max=np.array([.05]*3+[.5]*3),action_min=-np.ones(7),action_max=np.ones(7))

def row(step,p=(0,0,1.03),grasp=True,aperture=.03,success=False):
    return dict(timestep=step,bowl_position_m=list(p),bowl_rotation_world_from_object=np.eye(3).tolist(),eef_pose=pose(np.array(p)+[.04,0,.02],np.eye(3)).tolist(),grasp=grasp,gripper_aperture_m=aperture,tilt_deg=0.,libero_success=success)

class PlacementTests(unittest.TestCase):
    def make(self): return Placement([0,0,1],pose([-.04,0,-.02],np.eye(3)),controller())
    def enter(self,c):
        for i in range(3): c.observe(row(i),False,False,.05)
        self.assertEqual((c.phase,c.subphase),('PLACE','ALIGN'))
    def test_handoff_is_latched_and_requires_height_grasp(self):
        c=self.make()
        for i in range(4): c.observe(row(i,p=(0,0,1.01)),False,False,.05)
        self.assertEqual(c.phase,'TRANSPORT')
        self.enter(c); c.observe(row(4,p=(.2,0,1.03)),False,False,.05)
        self.assertEqual(c.phase,'PLACE')
    def test_transform_scaling_and_yaw_free_bounds(self):
        c=self.make(); self.enter(c)
        b=pose([.02,0,1.03],Rotation.from_euler('z',1.2).as_matrix()); e=b@np.linalg.inv(c.relative)
        a,d=c.action(b,e)
        self.assertLessEqual(np.linalg.norm(a[:3]*.05),.003+1e-12)
        np.testing.assert_allclose(a[3:6],0,atol=1e-12)
        self.assertEqual(a[6],1); self.assertFalse(d['guidance_active'])
    def test_release_requires_support_then_never_closes(self):
        c=self.make(); self.enter(c)
        for i in range(3,7): c.observe(row(i),False,False,.05)
        self.assertEqual(c.subphase,'LOWER')
        for i in range(7,12): c.observe(row(i,p=(0,0,1)),False,False,.05)
        self.assertEqual(c.phase,'PLACE')
        for i in range(12,16): c.observe(row(i,p=(0,0,1)),True,False,.05)
        self.assertEqual(c.phase,'RELEASE')
        for i in range(16,20):
            r=row(i,p=(0,0,1),grasp=False,aperture=.05); c.observe(r,True,False,.05)
            a,_=c.action(pose(r['bowl_position_m'],np.eye(3)),np.array(r['eef_pose']))
            self.assertEqual(a[6],-1)
        self.assertEqual(c.phase,'VERIFY')
    def test_slip_and_timeout_fail(self):
        c=self.make(); self.enter(c); c.observe(row(4),False,True,.05)
        self.assertEqual(c.failure,'placement_slip')
        c=self.make(); self.enter(c); c.observe(row(70,p=(.08,0,1.03)),False,False,.05)
        self.assertEqual(c.failure,'placement_align_timeout')

    def test_place_observer_ignores_policy_and_human_corrections(self):
        from evaluate_bowl_hybrid import HybridObserver
        from bowl_transport_guidance import Phases,Settings
        observer=HybridObserver.__new__(HybridObserver)
        observer.condition='C'; observer.phases=Phases(Settings()); observer.phases.phase='PLACE'; observer.phases.start=(0,0.)
        observer.placement=self.make(); self.enter(observer.placement)
        observer.guidance=object(); observer.slip_latched=False; observer.controller=controller()
        observer.rows=[row(3)]; observer.eef_pose=lambda:np.array(observer.rows[-1]['eef_pose'])
        observer.step_number=3; observer.capture=lambda obs,action:None
        received=[]
        observer.original_step=lambda action:(received.append(action.copy()) or {},0.,False,False,{})
        observer.step(np.array([1.,-1.,1.,1.,1.,1.,-1.]))
        self.assertFalse(observer.extra['guidance_active']); self.assertTrue(observer.extra['placement_control'])
        self.assertEqual(received[-1][6],1.)
        self.assertNotIn('correction',observer.extra)

    def test_transport_metrics_exclude_fall_and_stop_at_place(self):
        from evaluate_bowl_hybrid import analyze
        def records(phases):
            out=[]
            for i,phase in enumerate(phases):
                r=row(i);r.update(sim_time_s=i*.05,phase=phase,failure_reason=None)
                r['tilt_deg']=float(i);out.append(r)
            return out
        rows=records(['GRASP','TRANSPORT','TRANSPORT','PLACE','RELEASE'])
        rows[4]['tilt_deg']=90
        summary,_=analyze(rows);self.assertEqual(summary['transport_end_timestep'],3);self.assertEqual(summary['max_tilt_deg'],3)
        rows=records(['GRASP','TRANSPORT','TRANSPORT','TRANSPORT','FAILED'])
        for r in rows[3:]:r['grasp']=False;r['tilt_deg']=90;r['bowl_position_m'][2]=.5
        summary,acc=analyze(rows);self.assertEqual(summary['transport_end_timestep'],2);self.assertEqual(summary['max_tilt_deg'],2)
        self.assertIsNone(acc[3])
