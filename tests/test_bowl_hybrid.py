"""Offline hybrid integration checks; no model download or simulator required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bowl_human_reference import Reference,prepare,rotation
from bowl_transport_guidance import Guidance,Phases,Settings,pose
from calibrate_bowl_reference import gravity_rotation
from evaluate_bowl_hybrid import analyze,DynamicPrompt,CONTEXT,LIQUID,HybridObserver,compare
import evaluate_bowl_liquid as official
from unittest.mock import patch


def reference():
    t=np.arange(0,1.01,.05)
    p=np.column_stack([.2*t,np.zeros(len(t)),.04*np.sin(np.pi*t)])
    r=Rotation.from_euler('x',(10*np.sin(np.pi*t))[:,None],degrees=True).as_matrix()
    return Reference(t,p,r,dict(gravity_verified=True,mounting_verified=True,calibration_verified=True))


def controller():
    return dict(input_min=-np.ones(6),input_max=np.ones(6),output_min=-np.array([.05]*3+[.5]*3),output_max=np.array([.05]*3+[.5]*3),action_min=-np.ones(7),action_max=np.ones(7))


class HybridTests(unittest.TestCase):
    def test_gravity_rotation_and_pose_inverse(self):
        gravity=gravity_rotation([0,1,0]); np.testing.assert_allclose(gravity@[0,1,0],[0,0,1])
        rotation(gravity)
        g=pose([.1,.2,.4],Rotation.from_euler('xyz',[.2,.3,.4]).as_matrix())
        b=pose([.11,.23,.37],Rotation.from_euler('xyz',[.4,.1,.2]).as_matrix())
        rel=np.linalg.inv(g)@b
        np.testing.assert_allclose(b@np.linalg.inv(rel),g,atol=1e-12)
        with self.assertRaises(ValueError): rotation(np.diag([1,1,-1]))

    def test_alignment_preserves_tilt_time_and_endpoints(self):
        ref=reference(); aligned=ref.aligned(np.array([.1,.2,.4]),[.1,.6,.42])
        np.testing.assert_allclose(aligned.p[0],[.1,.2,.4]); np.testing.assert_allclose(aligned.p[-1],[.1,.6,.42])
        np.testing.assert_allclose(aligned.r[:,2,2],ref.r[:,2,2]); np.testing.assert_array_equal(aligned.t,ref.t)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ref.csv'; aligned.save(path); loaded=Reference.load(path)
            np.testing.assert_allclose(loaded.p,aligned.p); np.testing.assert_allclose(loaded.r,aligned.r,atol=1e-12)
            meta=json.loads(Path(str(path)+'.json').read_text()); meta['gravity_verified']=False; Path(str(path)+'.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError): Reference.load(path)

    def test_resampling_rotation_is_proper(self):
        ref=reference(); p,r=ref.sample([0,.123,1,1.5]); np.testing.assert_allclose(p[-1],ref.p[-1])
        for matrix in r: rotation(matrix)

    def test_state_requires_sustained_real_grasp_and_lift(self):
        machine=Phases(Settings(grasp_timeout_steps=10))
        for i in range(1,4): machine.update(i,i*.05,False,.1,True,False,False)
        self.assertEqual(machine.phase,'GRASP'); self.assertIsNone(machine.start)
        machine.update(4,.2,True,.03,True,False,False); machine.update(5,.25,False,.03,True,False,False)
        for i in range(6,9): machine.update(i,i*.05,True,.03,True,False,False)
        self.assertEqual(machine.phase,'TRANSPORT'); self.assertEqual(machine.start[0],8)
        for i in range(9,12): machine.update(i,i*.05,False,.03,True,False,False)
        self.assertEqual(machine.phase,'FAILED'); self.assertEqual(machine.failure,'confirmed_grasp_loss')

    def test_timeout_and_release(self):
        machine=Phases(Settings(grasp_timeout_steps=2)); machine.update(2,.1,False,0,False,False,False)
        self.assertEqual(machine.failure,'grasp_timeout')
        machine=Phases(Settings(confirm_steps=1)); machine.update(1,.05,True,.03,True,False,False); machine.update(2,.1,False,.02,False,True,False)
        self.assertEqual(machine.phase,'RELEASE'); machine.update(3,.15,False,0,False,True,True); self.assertEqual(machine.phase,'DONE')

    def test_guidance_bounds_ramp_slew_slip(self):
        start=pose([0,0,.4],np.eye(3)); eef=pose([0,0,.45],np.eye(3)); s=Settings()
        guide=Guidance(reference(),start,eef,[.2,0,.4],controller(),s)
        model=np.array([.5,.2,.1,.1,0,0,-1.])
        result,info=guide.apply(model,start,eef,0,.05)
        np.testing.assert_allclose(result[:6],model[:6]); self.assertEqual(result[6],1)
        result,info=guide.apply(model,start,eef,.1,.05)
        self.assertEqual(result.shape,(7,)); self.assertTrue(np.all(np.abs(result)<=1))
        self.assertLessEqual(np.max(np.abs(np.array(info['correction']))),s.correction_slew_per_s*.05+1e-10)
        result,info=guide.apply(model,pose([.1,0,.4],np.eye(3)),eef,.2,.05)
        self.assertTrue(info['slip_detected']); np.testing.assert_array_equal(result,model)

    def test_human_orientation_and_time_change_correction(self):
        start=pose([0,0,.4],np.eye(3)); eef=pose([0,0,.45],np.eye(3)); s=Settings(ramp_s=.01,correction_slew_per_s=10,correction_limit=1,position_blend=1,orientation_blend=1)
        first=Guidance(reference(),start,eef,[.2,0,.4],controller(),s)
        ref=reference(); ref.r=np.repeat(np.eye(3)[None],len(ref.t),axis=0)
        second=Guidance(ref,start,eef,[.2,0,.4],controller(),s)
        a,_=first.apply(np.zeros(7),start,eef,.5,.05); b,_=second.apply(np.zeros(7),start,eef,.5,.05)
        self.assertGreater(np.linalg.norm(a[3:6]-b[3:6]),.01)
        later,_=second.apply(np.zeros(7),start,eef,.9,.05)
        self.assertGreater(np.linalg.norm(later[:3]-b[:3]),.01)

    def test_metrics_exclude_pregrasp_and_failed_transport_is_null(self):
        rows=[]
        for i in range(9):
            t=i*.05
            rows.append(dict(timestep=i,sim_time_s=t,bowl_position_m=[t*t,0,.4 if i<2 else .43],tilt_deg=100 if i<2 else i,grasp=i>=2,phase='APPROACH' if i<2 else 'TRANSPORT',libero_success=False,failure_reason=None))
        rows[-1].update(phase='DONE',libero_success=True)
        summary,a=analyze(rows); self.assertEqual(summary['max_tilt_deg'],8); self.assertAlmostEqual(summary['max_acceleration_m_s2'],2); self.assertIsNone(a[2])
        for r in rows: r['phase']='FAILED'
        summary,_=analyze(rows); self.assertIsNone(summary['max_tilt_deg']); self.assertIsNone(summary['max_acceleration_m_s2'])

    def test_comparison_pairs_states_and_exports_unavailable_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for condition,tilt,acc in [('B',12.,2.),('C',8.,1.)]:
                output=root/condition; episode=output/'ground_truth/episode_000'; episode.mkdir(parents=True)
                summary=dict(condition=condition,max_tilt_deg=tilt,max_acceleration_m_s2=acc,transport_duration_s=3.,libero_success=True)
                (output/'summary.json').write_text(json.dumps(summary))
                init=dict(seed=0,init_state_index=0,init_state_sha256='same',bddl_sha256='same',local_opening_axis=[0,0,1])
                (episode/'initialization.json').write_text(json.dumps(init))
                (output/'experiment.json').write_text(json.dumps(dict(checkpoint_sha256={'model':'same'},settings={})))
                entry=dict(timestep=60,sim_time_s=3.,phase='TRANSPORT',bowl_position_m=[0,0,.4],bowl_rotation_world_from_object=np.eye(3).tolist(),eef_pose=np.eye(4).tolist(),gripper_qpos=[.01,-.01])
                (episode/'trajectory.jsonl').write_text(json.dumps(entry)+'\n')
            compare(root); result=json.loads((root/'comparison.json').read_text())
            self.assertEqual(result['delta_C_minus_B']['max_tilt_deg'],-4)
            self.assertEqual(result['grasp_entry_difference_B_C']['bowl_position_difference_m'],0)
            self.assertTrue((root/'comparison.csv').exists()); self.assertTrue((root/'comparison.png').exists())
            path=root/'C/ground_truth/episode_000/initialization.json'; init=json.loads(path.read_text()); init['init_state_sha256']='different'; path.write_text(json.dumps(init))
            with self.assertRaises(ValueError): compare(root)

    def test_observer_preserves_policy_actions_and_truncates_on_failure(self):
        observer=HybridObserver.__new__(HybridObserver)
        observer.condition='B'; observer.phases=Phases(Settings()); observer.slip_latched=False; observer.step_number=0
        received=[]
        def step(action):
            received.append(action.copy()); return {},0.,False,False,{'is_success':False}
        observer.original_step=step; observer.capture=lambda obs,action:None
        model=np.array([.4,-.2,.1,0,.3,0,1.])
        result=observer.step(model); np.testing.assert_array_equal(received[-1],model); self.assertFalse(result[3])
        observer.phases.phase='TRANSPORT'; observer.phases.start=(1,.05)
        result=observer.step(model); np.testing.assert_array_equal(received[-1],model); self.assertEqual(observer.extra['policy_instruction'],LIQUID)
        observer.phases.phase='FAILED'; observer.phases.failure='grasp_timeout'
        result=observer.step(model); self.assertTrue(result[3]); self.assertFalse(result[4]['is_success']); self.assertEqual(result[4]['hybrid_failure'],'grasp_timeout')

    def test_preparation_uses_object_poses_and_exact_unscaled_interval(self):
        import csv
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); detections=root/'detections.csv'; centers=root/'centers.csv'; cfg=root/'calibration.json'
            config=dict(gravity_verified=True,mounting_verified=True,calibration_verified=True,gravity_from_world_rotation=np.eye(3).tolist(),object_from_tag_rotations={'5':np.eye(3).tolist()})
            cfg.write_text(json.dumps(config))
            with detections.open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['frame','tag_id','status','error_px','world_relative_rx_rad','world_relative_ry_rad','world_relative_rz_rad'])
                for i in range(9): w.writerow([i,5,'observed',.1,.02*i,0,0])
            with centers.open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['frame','time_s','x_m','y_m','z_m'])
                for i in range(9): w.writerow([i,.05*i,.01*i,0,.2])
            ref=prepare(detections,centers,cfg,.025,.375,root/'reference.csv')
            self.assertAlmostEqual(ref.t[-1],.35); np.testing.assert_allclose(ref.p[0],[.005,0,.2]); np.testing.assert_allclose(ref.p[-1],[.075,0,.2])
            self.assertEqual(ref.metadata['timing_scale'],1)
            with self.assertRaises(ValueError):
                config['mounting_verified']=False; cfg.write_text(json.dumps(config)); prepare(detections,centers,cfg,.025,.375,root/'invalid.csv')

    def test_exact_prompt_switch_clears_policy_and_processor_state(self):
        class IDs:
            def __init__(self,values): self.values=values
            def __getitem__(self,key): return self
            def bool(self): return self
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.values
        class Tokenizer:
            def __call__(self,text,**kw): return {'input_ids':list(text.encode())}
            def decode(self,ids): return bytes(ids).decode()
        class Pipeline:
            def __init__(self): self.resets=0; self.prompts=[]
            def reset(self): self.resets+=1
            def __call__(self,batch):
                self.prompts.append(batch['task'][0]); ids=IDs(list((batch['task'][0]+'\n').encode()))
                return {'observation.language.tokens':[ids],'observation.language.attention_mask':[ids]}
        class Env:
            def __init__(self): self.started=False
            def call(self,name): return [dict(timestep=3,sim_time_s=.15,phase='TRANSPORT' if self.started else 'GRASP',transport_started=self.started)]
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{'BOWL_HYBRID_CONDITION':'B'}):
            pre=Pipeline(); policy=Pipeline(); post=Pipeline(); env=Env(); CONTEXT.update(env=env,policy=policy,postprocessor=post)
            prompt=DynamicPrompt(pre,Tokenizer(),official.ORIGINAL,d)
            prompt({'task':[official.ORIGINAL]}); env.started=True; prompt({'task':[official.ORIGINAL]}); prompt({'task':[official.ORIGINAL]})
            self.assertEqual(pre.prompts,[official.ORIGINAL,LIQUID,LIQUID]); self.assertEqual(policy.resets,1); self.assertEqual(post.resets,1)
            self.assertTrue((Path(d)/'prompt_original.json').exists()); self.assertTrue((Path(d)/'prompt_liquid.json').exists()); CONTEXT.clear()


if __name__=='__main__': unittest.main()
