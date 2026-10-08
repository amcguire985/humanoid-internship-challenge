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
from bowl_transport_guidance import Guidance,Phases,Settings,pose,physical_placement_goal,opening_axis_rotation_vector
from calibrate_bowl_reference import gravity_rotation
from evaluate_bowl_hybrid import analyze,DynamicPrompt,CONTEXT,LIQUID,HybridObserver,compare,write_condition_report
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

    def test_placement_uses_physical_collision_surface_not_extent_markers(self):
        from types import SimpleNamespace
        model=SimpleNamespace(body_name2id=lambda name:{'bowl':0,'plate':1}[name],geom_name2id=lambda name:{'bowl_bottom':0,'plate_center':1,'plate_rim':2}[name],geom_type=np.array([6,6,6]),geom_size=np.array([[.03,.03,.003],[.03,.03,.002],[.02,.02,.003]]))
        data=SimpleNamespace(body_xpos=np.array([[0.,0.,.95],[.1,.2,.90]]),body_xmat=np.tile(np.eye(3).reshape(1,9),(2,1)),geom_xpos=np.array([[0.,0.,.923],[.1,.2,.903],[.1,.25,.910]]),geom_xmat=np.tile(np.eye(3).reshape(1,9),(3,1)))
        base=SimpleNamespace(sim=SimpleNamespace(model=model,data=data))
        bowl=SimpleNamespace(root_body='bowl',contact_geoms=['bowl_bottom'],bottom_offset=[0,0,-.06])
        plate=SimpleNamespace(root_body='plate',contact_geoms=['plate_center','plate_rim'],top_offset=[0,0,.04])
        goal,audit=physical_placement_goal(base,bowl,plate)
        np.testing.assert_allclose(goal,[.1,.2,.935])
        self.assertEqual(audit['plate_support_geom'],'plate_center')
        self.assertLess(goal[2],1.)

    def test_guidance_diagnostics_are_strict_json_serializable(self):
        start=pose([0,0,.4],np.eye(3)); eef=pose([0,0,.45],np.eye(3))
        guide=Guidance(reference(),start,eef,[.2,0,.4],controller(),Settings())
        for elapsed in (0.,.05,float(guide.reference.t[-1])):
            _,info=guide.apply(np.array([0.,0.,0.,0.,0.,0.,1.]),start,eef,elapsed,.05)
            self.assertIs(type(info['reference_finished']),bool)
            self.assertIs(type(info['slip_detected']),bool)
            json.dumps(info,allow_nan=False)
        _,info=guide.apply(np.ones(7),pose([.1,0,.4],np.eye(3)),eef,.1,.05)
        json.dumps(info,allow_nan=False)

    def test_human_timing_retained_but_arbitrary_orientation_is_not_tracked(self):
        start=pose([0,0,.4],np.eye(3)); eef=pose([0,0,.45],np.eye(3)); s=Settings(ramp_s=.01,correction_slew_per_s=10,correction_limit=1,position_blend=1,orientation_blend=1)
        first=Guidance(reference(),start,eef,[.2,0,.4],controller(),s)
        ref=reference(); ref.r=np.repeat(np.eye(3)[None],len(ref.t),axis=0)
        second=Guidance(ref,start,eef,[.2,0,.4],controller(),s)
        a,_=first.apply(np.zeros(7),start,eef,.5,.05); b,_=second.apply(np.zeros(7),start,eef,.5,.05)
        np.testing.assert_allclose(a,b,atol=1e-12)
        np.testing.assert_array_equal(first.reference.p,second.reference.p)
        np.testing.assert_array_equal(first.reference_velocity,second.reference_velocity)
        np.testing.assert_array_equal(first.reference_acceleration,second.reference_acceleration)
        later,_=second.apply(np.zeros(7),start,eef,.9,.05)
        self.assertGreater(np.linalg.norm(later[:3]-b[:3]),.01)

    def test_opening_axis_yaw_invariance_and_world_composition(self):
        for yaw in (-179.,-45.,0.,90.,179.):
            matrix=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
            np.testing.assert_allclose(opening_axis_rotation_vector(matrix),0,atol=1e-12)
        matrix=Rotation.from_euler('xyz',[.2,-.3,.7]).as_matrix()
        delta=opening_axis_rotation_vector(matrix)
        np.testing.assert_allclose(Rotation.from_rotvec(delta).as_matrix()@matrix[:,2],[0,0,1],atol=1e-12)
        self.assertAlmostEqual(delta[2],0,places=12)
        spun=matrix@Rotation.from_euler('z',1.9).as_matrix()
        np.testing.assert_allclose(opening_axis_rotation_vector(spun),delta,atol=1e-12)
        frame=Rotation.from_euler('xyz',[.5,.2,-.8]).as_matrix()
        np.testing.assert_allclose(opening_axis_rotation_vector(frame@matrix,frame@np.array([0,0,1])),frame@delta,atol=1e-12)
        inverted=Rotation.from_euler('x',np.pi).as_matrix()
        np.testing.assert_allclose(Rotation.from_rotvec(opening_axis_rotation_vector(inverted)).as_matrix()@inverted[:,2],[0,0,1],atol=1e-12)

    def test_yaw_reference_does_not_create_tilt_or_grasp_twist_corrections(self):
        t=np.linspace(0,1,21); p=np.column_stack([.2*t,np.zeros(21),np.zeros(21)])
        r=Rotation.from_euler('z',np.linspace(-179,179,21)[:,None],degrees=True).as_matrix()
        ref=Reference(t,p,r)
        bowl=pose([0,0,.4],np.eye(3)); eef=pose([.02,0,.45],Rotation.from_euler('x',np.pi).as_matrix())
        guide=Guidance(ref,bowl,eef,[.2,0,.4],controller(),Settings(ramp_s=.01))
        # Relative grasp yaw drift must not trigger an orientation-restoration objective.
        drifted_eef=eef.copy(); drifted_eef[:3,:3]=Rotation.from_euler('z',.1).as_matrix()@eef[:3,:3]
        result,info=guide.apply(np.zeros(7),bowl,drifted_eef,.5,.05)
        np.testing.assert_allclose(result[3:6],0,atol=1e-12)
        self.assertFalse(info['human_orientation_used_for_rotation'])

    def test_reference_endpoint_and_quaternion_sign_are_continuous(self):
        t=np.array([0.,.5,1.]); p=np.column_stack([t,np.zeros(3),np.zeros(3)])
        q=Rotation.from_euler('z',np.array([179.,180.,181.])[:,None],degrees=True).as_quat(); q[1]*=-1
        ref=Reference(t,p,Rotation.from_quat(q).as_matrix())
        grid=np.linspace(0,1,101); _,matrices=ref.sample(grid)
        increments=(Rotation.from_matrix(matrices[1:])*Rotation.from_matrix(matrices[:-1]).inv()).magnitude()
        self.assertLess(np.max(increments),np.radians(.03))
        a,b=ref.sample([1-1e-8,1,1+1e-8])
        np.testing.assert_allclose(a[1],a[2]); np.testing.assert_allclose(b[1],b[2]); np.testing.assert_allclose(b[0],b[1],atol=1e-8)
        for m in matrices: np.testing.assert_allclose(opening_axis_rotation_vector(m),0,atol=1e-12)

    def test_osc_world_delta_has_correct_tilt_reducing_sign(self):
        bowl=pose([0,0,.4],Rotation.from_euler('x',.3).as_matrix())
        eef=pose([0,0,.45],Rotation.from_euler('xyz',[2.8,.1,.7]).as_matrix())
        guide=Guidance(reference(),bowl,eef,[.2,0,.4],controller(),Settings(ramp_s=.01))
        action,info=guide.apply(np.zeros(7),bowl,eef,.5,.05)
        # Ideal one-step world delta, not a simulator performance prediction.
        delta=.5*action[3:6]
        predicted=Rotation.from_rotvec(delta).as_matrix()@bowl[:3,:3]
        self.assertLess(np.arccos(predicted[2,2]),.3)
        self.assertAlmostEqual(action[5],0,places=12)
        np.testing.assert_allclose(np.array(info['desired_eef_pose'])[:3,:3],Rotation.from_rotvec(info['opening_axis_error_world_rad']).as_matrix()@eef[:3,:3],atol=1e-12)

    def test_orientation_analysis_handles_absent_transport(self):
        from analyze_bowl_rotation import analyze as analyze_rotation
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trajectory.jsonl'; path.write_text(json.dumps(dict(phase='FAILED',libero_success=False))+'\n')
            report=analyze_rotation(path,Path(d)/'diagnostics')
            self.assertFalse(report['transport_detected']); self.assertTrue((Path(d)/'diagnostics/orientation_diagnosis.json').exists())

    def test_orientation_analysis_allows_disabled_guidance_without_target(self):
        from analyze_bowl_rotation import analyze as analyze_rotation
        with tempfile.TemporaryDirectory() as d:
            rows=[]
            for i in range(3):
                rows.append(dict(timestep=i,sim_time_s=i*.05,phase='TRANSPORT' if i<2 else 'FAILED',bowl_rotation_world_from_object=np.eye(3).tolist(),bowl_quaternion_wxyz=[1,0,0,0],eef_pose=np.eye(4).tolist(),tilt_deg=0.,grasp=True,libero_success=False,action=[0]*7,policy_action=[0]*7,desired_bowl_rotation=np.eye(3).tolist(),guidance_disabled_reason='rigid_grasp_transform_drift'))
            path=Path(d)/'trajectory.jsonl'; path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            report=analyze_rotation(path,Path(d)/'diagnostics')
            self.assertEqual(report['reference_samples_without_control_target'],3)
            self.assertIsNone(report['max_rotational_correction'])
            self.assertTrue((Path(d)/'diagnostics/orientation_signals.csv').exists())

    def test_release_handoff_slew_and_no_translation_or_gripper_changes(self):
        bowl=pose([0,0,.4],np.eye(3)); eef=pose([0,0,.45],np.eye(3))
        guide=Guidance(reference(),bowl,eef,[.2,0,.4],controller(),Settings())
        guide.previous_correction=np.array([.1,-.1,.1,.15,-.12,.06]); policy=np.array([.4,-.2,.1,.02,.01,-.03,-1.])
        previous=guide.previous_correction[3:].copy()
        for _ in range(8):
            result,info=guide.release_orientation(policy,.05)
            np.testing.assert_array_equal(result[:3],policy[:3]); self.assertEqual(result[6],policy[6])
            correction=np.array(info['correction'])[3:]
            self.assertLessEqual(np.max(np.abs(correction-previous)),.03000000001)
            previous=correction; json.dumps(info,allow_nan=False)
        np.testing.assert_allclose(result,policy,atol=1e-12)

    def test_metrics_exclude_pregrasp_and_failed_transport_is_null(self):
        rows=[]
        for i in range(9):
            t=i*.05
            rows.append(dict(timestep=i,sim_time_s=t,bowl_position_m=[t*t,0,.4 if i<2 else .43],tilt_deg=100 if i<2 else i,grasp=i>=2,phase='APPROACH' if i<2 else 'TRANSPORT',libero_success=False,failure_reason=None))
        rows[-1].update(phase='DONE',libero_success=True)
        summary,a=analyze(rows); self.assertEqual(summary['max_tilt_deg'],8); self.assertAlmostEqual(summary['max_acceleration_m_s2'],2); self.assertIsNone(a[2])
        for r in rows: r['phase']='FAILED'
        summary,_=analyze(rows); self.assertIsNone(summary['max_tilt_deg']); self.assertIsNone(summary['max_acceleration_m_s2'])

    def test_parent_wrapper_report_exists_with_unavailable_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            write_condition_report(Path(d),dict(libero_success=False,transport_detected=False,max_tilt_deg=None,max_acceleration_m_s2=None,transport_duration_s=None,failure_reason='grasp_timeout'))
            report=(Path(d)/'comparison.md').read_text()
            self.assertIn('| max_tilt_deg | unavailable |',report)
            self.assertIn('| failure_reason | grasp_timeout |',report)

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
