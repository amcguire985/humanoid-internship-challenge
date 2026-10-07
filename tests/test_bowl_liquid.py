"""Offline checks for the official-evaluator observer and transport-only metrics."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_bowl_liquid as experiment


def row(i,z,grasp=True,command=1,aperture=.02):
    t=i*.05
    return dict(timestep=i,sim_time_s=t,bowl_position_m=[t*t,0,z],tilt_deg=float(i),
        gripper_aperture_m=.08 if i==0 else aperture,action=None if i==0 else [0]*6+[command],
        grasp=False if i==0 else grasp,libero_success=i==7)


class TransportTests(unittest.TestCase):
    def test_quadratic_acceleration_only_inside_transport(self):
        rows=[row(i,.4 if i<2 else .43) for i in range(8)]
        # Approach tilt and acceleration must not enter transport maxima.
        rows[0]['tilt_deg']=170;rows[0]['bowl_position_m'][0]=-100
        summary,trace=experiment.analyze_transport(rows)
        self.assertEqual(summary['transport_start_timestep'],2)
        self.assertEqual(summary['max_tilt_deg'],7)
        self.assertAlmostEqual(summary['max_acceleration_m_s2'],2)
        self.assertIsNone(trace[2]['acceleration_raw_m_s2'])
        self.assertIsNone(trace[7]['acceleration_raw_m_s2'])
        self.assertTrue(summary['libero_success'])

    def test_no_lift_has_unavailable_metrics_not_zero(self):
        summary,trace=experiment.analyze_transport([row(i,.4) for i in range(8)])
        self.assertFalse(summary['transport_detected'])
        self.assertIsNone(summary['max_tilt_deg'])
        self.assertIsNone(summary['max_acceleration_m_s2'])

    def test_fallback_is_explicit(self):
        summary,_=experiment.analyze_transport([row(i,.4 if i==0 else .43,grasp=None) for i in range(8)])
        self.assertIn('fallback',summary['transport_start_definition'])
        self.assertIsNotNone(summary['grasp_warning'])

    def test_release_excludes_opening_sample(self):
        rows=[row(i,.4 if i<2 else .43) for i in range(8)]
        rows[5].update(grasp=False,action=[0]*6+[-1],gripper_aperture_m=.04,tilt_deg=150)
        summary,_=experiment.analyze_transport(rows)
        self.assertEqual(summary['release_timestep'],5)
        self.assertEqual(summary['transport_end_timestep'],4)
        self.assertEqual(summary['max_tilt_deg'],4)

    def test_true_nonuniform_times(self):
        rows=[row(i,.4 if i==0 else .43) for i in range(8)]
        for r,t in zip(rows,[0,.05,.12,.17,.21,.29,.34,.42]):
            r['sim_time_s']=t;r['bowl_position_m'][0]=t*t
        summary,_=experiment.analyze_transport(rows)
        self.assertAlmostEqual(summary['max_acceleration_m_s2'],2)
        rows[3]['sim_time_s']=rows[2]['sim_time_s']
        with self.assertRaises(ValueError):experiment.analyze_transport(rows)

    def test_yaw_does_not_count_as_tilt(self):
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        self.assertEqual(experiment.compute_upright_deviation_deg(rotation),0)


class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    def __getitem__(self,index):return Tensor(self.value[index.value if isinstance(index,Tensor) else index])
    def bool(self):return Tensor(self.value.astype(bool))
    def detach(self):return self
    def cpu(self):return self
    def tolist(self):return self.value.tolist()


class PromptTests(unittest.TestCase):
    def test_policy_only_override_and_full_token_check(self):
        class Tokenizer:
            def __call__(self,text,**kwargs):return {'input_ids':[ord(c) for c in text]}
            def decode(self,ids):return ''.join(chr(i) for i in ids)
        class Pipeline:
            def __call__(self,batch):
                self.prompt=batch['task'][0]
                ids=Tokenizer()(self.prompt+'\n')['input_ids']
                return {'observation.language.tokens':Tensor([ids]),
                        'observation.language.attention_mask':Tensor([[True]*len(ids)])}
            def reset(self):self.was_reset=True
        with tempfile.TemporaryDirectory() as temp:
            pipeline=Pipeline();wrapper=experiment.PromptProcessor(pipeline,Tokenizer(),experiment.LIQUID,temp)
            observation={'task':[experiment.ORIGINAL]}
            wrapper(observation)
            self.assertEqual(observation['task'],[experiment.ORIGINAL])
            self.assertEqual(pipeline.prompt,experiment.LIQUID)
            wrapper.reset();self.assertTrue(pipeline.was_reset)
            evidence=json.loads((Path(temp)/'prompt_verification.json').read_text())
            self.assertEqual(evidence['decoded_tokens'],experiment.LIQUID+'\n')
            with self.assertRaises(ValueError):wrapper({'task':['different task']})

    def test_truncation_fails_before_action(self):
        tokenizer=lambda text,**kw:{'input_ids':[1,2,3]}
        pipeline=lambda batch:{'observation.language.tokens':Tensor([[1,2]]),
            'observation.language.attention_mask':Tensor([[True,True]])}
        with tempfile.TemporaryDirectory() as temp:
            wrapper=experiment.PromptProcessor(pipeline,tokenizer,experiment.LIQUID,temp)
            with self.assertRaises(ValueError):wrapper({'task':[experiment.ORIGINAL]})


class ObserverTests(unittest.TestCase):
    def test_observer_preserves_official_reset_step_and_task(self):
        class Env:
            init_state_id=0;task='bowl';task_description=experiment.ORIGINAL;num_steps_wait=10
            _init_states=[np.zeros(2)]
            def __init__(self,root):
                self._task_bddl_file=str(root/'task.bddl');Path(self._task_bddl_file).write_text('unchanged')
                sim=SimpleNamespace(model=SimpleNamespace(body_name2id=lambda name:0),
                    data=SimpleNamespace(time=.5,body_xpos=np.array([[0.,0.,.4]]),body_xmat=np.eye(3).reshape(1,9),body_xquat=np.array([[1.,0.,0.,0.]])))
                obj=SimpleNamespace(root_body='bowl',top_offset=np.array([0,0,.04]),bottom_offset=np.array([0,0,-.06]))
                base=SimpleNamespace(sim=sim,objects_dict={experiment.OBJECT:obj},robots=[SimpleNamespace(gripper=object())],_check_grasp=lambda *a:False)
                self._env=SimpleNamespace(env=base,check_success=lambda:False)
                self.reset_count=0;self.step_count=0
            def obs(self):return {'robot_state':{'gripper':{'qpos':np.array([.04,-.04])}}}
            def reset(self,**kwargs):self.reset_count+=1;self.init_state_id+=1;return self.obs(),{'original':True}
            def step(self,action):self.step_count+=1;self._env.env.sim.data.time+=.05;return self.obs(),0,False,False,{'original':True}
            def close(self):pass
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);env=Env(root);observer=experiment.BowlObserver(env,root/'trace')
            reset=observer.reset(seed=0);result=observer.step(np.zeros(7));observer.close()
            self.assertEqual(env.reset_count,1);self.assertEqual(env.step_count,1)
            self.assertEqual(env.task_description,experiment.ORIGINAL)
            self.assertEqual(reset[1],{'original':True});self.assertEqual(result[-1],{'original':True})
            rows=[json.loads(s) for s in (root/'trace/episode_000/trajectory.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2);self.assertEqual(rows[0]['timestep'],0)


if __name__=='__main__':unittest.main()
