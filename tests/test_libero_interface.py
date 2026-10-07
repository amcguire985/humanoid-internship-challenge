"""Interface bookkeeping tests; no GPU, checkpoint download or simulator required."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import validate_libero_interface as validation


class InterfaceTests(unittest.TestCase):
    def test_observation_transport_is_raw_rgb_and_exact_state(self):
        obs={camera:np.zeros((256,256,3),dtype=np.uint8) for camera in validation.CAMERAS}
        obs['agentview_image'][0,0]=[255,17,9]
        obs.update(robot0_eef_pos=[1,2,3],robot0_eef_quat=[0,0,0,1],robot0_gripper_qpos=[.04,-.04])
        import base64
        encoded=validation.encode_observation(obs)
        image=np.frombuffer(base64.b64decode(encoded['images']['agentview_image']),dtype=np.uint8).reshape(256,256,3)
        np.testing.assert_array_equal(image[0,0],[255,17,9])
        self.assertEqual(encoded['state']['robot0_eef_quat'],[0,0,0,1])
        obs['agentview_image']=np.zeros((128,128,3),dtype=np.uint8)
        with self.assertRaises(ValueError):validation.encode_observation(obs)

    def test_action_preserves_sign_magnitude_and_zero(self):
        action=np.array([.8,-.9,.2,.1,-.2,.3,.4])
        np.testing.assert_array_equal(validation.final_libero_action(action,-np.ones(7),np.ones(7)),action)
        action[6]=-.2
        self.assertEqual(validation.final_libero_action(action,-np.ones(7),np.ones(7))[6],-.2)
        action[6]=0
        self.assertEqual(validation.final_libero_action(action,-np.ones(7),np.ones(7))[6],0)
        action[0]=2
        self.assertEqual(validation.final_libero_action(action,-np.ones(7),np.ones(7))[0],1)
        with self.assertRaises(ValueError):validation.final_libero_action([np.nan]*7,-np.ones(7),np.ones(7))

    def test_rejects_wrong_checkpoint_statistics(self):
        flat={key+'.'+name:np.ones(size) for key,size in [('observation.state',8),('action',7)] for name in ['mean','std']}
        self.assertEqual(len(validation.checked_stats(flat)['observation.state']['mean']),8)
        flat['observation.state.mean']=np.ones(18)
        with self.assertRaises(ValueError):validation.checked_stats(flat)

    def test_standalone_probe_distinguishes_commands_from_motion(self):
        class Env:
            aperture=.08
            def step(self,action):
                self.aperture=float(np.clip(self.aperture-action[6]*.001,0,.08))
                return {'robot0_gripper_qpos':[self.aperture/2,-self.aperture/2]},0,False,{}
        result=validation.gripper_probe(Env(),{'robot0_gripper_qpos':[.04,-.04]},50)
        self.assertTrue(result['physically_closes']);self.assertTrue(result['physically_reopens'])

    def test_default_is_single_ordinary_rollout_and_checkpoint_horizon(self):
        args=validation.parser().parse_args([])
        self.assertEqual(args.num_rollouts,1)
        self.assertIsNone(args.execute_steps)
        self.assertEqual(args.checkpoint,'lerobot/smolvla_libero')
        self.assertFalse(hasattr(args,'policy_instruction'))

    def test_rollout_action_chain_and_original_prompt(self):
        class Env:
            def __init__(self,**kwargs):
                self.count=0;self.z=.8
                controller=SimpleNamespace(scale_action=lambda a:a*np.r_[[.05]*3,[.5]*3])
                self.sim=SimpleNamespace(model=SimpleNamespace(body_name2id=lambda name:0),
                                         data=SimpleNamespace(body_xpos=np.array([[0.,0.,self.z]])))
                robot=SimpleNamespace(controller=controller,gripper=object())
                self.env=SimpleNamespace(sim=self.sim,robots=[robot],objects_dict={validation.OBJECT:SimpleNamespace(root_body='mug')},
                                         action_spec=(-np.ones(7),np.ones(7)),_check_grasp=lambda *args:self.count>10)
            def obs(self):
                return dict(robot0_eef_pos=np.array([0.,0.,self.z]),robot0_gripper_qpos=np.array([.04,-.04]) if self.count<=10 else np.array([.01,-.01]))
            def seed(self,seed):pass
            def reset(self):pass
            def set_init_state(self,state):return self.obs()
            def check_success(self):return self.count>=12
            def close(self):pass
            def step(self,action):
                self.count+=1
                if self.count>10:self.z+=.025;self.sim.data.body_xpos[0,2]=self.z
                return self.obs(),0,False,{}
        class Runtime:
            metadata={'checkpoint':validation.CHECKPOINT,'revision':validation.REVISION,'libero_trained':True,'n_action_steps':2,'chunk_size':2}
            def reset(self,seed):pass
            def predict(self,obs,task):
                self.task=task
                return dict(raw_policy_output=np.ones((2,7)).tolist(),normalized_action=np.ones((2,7)).tolist(),
                            denormalized_action=np.full((2,7),.8).tolist(),inference_seconds=.01)
        task=SimpleNamespace(name=validation.TASK,language='put the white mug on the plate',problem_folder='suite',bddl_file='task.bddl')
        suite=SimpleNamespace(n_tasks=1,get_task=lambda i:task,get_task_init_states=lambda i:[np.zeros(1)])
        modules={'libero':SimpleNamespace(),'libero.libero':SimpleNamespace(
            benchmark=SimpleNamespace(get_benchmark_dict=lambda:{'libero_90':lambda **kw:suite}),get_libero_path=lambda key:str(root)),
            'libero.libero.envs':SimpleNamespace(OffScreenRenderEnv=Env),
            'evaluate_libero_bc':SimpleNamespace(rendering_settings=lambda sim:None)}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'suite').mkdir();(root/'suite/task.bddl').write_text('goal unchanged')
            args=validation.parser().parse_args(['--output',str(root/'run')])
            runtime=Runtime()
            with patch.dict(sys.modules,modules),patch.object(validation,'controller_audit',lambda env:{}),contextlib.redirect_stdout(io.StringIO()):
                validation.evaluate(args,runtime)
            self.assertEqual(runtime.task,task.language)
            report=json.loads((root/'run/validation_report.json').read_text())
            self.assertEqual(report['libero_successes'],1)
            self.assertEqual(report['rollouts_with_grasp'],1)
            self.assertEqual(report['rollouts_with_lift'],1)
            rows=[json.loads(l) for l in (root/'run/rollout_000/trajectory.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['final_libero_action'][0],.8)
            self.assertEqual(rows[0]['normalized_action'][0],1)
            self.assertEqual(rows[0]['denormalized_action'][0],.8)
            self.assertAlmostEqual(rows[0]['controller_scaled_delta'][0],.04)
            self.assertAlmostEqual(rows[0]['eef_displacement_m'][2],.025)


if __name__=='__main__':unittest.main()
