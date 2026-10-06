import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upright_orientation import compute_upright_deviation_deg, get_mug_orientation
from evaluate_upright_mug import resolve_task, summarize, plot_trace, INSTRUCTION, parser


class OrientationTests(unittest.TestCase):
    def test_known_tilts_and_yaw_invariance(self):
        for angle in (0,30,90,180):
            a=np.deg2rad(angle)
            r=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
            for yaw in (0,37,180):
                b=np.deg2rad(yaw)
                z=np.array([[np.cos(b),-np.sin(b),0],[np.sin(b),np.cos(b),0],[0,0,1]])
                self.assertAlmostEqual(compute_upright_deviation_deg(z@r),angle,places=5)

    def test_invalid_rotation_and_axes(self):
        for rotation in (np.zeros((3,3)),np.eye(4),np.eye(3)*np.nan,np.diag([1,1,-1])):
            with self.assertRaises(ValueError): compute_upright_deviation_deg(rotation)
        with self.assertRaises(ValueError): compute_upright_deviation_deg(np.eye(3),world_up=(0,0,0))

    def test_root_body_and_quaternion_convention(self):
        model=SimpleNamespace(body_name2id=lambda name: 1 if name=='mug_root' else 0)
        data=SimpleNamespace(body_xmat=np.array([np.eye(3).ravel(),np.eye(3).ravel()]),
                             body_xquat=np.array([[0,0,0,1],[1,0,0,0]]))
        env=SimpleNamespace(env=SimpleNamespace(objects_dict={'porcelain_mug_1':SimpleNamespace(root_body='mug_root')},
                                                sim=SimpleNamespace(model=model,data=data)))
        r,q=get_mug_orientation(env)
        np.testing.assert_array_equal(q,[1,0,0,0]);np.testing.assert_array_equal(r,np.eye(3))
        q[0]=0
        self.assertEqual(data.body_xquat[1,0],1)

    def test_task_lookup_survives_reordering(self):
        tasks=[SimpleNamespace(name='other'),SimpleNamespace(name='mug')]
        suite=SimpleNamespace(n_tasks=2,get_task=lambda i:tasks[i])
        self.assertEqual(resolve_task(suite,'mug',None)[0],1)
        self.assertEqual(resolve_task(suite,'ignored',0)[1].name,'other')
        with self.assertRaises(ValueError): resolve_task(suite,'missing',None)

    def test_success_is_independent_of_tilt(self):
        result=summarize([{'libero_success':True,'max_mug_tilt_deg':90},
                          {'libero_success':False,'max_mug_tilt_deg':0}])
        self.assertEqual(result['libero_success_rate'],.5)
        self.assertEqual(result['median_max_tilt_deg'],45)
        self.assertEqual(result['max_max_tilt_deg'],90)
        self.assertIn('full of liquid',parser().parse_args([]).policy_instruction)
        self.assertEqual(parser().parse_args([]).policy_instruction,INSTRUCTION)

    def test_rollout_records_prompt_peak_and_unmodified_success(self):
        import tempfile, json
        from unittest.mock import patch
        from evaluate_upright_mug import evaluate
        class Env:
            def __init__(self, **kwargs):
                self.count=0
                self.env=SimpleNamespace(sim=SimpleNamespace(render=lambda **kw:np.zeros((128,128,3),dtype=np.uint8)))
            def seed(self,seed): pass
            def reset(self): pass
            def set_init_state(self,state): return {'robot0_eef_pos':np.zeros(3)}
            def check_success(self): return self.count==2
            def step(self,action):
                self.count+=1
                return {'robot0_eef_pos':np.zeros(3)},0,False,{}
            def close(self): pass
        class Runtime:
            metadata={'chunk_size':4}
            def reset(self,seed): self.seed=seed
            def predict(self,rgb,state,task):
                self.task=task
                return np.zeros((4,7))
        task=SimpleNamespace(name='mug',language='original mug instruction',problem_folder='suite',bddl_file='mug.bddl')
        suite=SimpleNamespace(n_tasks=1,get_task=lambda i:task,get_task_init_states=lambda i:[np.zeros(1)])
        runtime=Runtime()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'suite').mkdir(); (root/'suite/mug.bddl').write_text('unchanged goal')
            stats=root/'stats.json';stats.write_text('{}')
            args=parser().parse_args(['--task-name','mug','--num-rollouts','1','--settle-steps','0',
                                     '--stats',str(stats),'--output',str(root/'output')])
            def pose(env,obs,timestep,stage,action=None):
                return dict(timestep=timestep,stage=stage,sim_time_s=timestep/20,
                            mug_upright_deviation_deg=90 if timestep==1 else 0,
                            libero_success=env.check_success())
            modules={'libero':SimpleNamespace(), 'libero.libero':SimpleNamespace(
                benchmark=SimpleNamespace(get_benchmark_dict=lambda:{'libero_90':lambda **kw:suite}),
                get_libero_path=lambda key:str(root)),
                'libero.libero.envs':SimpleNamespace(OffScreenRenderEnv=Env),
                'evaluate_libero_bc':SimpleNamespace(proprio=lambda env,obs:np.zeros(18),rendering_settings=lambda sim:None)}
            with patch.dict(sys.modules,modules), patch('evaluate_upright_mug.record_pose',pose):
                evaluate(args,runtime)
            report=json.loads((root/'output/rollout_000/summary.json').read_text())
            self.assertTrue(report['libero_success'])
            self.assertEqual(report['max_mug_tilt_deg'],90)
            self.assertEqual(report['timestep_of_max_tilt'],1)
            self.assertEqual(runtime.task,INSTRUCTION)
            trace=(root/'output/rollout_000/trajectory.jsonl').read_text().splitlines()
            self.assertEqual(len(trace),3)

    def test_headless_plot(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tilt.png'
            plot_trace([{'timestep':0,'mug_upright_deviation_deg':0},
                        {'timestep':1,'mug_upright_deviation_deg':30}],path)
            self.assertGreater(path.stat().st_size,0)


if __name__=='__main__': unittest.main()
