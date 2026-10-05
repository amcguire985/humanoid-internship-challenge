"""Exactly one nominal rollout; separates legacy simulator and modern policy Python."""
import argparse
import base64
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def worker(training):
    # Model/library startup output cannot contaminate the JSON protocol on stdout.
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('SmolVLA evaluation worker requires CUDA')
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
        from prepare_libero_smolvla import IMAGE, STATE
        torch.manual_seed(0)
        config=SmolVLAConfig.from_pretrained(training/'pretrained_model')
        config.device='cuda';config.load_vlm_weights=False
        policy=SmolVLAPolicy.from_pretrained(training/'pretrained_model',config=config,strict=True)
        stats=json.loads((training/'dataset_stats.json').read_text())
        stats={k:{n:torch.tensor(v) for n,v in s.items()} for k,s in stats.items()}
        pre,post=make_smolvla_pre_post_processors(config,stats)
        manifest=json.loads((training/'source_manifest.json').read_text())
        task=manifest['episodes'][0]['task']; policy.eval();policy.reset()
    print(json.dumps(dict(architecture='Pretrained SmolVLA adapted on two LIBERO demos',chunk_size=config.chunk_size)),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        rgb=np.frombuffer(base64.b64decode(request['rgb']),dtype=np.uint8).reshape(128,128,3).copy()
        batch={IMAGE:torch.from_numpy(rgb).permute(2,0,1).float()/255,
               STATE:torch.tensor(request['state'],dtype=torch.float32),'task':task}
        with contextlib.redirect_stdout(sys.stderr),torch.no_grad():
            action=post(policy.predict_action_chunk(pre(batch)))[0].cpu().numpy()
        print(json.dumps(action.tolist()),flush=True)


class WorkerRuntime:
    def __init__(self,python,training):
        self.process=subprocess.Popen([python,str(Path(__file__).resolve()),'--worker','--training',str(training)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        self.metadata=self.read()
    def read(self):
        line=self.process.stdout.readline()
        if not line: raise RuntimeError('SmolVLA worker stopped; inspect stderr. Do not repeat the evaluation.')
        return json.loads(line)
    def predict(self,rgb,state):
        self.process.stdin.write(json.dumps(dict(rgb=base64.b64encode(rgb.tobytes()).decode(),state=state.tolist()))+'\n')
        self.process.stdin.flush()
        return np.asarray(self.read(),dtype=np.float32)
    def close(self):
        self.process.stdin.close()
        try: self.process.wait(timeout=10)
        except subprocess.TimeoutExpired: self.process.terminate();self.process.wait(timeout=10)


def evaluate(args):
    from evaluate_libero_bc import evaluate as run_nominal
    # Training must finish before even initializing a simulator.
    if not (args.training/'training.json').exists(): raise ValueError('No completed fine-tuning run')
    if args.output.exists(): raise ValueError('Refusing to repeat an evaluation')
    runtime=WorkerRuntime(args.policy_python,args.training)
    try:
        run_nominal(args.training/'pretrained_model',args.output,nominal_only=True,execute_steps=4,runtime=runtime)
    finally: runtime.close()
    import csv
    rows=list(csv.DictReader((args.output/'nominal/rollout.csv').open()))
    eef=np.array([[float(r['eef_'+axis]) for axis in 'xyz'] for r in rows])
    distance=np.array([float(r['eef_object_distance_m']) for r in rows])
    metrics=json.loads((args.output/'nominal/metrics.json').read_text())
    displacement=float(np.linalg.norm(eef-eef[0],axis=1).max())
    metrics.update(leaves_starting_pose=displacement>=.02,maximum_displacement_from_first_post_action_m=displacement,
        approaches_object=bool(distance.min()<distance[0]-.05),
        approach_definition='At least 5 cm reduction in EEF-to-object-center distance',
        commands_close=metrics['close_command_count']>0)
    metrics['first_meaningful_failure']=None if metrics['task_completed'] else (
        'Does not leave start' if not metrics['leaves_starting_pose'] else
        'Does not approach object' if not metrics['approaches_object'] else
        'No close command' if not metrics['commands_close'] else
        'No acquisition' if not metrics['object_acquired'] else
        'No lift' if not metrics['object_lifted'] else metrics['failure_phase'])
    (args.output/'result.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,default=Path('results/libero_smolvla/training'))
    parser.add_argument('--output',type=Path,default=Path('results/libero_smolvla/evaluation'))
    parser.add_argument('--policy-python',default='python')
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker: worker(args.training)
    else: evaluate(args)
