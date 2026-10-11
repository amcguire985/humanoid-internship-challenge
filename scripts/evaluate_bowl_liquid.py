"""Observe and override policy language around the pinned official LIBERO evaluator.

No alternate simulation loop. The original env reset/step and evaluator execute.
"""
import argparse
from functools import partial
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import numpy as np
from upright_orientation import compute_upright_deviation_deg, get_mug_orientation

REVISION = '8c920c4270460851cedd2737657584586d3dc66f'
ORIGINAL = 'pick up the black bowl from table center and place it on the plate'
CONSTRAINT = 'the bowl is full of liquid, do not spill it'
LIQUID = ORIGINAL + '. ' + CONSTRAINT
OBJECT = 'akita_black_bowl_1'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


class PromptProcessor:
    """Override only policy language; assert that every original token survives."""
    def __init__(self, pipeline, tokenizer, prompt, output):
        self.pipeline=pipeline; self.tokenizer=tokenizer; self.prompt=prompt
        self.output=Path(output); self.checked=False

    def __getattr__(self, name):
        return getattr(self.pipeline, name)

    def __call__(self, observation):
        tasks=observation['task']
        if tasks != [ORIGINAL]:
            raise ValueError('Unexpected LIBERO environment instruction: '+str(tasks))
        batch=dict(observation); batch['task']=[self.prompt]
        processed=self.pipeline(batch)
        ids=processed['observation.language.tokens'][0]
        mask=processed['observation.language.attention_mask'][0]
        actual=ids[mask.bool()].detach().cpu().tolist()
        expected=self.tokenizer(self.prompt+'\n', add_special_tokens=True)['input_ids']
        if actual != expected:
            raise ValueError('Policy prompt was altered or truncated during tokenization')
        if not self.checked:
            diagnostic=dict(environment_instruction=tasks[0],policy_instruction=self.prompt,
                unmasked_token_ids=actual,decoded_tokens=self.tokenizer.decode(actual),
                token_verification='Exact equality against full prompt plus SmolVLA newline')
            write_json(self.output/'prompt_verification.json', diagnostic)
            print('LIBERO environment instruction:',tasks[0],flush=True)
            print('Instruction actually passed to SmolVLA:',self.prompt,flush=True)
            self.checked=True
        return processed


class BowlObserver:
    """Reads simulator fields only, after the official reset/settle or action."""
    def __init__(self, env, root):
        self.env=env; self.root=Path(root); self.stream=None; self.episode=-1
        self.original_reset=env.reset; self.original_step=env.step; self.original_close=env.close

    def reset(self, *args, **kwargs):
        self.finish()
        index=int(self.env.init_state_id % len(self.env._init_states))
        result=self.original_reset(*args, **kwargs)
        self.episode+=1; self.step_number=0
        directory=self.root/f'episode_{self.episode:03d}'
        directory.mkdir(parents=True,exist_ok=False)
        base=self.env._env.env; obj=base.objects_dict[OBJECT]
        up=np.asarray(obj.top_offset)-np.asarray(obj.bottom_offset)
        up=up/np.linalg.norm(up)
        if not np.allclose(up,[0,0,1],atol=1e-5):
            raise ValueError('Unexpected bowl opening axis; requires asset audit')
        self.local_up=up; self.rows=[]
        self.stream=(directory/'trajectory.jsonl').open('w',encoding='utf-8')
        state=np.asarray(self.env._init_states[index])
        bddl=Path(self.env._task_bddl_file)
        write_json(directory/'initialization.json',dict(episode=self.episode,seed=None if kwargs.get('seed') is None else int(kwargs['seed']),
            init_state_index=index,init_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
            task_name=self.env.task,environment_instruction=self.env.task_description,
            bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),object_name=OBJECT,
            local_opening_axis=up.tolist(),bottom_offset=np.asarray(obj.bottom_offset).tolist(),
            top_offset=np.asarray(obj.top_offset).tolist(),settle_steps=self.env.num_steps_wait))
        self.capture(result[0],None)
        return result

    def capture(self, observation, action):
        base=self.env._env.env; obj=base.objects_dict[OBJECT]
        body=base.sim.model.body_name2id(obj.root_body)
        rotation,quat=get_mug_orientation(self.env._env,OBJECT)
        qpos=np.asarray(observation['robot_state']['gripper']['qpos'])
        try:
            grasp=bool(base._check_grasp(base.robots[0].gripper,obj))
        except (AttributeError,NotImplementedError):
            grasp=None
        row=dict(timestep=self.step_number,sim_time_s=float(base.sim.data.time),
            bowl_position_m=np.asarray(base.sim.data.body_xpos[body]).tolist(),
            bowl_quaternion_wxyz=quat.tolist(),bowl_rotation_world_from_object=rotation.tolist(),
            tilt_deg=compute_upright_deviation_deg(rotation,self.local_up),
            gripper_qpos=qpos.tolist(),gripper_aperture_m=float(abs(qpos[0]-qpos[1])),
            grasp=grasp,action=None if action is None else np.asarray(action).tolist(),
            libero_success=bool(base.check_success()) if hasattr(base,'check_success') else bool(self.env._env.check_success()))
        self.rows.append(row)
        self.stream.write(json.dumps(row,allow_nan=False)+'\n'); self.stream.flush()

    def step(self, action):
        result=self.original_step(action)
        self.step_number+=1; self.capture(result[0],action)
        return result

    def finish(self):
        if self.stream:
            self.stream.close(); self.stream=None

    def close(self):
        self.finish(); return self.original_close()


def create_observed_env(factory):
    """Called inside the official async worker; no extra reset or env.step."""
    env=factory()
    observer=BowlObserver(env,os.environ['BOWL_EXPERIMENT_OUTPUT'])
    env.reset=observer.reset; env.step=observer.step; env.close=observer.close
    return env


def analyze_transport(rows):
    """Unfiltered central second difference, wholly inside the transport interval."""
    n=len(rows); t=np.array([r['sim_time_s'] for r in rows],float)
    p=np.array([r['bowl_position_m'] for r in rows],float)
    if n<2 or not np.isfinite(p).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Invalid or non-increasing ground-truth trajectory')
    height=p[:,2]-p[0,2]
    initial_aperture=rows[0]['gripper_aperture_m']
    closed=np.array([r['action'] is not None and r['action'][6]>0 and
        r['gripper_aperture_m']<initial_aperture-.001 for r in rows])
    grasp=np.array([r['grasp'] is True for r in rows])
    # Prefer a detected grasp plus >=2 cm lift; otherwise transparently label fallback.
    candidates=np.flatnonzero(grasp & (height>=.02))
    method='confirmed_grasp_and_2cm_lift'
    if not len(candidates):
        candidates=np.flatnonzero(closed & (height>=.02))
        method='close_command_and_aperture_reduction_and_2cm_lift_fallback'
    start=int(candidates[0]) if len(candidates) else None
    end=None; release=None; end_reason=None
    raw_acceleration=[None]*n
    if start is not None:
        end=n-1;end_reason='success_or_episode_end'
        for i in range(start+1,n):
            a=rows[i]['action']
            if a is not None and a[6]<0 and rows[i]['gripper_aperture_m']>rows[start]['gripper_aperture_m']+.001 and rows[i]['grasp'] is not True:
                release=i;end=i-1;end_reason='opening_and_aperture_increase_without_detected_grasp';break
        for i in range(start+1,end):
            h0=t[i]-t[i-1];h1=t[i+1]-t[i]
            acceleration=2*((p[i+1]-p[i])/h1-(p[i]-p[i-1])/h0)/(h0+h1)
            raw_acceleration[i]=float(np.linalg.norm(acceleration))
    values=[x for x in raw_acceleration if x is not None]
    summary=dict(libero_success=any(r['libero_success'] for r in rows),
        grasp_ever=bool(grasp.any()),max_height_increase_m=float(max(0,height.max())),
        transport_start_timestep=None if start is None else rows[start]['timestep'],
        transport_end_timestep=None if end is None else rows[end]['timestep'],
        release_timestep=None if release is None else rows[release]['timestep'],
        transport_start_definition=method,transport_end_definition=end_reason,
        transport_detected=start is not None,
        max_tilt_deg=None if start is None else max(r['tilt_deg'] for r in rows[start:end+1]),
        max_acceleration_m_s2=max(values) if values else None,
        acceleration_method='Unsmoothed nonuniform-time three-point central second difference; transport interior only; endpoints unavailable',
        sampled_time_step_range_s=[float(np.diff(t).min()),float(np.diff(t).max())],
        sampling_limit='One sample per official control step, not physics substeps; acceleration is a sampled estimate, not continuous-time peak',
        baseline_height_definition='First observation after official reset and 10 settling steps',
        smoothing='None',grasp_warning='Fallback does not prove grasp; height change may include pushing' if 'fallback' in method else None)
    trace=[]
    for i,r in enumerate(rows):
        trace.append(dict(timestep=r['timestep'],sim_time_s=t[i],time_since_settled_reset_s=float(t[i]-t[0]),
            height_increase_m=float(height[i]),tilt_deg=r['tilt_deg'],
            in_transport=start is not None and start<=i<=end,
            acceleration_raw_m_s2=raw_acceleration[i],acceleration_processed_m_s2=raw_acceleration[i]))
    return summary,trace


def process_outputs(output, baseline_root, condition, episodes=1):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    official=json.loads((output/'eval_info.json').read_text())
    successes=official['per_task'][0]['metrics']['successes']
    reports=[]
    for directory in sorted((output/'ground_truth').glob('episode_*')):
        rows=[json.loads(line) for line in (directory/'trajectory.jsonl').read_text().splitlines()]
        summary,trace=analyze_transport(rows)
        init=json.loads((directory/'initialization.json').read_text())
        episode=init['episode']
        if init['init_state_index']!=episode or init['seed']!=episode:
            raise ValueError('Official reset selection differs from expected baseline pairing')
        if episode>=len(successes):raise ValueError('Extra/reset episode unexpectedly recorded')
        if summary['libero_success'] != successes[episode]:raise ValueError('Trace success disagrees with official evaluator')
        summary.update(episode=episode,condition=condition,seed=init['seed'],init_state_index=init['init_state_index'])
        write_json(directory/'summary.json',summary);reports.append(summary)
        with (directory/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(trace[0]));writer.writeheader();writer.writerows(trace)
        for field,label,name in [('tilt_deg','Bowl tilt (degrees)','tilt.png'),('acceleration_raw_m_s2','Bowl acceleration (m/sÃ‚Â²)','acceleration.png')]:
            fig,ax=plt.subplots()
            selected=[r for r in trace if r['in_transport']]
            ax.plot([r['time_since_settled_reset_s'] for r in selected],
                [np.nan if r[field] is None else r[field] for r in selected])
            if not selected:ax.text(.5,.5,'No transport detected',transform=ax.transAxes,ha='center')
            ax.set(xlabel='Simulation time since settled reset (s)',ylabel=label,title=f'{condition} episode {episode}: transport phase')
            fig.tight_layout();fig.savefig(directory/name);plt.close(fig)
    if len(reports)!=episodes:raise ValueError('Unexpected recorded episode count')
    baseline_path=baseline_root/'libero_spatial_2/eval_info.json'
    baseline=json.loads(baseline_path.read_text())['per_task'][0]['metrics']['successes']
    if baseline != [True,True,True]:raise ValueError('Expected the validated three-success original baseline')
    table=['| Episode | Prompt | LIBERO success | Max tilt (deg) | Max acceleration (m/s?) |',
           '|---|---|---|---|---|']
    for r in reports:
        tilt='unavailable' if r['max_tilt_deg'] is None else f"{r['max_tilt_deg']:.3f}"
        acc='unavailable' if r['max_acceleration_m_s2'] is None else f"{r['max_acceleration_m_s2']:.3f}"
        table.append(f"| {r['episode']} | {condition} | {r['libero_success']} | {tilt} | {acc} |")
    (output/'comparison.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    write_json(output/'comparison.json',dict(episodes=reports,condition=condition,
        historical_original_successes=baseline,
        limitation='Use the new logged original run for paired dynamics; historical baseline lacks object poses'))


def compare_pair(original, liquid, output):
    """Compare already completed matched episodes; never launches a simulation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv
    output=Path(output)
    original=Path(original);liquid=Path(liquid)
    a=json.loads((original/'comparison.json').read_text())['episodes']
    b=json.loads((liquid/'comparison.json').read_text())['episodes']
    if len(a)!=1 or len(b)!=1:raise ValueError('This comparison expects one episode per condition')
    for path,prompt in [(original,ORIGINAL),(liquid,LIQUID)]:
        verification=json.loads((path/'prompt_verification.json').read_text())
        if verification['policy_instruction']!=prompt:raise ValueError('Incorrect comparison prompt')
    ia=json.loads((original/'ground_truth/episode_000/initialization.json').read_text())
    ib=json.loads((liquid/'ground_truth/episode_000/initialization.json').read_text())
    for key in ['seed','init_state_index','init_state_sha256','bddl_sha256','local_opening_axis']:
        if ia[key]!=ib[key]:raise ValueError('Unmatched conditions: '+key)
    ea=json.loads((original/'experiment.json').read_text())
    eb=json.loads((liquid/'experiment.json').read_text())
    if ea['checkpoint_sha256']!=eb['checkpoint_sha256']:raise ValueError('Different checkpoints')
    differences={}
    for field in ['max_tilt_deg','max_acceleration_m_s2']:
        differences['delta_'+field]=None if a[0][field] is None or b[0][field] is None else b[0][field]-a[0][field]
    write_json(output/'paired_comparison.json',dict(original=a[0],liquid=b[0],**differences,
        interpretation='Liquid minus original; unavailable if either run has no valid transport metric. One pair cannot establish a general effect.'))
    lines=['| Prompt | LIBERO success | Max tilt (deg) | Max acceleration (m/s?) |','|---|---|---|---|']
    for name,r in [('original',a[0]),('original + liquid',b[0])]:
        values=['unavailable' if r[k] is None else f"{r[k]:.3f}" for k in ['max_tilt_deg','max_acceleration_m_s2']]
        lines.append(f"| {name} | {r['libero_success']} | {values[0]} | {values[1]} |")
    lines.extend(['','Liquid minus original: '+json.dumps(differences),'','No significance testing; one matched pair.'])
    (output/'paired_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for field,label,filename in [('tilt_deg','Bowl tilt (degrees)','paired_tilt.png'),('acceleration_raw_m_s2','Bowl acceleration (m/s?)','paired_acceleration.png')]:
        fig,ax=plt.subplots()
        for path,name in [(original,'original'),(liquid,'original + liquid')]:
            with (path/'ground_truth/episode_000/metrics.csv').open(encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            selected=[r for r in rows if r['in_transport']=='True' and r[field]]
            ax.plot([float(r['time_since_settled_reset_s']) for r in selected],
                    [float(r[field]) for r in selected],label=name)
        ax.set(xlabel='Simulation time since settled reset (s)',ylabel=label)
        ax.legend();fig.tight_layout();fig.savefig(output/filename);plt.close(fig)
    print((output/'paired_comparison.md').read_text(),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--condition',choices=['original','liquid'],default='liquid')
    parser.add_argument('--episodes',type=int,choices=[1,3],default=1)
    args,official_args=parser.parse_known_args()
    output_option=next((a.split('=',1)[1] for a in official_args if a.startswith('--output_dir=')),None)
    if not output_option:raise ValueError('Use --output_dir=<new directory>')
    output=Path(output_option);output.mkdir(parents=True,exist_ok=False)
    # Verify the remote runtime matches the successful original run before rollout.
    baseline_packages=(args.baseline_root/'packages.txt').read_text().splitlines()
    expected={line.split('==',1)[0].lower().replace('_','-'):line.split('==',1)[1] for line in baseline_packages if '==' in line}
    for package in ['hf-libero','robosuite','mujoco','torch','torchvision','transformers','numpy','gymnasium']:
        if metadata.version(package)!=expected[package]:raise ValueError('Runtime differs from original baseline: '+package)
    direct=json.loads(metadata.distribution('lerobot').read_text('direct_url.json'))
    if direct.get('vcs_info',{}).get('commit_id')!=REVISION:raise ValueError('Unvalidated LeRobot source revision')
    required=['--env.type=libero','--env.task=libero_spatial','--env.task_ids=[2]',
        '--env.control_mode=relative','--env.init_states=true','--env.hard_reset=true',
        '--env.observation_height=256','--env.observation_width=256','--env.max_parallel_tasks=1',
        '--eval.batch_size=1',f'--eval.n_episodes={args.episodes}','--seed=0','--policy.device=cuda','--policy.load_vlm_weights=false']
    if any(a not in official_args for a in required):raise ValueError('Use the documented matched official configuration')
    allowed={a.split('=',1)[0] for a in required}|{'--policy.path','--output_dir'}
    keys=[a.split('=',1)[0] for a in official_args]
    if len(set(keys))!=len(keys) or any(k not in allowed for k in keys):
        raise ValueError('Unexpected or duplicate official override; use the documented cell unchanged')
    baseline_info=args.baseline_root/'libero_spatial_2/eval_info.json'
    if not baseline_info.is_file():raise FileNotFoundError('Need original eval_info.json to pair success outcomes')
    manifest=json.loads((args.baseline_root/'sanity_manifest.json').read_text())
    if manifest['checkpoint_revision']!='6721902bc4d61e50a3bfdb11dfb4cb626f05d102' or manifest['episodes']!=3 or manifest['seed']!=0 or manifest['task_index']!=2:
        raise ValueError('Unexpected baseline manifest')
    policy_path=next(a.split('=',1)[1] for a in official_args if a.startswith('--policy.path='))
    checkpoint=Path(policy_path)
    config=json.loads((checkpoint/'config.json').read_text())
    if (config['n_action_steps'],config['chunk_size'],config['num_steps'])!=(1,50,10):raise ValueError('Checkpoint action configuration changed')
    # snapshot_download(local_dir=...) stores revision/ETag evidence beside each file.
    # Validate that evidence without fetching a second copy of the model weights.
    checkpoint_hashes={}
    for name in ['config.json','model.safetensors','policy_preprocessor.json','policy_postprocessor.json',
        'policy_preprocessor_step_5_normalizer_processor.safetensors','policy_postprocessor_step_1_unnormalizer_processor.safetensors']:
        evidence=checkpoint/'.cache/huggingface/download'/f'{name}.metadata'
        lines=evidence.read_text().splitlines()
        if lines[0]!=manifest['checkpoint_revision']:raise ValueError('Checkpoint revision mismatch: '+name)
        path=checkpoint/name
        with path.open('rb') as f:
            sha256=hashlib.file_digest(f,'sha256').hexdigest()
        etag=lines[1].strip('"')
        if len(etag)==64:
            actual=sha256
        elif len(etag)==40:
            blob=path.read_bytes();actual=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
        else:raise ValueError('Unexpected checkpoint ETag')
        if actual!=etag:raise ValueError('Checkpoint file changed: '+name)
        checkpoint_hashes[name]=sha256
    ground_truth=output/'ground_truth';ground_truth.mkdir()
    os.environ['BOWL_EXPERIMENT_OUTPUT']=str(ground_truth.resolve())
    from lerobot.envs import libero as libero_module
    from lerobot.scripts import lerobot_eval as evaluator
    original_factories=libero_module._make_env_fns
    def factories(**kwargs):
        return [partial(create_observed_env,factory=f) for f in original_factories(**kwargs)]
    libero_module._make_env_fns=factories
    original_processors=evaluator.make_pre_post_processors
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(config['vlm_model_name'])
    prompt=ORIGINAL if args.condition=='original' else LIQUID
    def processors(*a,**kw):
        pre,post=original_processors(*a,**kw)
        return PromptProcessor(pre,tokenizer,prompt,output),post
    evaluator.make_pre_post_processors=processors
    write_json(output/'experiment.json',dict(condition=args.condition,policy_instruction=prompt,
        environment_instruction=ORIGINAL,checkpoint_sha256=checkpoint_hashes,baseline_root=str(args.baseline_root),original_manifest=manifest,
        matched_seeds=list(range(args.episodes)),expected_initial_state_indices=list(range(args.episodes)),
        prompt_design='Original task instruction unchanged; liquid condition appends the physical constraint',
        baseline_limitation='Original official recording=False; ground-truth bowl poses unavailable',
        override_location='Policy preprocessor input only; environment task_description and BDDL unchanged'))
    sys.argv=[sys.argv[0]]+official_args
    evaluator.main()
    process_outputs(output,args.baseline_root,args.condition,args.episodes)
    print((output/'comparison.md').read_text(),flush=True)


if __name__=='__main__':
    main()
