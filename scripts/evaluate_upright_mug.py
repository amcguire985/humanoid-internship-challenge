"""Pretrained SmolVLA upright-mug baseline; no training or modified task predicate."""
import argparse
import base64
import contextlib
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
from evaluate_libero_smolvla import WorkerRuntime
from upright_orientation import compute_upright_deviation_deg, get_mug_orientation

TASK = 'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate'
INSTRUCTION = ('Pick up the white mug and place it on the plate. The mug is full of liquid, '
               'so keep it upright throughout the entire motion and do not spill it.')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def worker(args):
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
        from transformers import AutoTokenizer
        from huggingface_hub import model_info
        from prepare_libero_smolvla import IMAGE, STATE
        if not torch.cuda.is_available():
            raise RuntimeError('Policy worker requires CUDA')
        revision = args.revision or model_info(args.checkpoint).sha
        config = SmolVLAConfig.from_pretrained(args.checkpoint, revision=revision)
        # Same LIBERO feature adapter as train_libero_smolvla, but no optimizer.
        config.input_features = {IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3,128,128)),
                                 STATE: PolicyFeature(type=FeatureType.STATE, shape=(18,))}
        config.output_features = {'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
        config.device = 'cuda'
        config.load_vlm_weights = False
        # The longer constraint MUST survive tokenization (base default is only 48).
        tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name)
        config.tokenizer_max_length = max(128, len(tokenizer.encode(args.policy_instruction + '\n')) + 8)
        policy = SmolVLAPolicy.from_pretrained(args.checkpoint, revision=revision, config=config, strict=True)
        stats = json.loads(args.stats.read_text(encoding='utf-8-sig'))
        for key, size in [(STATE,18), ('action',7)]:
            for name in ['mean', 'std']:
                a = np.asarray(stats[key][name])
                if a.shape != (size,) or not np.isfinite(a).all() or (name == 'std' and np.any(a <= 0)):
                    raise ValueError('Invalid normalization statistics: ' + key + '/' + name)
        stats = {k:{n:torch.tensor(v) for n,v in s.items()} for k,s in stats.items()}
        pre, post = make_smolvla_pre_post_processors(config, stats)
        policy.eval()
    print(json.dumps(dict(checkpoint=args.checkpoint, revision=revision, chunk_size=config.chunk_size,
                          tokenizer_max_length=config.tokenizer_max_length,
                          policy_instruction=args.policy_instruction)), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr), torch.no_grad():
            if request['op'] == 'reset':
                torch.manual_seed(request['seed']); np.random.seed(request['seed']); policy.reset()
                result = {'reset': True}
            else:
                if request['task'] != args.policy_instruction:
                    raise ValueError('Unexpected policy prompt')
                rgb = np.frombuffer(base64.b64decode(request['rgb']), dtype=np.uint8).reshape(128,128,3).copy()
                batch = {IMAGE:torch.from_numpy(rgb).permute(2,0,1).float()/255,
                         STATE:torch.tensor(request['state'],dtype=torch.float32), 'task':request['task']}
                processed = pre(batch)
                # Verify the actual processor's unmasked token stream includes the whole prompt.
                expected = tokenizer(args.policy_instruction+'\n', add_special_tokens=True)['input_ids']
                ids = processed['observation.language.tokens'][0]
                mask = processed['observation.language.attention_mask'][0]
                actual = ids[mask.bool()].cpu().tolist()
                if actual != expected:
                    raise ValueError('Policy processor altered or truncated the instruction')
                result = post(policy.predict_action_chunk(processed))[0].cpu().numpy().tolist()
        print(json.dumps(result), flush=True)


class MugRuntime(WorkerRuntime):
    def __init__(self, args):
        import subprocess
        self.process = subprocess.Popen([args.policy_python, str(Path(__file__).resolve()), '--worker',
            '--checkpoint', args.checkpoint, '--stats', str(args.stats.resolve()),
            '--policy-instruction', args.policy_instruction] + (['--revision',args.revision] if args.revision else []),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            self.metadata = self.read()
        except BaseException:
            self.close(); raise

    def request(self, value):
        self.process.stdin.write(json.dumps(value)+'\n'); self.process.stdin.flush()
        return self.read()

    def reset(self, seed):
        self.request(dict(op='reset',seed=seed))

    def predict(self, rgb, state, task):
        return np.asarray(self.request(dict(op='predict', rgb=base64.b64encode(rgb.tobytes()).decode(),
                                           state=state.tolist(),task=task)), dtype=float)


def resolve_task(suite, task_name, task_index):
    if task_index is not None:
        task = suite.get_task(task_index)
        return task_index, task
    matches = [(i,suite.get_task(i)) for i in range(suite.n_tasks) if suite.get_task(i).name == task_name]
    if len(matches) != 1:
        raise ValueError('Task identifier not uniquely found: '+task_name)
    return matches[0]


def record_pose(env, obs, timestep, stage, action=None):
    rotation, quat = get_mug_orientation(env, env._orientation_object_name)
    return dict(timestep=timestep, stage=stage, sim_time_s=float(env.env.sim.data.time),
                mug_quat_wxyz=quat.tolist(), mug_rotation_world_from_object=rotation.tolist(),
                mug_upright_deviation_deg=compute_upright_deviation_deg(rotation),
                eef_position_m=np.asarray(obs['robot0_eef_pos']).tolist(),
                action=None if action is None else action.tolist(), libero_success=bool(env.check_success()))


def summarize(rows):
    maxima = np.array([r['max_mug_tilt_deg'] for r in rows])
    return dict(num_rollouts=len(rows), libero_success_rate=float(np.mean([r['libero_success'] for r in rows])),
                mean_max_tilt_deg=float(maxima.mean()), median_max_tilt_deg=float(np.median(maxima)),
                min_max_tilt_deg=float(maxima.min()), max_max_tilt_deg=float(maxima.max()))


def plot_trace(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    angles = [r['mug_upright_deviation_deg'] for r in rows]
    peak = rows[int(np.argmax(angles))]
    fig, ax = plt.subplots()
    ax.plot([r['timestep'] for r in rows], angles)
    ax.scatter([peak['timestep']], [peak['mug_upright_deviation_deg']], color='red')
    ax.annotate(f"Max: {peak['mug_upright_deviation_deg']:.2f} deg", (peak['timestep'],peak['mug_upright_deviation_deg']))
    ax.set(xlabel='Control timestep (negative = reset settling)', ylabel='Mug upright deviation (degrees)')
    fig.tight_layout(); fig.savefig(output); plt.close(fig)


def evaluate(args, runtime=None):
    os.environ.setdefault('MUJOCO_GL','osmesa')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from evaluate_libero_bc import proprio, rendering_settings
    import hashlib
    suite = benchmark.get_benchmark_dict()[args.suite](task_order_index=0)
    index, task = resolve_task(suite,args.task_name,args.task_index)
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    states = suite.get_task_init_states(index)
    if args.num_rollouts > len(states):
        raise ValueError('Requested more rollouts than distinct benchmark initializations')
    args.output.mkdir(parents=True,exist_ok=False)
    owned = runtime is None
    runtime = runtime or MugRuntime(args)
    reports = []
    try:
        if not 1 <= args.execute_steps <= runtime.metadata['chunk_size']:
            raise ValueError('execute-steps exceeds predicted chunk size')
        manifest = dict(suite=args.suite, task_index=index, task_name=task.name,
            original_task_instruction=task.language, policy_instruction=args.policy_instruction,
            object_name=args.object_name, object_local_up=[0,0,1], world_up=[0,0,1],
            quaternion_order='wxyz', rotation='world_from_object', policy=runtime.metadata,
            normalization_stats=str(args.stats.resolve()), stats_sha256=hashlib.sha256(args.stats.read_bytes()).hexdigest(),
            bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(), horizon=args.horizon,
            execute_steps=args.execute_steps, settle_steps=args.settle_steps,
            adapter='agentview RGB 128x128 flipped vertically; 18D joint pos/vel + gripper pos/vel; normalized OSC 7D actions',
            success_definition='Unmodified env.check_success() at termination; success stops rollout',
            sampling='Every env.step endpoint, including initial and settling states; not physics substeps',
            baseline='Untrained LIBERO adapter on pretrained SmolVLA; normalization borrowed from robot-native bowl data')
        write_json(args.output/'experiment.json',manifest)
        print(json.dumps(manifest,indent=2),flush=True)
        for rollout_id in range(args.num_rollouts):
            seed = args.seed + rollout_id
            # Permuted benchmark states: distinct starts, reproducible order.
            init_index = int(np.random.default_rng(args.seed).permutation(len(states))[rollout_id])
            directory = args.output/f'rollout_{rollout_id:03d}'; directory.mkdir()
            print(f'Rollout {rollout_id}, seed {seed}, init {init_index}\nOriginal: {task.language}\nPolicy: {args.policy_instruction}',flush=True)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=128,camera_widths=128)
            rows = []
            try:
                env.seed(seed); env.reset(); obs = env.set_init_state(states[init_index])
                env._orientation_object_name = args.object_name
                runtime.reset(seed)
                rendering_settings(env.env.sim)
                rows.append(record_pose(env,obs,-args.settle_steps,'initial'))
                for step in range(args.settle_steps):
                    action = np.zeros(7); action[6] = -1
                    obs,_,_,_ = env.step(action)
                    rows.append(record_pose(env,obs,step-args.settle_steps+1,'settling',action))
                for step in range(args.horizon):
                    if env.check_success():
                        break
                    if step % args.execute_steps == 0:
                        rgb = env.env.sim.render(width=128,height=128,camera_name='agentview')[::-1].copy()
                        chunk = runtime.predict(rgb,proprio(env,obs),args.policy_instruction)
                        if chunk.shape != (runtime.metadata['chunk_size'],7) or not np.isfinite(chunk).all():
                            raise ValueError('Invalid policy action chunk')
                    action = chunk[step % args.execute_steps].copy()
                    # Preserve existing evaluator's OSC clipping and gripper convention.
                    action[:6] = np.clip(action[:6],-.5,.5)
                    action[6] = 1 if action[6] >= 0 else -1
                    obs,_,done,_ = env.step(action)
                    rows.append(record_pose(env,obs,step+1,'policy',action))
                    if done or rows[-1]['libero_success']:
                        break
                peak = max(rows,key=lambda r:r['mug_upright_deviation_deg'])
                report = dict(rollout_id=rollout_id,seed=seed,init_state_index=init_index,
                    original_task_instruction=task.language,policy_instruction=args.policy_instruction,
                    libero_success=bool(env.check_success()),max_mug_tilt_deg=peak['mug_upright_deviation_deg'],
                    timestep_of_max_tilt=peak['timestep'],sim_time_of_max_tilt_s=peak['sim_time_s'],
                    policy_steps=sum(r['stage']=='policy' for r in rows))
                write_json(directory/'summary.json',report)
                plot_trace(rows,directory/'tilt.png')
                reports.append(report)
                with (args.output/'rollouts.csv').open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(report)); writer.writeheader(); writer.writerows(reports)
                write_json(args.output/'aggregate.json',summarize(reports))
                print(json.dumps(report),flush=True)
            finally:
                if rows:
                    with (directory/'trajectory.jsonl').open('w',encoding='utf-8') as f:
                        for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
                env.close()
        print(json.dumps(summarize(reports),indent=2),flush=True)
    finally:
        if owned: runtime.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',default='libero_90')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--task-name',default=TASK)
    group.add_argument('--task-index',type=int)
    p.add_argument('--object-name',default='porcelain_mug_1')
    p.add_argument('--policy-instruction',default=INSTRUCTION)
    p.add_argument('--checkpoint',default='lerobot/smolvla_base')
    p.add_argument('--revision')
    p.add_argument('--stats',type=Path,default=Path('results/libero_smolvla/training/dataset_stats.json'))
    p.add_argument('--num-rollouts',type=int,default=20)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--horizon',type=int,default=600)
    p.add_argument('--settle-steps',type=int,default=10)
    p.add_argument('--execute-steps',type=int,default=4)
    p.add_argument('--policy-python',default='python')
    p.add_argument('--output',type=Path,default=Path('results/upright_mug_baseline'))
    p.add_argument('--worker',action='store_true')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if min(args.num_rollouts,args.horizon,args.execute_steps) < 1 or args.settle_steps < 0 or args.seed < 0:
        raise SystemExit('Rollouts, horizon and execute-steps must be positive; seed/settling nonnegative')
    if args.worker: worker(args)
    else: evaluate(args)
