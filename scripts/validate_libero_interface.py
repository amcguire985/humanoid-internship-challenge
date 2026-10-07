"""Audit a LIBERO-trained SmolVLA on the ordinary white-mug task, without training."""
import argparse
import base64
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from evaluate_libero_smolvla import WorkerRuntime
from evaluate_upright_mug import TASK, resolve_task, write_json

CHECKPOINT = 'HuggingFaceVLA/smolvla_libero'
REVISION = '6721902bc4d61e50a3bfdb11dfb4cb626f05d102'
CAMERAS = ('agentview_image', 'robot0_eye_in_hand_image')
POLICY_IMAGES = ('observation.images.image', 'observation.images.image2')
OBJECT = 'porcelain_mug_1'


def encode_observation(obs):
    """Send raw RGB and raw robosuite state. Official LeRobot handles frame mapping."""
    images = {}
    for key in CAMERAS:
        image = np.asarray(obs[key])
        if image.shape != (256,256,3) or image.dtype != np.uint8:
            raise ValueError('Expected uint8 RGB 256x256x3: '+key)
        images[key] = base64.b64encode(image.tobytes()).decode()
    state = {}
    for key, size in [('robot0_eef_pos',3),('robot0_eef_quat',4),('robot0_gripper_qpos',2)]:
        value = np.asarray(obs[key],dtype=np.float32)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError('Invalid state field: '+key)
        state[key] = value.tolist()
    return dict(images=images,state=state)


def final_libero_action(denormalized, low, high):
    """No sign inversion, gripper threshold, extra 0.5 clamp, or pose rescaling.

    OSC_POSE performs its own conversion from dimensionless input into world-frame
    delta position (metres) / delta axis-angle (radians). Panda uses +close/-open.
    """
    action = np.asarray(denormalized,dtype=float)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError('Expected finite 7D denormalized action')
    return np.clip(action,low,high)


def checked_stats(flat, fields=(('observation.state',8),('action',7))):
    """Validate checkpoint-owned normalization, never substitute bowl statistics."""
    result = {}
    for key,size in fields:
        result[key] = {}
        for name in ('mean','std'):
            value = flat[key+'.'+name]
            value = value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
            if value.shape != (size,) or not np.isfinite(value).all() or (name=='std' and np.any(value<=0)):
                raise ValueError('Unexpected checkpoint normalization: '+key+'.'+name)
            result[key][name] = value.tolist()
    return result


def worker(args):
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from lerobot.configs import FeatureType
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor.env_processor import LiberoProcessorStep
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required in the policy worker')
        root = Path(snapshot_download(args.checkpoint,revision=args.revision,
            allow_patterns=['*.json','*.safetensors']))
        pre_json = json.loads((root/'policy_preprocessor.json').read_text())
        post_json = json.loads((root/'policy_postprocessor.json').read_text())
        norm = next(s for s in pre_json['steps'] if s['registry_name']=='normalizer_processor')
        unnorm = next(s for s in post_json['steps'] if s['registry_name']=='unnormalizer_processor')
        flat = load_file(str(root/norm['state_file']))
        stats = checked_stats(flat)
        post_stats = checked_stats(load_file(str(root/unnorm['state_file'])), fields=(('action',7),))
        if stats['action'] != post_stats['action']:
            raise ValueError('Pre/post action statistics disagree')
        config = SmolVLAConfig.from_pretrained(root)
        original_features = json.loads((root/'config.json').read_text())['input_features']
        if config.output_features['action'].shape != (7,) or config.adapt_to_pi_aloha:
            raise ValueError('Unsupported action parameterization')
        if config.normalization_mapping[FeatureType.STATE].value != 'MEAN_STD' or config.normalization_mapping[FeatureType.ACTION].value != 'MEAN_STD':
            raise ValueError('Expected audited MEAN_STD checkpoint')
        if config.input_features['observation.state'].shape != (8,):
            raise ValueError('Expected checkpoint-native 8D state')
        if tuple(config.image_features) != POLICY_IMAGES or any(
                config.image_features[k].shape != (3,256,256) for k in POLICY_IMAGES):
            raise ValueError('Unexpected checkpoint camera order or shape')
        if config.empty_cameras != 0:
            raise ValueError('Expected no fabricated cameras')
        config.device='cuda'; config.load_vlm_weights=False
        pre,post=make_pre_post_processors(config,pretrained_path=root)
        policy=SmolVLAPolicy.from_pretrained(root,config=config,strict=True)
        policy.eval(); policy.reset()
        env_processor=LiberoProcessorStep()
        # Verify the ACTUAL saved postprocessor can produce open and close values.
        mean=torch.tensor(stats['action']['mean']);std=torch.tensor(stats['action']['std'])
        targets=torch.zeros((2,7));targets[:,6]=torch.tensor([-1.,1.])
        normalized=(targets-mean)/std
        recovered=post(normalized).cpu()
        if not torch.allclose(recovered,targets,atol=1e-5):
            raise ValueError('Checkpoint action denormalization failed round-trip')
        metadata=dict(checkpoint=args.checkpoint,revision=args.revision,libero_trained=True,
            training_task_coverage='LIBERO checkpoint; competence on this libero_90 task must be measured',
            original_input_features=original_features,state_metadata_correction=None,
            chunk_size=config.chunk_size,n_action_steps=config.n_action_steps,num_steps=config.num_steps,
            versions={name:__import__('importlib.metadata',fromlist=['version']).version(name) for name in ('lerobot','transformers','torch')},
            normalization_source='Checkpoint serialized processors and their safetensors, not local data',
            normalization=stats,processor_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in
                [root/'policy_preprocessor.json',root/'policy_postprocessor.json',root/norm['state_file'],root/unnorm['state_file']]},
            gripper_roundtrip=dict(normalized=normalized[:,6].tolist(),denormalized=recovered[:,6].tolist()),
            camera_order=['agentview -> observation.images.image','eye_in_hand -> observation.images.image2'],
            preprocessing='Official LiberoProcessorStep: rotate raw RGB 180 degrees; xyzw quaternion to axis-angle; EEF pos(3)+axis-angle(3)+gripper qpos(2)',
            image_normalization='uint8 RGB -> CHW float32 /255; model padded resize to 512x512 then 2*x-1',
            action_order=['dx','dy','dz','drot_x','drot_y','drot_z','gripper'],
            gripper_convention='After checkpoint denormalization: positive closes; negative opens; zero holds')
    print(json.dumps(metadata),flush=True)
    first=True
    for line in sys.stdin:
        request=json.loads(line)
        with contextlib.redirect_stdout(sys.stderr),torch.no_grad():
            if request['op']=='reset':
                torch.manual_seed(request['seed']);policy.reset();first=True
                reply={'reset':True}
            else:
                start=time.perf_counter()
                batch={}
                for camera,key in zip(CAMERAS,POLICY_IMAGES):
                    image=np.frombuffer(base64.b64decode(request['images'][camera]),dtype=np.uint8).reshape(256,256,3).copy()
                    batch[key]=torch.from_numpy(image).permute(2,0,1).float().unsqueeze(0)/255
                state=request['state']
                batch['observation.robot_state']={'eef':{
                    'pos':torch.tensor([state['robot0_eef_pos']]),'quat':torch.tensor([state['robot0_eef_quat']])},
                    'gripper':{'qpos':torch.tensor([state['robot0_gripper_qpos']])}}
                mapped=env_processor.observation(batch)
                model_batch={k:v.squeeze(0) for k,v in mapped.items()}
                model_batch['task']=request['task']
                processed=pre(model_batch)
                if processed['observation.state'].shape!=(1,8):
                    raise ValueError('Final state is not 8D')
                expected_state=(mapped['observation.state'].to('cuda')-torch.tensor(stats['observation.state']['mean'],device='cuda'))/(torch.tensor(stats['observation.state']['std'],device='cuda')+1e-8)
                if not torch.allclose(processed['observation.state'],expected_state,atol=1e-5):
                    raise ValueError('Loaded state normalization differs from checkpoint tensors')
                raw=policy.predict_action_chunk(processed)
                denormalized=post(raw.clone())
                expected_action=raw*torch.tensor(stats['action']['std'],device=raw.device)+torch.tensor(stats['action']['mean'],device=raw.device)
                if not torch.allclose(denormalized.cpu(),expected_action.cpu(),atol=1e-5):
                    raise ValueError('Loaded action denormalization differs from checkpoint tensors')
                reply=dict(raw_policy_output=raw[0].cpu().tolist(),normalized_action=raw[0].cpu().tolist(),
                    denormalized_action=denormalized[0].cpu().tolist(),inference_seconds=time.perf_counter()-start)
                if first:
                    images,masks=policy.prepare_images(processed)
                    reply['observation_diagnostic']={k:dict(shape=list(v.shape),dtype=str(v.dtype),
                        min=float(v.min()),max=float(v.max())) for k,v in processed.items() if torch.is_tensor(v)}
                    reply['observation_diagnostic'].update(state_before_normalization=mapped['observation.state'][0].tolist(),
                        normalized_state=processed['observation.state'][0].cpu().tolist(),
                        model_image_order=[k for k in config.image_features if k in processed],
                        internal_image_shapes=[list(v.shape) for v in images],
                        internal_image_masks=[v.cpu().tolist() for v in masks],
                        padded_state_shape=list(policy.prepare_state(processed).shape),task=request['task'])
                    first=False
        print(json.dumps(reply,allow_nan=False),flush=True)


class ValidationRuntime(WorkerRuntime):
    def __init__(self,args):
        self.process=subprocess.Popen([args.policy_python,str(Path(__file__).resolve()),'--worker',
            '--checkpoint',args.checkpoint,'--revision',args.revision],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,text=True,bufsize=1)
        try:self.metadata=self.read()
        except BaseException:self.close();raise

    def request(self,request):
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        return self.read()

    def reset(self,seed):return self.request(dict(op='reset',seed=seed))
    def predict(self,obs,task):return self.request(dict(op='predict',task=task,**encode_observation(obs)))


def controller_audit(env):
    from replay_libero_transport import inspect_controller
    scaling=inspect_controller(env)
    for key,target in [('input_min',[-1]*6),('input_max',[1]*6),
                       ('output_min',[-.05]*3+[-.5]*3),('output_max',[.05]*3+[.5]*3)]:
        if not np.allclose(scaling[key],target):
            raise ValueError('Unexpected LIBERO controller scale: '+key)
    robot=env.env.robots[0]
    if type(robot.gripper).__name__!='PandaGripper':
        raise ValueError('Expected audited PandaGripper implementation')
    signs={}
    for sign in [-1,1]:
        gripper=copy.deepcopy(robot.gripper)
        gripper.current_action=np.zeros(2)
        signs[str(sign)]=np.asarray(gripper.format_action(np.array([sign]))).tolist()
    if signs['1'][0]>=0 or signs['1'][1]<=0 or signs['-1'][0]<=0 or signs['-1'][1]>=0:
        raise ValueError('Unexpected Panda gripper sign mapping')
    return dict(scales={k:v.tolist() for k,v in scaling.items()},gripper_format_action_from_zero=signs,
                controller='World-frame fixed-impedance OSC_POSE delta control',control_frequency_hz=env.env.control_freq)


def gripper_probe(env,obs,steps):
    """Separate diagnostic only. Restore benchmark state before any policy rollout."""
    records=[]
    for sign in [-1,1,-1]:
        before=np.asarray(obs['robot0_gripper_qpos']).copy()
        for _ in range(steps):
            obs,_,_,_=env.step(np.r_[np.zeros(6),sign])
        after=np.asarray(obs['robot0_gripper_qpos']).copy()
        records.append(dict(command=sign,qpos_before=before.tolist(),qpos_after=after.tolist(),
                            aperture_before_m=float(abs(before[0]-before[1])),aperture_after_m=float(abs(after[0]-after[1]))))
    return dict(steps_per_command=steps,commands=records,
                physically_closes=records[1]['aperture_after_m']<records[1]['aperture_before_m']-.001,
                physically_reopens=records[2]['aperture_after_m']>records[2]['aperture_before_m']+.001)


def aggregate(reports,metadata):
    return dict(checkpoint=metadata['checkpoint'],libero_trained=metadata['libero_trained'],
                checkpoint_revision=metadata['revision'],
                observation_mapping=metadata.get('preprocessing'),camera_order=metadata.get('camera_order'),
                action_order=metadata.get('action_order'),normalization_source=metadata.get('normalization_source'),
                gripper_convention=metadata.get('gripper_convention'),num_rollouts=len(reports),
                rollouts_with_close_commands=sum(r['gripper_ever_commanded_close'] for r in reports),
                rollouts_with_finger_closure=sum(r['finger_aperture_reduced'] for r in reports),
                rollouts_with_grasp=sum(r['grasp_ever'] for r in reports),
                rollouts_with_lift=sum(r['mug_lifted'] for r in reports),
                libero_successes=sum(r['libero_success'] for r in reports),
                libero_success_rate=sum(r['libero_success'] for r in reports)/len(reports) if reports else None,
                ordinary_baseline_validated=any(r['libero_success'] and r['grasp_ever'] and r['mug_lifted'] and r['finger_aperture_reduced'] for r in reports),
                next_step='Inspect failures; do not run orientation experiments' if not any(r['libero_success'] and r['grasp_ever'] and r['mug_lifted'] and r['finger_aperture_reduced'] for r in reports)
                          else 'Ordinary-task success observed; review action/observation diagnostics before constraint comparison')


def evaluate(args,runtime=None):
    if not args.no_video:
        from video_paths import video_root
        video_root()  # Fail before model download if external video storage is unset.
    os.environ.setdefault('MUJOCO_GL','osmesa')
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from evaluate_libero_bc import rendering_settings
    suite=benchmark.get_benchmark_dict()['libero_90'](task_order_index=0)
    index,task=resolve_task(suite,TASK,None)
    states=suite.get_task_init_states(index)
    if args.num_rollouts>len(states):raise ValueError('Too many rollouts for distinct initial states')
    args.output.mkdir(parents=True,exist_ok=False)
    bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    if args.gripper_probe_only:
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
        try:
            env.seed(args.seed);env.reset();obs=env.set_init_state(states[0])
            result=dict(controller=controller_audit(env),probe=gripper_probe(env,obs,50))
            write_json(args.output/'gripper_probe.json',result);print(json.dumps(result,indent=2),flush=True)
        finally:env.close()
        return
    owned=runtime is None
    runtime=runtime or ValidationRuntime(args)
    reports=[]
    try:
        steps=runtime.metadata['n_action_steps'] if args.execute_steps is None else args.execute_steps
        if not 1<=steps<=runtime.metadata['chunk_size']:raise ValueError('Invalid execute-steps')
        manifest=dict(task_name=task.name,task_index=index,suite='libero_90',
            original_task_instruction=task.language,policy_instruction=task.language,
            checkpoint=runtime.metadata,seed=args.seed,horizon=args.horizon,execute_steps=steps,
            observation_resolution=[256,256],settle_steps=10,
            lift_definition='Mug root-body Z increases by >= 0.02 m from settled start; diagnostic only',
            action_processing='Checkpoint postprocessor then only clip to environment bounds; no extra scaling or sign threshold',
            bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest())
        write_json(args.output/'experiment.json',manifest)
        init_order=np.random.default_rng(args.seed).permutation(len(states))
        for number in range(args.num_rollouts):
            seed=args.seed+number;directory=args.output/f'rollout_{number:03d}';directory.mkdir()
            env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
            started=time.perf_counter();rows=[];video=None
            try:
                env.seed(seed);env.reset();obs=env.set_init_state(states[int(init_order[number])])
                audit=controller_audit(env);write_json(directory/'controller.json',audit)
                sanity=dict(checkpoint=runtime.metadata['checkpoint'],task_name=task.name,task_index=index,
                    instruction=task.language,camera_keys=list(CAMERAS),policy_images=list(POLICY_IMAGES),
                    state_shape=[8],image_shapes=[[256,256,3]]*2,action_dimension=7,
                    action_order=runtime.metadata.get('action_order'),gripper=runtime.metadata.get('gripper_convention'),
                    normalization_source=runtime.metadata.get('normalization_source'),controller=audit,
                    chunk_size=runtime.metadata['chunk_size'],n_action_steps=runtime.metadata['n_action_steps'],
                    execute_steps=steps,num_steps=runtime.metadata.get('num_steps'),horizon=args.horizon)
                if number==0:print('Runtime sanity:',json.dumps(sanity),flush=True)
                write_json(args.output/'runtime_sanity.json',sanity)
                rendering_settings(env.env.sim)
                for _ in range(10):obs,_,_,_=env.step(np.r_[np.zeros(6),-1])
                runtime.reset(seed)
                if not args.no_video:
                    video=RolloutVideo(args.output,number,env.env.control_freq)
                    video.append(obs)
                obj=env.env.objects_dict[OBJECT]
                body=env.env.sim.model.body_name2id(obj.root_body)
                start_z=float(env.env.sim.data.body_xpos[body][2])
                initial_aperture=float(abs(np.diff(obs['robot0_gripper_qpos'])[0]))
                chunk=None;stop='horizon'
                with (directory/'trajectory.jsonl').open('w',encoding='utf-8') as trace:
                    for step in range(args.horizon):
                        if env.check_success():stop='libero_success';break
                        if step%steps==0:
                            chunk=runtime.predict(obs,task.language)
                            raw=np.asarray(chunk['raw_policy_output']);normalized=np.asarray(chunk['normalized_action']);denorm=np.asarray(chunk['denormalized_action'])
                            expected=(runtime.metadata['chunk_size'],7)
                            if any(a.shape!=expected or not np.isfinite(a).all() for a in [raw,normalized,denorm]):
                                raise ValueError('Invalid predicted chunk')
                            if 'observation_diagnostic' in chunk:
                                write_json(directory/'observation.json',chunk['observation_diagnostic'])
                                print('Model observation:',json.dumps(chunk['observation_diagnostic']),flush=True)
                            print(f'Rollout {number}: step {step}/{args.horizon}, inference {chunk["inference_seconds"]:.2f}s, elapsed {time.perf_counter()-started:.1f}s',flush=True)
                        i=step%steps
                        action=final_libero_action(denorm[i],*env.env.action_spec)
                        before=np.asarray(obs['robot0_eef_pos']).copy()
                        obs,_,done,_=env.step(action)
                        position=np.array(env.env.sim.data.body_xpos[body])
                        eef=np.asarray(obs['robot0_eef_pos']);qpos=np.asarray(obs['robot0_gripper_qpos'])
                        from upright_orientation import get_mug_orientation, compute_upright_deviation_deg
                        rotation,_=get_mug_orientation(env,OBJECT)
                        tilt=compute_upright_deviation_deg(rotation)
                        if video:video.append(obs)
                        grasp=bool(env.env._check_grasp(env.env.robots[0].gripper,obj))
                        row=dict(timestep=step+1,raw_policy_output=raw[i].tolist(),normalized_action=normalized[i].tolist(),
                            denormalized_action=denorm[i].tolist(),final_libero_action=action.tolist(),
                            controller_scaled_delta=env.env.robots[0].controller.scale_action(action[:6]).tolist(),
                            eef_position_m=eef.tolist(),eef_displacement_m=(eef-before).tolist(),
                            gripper_qpos=qpos.tolist(),gripper_aperture_m=float(abs(qpos[0]-qpos[1])),
                            mug_position_m=position.tolist(),mug_height_increase_m=float(position[2]-start_z),
                            eef_mug_distance_m=float(np.linalg.norm(eef-position)),grasp=grasp,
                            mug_tilt_deg=tilt,libero_success=bool(env.check_success()))
                        rows.append(row);trace.write(json.dumps(row,allow_nan=False)+'\n');trace.flush()
                        if step<args.diagnostic_steps:print('Action diagnostic:',json.dumps(row),flush=True)
                        if row['libero_success'] or done:
                            stop='libero_success' if row['libero_success'] else 'environment_done';break
                report=dict(rollout_id=number,seed=seed,init_state_index=int(init_order[number]),
                    policy_instruction=task.language,steps=len(rows),termination=stop,
                    libero_success=bool(env.check_success()),
                    gripper_ever_commanded_close=any(r['final_libero_action'][6]>0 for r in rows),
                    finger_aperture_reduced=any(r['gripper_aperture_m']<initial_aperture-.001 for r in rows),
                    grasp_ever=any(r['grasp'] for r in rows),
                    minimum_eef_mug_distance_m=min((r['eef_mug_distance_m'] for r in rows),default=None),
                    max_mug_height_increase_m=max([0]+[r['mug_height_increase_m'] for r in rows]),
                    mug_lifted=any(r['mug_height_increase_m']>=.02 for r in rows),
                    max_mug_tilt_deg=max((r['mug_tilt_deg'] for r in rows),default=None),
                    videos=video.paths if video else [],wall_seconds=time.perf_counter()-started)
                write_json(directory/'summary.json',report);reports.append(report)
                write_json(args.output/'validation_report.json',aggregate(reports,runtime.metadata))
                write_results_table(args.output,reports)
                print('Rollout summary:',json.dumps(report),flush=True)
            except BaseException as error:
                write_json(directory/'error.json',dict(type=type(error).__name__,message=str(error),recorded_steps=len(rows)))
                raise
            finally:
                if video:video.close()
                env.close()
        print(json.dumps(aggregate(reports,runtime.metadata),indent=2),flush=True)
    finally:
        if owned:runtime.close()



class RolloutVideo:
    """Stream both policy-oriented RGB camera views to external storage."""
    def __init__(self, output, number, fps):
        import cv2
        from video_paths import output_video
        self.cv2=cv2;self.writers=[];self.paths=[]
        try:
            for camera in CAMERAS:
                path=output_video(output,f'rollout_{number:03d}_{camera}.mp4')
                if path.exists():raise FileExistsError(path)
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(256,256))
                self.writers.append(writer)
                if not writer.isOpened():raise RuntimeError('Cannot open video: '+str(path))
                self.paths.append(str(path))
        except BaseException:
            self.close();raise

    def append(self, obs):
        for camera,writer in zip(CAMERAS,self.writers):
            rgb=np.ascontiguousarray(np.asarray(obs[camera])[::-1,::-1])
            writer.write(self.cv2.cvtColor(rgb,self.cv2.COLOR_RGB2BGR))

    def close(self):
        for writer in self.writers:writer.release()


def write_results_table(output,reports):
    lines=['| Rollout | LIBERO success | Gripper closed | Mug lifted | Max mug height increase |',
           '|---|---|---|---|---|']
    for r in reports:
        lines.append(f"| {r['rollout_id']} | {r['libero_success']} | {r['finger_aperture_reduced']} | {r['mug_lifted']} | {r['max_mug_height_increase_m']:.4f} m |")
    (output/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',choices=[CHECKPOINT],default=CHECKPOINT)
    p.add_argument('--revision',default=REVISION)
    p.add_argument('--num-rollouts',type=int,default=5)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--horizon',type=int,default=300)
    p.add_argument('--execute-steps',type=int,default=None)
    p.add_argument('--diagnostic-steps',type=int,default=5)
    p.add_argument('--policy-python',default='python')
    p.add_argument('--output',type=Path,default=Path('results/libero_interface_validation'))
    p.add_argument('--gripper-probe-only',action='store_true')
    p.add_argument('--no-video',action='store_true',help='Explicitly disable validation videos')
    p.add_argument('--worker',action='store_true')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.num_rollouts<1 or args.horizon<1 or args.seed<0 or args.diagnostic_steps<0:
        raise SystemExit('Invalid rollout count, horizon, seed or diagnostic count')
    if args.worker:worker(args)
    else:evaluate(args)
