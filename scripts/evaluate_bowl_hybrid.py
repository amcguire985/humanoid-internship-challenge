"""A/B/C hybrid hooks around the validated, pinned official LeRobot evaluator."""
import argparse
import csv
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import evaluate_bowl_liquid as official
from bowl_human_reference import Reference
from bowl_transport_guidance import Guidance, Phases, Settings, pose, physical_placement_goal
from replay_libero_transport import inspect_controller

LIQUID=official.ORIGINAL+'. The bowl is full of liquid. Do not spill it.'
CONTEXT={}


class DynamicPrompt(official.PromptProcessor):
    def __init__(self,pipeline,tokenizer,prompt,output):
        super().__init__(pipeline,tokenizer,official.ORIGINAL,output)
        self.previous=None

    def reset(self):
        self.previous=None; return self.pipeline.reset()

    def __call__(self,observation):
        state=CONTEXT['env'].call('hybrid_status')[0]
        switched=state['transport_started'] and os.environ['BOWL_HYBRID_CONDITION']!='A'
        instruction=LIQUID if switched else official.ORIGINAL
        if instruction!=self.previous:
            if self.previous is not None:
                # Clear SmolVLA action queues and temporal state before new inference.
                CONTEXT['policy'].reset(); self.pipeline.reset(); CONTEXT['postprocessor'].reset()
            self.prompt=instruction; self.checked=False
            self.previous=instruction
            event=dict(timestep=state['timestep'],sim_time_s=state['sim_time_s'],phase=state['phase'],policy_instruction=instruction,environment_instruction=official.ORIGINAL,policy_reset=switched)
            with (self.output/'prompt_transitions.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(event)+'\n')
        result=super().__call__(observation)
        # Keep separate exact token evidence for both prompts.
        name='prompt_liquid.json' if switched else 'prompt_original.json'
        target=self.output/name
        if not target.exists(): target.write_bytes((self.output/'prompt_verification.json').read_bytes())
        return result


class HybridObserver(official.BowlObserver):
    def __init__(self,env,root):
        super().__init__(env,root)
        self.condition=os.environ['BOWL_HYBRID_CONDITION']
        self.settings=Settings(**json.loads(os.environ['BOWL_HYBRID_SETTINGS'])).validate()
        self.reference=Reference.load(os.environ['BOWL_HYBRID_REFERENCE']) if self.condition=='C' else None
        self.videos={}; self.guidance=None; self.status={}; self.extra={}

    def reset(self,*args,**kwargs):
        self.phases=Phases(self.settings); self.guidance=None; self.extra={}; self.initial_height=None; self.slip_latched=False
        result=super().reset(*args,**kwargs)
        self.controller=inspect_controller(self.env._env)
        official.write_json(self.directory/'controller.json',{k:v.tolist() for k,v in self.controller.items()})
        if abs(self.env._env.env.control_freq-20)>1e-6: raise ValueError('Expected validated 20 Hz controller')
        return result

    def eef_pose(self):
        base=self.env._env.env; site=base.robots[0].eef_site_id
        return pose(base.sim.data.site_xpos[site],base.sim.data.site_xmat[site].reshape(3,3))

    def capture(self,observation,action):
        # Reuse the audited ground-truth sampler, then add state-machine diagnostics.
        stream=self.stream; self.stream=io.StringIO()
        try: super().capture(observation,action)
        finally: self.stream=stream
        row=self.rows[-1]; self.directory=self.root/f'episode_{self.episode:03d}'
        if self.initial_height is None: self.initial_height=row['bowl_position_m'][2]
        eef=self.eef_pose(); row['eef_pose']=eef.tolist()
        closing=action is not None and action[6]>0
        released=action is not None and action[6]<0 and row['grasp'] is not True and row['gripper_aperture_m']>self.transport_aperture+.001 if self.phases.start is not None else False
        old_phase=self.phases.phase
        self.phases.update(self.step_number,row['sim_time_s'],row['grasp'],row['bowl_position_m'][2]-self.initial_height,closing,released,row['libero_success'])
        if old_phase!='TRANSPORT' and self.phases.phase=='TRANSPORT':
            self.transport_aperture=row['gripper_aperture_m']
            bowl=pose(row['bowl_position_m'],row['bowl_rotation_world_from_object'])
            self.gripper_bowl=np.linalg.inv(eef)@bowl
            base=self.env._env.env
            np.savez_compressed(self.directory/'grasp_state_diagnostic.npz',sim_state=base.sim.get_state().flatten(),gripper_bowl=self.gripper_bowl,eef_pose=eef,bowl_pose=bowl)
            official.write_json(self.directory/'grasp_snapshot.json',dict(timestep=self.step_number,sim_time_s=row['sim_time_s'],restore_supported=False,reason='MuJoCo state alone omits controller goals, actuator bookkeeping, and policy/RNG state. Comparison uses matched initial states instead of unvalidated restoration.'))
            if self.condition=='C':
                plates=[obj for name,obj in base.objects_dict.items() if 'plate' in name.lower()]
                if len(plates)!=1: raise ValueError('Ambiguous plate placement target')
                plate=plates[0]; body=base.sim.model.body_name2id(plate.root_body)
                bowl_obj=base.objects_dict[official.OBJECT]
                goal,placement_audit=physical_placement_goal(base,bowl_obj,plate)
                official.write_json(self.directory/'placement_target.json',placement_audit)
                self.guidance=Guidance(self.reference,bowl,eef,goal,self.controller,self.settings)
                self.guidance.reference.save(self.directory/'aligned_reference.csv')
                from bowl_human_reference import preview
                preview(self.guidance.reference,self.directory/'aligned_reference.png')
        if self.phases.start is not None:
            relative=np.linalg.inv(eef)@pose(row['bowl_position_m'],row['bowl_rotation_world_from_object'])
            row['grasp_transform_translation_drift_m']=float(np.linalg.norm(relative[:3,3]-self.gripper_bowl[:3,3]))
            row['grasp_transform_rotation_drift_deg']=float(np.degrees(Rotation.from_matrix(relative[:3,:3]@self.gripper_bowl[:3,:3].T).magnitude()))
        if self.step_number>=self.env._max_episode_steps and self.phases.phase not in ('DONE','FAILED'):
            self.phases.phase='FAILED'; self.phases.failure='episode_horizon'
        if self.phases.start is not None:
            row['slip_detected']=bool(row['grasp_transform_translation_drift_m']>self.settings.slip_translation_m or row['grasp_transform_rotation_drift_deg']>self.settings.slip_rotation_deg)
        row.update(phase=self.phases.phase,phase_before_step=old_phase,transport_started=self.phases.start is not None,failure_reason=self.phases.failure,**self.extra)
        self.status={k:row[k] for k in ('timestep','sim_time_s','phase','transport_started')}
        stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
        self.record_videos(observation)

    def record_videos(self,observation):
        import cv2
        for key,filename in [('image','agentview.mp4'),('image2','wrist.mp4')]:
            pixels=observation['pixels'].get(key)
            if pixels is None: continue
            frame=np.asarray(pixels)[::-1,::-1]
            if key not in self.videos:
                h,w=frame.shape[:2]; writer=cv2.VideoWriter(str(self.directory/filename),cv2.VideoWriter_fourcc(*'mp4v'),20,(w,h))
                if not writer.isOpened(): raise RuntimeError('Video encoder could not open '+filename)
                self.videos[key]=writer
            self.videos[key].write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

    def step(self,action):
        original=np.asarray(action,float).copy(); actual=original.copy(); self.extra={'policy_action':original.tolist(),'guidance_active':False}
        if self.condition=='C' and self.phases.phase=='TRANSPORT' and not self.slip_latched:
            row=self.rows[-1]; dt=1/self.env._env.env.control_freq
            bowl=pose(row['bowl_position_m'],row['bowl_rotation_world_from_object'])
            elapsed=row['sim_time_s']-self.phases.start[1]
            actual,diagnostics=self.guidance.apply(original,bowl,self.eef_pose(),elapsed,dt)
            self.extra.update(diagnostics,guidance_active=not diagnostics['slip_detected'])
            self.extra['guidance_slip_detected']=self.extra.pop('slip_detected')
            if diagnostics['slip_detected']:
                self.slip_latched=True; self.extra['guidance_disabled_reason']='rigid_grasp_transform_drift'
        elif self.condition=='C' and self.phases.phase=='RELEASE' and not self.slip_latched:
            actual,diagnostics=self.guidance.release_orientation(original,1/self.env._env.env.control_freq)
            self.extra.update(diagnostics)
        elif self.slip_latched: self.extra['guidance_disabled_reason']='rigid_grasp_transform_drift_latched'
        if self.condition=='C':
            self.extra['policy_rotation_action']=original[3:6].tolist()
            self.extra['executed_rotation_action']=actual[3:6].tolist()
            self.extra['executed_rotation_delta_world_rad']=(((np.clip(actual[3:6],self.controller['input_min'][3:],self.controller['input_max'][3:])-(self.controller['input_max'][3:]+self.controller['input_min'][3:])/2)*(self.controller['output_max'][3:]-self.controller['output_min'][3:])/(self.controller['input_max'][3:]-self.controller['input_min'][3:]))+(self.controller['output_max'][3:]+self.controller['output_min'][3:])/2).tolist()
        self.extra['policy_instruction']=LIQUID if self.phases.start is not None and self.condition!='A' else official.ORIGINAL
        result=super().step(actual)
        if self.phases.phase=='FAILED' and not result[2]:
            observation,reward,terminated,truncated,info=result
            info=dict(info,hybrid_failure=self.phases.failure)
            return observation,reward,terminated,True,info
        return result

    def finish(self):
        for writer in getattr(self,'videos',{}).values(): writer.release()
        self.videos={}; super().finish()


def create_env(factory):
    env=factory(); observer=HybridObserver(env,os.environ['BOWL_EXPERIMENT_OUTPUT'])
    env.reset=observer.reset; env.step=observer.step; env.close=observer.close
    env.hybrid_status=lambda:observer.status
    return env


def analyze(rows):
    # Require the actual confirmed phase transition; never infer transport from pushing.
    starts=[i for i,r in enumerate(rows) if r['phase']=='TRANSPORT']
    summary=dict(libero_success=any(r['libero_success'] for r in rows),grasp_ever=any(r['grasp'] is True for r in rows),transport_detected=bool(starts),max_height_increase_m=float(max(0,max(r['bowl_position_m'][2] for r in rows)-rows[0]['bowl_position_m'][2])),failure_reason=rows[-1]['failure_reason'])
    accelerations=[None]*len(rows); start=end=loss=None
    if starts:
        start=starts[0]; end=len(rows)-1
        for i in range(start+1,len(rows)):
            if rows[i]['phase']=='RELEASE': end=i-1; break
            if rows[i]['phase']=='DONE': end=i; break
            if rows[i]['phase']=='FAILED': end=i; break
        loss=next((i for i in range(start,end+1) if rows[i]['grasp'] is not True),None)
        for i in range(start+1,end):
            t0,t1,t2=[rows[j]['sim_time_s'] for j in (i-1,i,i+1)]
            if not t0<t1<t2: raise ValueError('Non-increasing simulator time')
            p0,p1,p2=[np.array(rows[j]['bowl_position_m']) for j in (i-1,i,i+1)]
            accelerations[i]=float(np.linalg.norm(2*((p2-p1)/(t2-t1)-(p1-p0)/(t1-t0))/(t2-t0)))
    tilt_peak=max(range(start,end+1),key=lambda i:rows[i]['tilt_deg']) if start is not None else None
    acc_indices=[i for i,a in enumerate(accelerations) if a is not None]
    acc_peak=max(acc_indices,key=lambda i:accelerations[i]) if acc_indices else None
    summary.update(transport_start_timestep=start,transport_end_timestep=end,transport_duration_s=None if start is None else rows[end]['sim_time_s']-rows[start]['sim_time_s'],max_tilt_deg=None if tilt_peak is None else rows[tilt_peak]['tilt_deg'],max_acceleration_m_s2=None if acc_peak is None else accelerations[acc_peak],first_possible_grasp_loss_timestep=loss,tilt_peak_timestep=tilt_peak,acceleration_peak_timestep=acc_peak,tilt_peak_after_possible_grasp_loss=None if tilt_peak is None else loss is not None and tilt_peak>=loss,acceleration_peak_after_possible_grasp_loss=None if acc_peak is None else loss is not None and acc_peak>=loss,slip_detected=any(r.get('slip_detected',False) for r in rows if r.get('phase_before_step')=='TRANSPORT'),acceleration_method='Unsmoothed nonuniform-time central three-point second difference, transport interior only; sampled at 20 Hz, not physics substeps',guidance_disabled=any(r.get('guidance_disabled_reason') for r in rows))
    return summary,accelerations



def write_condition_report(output, summary):
    """Fulfil the parent official wrapper's printable report contract."""
    fields=('libero_success','transport_detected','max_tilt_deg','max_acceleration_m_s2','transport_duration_s','failure_reason')
    lines=['| Metric | Value |','|---|---|']
    for key in fields:
        value=summary.get(key)
        lines.append(f"| {key} | {'unavailable' if value is None else value} |")
    Path(output,'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def process_outputs(output,baseline_root,condition,episodes=1):
    directory=output/'ground_truth/episode_000'
    rows=[json.loads(line) for line in (directory/'trajectory.jsonl').read_text().splitlines()]
    summary,acceleration=analyze(rows)
    success=json.loads((output/'eval_info.json').read_text())['per_task'][0]['metrics']['successes']
    if success!=[summary['libero_success']]: raise ValueError('Official success mismatch')
    summary['condition']=os.environ['BOWL_HYBRID_CONDITION']; summary['state_pairing']='Matched seed 0 and initial state 0; grasp snapshots are diagnostic, not restored'
    initialization=json.loads((directory/'initialization.json').read_text())
    if initialization['seed']!=0 or initialization['init_state_index']!=0: raise ValueError('Unexpected matched state')
    official.write_json(directory/'summary.json',summary); official.write_json(output/'summary.json',summary)
    with (directory/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['timestep','sim_time_s','phase','tilt_deg','acceleration_m_s2','grasp','x_m','y_m','z_m'])
        for r,a in zip(rows,acceleration): writer.writerow([r['timestep'],r['sim_time_s'],r['phase'],r['tilt_deg'],a,r['grasp'],*r['bowl_position_m']])
    plot_rollout(directory,rows,acceleration,summary)
    manifest=json.loads((output/'experiment.json').read_text()); manifest.update(condition=summary['condition'],prompt_design='Original until confirmed grasp/lift; liquid appended for B/C transport',policy_instruction_transport=official.ORIGINAL if summary['condition']=='A' else LIQUID,settings=json.loads(os.environ['BOWL_HYBRID_SETTINGS']),human_reference=os.environ.get('BOWL_HYBRID_REFERENCE'),pairing=summary['state_pairing'],weights_updated=False,guidance_algorithm='gravity_opening_axis_v3' if summary['condition']=='C' else 'collision_surface_placement_v2')
    if summary['condition']=='C':
        refpath=Path(os.environ['BOWL_HYBRID_REFERENCE'])
        manifest['reference_sha256']=hashlib.sha256(refpath.read_bytes()).hexdigest()
        manifest['reference_metadata']=json.loads(Path(str(refpath)+'.json').read_text())
    official.write_json(output/'experiment.json',manifest)
    write_condition_report(output,summary)
    if summary['condition']=='C':
        from analyze_bowl_rotation import analyze as analyze_rotation
        try:
            analyze_rotation(directory/'trajectory.jsonl',directory/'orientation_analysis')
        except Exception as exc:
            # Optional plots must not invalidate a completed rollout or its primary report.
            official.write_json(output/'orientation_analysis_error.json',dict(error_type=type(exc).__name__,error=str(exc),simulation_rerun_needed=False))
            print('Orientation analysis failed; rollout and summary are saved:',str(exc),file=sys.stderr,flush=True)

    print(json.dumps(summary,indent=2),flush=True)


def plot_rollout(directory,rows,acceleration,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.array([r['sim_time_s']-rows[0]['sim_time_s'] for r in rows]); p=np.array([r['bowl_position_m'] for r in rows])
    desired=np.array([r.get('desired_bowl_position_m',[np.nan]*3) for r in rows])
    tilt=np.array([r['tilt_deg'] for r in rows]); desired_tilt=np.array([np.degrees(np.arccos(np.clip(np.array(r['desired_bowl_rotation'])[2,2],-1,1))) if 'desired_bowl_rotation' in r else np.nan for r in rows])
    control_tilt=np.array([np.degrees(np.arccos(np.clip(np.array(r['orientation_target_bowl_rotation'])[2,2],-1,1))) if 'orientation_target_bowl_rotation' in r else np.nan for r in rows])
    for name,value,label in [('tilt',tilt,'Bowl tilt (deg)'),('acceleration',acceleration,'Transport acceleration (m/s2)')]:
        fig,ax=plt.subplots(); ax.plot(t,value,label='Actual')
        if name=='tilt':
            ax.plot(t,desired_tilt,label='Human reference')
            if np.isfinite(control_tilt).any(): ax.plot(t,control_tilt,label='Opening-axis control target')
        if summary['transport_start_timestep'] is not None: ax.axvspan(t[summary['transport_start_timestep']],t[summary['transport_end_timestep']],alpha=.1,color='green')
        else: ax.text(.5,.9,'No confirmed transport',transform=ax.transAxes,ha='center')
        ax.set(xlabel='Simulator time since reset (s)',ylabel=label); ax.legend(); fig.tight_layout(); fig.savefig(directory/(name+'.png')); plt.close(fig)
    fig,axes=plt.subplots(3,1,sharex=True,figsize=(9,7))
    for i,ax in enumerate(axes): ax.plot(t,p[:,i],label='Actual'); ax.plot(t,desired[:,i],label='Human target'); ax.set_ylabel('xyz'[i]+' (m)')
    axes[0].legend(); axes[-1].set_xlabel('Simulator time (s)'); fig.tight_layout(); fig.savefig(directory/'desired_actual_position.png'); plt.close(fig)
    fig,ax=plt.subplots(); ax.plot(t,tilt,label='Actual'); ax.plot(t,desired_tilt,label='Human reference')
    if np.isfinite(control_tilt).any(): ax.plot(t,control_tilt,label='Opening-axis control target')
    ax.set(xlabel='Simulator time (s)',ylabel='Gravity-relative tilt (deg)'); ax.legend(); fig.tight_layout(); fig.savefig(directory/'desired_actual_orientation.png'); plt.close(fig)


def compare(root):
    root=Path(root); paths=[root/k for k in 'ABC' if (root/k/'summary.json').exists()]
    summaries=[json.loads((p/'summary.json').read_text()) for p in paths]
    if not summaries: raise ValueError('No completed conditions')
    inits=[json.loads((p/'ground_truth/episode_000/initialization.json').read_text()) for p in paths]
    manifests=[json.loads((p/'experiment.json').read_text()) for p in paths]
    for init,manifest in zip(inits[1:],manifests[1:]):
        for key in ('seed','init_state_index','init_state_sha256','bddl_sha256','local_opening_axis'):
            if init[key]!=inits[0][key]: raise ValueError('Unmatched initial state: '+key)
        if manifest['checkpoint_sha256']!=manifests[0]['checkpoint_sha256']: raise ValueError('Checkpoint mismatch')
        if manifest['settings']!=manifests[0]['settings']: raise ValueError('Different controller settings')
        if manifest.get('guidance_algorithm')!=manifests[0].get('guidance_algorithm'): raise ValueError('Different guidance algorithm versions; preserve the earlier failed run separately')
    # Quantify divergence at phase entry rather than claiming shared grasp state.
    grasp_poses={}
    for path in paths:
        rows=[json.loads(line) for line in (path/'ground_truth/episode_000/trajectory.jsonl').read_text().splitlines()]
        entry=next((r for r in rows if r['phase']=='TRANSPORT'),None)
        grasp_poses[path.name]=None if entry is None else {k:entry[k] for k in ('timestep','sim_time_s','bowl_position_m','bowl_rotation_world_from_object','eef_pose','gripper_qpos')}
    grasp_difference=None
    b=grasp_poses.get('B'); c=grasp_poses.get('C')
    if b is not None and c is not None:
        grasp_difference=dict(timestep_delta_C_minus_B=c['timestep']-b['timestep'],bowl_position_difference_m=float(np.linalg.norm(np.array(c['bowl_position_m'])-b['bowl_position_m'])),bowl_orientation_difference_deg=float(np.degrees(Rotation.from_matrix(np.array(c['bowl_rotation_world_from_object'])@np.array(b['bowl_rotation_world_from_object']).T).magnitude())),eef_position_difference_m=float(np.linalg.norm(np.array(c['eef_pose'])[:3,3]-np.array(b['eef_pose'])[:3,3])),gripper_qpos_difference_m=float(np.linalg.norm(np.array(c['gripper_qpos'])-b['gripper_qpos'])))
    by_condition={s['condition']:s for s in summaries}
    differences={}
    if 'B' in by_condition and 'C' in by_condition:
        for key in ('max_tilt_deg','max_acceleration_m_s2','transport_duration_s'):
            b_value=by_condition['B'][key]; c_value=by_condition['C'][key]
            differences[key]=None if b_value is None or c_value is None else c_value-b_value
    fields=list(summaries[0])
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(summaries)
    official.write_json(root/'comparison.json',dict(conditions=summaries,grasp_entries=grasp_poses,grasp_entry_difference_B_C=grasp_difference,delta_C_minus_B=differences,primary_comparison='B versus C',interpretation='One episode per condition is an integration check, not evidence of a general effect. No liquid physics; tilt/acceleration are spill-risk proxies.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3))
    for ax,key,label in zip(axes,['max_tilt_deg','max_acceleration_m_s2','transport_duration_s'],['Max tilt (deg)','Max acceleration (m/s2)','Transport duration (s)']):
        for i,s in enumerate(summaries):
            if s[key] is None: ax.text(i,0,'N/A',ha='center')
            else: ax.bar(i,s[key])
        ax.set(xticks=range(len(summaries)),xticklabels=[s['condition'] for s in summaries],ylabel=label)
    fig.tight_layout(); fig.savefig(root/'comparison.png'); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hybrid-condition',choices=list('ABC'),required=True)
    parser.add_argument('--human-reference',type=Path)
    parser.add_argument('--guidance-config',type=Path,default=Path('config/smolvla/bowl_hybrid.json'))
    args,remaining=parser.parse_known_args()
    settings=Settings(**json.loads(args.guidance_config.read_text(encoding='utf-8-sig'))).validate()
    if args.hybrid_condition=='C':
        if not args.human_reference: parser.error('C requires --human-reference')
        Reference.load(args.human_reference)
    os.environ['BOWL_HYBRID_CONDITION']=args.hybrid_condition
    os.environ['BOWL_HYBRID_SETTINGS']=json.dumps(vars(settings))
    if args.human_reference: os.environ['BOWL_HYBRID_REFERENCE']=str(args.human_reference.resolve())
    from lerobot.scripts import lerobot_eval as evaluator
    original_rollout=evaluator.rollout; signature=inspect.signature(original_rollout)
    def rollout(*a,**kw):
        bound=signature.bind(*a,**kw); CONTEXT.update({key:bound.arguments[key] for key in ('env','policy','postprocessor')})
        try: return original_rollout(*a,**kw)
        finally: CONTEXT.clear()
    evaluator.rollout=rollout
    official.PromptProcessor=DynamicPrompt; official.create_observed_env=create_env; official.process_outputs=process_outputs
    sys.argv=[sys.argv[0],'--condition','original','--episodes','1']+remaining
    official.main()


if __name__=='__main__': main()
