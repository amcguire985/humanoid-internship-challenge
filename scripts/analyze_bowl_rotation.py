"""Offline orientation diagnosis from an existing C JSONL; never starts a simulator."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def yaw_tilt(matrix):
    r=np.asarray(matrix)
    return float(np.degrees(np.arctan2(r[1,0],r[0,0]))),float(np.degrees(np.arccos(np.clip(r[2,2],-1,1))))


def analyze(path,output):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    start=next((i for i,r in enumerate(rows) if r['phase']=='TRANSPORT'),None)
    if start is None:
        output=Path(output); output.mkdir(parents=True,exist_ok=True)
        report=dict(source=str(path),transport_detected=False,orientation_diagnosis='Unavailable: no confirmed transport',libero_success=any(r.get('libero_success',False) for r in rows))
        (output/'orientation_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    initial=rows[start]
    gb_rotation=np.array(initial['eef_pose'])[:3,:3].T@np.array(initial['bowl_rotation_world_from_object'])
    samples=[]; q_errors=[]; reconstruction_errors=[]
    for i,r in enumerate(rows):
        bowl=np.array(r['bowl_rotation_world_from_object']); eef=np.array(r['eef_pose'])[:3,:3]
        quat=np.array(r['bowl_quaternion_wxyz']); q_errors.append(float(np.max(np.abs(Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()-bowl))))
        sample=dict(timestep=r['timestep'],sim_time_s=r['sim_time_s'],phase=r['phase'],bowl_yaw_deg=yaw_tilt(bowl)[0],bowl_tilt_deg=yaw_tilt(bowl)[1],gripper_yaw_deg=yaw_tilt(eef)[0],relative_gripper_bowl_rotation_vector_deg=np.degrees(Rotation.from_matrix(eef.T@bowl).as_rotvec()).tolist(),relative_grasp_drift_deg=r.get('grasp_transform_rotation_drift_deg'),human_yaw_deg=None,human_tilt_deg=None,applied_target_yaw_deg=None,applied_target_tilt_deg=None,human_orientation_step_deg=None)
        for name,vector in [('policy',r.get('policy_action')),('correction',r.get('correction')),('command',r.get('action'))]:
            for j,axis in enumerate('xyz'): sample[name+'_rotation_'+axis]=None if vector is None else vector[j+3]
        if 'desired_bowl_rotation' in r:
            human=np.array(r['desired_bowl_rotation']); sample['human_yaw_deg'],sample['human_tilt_deg']=yaw_tilt(human)
            applied=np.array(r['orientation_target_bowl_rotation']) if 'orientation_target_bowl_rotation' in r else np.array(r['desired_eef_pose'])[:3,:3]@gb_rotation
            sample['applied_target_yaw_deg'],sample['applied_target_tilt_deg']=yaw_tilt(applied)
            if i>0 and 'desired_bowl_rotation' in rows[i-1]: sample['human_orientation_step_deg']=float(np.degrees(Rotation.from_matrix(human@np.array(rows[i-1]['desired_bowl_rotation']).T).magnitude()))
            reconstructed=np.array(r['policy_action'])[:6]+np.array(r['correction'])
            reconstruction_errors.append(float(np.max(np.abs(reconstructed-np.array(r['action'])[:6]))))
        samples.append(sample)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    with (output/'orientation_signals.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0])); writer.writeheader(); writer.writerows(samples)
    guided=[r for r in rows if 'correction' in r]; human_samples=[s for s in samples if s['human_yaw_deg'] is not None]
    release=next((i for i,r in enumerate(rows) if r['phase']=='RELEASE'),None)
    prior=release if release is not None else len(rows)-1
    next_step=prior+1 if prior+1<len(rows) else None
    first=human_samples[0]
    mismatch=float((first['human_yaw_deg']-samples[start]['bowl_yaw_deg']+180)%360-180)
    def yaw_path(key,subset):
        values=np.array([s[key] for s in subset]); return float(np.degrees(np.unwrap(np.radians(values))[-1]-np.unwrap(np.radians(values))[0]))
    report=dict(source=str(path),source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),transport_start_step=start,release_step=release,initial_bowl_yaw_deg=samples[start]['bowl_yaw_deg'],first_human_yaw_deg=first['human_yaw_deg'],initial_yaw_mismatch_deg=mismatch,bowl_yaw_change_during_guidance_deg=yaw_path('bowl_yaw_deg',samples[start:prior+1]),human_yaw_change_during_guidance_deg=yaw_path('human_yaw_deg',human_samples),max_human_adjacent_orientation_step_deg=max(s['human_orientation_step_deg'] or 0 for s in samples),quaternion_wxyz_matrix_max_error=max(q_errors),policy_plus_correction_command_max_error=max(reconstruction_errors,default=0),max_rotational_correction=max(float(np.max(np.abs(r['correction'][3:6]))) for r in guided),rotational_correction_at_limit_samples=sum(bool(np.any(np.abs(r['correction'][3:6])>=.15-1e-9)) for r in guided),correction_gravity_component_sum=sum(float(r['correction'][5]) for r in guided),policy_gravity_component_sum=sum(float(r['policy_action'][5]) for r in guided),release_correction_before=None if release is None else rows[release].get('correction'),release_correction_after=None if next_step is None else rows[next_step].get('correction',[0]*6),max_relative_rotation_drift_before_step_200_deg=max(r.get('grasp_transform_rotation_drift_deg',0) for r in rows[start:201]),placement_tilt_at_step_240_deg=rows[240]['tilt_deg'] if len(rows)>240 else None,release_tilt_deg=None if release is None else rows[release]['tilt_deg'],final_tilt_deg=rows[-1]['tilt_deg'],final_grasp=rows[-1]['grasp'],libero_success=any(r['libero_success'] for r in rows),missing_signals=['Internal OSC goals/torques and per-contact forces were not recorded; exact controller.json and correctly paired camera videos must be checked separately','Human orientation after RELEASE is absent from this trace; logged last used target is preserved'],video_identity='This report uses JSONL only; pair any video with its exact run before interpreting frame timing')
    report['max_tilt_last_40_steps_deg']=max(r['tilt_deg'] for r in rows[-40:])
    report['last_40_steps_definition']='Diagnostic fixed sample window, includes release and post-release; not the transport metric'
    report['max_tilt_from_release_deg']=None if release is None else max(r['tilt_deg'] for r in rows[release:])
    (output/'orientation_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    plots(samples,start,release,output)
    print(json.dumps(report,indent=2))
    return report


def plots(samples,start,release,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.array([s['sim_time_s'] for s in samples]); t-=t[0]
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
    for key,label in [('bowl_yaw_deg','Bowl'),('gripper_yaw_deg','Gripper'),('human_yaw_deg','Human reference'),('applied_target_yaw_deg','Applied target')]:
        values=np.array([np.nan if s[key] is None else s[key] for s in samples]); valid=np.isfinite(values)
        axes[0].plot(t[valid],np.degrees(np.unwrap(np.radians(values[valid]))),label=label)
    axes[0].set_ylabel('Yaw (deg)'); axes[0].legend()
    for key,label in [('bowl_tilt_deg','Actual bowl'),('human_tilt_deg','Human reference'),('applied_target_tilt_deg','Applied target')]: axes[1].plot(t,[s[key] for s in samples],label=label)
    axes[1].set_ylabel('Opening-axis tilt (deg)'); axes[1].legend()
    for key,label in [('policy_rotation_z','Policy world-Z'),('correction_rotation_z','Hybrid world-Z'),('command_rotation_z','Command world-Z')]: axes[2].plot(t,[s[key] for s in samples],label=label)
    axes[2].set_ylabel('Normalized rotation'); axes[2].legend()
    axes[3].plot(t,[s['relative_grasp_drift_deg'] for s in samples]); axes[3].set(ylabel='Relative rotation drift (deg)',xlabel='Simulator time since settled reset (s)')
    for ax in axes:
        ax.axvline(t[start],color='green',ls='--')
        if release is not None: ax.axvline(t[release],color='red',ls='--')
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/'orientation_sources.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for j,axis in enumerate('xyz'):
        for name in ('policy','correction','command'): axes[j].plot(t,[s[name+'_rotation_'+axis] for s in samples],label=name)
        axes[j].set_ylabel('World '+axis+' rotation'); axes[j].grid(alpha=.2)
        if release is not None: axes[j].axvline(t[release],color='red',ls='--')
    axes[0].legend(); axes[-1].set_xlabel('Simulator time (s)'); fig.tight_layout(); fig.savefig(output/'rotation_actions.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--trajectory',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args(); analyze(args.trajectory,args.output)
