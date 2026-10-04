"""Human position replay through verified world-frame delta OSC_POSE; no grasping."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def load_human_trajectory(path, max_gap=.15):
    """Load valid relative positions; end replay at the first unresolved gap."""
    with Path(path).open(newline='',encoding='utf-8-sig') as handle:
        rows=list(csv.DictReader(handle))
    times=[]; positions=[]; previous=None; stop=None
    for row in rows:
        if not all(row.get(k,'').strip() for k in ('time','x_rel','y_rel','z_rel')):
            stop='missing_source_position'; break
        values=np.array([float(row[k]) for k in ('time','x_rel','y_rel','z_rel')])
        if not np.isfinite(values).all():
            raise ValueError('Source contains nonfinite values')
        if previous is not None:
            if values[0]<=previous: raise ValueError('Source timestamps must increase')
            if values[0]-previous>max_gap:
                stop='unresolved_source_gap'; break
        times.append(values[0]); positions.append(values[1:]); previous=values[0]
    if len(times)<2: raise ValueError('Need two continuous valid source samples')
    positions=np.array(positions)
    if np.linalg.norm(positions[0])>1e-6:
        raise ValueError('Relative trajectory must start at zero; normalize explicitly first')
    return np.array(times)-times[0],positions,dict(source_rows=len(rows),replayed_source_rows=len(times),source_stop_reason=stop)


def map_to_libero_frame(positions, permutation, signs, scales):
    """Explicit p_robot=S @ p_human; rows of S correspond to robot x,y,z."""
    if sorted(permutation)!=[0,1,2] or len(signs)!=3 or any(s not in (-1,1) for s in signs):
        raise ValueError('Mapping requires a permutation of 0,1,2 and three signs +/-1')
    scales=np.asarray(scales,dtype=float)
    if scales.shape!=(3,) or not np.isfinite(scales).all() or np.any(scales<=0):
        raise ValueError('Need three finite positive scales')
    matrix=np.diag(np.asarray(signs)*scales)@np.eye(3)[permutation]
    return np.asarray(positions)@matrix.T,matrix


def build_desired_eef_trajectory(times, mapped, start, frequency, time_scale=2, max_speed=.1):
    """Resample at policy frequency, slowing enough to bound desired Cartesian speed."""
    if frequency<=0 or time_scale<=0 or max_speed<=0: raise ValueError('Timing and speed limits must be positive')
    peak=float(np.max(np.linalg.norm(np.diff(mapped,axis=0),axis=1)/np.diff(times)))
    factor=max(time_scale,peak/max_speed)
    duration=float(times[-1]*factor)
    grid=np.arange(int(np.ceil(duration*frequency))+1)/frequency
    source_time=np.minimum(grid/factor,times[-1])
    desired=np.column_stack([np.interp(source_time,times,mapped[:,i]) for i in range(3)])+start
    return grid,desired,factor


def inspect_controller(env):
    """Reject unsupported APIs instead of guessing scale, frame or delta semantics."""
    robot=env.robots[0]
    controller=getattr(robot,'controller',None)
    if controller is None or getattr(controller,'name',None)!='OSC_POSE':
        raise ValueError('Requires the classic robosuite world-frame OSC_POSE controller')
    if not getattr(controller,'use_delta',False) or getattr(controller,'impedance_mode',None)!='fixed':
        raise ValueError('Requires delta control and fixed impedance')
    if getattr(controller,'input_ref_frame','world')!='world':
        raise ValueError('Base-frame action convention unsupported; use a verified world-frame controller')
    low,high=env.env.action_spec
    if np.asarray(low).shape!=(7,) or np.asarray(high).shape!=(7,):
        raise ValueError('Expected exactly six pose commands plus one gripper command')
    config={k:np.broadcast_to(np.asarray(getattr(controller,k),dtype=float),(6,)).copy()
            for k in ('input_min','input_max','output_min','output_max')}
    if any(not np.isfinite(v).all() for v in config.values()) or np.any(config['output_max']<=config['output_min']):
        raise ValueError('Invalid controller scaling')
    config['action_min']=np.array(low); config['action_max']=np.array(high)
    return config


def desired_pose_to_action(desired,actual,target_rotation,current_rotation,controller,clip=.5,gripper=-1):
    """Invert robosuite affine scaling for position and world-frame axis-angle error."""
    delta=np.r_[desired-actual,Rotation.from_matrix(target_rotation@current_rotation.T).as_rotvec()]
    imin,imax=controller['input_min'],controller['input_max']
    omin,omax=controller['output_min'],controller['output_max']
    command=(delta-(omin+omax)/2)*((imax-imin)/(omax-omin))+(imin+imax)/2
    lower=np.maximum(np.maximum(imin,controller['action_min'][:6]),-clip)
    upper=np.minimum(np.minimum(imax,controller['action_max'][:6]),clip)
    if np.any(lower>upper): raise ValueError('Action bounds do not intersect')
    action=np.r_[np.clip(command,lower,upper),gripper]
    return action,bool(np.any(np.abs(action[:6]-command)>1e-12))


def robot_contact(env):
    """Stop on robot/scene contact; internal robot contacts are ignored."""
    sim=env.sim
    for contact in sim.data.contact[:sim.data.ncon]:
        names=[sim.model.geom_id2name(int(g)) or '' for g in (contact.geom1,contact.geom2)]
        robot=[name.startswith(('robot0_','gripper0_')) for name in names]
        if any(robot) and not all(robot): return names
    return None


def evaluate_tracking(records,completed):
    desired=np.array([r['desired'] for r in records]); actual=np.array([r['actual'] for r in records])
    errors=np.linalg.norm(actual-desired,axis=1)
    return dict(steps=len(records),completed=completed,mean_error_m=float(errors.mean()),
                rmse_m=float(np.sqrt(np.mean(errors**2))),maximum_error_m=float(errors.max()),
                final_position_error_m=float(errors[-1]),clipped_steps=sum(r['clipped'] for r in records)),errors


def plot_results(records,errors,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.array([r['time'] for r in records]); d=np.array([r['desired'] for r in records]); a=np.array([r['actual'] for r in records])
    fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(111,projection='3d')
    ax.plot(*d.T,label='Desired'); ax.plot(*a.T,label='Actual'); ax.scatter(*a[0],color='green',label='Start'); ax.scatter(*a[-1],color='red',label='End')
    ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)'); ax.legend(); fig.tight_layout(); fig.savefig(output/'tracking_3d.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
    for i in range(3):
        axes[i].plot(t,d[:,i],label='Desired'); axes[i].plot(t,a[:,i],label='Actual'); axes[i].set_ylabel('xyz'[i]+' (m)'); axes[i].grid(alpha=.3)
    axes[0].legend(); axes[3].plot(t,errors*1000); axes[3].set_ylabel('Error (mm)'); axes[3].set_xlabel('Simulation time (s)')
    fig.tight_layout(); fig.savefig(output/'tracking_timeseries.png',dpi=160); plt.close(fig)


def replay_trajectory(env,times,mapped,settings,output):
    """Closed-loop position/orientation replay; log post-step actual versus that step's target."""
    obs=env.reset()
    controller=inspect_controller(env)
    frequency=float(env.env.control_freq)
    start=np.asarray(obs['robot0_eef_pos']).copy()
    target_rotation=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
    grid,desired,factor=build_desired_eef_trajectory(times,mapped,start,frequency,settings['time_scale'],settings['max_eef_speed_m_s'])
    lower=np.array(settings['workspace_min']); upper=np.array(settings['workspace_max'])
    if lower.shape!=(3,) or upper.shape!=(3,) or np.any(lower>=upper): raise ValueError('Invalid workspace limits')
    if np.any(desired<lower) or np.any(desired>upper): raise ValueError('Desired path leaves workspace; edit mapping or bounds')
    grip=settings['gripper']
    if grip!=-1 or not controller['action_min'][6]<=grip<=controller['action_max'][6]: raise ValueError('This test requires open gripper command -1')
    records=[dict(step=0,time=0.,desired=start.copy(),actual=start.copy(),action=np.zeros(7),clipped=False)]
    stop=None
    # Optional final hold lets the controller settle; its target stays at the last position.
    targets=list(desired[1:])+[desired[-1]]*int(settings['settle_seconds']*frequency)
    for step,target in enumerate(targets,1):
        actual=np.asarray(obs['robot0_eef_pos'])
        if not np.isfinite(actual).all() or np.any(actual<lower) or np.any(actual>upper): stop='workspace_exit'; break
        contact=robot_contact(env)
        if contact: stop='robot_scene_contact:'+str(contact); break
        action,clipped=desired_pose_to_action(target,actual,target_rotation,Rotation.from_quat(obs['robot0_eef_quat']).as_matrix(),controller,settings['action_clip'],grip)
        obs,reward,done,info=env.step(action)
        records.append(dict(step=step,time=step/frequency,desired=target.copy(),actual=np.asarray(obs['robot0_eef_pos']).copy(),action=action,clipped=clipped))
        if done: stop='environment_done'; break
        if robot_contact(env): stop='robot_scene_contact'; break
        actual=np.asarray(obs['robot0_eef_pos'])
        if not np.isfinite(actual).all() or np.any(actual<lower) or np.any(actual>upper): stop='workspace_exit'; break
    if not all(np.isfinite(r['actual']).all() for r in records): raise RuntimeError('Nonfinite simulator state')
    output.mkdir(parents=True,exist_ok=True)
    report,errors=evaluate_tracking(records,stop is None)
    report.update(stop_reason=stop,eef_start=start.tolist(),fixed_orientation_matrix=target_rotation.tolist(),
                  time_scale_used=factor,control_frequency_hz=frequency,controller={k:v.tolist() for k,v in controller.items()},
                  endpoint_error_to_full_requested_target_m=float(np.linalg.norm(records[-1]['actual']-desired[-1])))
    with (output/'replay.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle); writer.writerow(['step','time',*[f'desired_{a}' for a in 'xyz'],*[f'actual_{a}' for a in 'xyz'],*[f'action_{i}' for i in range(7)],'error_m','clipped'])
        for r,error in zip(records,errors): writer.writerow([r['step'],r['time'],*r['desired'],*r['actual'],*r['action'],error,r['clipped']])
    plot_results(records,errors,output)
    return report


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory',type=Path); parser.add_argument('--bddl',type=Path,help='Existing LIBERO scene/task file')
    parser.add_argument('--config',type=Path,default=root/'config/libero_transport.json')
    parser.add_argument('--output',type=Path,default=root/'results/libero_transport')
    parser.add_argument('--inspect-only',action='store_true',help='Save mapping/preflight without starting simulation')
    args=parser.parse_args(); settings=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if any(not np.isfinite(settings[k]) or settings[k]<=0 for k in ('time_scale','action_clip','max_source_gap_s','max_eef_speed_m_s')) or settings['action_clip']>1 or settings['settle_seconds']<0:
        parser.error('Invalid replay limits')
    times,positions,source=load_human_trajectory(args.trajectory,settings['max_source_gap_s'])
    mapped,matrix=map_to_libero_frame(positions,settings['axis_permutation'],settings['axis_signs'],settings['scale_factors'])
    args.output.mkdir(parents=True,exist_ok=True)
    preflight=dict(source=str(args.trajectory),source_info=source,mapping_matrix=matrix.tolist(),settings=settings,
                   mapped_min_m=mapped.min(axis=0).tolist(),mapped_max_m=mapped.max(axis=0).tolist(),simulation_executed=False)
    (args.output/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    if args.inspect_only: print(json.dumps(preflight,indent=2)); return
    if not args.bddl or not args.bddl.exists(): parser.error('Provide an existing --bddl scene file from your LIBERO installation')
    try:
        from libero.libero.envs import ControlEnv
    except ImportError as exc:
        raise RuntimeError('Run this script in your working LIBERO/robosuite environment; no simulation results have been generated') from exc
    env=ControlEnv(bddl_file_name=str(args.bddl.resolve()),controller='OSC_POSE',robots=['Panda'],
                   use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,ignore_done=False,horizon=100000)
    try: report=replay_trajectory(env,times,mapped,settings,args.output)
    finally: env.close()
    report.update(preflight,simulation_executed=True)
    report['completed_full_source']=report['completed'] and source['source_stop_reason'] is None
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
