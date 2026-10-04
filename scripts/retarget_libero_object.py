"""Transport-only human object retargeting to LIBERO Spatial task 0; no grasping."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from replay_libero_transport import (
    map_to_libero_frame, build_desired_eef_trajectory, inspect_controller,
    desired_pose_to_action, robot_contact, evaluate_tracking, plot_results,
)

ROOT = Path(__file__).resolve().parents[1]
TASK = 'pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate'


def load_transport(csv_path, metadata_path, max_gap=.15, timing_config=None):
    """Use processed object positions ONLY within the metadata transport interval.

    transport.csv also includes pickup/lowering; cleaned_trajectory.csv lacks
    validity/phase information. processed_demo.csv is the authoritative input.
    Missing/invalid samples are rejected, never silently bridged or truncated.
    """
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8-sig'))
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    timestamps = np.array([float(r['time']) for r in rows])
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Source timestamps must be finite and strictly increasing')
    if timing_config is not None:
        from process_task_demo import validate_config, resolve_phase_bounds, manual_phase_overrides, EVENT_BOUNDARIES
        config = validate_config(timing_config)
        bounds = resolve_phase_bounds(timestamps, metadata['automatic_phase_estimates'], config)
        overrides = manual_phase_overrides(config)
        metadata = {**metadata, **bounds, 'manual_event_annotations': {k: config[k] for k in EVENT_BOUNDARIES},
                    'phase_boundary_sources': {key: dict(source='manual_override' if key in overrides else 'automatic',
                        requested_time_s=overrides.get(key), time_s=value) for key, value in bounds.items()},
                    'runtime_timing_config': config}
    begin, end = float(metadata['transport_start']), float(metadata['transport_end'])
    if not np.isfinite([begin, end]).all() or end <= begin:
        raise ValueError('Invalid transport interval')
    rows = [r for r, t in zip(rows, timestamps) if begin-1e-8 <= t <= end+1e-8]
    if len(rows) < 2:
        raise ValueError('Need at least two transport samples')
    times = np.array([float(r['time']) for r in rows])
    if abs(times[0]-begin) > 1e-8 or abs(times[-1]-end) > 1e-8:
        raise ValueError('Transport boundaries must match source samples')
    if np.any(np.diff(times) > max_gap):
        raise ValueError('Unresolved transport timestamp gap')
    if any(r['valid_processed'] != '1' for r in rows):
        raise ValueError('Invalid processed transport samples')
    positions = np.array([[float(r['object_'+a]) for a in 'xyz'] for r in rows])
    goal = np.asarray(metadata['goal_position'], dtype=float)
    if goal.shape != (3,) or not np.isfinite(positions).all() or not np.isfinite(goal).all():
        raise ValueError('Invalid transport positions/goal')
    return times-times[0], positions, goal, [r['phase'] for r in rows], metadata


def smoothstep(u):
    """Quintic blend: zero first and second derivatives at both endpoints."""
    return u**3 * (10 + u * (-15 + 6*u))


def retarget_transport(times, positions, human_goal, sim_start, sim_goal, settings):
    """Align task bases, then distribute endpoint residual smoothly over time.

    A is a signed axis permutation; human vertical is A's third output axis.
    q=A(p-p0), d=A(goal-p0), u=q_xy.e_h/|d_xy|, l=q_xy.n_h.
    B=s0+u*(sg-s0)+horizontal_scale*l*n_s
       +vertical_scale*(q_z-u*d_z)*ez.
    P=B+smoothstep(t/T)*(sg-B[-1]). No last-sample snap.
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    human_goal, sim_start, sim_goal = [np.asarray(p, dtype=float) for p in (human_goal, sim_start, sim_goal)]
    if (times.ndim != 1 or len(times) < 2 or positions.shape != (len(times), 3)
            or any(p.shape != (3,) for p in (human_goal, sim_start, sim_goal))
            or not all(np.isfinite(p).all() for p in (times, positions, human_goal, sim_start, sim_goal))
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid retargeting samples')
    horizontal, vertical = settings['horizontal_scale'], settings['vertical_scale']
    if not np.isfinite([horizontal, vertical]).all() or min(horizontal, vertical) <= 0:
        raise ValueError('Shape scales must be finite and positive')
    q, matrix = map_to_libero_frame(positions-positions[0], settings['axis_permutation'], settings['axis_signs'], [1, 1, 1])
    dh = matrix @ (human_goal-positions[0])
    ds = sim_goal-sim_start
    lh, ls = np.linalg.norm(dh[:2]), np.linalg.norm(ds[:2])
    if min(lh, ls) < 1e-6:
        raise ValueError('Task requires nonzero horizontal start-to-goal distances')
    eh, es = dh[:2]/lh, ds[:2]/ls
    nh, ns = np.array([-eh[1], eh[0]]), np.array([-es[1], es[0]])
    progress = q[:, :2] @ eh / lh
    lateral = q[:, :2] @ nh
    lift = q[:, 2]-progress*dh[2]
    base = sim_start + progress[:, None]*ds
    base[:, :2] += horizontal*lateral[:, None]*ns
    base[:, 2] += vertical*lift
    residual = sim_goal-base[-1]
    blend = smoothstep((times-times[0])/(times[-1]-times[0]))
    path = base+blend[:, None]*residual
    report = dict(axis_matrix=matrix.tolist(), human_start=positions[0].tolist(), human_goal=human_goal.tolist(),
                  sim_start=sim_start.tolist(), sim_goal=sim_goal.tolist(),
                  human_start_to_goal_distance_m=float(np.linalg.norm(dh)),
                  libero_start_to_goal_distance_m=float(np.linalg.norm(ds)),
                  horizontal_scale_factor=float(ls/lh), lateral_scale_factor=float(horizontal),
                  vertical_scale_factor=float(vertical),
                  human_transport_endpoint_progress=float(progress[-1]),
                  human_transport_endpoint_goal_error_m=float(np.linalg.norm(positions[-1]-human_goal)),
                  endpoint_correction_m=residual.tolist(), endpoint_correction_norm_m=float(np.linalg.norm(residual)),
                  start_error_m=float(np.linalg.norm(path[0]-sim_start)),
                  final_error_m=float(np.linalg.norm(path[-1]-sim_goal)),
                  maximum_lift_m=float(np.max(path[:, 2]-sim_start[2])),
                  path_length_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()))
    return path, progress, report


def validate_settings(settings):
    for key in ('time_scale', 'action_clip', 'max_source_gap_s', 'max_eef_speed_m_s',
                'max_command_step_m', 'approach_seconds', 'start_tolerance_m', 'scene_settle_seconds'):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError('Invalid positive setting: '+key)
    if settings['action_clip'] > 1 or settings['gripper'] != -1:
        raise ValueError('Requires bounded actions and open gripper (-1)')
    if not np.isfinite(settings['settle_seconds']) or settings['settle_seconds'] < 0:
        raise ValueError('Invalid settle duration')
    for key in ('workspace_min', 'workspace_max', 'grasp_offset', 'target_offset'):
        value = np.asarray(settings[key])
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('Invalid vector: '+key)
    if np.any(np.asarray(settings['workspace_min']) >= settings['workspace_max']):
        raise ValueError('Invalid workspace bounds')


def make_plots(path, offset, records, errors, output, start, goal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_results(records, errors, output)
    actual = np.array([r['actual'] for r in records if r['stage'] in ('transport', 'hold')])
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection='3d')
    ax.plot(*path.T, label='Retargeted object (virtual)')
    ax.plot(*(path+offset).T, label='Desired Panda EEF')
    if len(actual): ax.plot(*actual.T, label='Actual Panda EEF', linestyle='--')
    ax.scatter(*start, label='Bowl start', s=60); ax.scatter(*goal, label='Plate target', s=70, marker='*')
    ax.set(xlabel='World x (m)', ylabel='World y (m)', zlabel='World z (m)')
    ax.set_box_aspect((1.2, .8, 1.1))
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.legend(); fig.tight_layout(); fig.savefig(output/'retargeting_3d.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(*path[:, :2].T, label='Retargeted object')
    ax.scatter(*start[:2], label='Bowl start'); ax.scatter(*goal[:2], marker='*', s=100, label='Plate target')
    ax.set(xlabel='World x (m)', ylabel='World y (m)', title='Task geometry'); ax.axis('equal'); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(output/'top_down.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot([r['time'] for r in records], errors*1000)
    ax.set(xlabel='Simulation time (s)', ylabel='EEF error (mm)'); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(output/'tracking_error.png', dpi=160); plt.close(fig)


def replay(env, times, human, human_goal, phases, settings, output):
    obs = env.reset()
    controller = inspect_controller(env)
    if not controller['action_min'][6] <= -1 <= controller['action_max'][6]:
        raise ValueError('Open gripper action unavailable')
    frequency = float(env.env.control_freq)
    initial = np.asarray(obs['robot0_eef_pos']).copy()
    rotation = Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
    lower, upper = np.array(settings['workspace_min']), np.array(settings['workspace_max'])
    records = []
    # Hold the robot while reset objects settle under gravity; log every step.
    for step in range(1, int(np.ceil(settings['scene_settle_seconds']*frequency))+1):
        actual = np.asarray(obs['robot0_eef_pos'])
        if not np.isfinite(actual).all() or np.any(actual < lower) or np.any(actual > upper) or robot_contact(env):
            raise RuntimeError('Unsafe scene initialization')
        delta = initial-actual
        limited = actual+delta*min(1., settings['max_command_step_m']/max(np.linalg.norm(delta), 1e-12))
        action, clipped = desired_pose_to_action(limited, actual, rotation,
            Rotation.from_quat(obs['robot0_eef_quat']).as_matrix(), controller, settings['action_clip'], -1)
        obs, _, done, _ = env.step(action)
        records.append(dict(step=step, time=step/frequency, stage='scene_settle', desired=initial.copy(),
            actual=np.asarray(obs['robot0_eef_pos']).copy(), action=action, clipped=clipped,
            command_limited=bool(np.linalg.norm(delta) > settings['max_command_step_m']),
            source_time=0., source_progress=0., task_progress=0., phase=phases[0],
            orientation_error_rad=float(Rotation.from_matrix(rotation@Rotation.from_quat(obs['robot0_eef_quat']).as_matrix().T).magnitude())))
        if done or robot_contact(env): raise RuntimeError('Scene initialization stopped')
    start = np.asarray(obs['akita_black_bowl_1_pos']).copy()
    goal = np.asarray(obs['plate_1_pos']).copy()+settings['target_offset']
    path, progress, mapping = retarget_transport(times, human, human_goal, start, goal, settings)
    for record in records:
        record['object'] = start.copy()
    offset = np.array(settings['grasp_offset'])
    grid, object_targets, stretch = build_desired_eef_trajectory(
        times, path-path[0], path[0], frequency, settings['time_scale'], settings['max_eef_speed_m_s'])
    eef_targets = object_targets+offset
    source_times = np.minimum(grid/stretch, times[-1])
    approach_start = np.asarray(obs['robot0_eef_pos']).copy()
    # Quintic approach, duration accounts for peak blend derivative 1.875.
    duration = max(settings['approach_seconds'], 1.875*np.linalg.norm(eef_targets[0]-approach_start)/settings['max_eef_speed_m_s'])
    approach = approach_start+smoothstep(np.linspace(0, 1, int(np.ceil(duration*frequency))+1))[:, None]*(eef_targets[0]-approach_start)
    lower, upper = np.array(settings['workspace_min']), np.array(settings['workspace_max'])
    planned = np.vstack([approach, eef_targets])
    if np.any(planned < lower) or np.any(planned > upper):
        raise ValueError('Planned EEF path leaves workspace; revise mapping/offset')
    targets = [(p, 'approach', 0.) for p in approach[1:]]
    targets += [(eef_targets[0], 'approach', 0.)]*int(settings['settle_seconds']*frequency)
    targets += [(p, 'transport', t) for p, t in zip(eef_targets, source_times)]
    targets += [(eef_targets[-1], 'hold', times[-1])]*int(settings['settle_seconds']*frequency)
    np.savetxt(output/'retargeted_trajectory.csv', np.column_stack([grid, object_targets, eef_targets, source_times, np.interp(source_times, times, progress)]),
               delimiter=',', header='time,object_x,object_y,object_z,eef_x,eef_y,eef_z,source_time,task_progress', comments='')
    initialization_steps = len(records)
    stop = None
    previous_stage = 'approach'
    for step, (target, stage, source_t) in enumerate(targets, initialization_steps+1):
        actual = np.asarray(obs['robot0_eef_pos'])
        if not np.isfinite(actual).all() or np.any(actual < lower) or np.any(actual > upper):
            stop = 'workspace_exit'; break
        if robot_contact(env): stop = 'robot_scene_contact'; break
        if stage == 'transport' and previous_stage == 'approach' and np.linalg.norm(actual-eef_targets[0]) > settings['start_tolerance_m']:
            stop = 'approach_failed_to_reach_start'; break
        error = target-actual
        command_target = actual+error*min(1., settings['max_command_step_m']/max(np.linalg.norm(error), 1e-12))
        action, clipped = desired_pose_to_action(command_target, actual, rotation,
            Rotation.from_quat(obs['robot0_eef_quat']).as_matrix(), controller, settings['action_clip'], -1)
        obs, reward, done, info = env.step(action)
        object_target = np.array([np.interp(source_t, times, path[:, i]) for i in range(3)])
        records.append(dict(step=step, time=step/frequency, stage=stage, desired=target.copy(),
            actual=np.asarray(obs['robot0_eef_pos']).copy(), object=object_target,
            action=action, clipped=clipped, command_limited=bool(np.linalg.norm(error) > settings['max_command_step_m']),
            source_time=source_t, source_progress=float(source_t/times[-1]),
            task_progress=float(np.interp(source_t, times, progress)),
            phase=phases[min(np.searchsorted(times, source_t, side='right')-1, len(phases)-1)],
            orientation_error_rad=float(Rotation.from_matrix(rotation@Rotation.from_quat(obs['robot0_eef_quat']).as_matrix().T).magnitude())))
        previous_stage = stage
        if step % 100 == 0:
            print(f'{stage} step {step}/{len(targets)+initialization_steps}: {np.linalg.norm(target-records[-1]["actual"])*1000:.2f} mm', flush=True)
        if robot_contact(env): stop = 'robot_scene_contact'; break
        if done: stop = 'environment_done'; break
        actual_after = records[-1]['actual']
        if not np.isfinite(actual_after).all() or np.any(actual_after < lower) or np.any(actual_after > upper):
            stop = 'workspace_exit'; break
    if not records: raise RuntimeError('No steps executed: '+str(stop))
    metrics, errors = evaluate_tracking(records, stop is None)
    by_stage = {}
    for stage in ('scene_settle', 'approach', 'transport', 'hold'):
        subset = [r for r in records if r['stage'] == stage]
        if subset: by_stage[stage] = evaluate_tracking(subset, stop is None)[0]
    metrics.update(stop_reason=stop, stages=by_stage, retargeting=mapping,
        requested_steps=len(targets)+initialization_steps, time_scale_used=stretch, control_frequency_hz=frequency,
        fixed_orientation_matrix=rotation.tolist(), eef_initial=initial.tolist(),
        controller={k: v.tolist() for k, v in controller.items()},
        observation_keys=list(obs.keys()), command_limited_steps=sum(r['command_limited'] for r in records),
        maximum_orientation_error_rad=max(r['orientation_error_rad'] for r in records),
        final_eef_error_to_requested_goal_m=float(np.linalg.norm(records[-1]['actual']-eef_targets[-1])),
        simulated_object_displacement_m=float(np.linalg.norm(np.asarray(obs['akita_black_bowl_1_pos'])-start)))
    with (output/'replay.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['step', 'time', 'stage', 'source_time', 'source_progress', 'task_progress', 'human_phase',
            *['desired_object_'+a for a in 'xyz'], *['retargeted_object_'+a for a in 'xyz'],
            *['desired_eef_'+a for a in 'xyz'], *['actual_eef_'+a for a in 'xyz'],
            *['error_'+a for a in 'xyz'], 'error_m', *['action_'+str(i) for i in range(7)],
            'clipped', 'command_limited', 'orientation_error_rad'])
        for r, e in zip(records, errors):
            writer.writerow([r['step'], r['time'], r['stage'], r['source_time'], r['source_progress'], r['task_progress'], r['phase'],
                *r['object'], *r['object'], *r['desired'], *r['actual'], *(r['desired']-r['actual']), e,
                *r['action'], r['clipped'], r['command_limited'], r['orientation_error_rad']])
    make_plots(object_targets, offset, records, errors, output, start, goal)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', type=Path, default=ROOT/'results/test_007_block_only_raw/task_demo')
    parser.add_argument('--config', type=Path, default=ROOT/'config/libero_retarget.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/libero_retarget')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    validate_settings(settings)
    times, human, goal, phases, metadata = load_transport(args.demo/'processed_demo.csv', args.demo/'metadata.json', settings['max_source_gap_s'])
    print('Source quality flags:', metadata.get('quality', {}).get('review_flags', []), flush=True)
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = benchmark.get_benchmark_dict()['libero_spatial']().get_task(0)
    if task.name != TASK: raise ValueError('Unexpected task ordering: '+task.name)
    bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    np.random.seed(args.seed)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), use_camera_obs=False,
                             camera_heights=128, camera_widths=128, horizon=100000)
    env.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        report = replay(env, times, human, goal, phases, settings, args.output)
    finally:
        env.close()
    report.update(task=task.name, bddl=str(bddl), seed=args.seed, settings=settings,
                  source=str(args.demo/'processed_demo.csv'), source_metadata=metadata,
                  transport_samples=len(times), transport_interval=[metadata['transport_start'], metadata['transport_end']],
                  target_definition='plate_1_pos + target_offset; geometric anchor, not a contact placement pose')
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_metadata', 'observation_keys', 'controller')}, indent=2))
    if not report['completed']: raise SystemExit('Replay stopped: '+str(report['stop_reason']))


if __name__ == '__main__':
    main()
