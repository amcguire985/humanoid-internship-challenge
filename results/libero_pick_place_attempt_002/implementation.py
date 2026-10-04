"""One deterministic Panda rim-grasp / human-transport / release attempt in LIBERO.

No parameter search, policy learning, object teleportation, or grasp attachment.
"""
import argparse
import csv
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from replay_libero_transport import inspect_controller, desired_pose_to_action, build_desired_eef_trajectory
from retarget_libero_object import ROOT, TASK, load_transport, retarget_transport, smoothstep

OBJECT = 'akita_black_bowl_1'
TARGET = 'plate_1'
PHASES = ('SCENE_SETTLE', 'PREGRASP', 'DESCEND', 'GRASP', 'GRASP_SETTLE', 'LIFT',
          'TRANSPORT', 'PLACE', 'RELEASE', 'RELEASE_SETTLE', 'RETRACT')
CARRY_PHASES = ('LIFT', 'TRANSPORT', 'PLACE')


class ManipulationAbort(RuntimeError):
    pass


def validate_config(c):
    positive = ('pregrasp_height', 'max_grasp_offset_adjustment_m', 'postgrasp_lift_height',
                'minimum_object_lift_m', 'transport_clearance_height', 'approach_step_size',
                'max_eef_speed_m_s', 'max_command_step_m', 'action_clip', 'grasp_offset_drift_limit_m',
                'retract_height', 'position_tolerance_m', 'tracking_error_abort_m',
                'target_region_radius_m', 'placement_height_tolerance_m', 'maximum_endpoint_correction_m')
    for key in positive:
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Expected positive finite '+key)
    for key in ('scene_settle_steps', 'grasp_close_steps', 'grasp_settle_steps', 'grasp_confirm_steps',
                'grasp_loss_grace_steps', 'release_open_steps', 'release_settle_steps',
                'retract_settle_steps', 'move_settle_steps', 'tracking_error_abort_steps'):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError('Expected positive integer '+key)
    if c['grasp_confirm_steps'] > c['grasp_settle_steps']:
        raise ValueError('Grasp confirmation must fit in GRASP_SETTLE')
    if c['action_clip'] > 1 or c['grasp_offset_mode'] not in ('configured', 'measured_after_grasp'):
        raise ValueError('Invalid action clip or grasp offset mode')
    for key in ('grasp_offset', 'workspace_min', 'workspace_max'):
        if np.asarray(c[key]).shape != (3,) or not np.isfinite(c[key]).all():
            raise ValueError('Expected finite vector '+key)
    if np.any(np.array(c['workspace_min']) >= c['workspace_max']):
        raise ValueError('Invalid workspace')
    if c['pregrasp_height'] <= c['grasp_offset'][2]:
        raise ValueError('Pregrasp must be above the grasp pose')
    if not np.isfinite(c['placement_object_height_offset']):
        raise ValueError('Invalid placement height')
    quat = np.array(c['fixed_eef_quaternion_xyzw'])
    if quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1):
        raise ValueError('Expected unit xyzw quaternion')
    for phase in PHASES:
        if type(c['phase_max_steps'][phase]) is not int or c['phase_max_steps'][phase] <= 0:
            raise ValueError('Invalid step budget for '+phase)


def verify_gripper(robot, controller):
    """Check installed Panda sign integration without changing its live state."""
    grip = robot.gripper
    if type(grip).__name__ != 'PandaGripper' or grip.dof != 1:
        raise ValueError('This script requires the inspected PandaGripper')
    if not controller['action_min'][6] <= -1 < 1 <= controller['action_max'][6]:
        raise ValueError('Gripper action range must include [-1, 1]')
    saved = grip.current_action.copy()
    try:
        grip.current_action = np.zeros(2)
        opening = grip.format_action(np.array([-1.])).copy()
        grip.current_action = np.zeros(2)
        closing = grip.format_action(np.array([1.])).copy()
    finally:
        grip.current_action = saved
    if not (opening[0] > 0 > opening[1] and closing[0] < 0 < closing[1]):
        raise ValueError('Unexpected Panda gripper sign convention')
    return dict(open_command=-1, close_command=1, opening_internal_action=opening.tolist(),
                closing_internal_action=closing.tolist(), semantics='Sign-integrated aperture command, not target aperture')


def collision_bounds(env, obj):
    """World bounds of compiled box collision geoms, not inaccurate asset sites."""
    sim = env.env.sim
    points = []
    for name in obj.contact_geoms:
        i = sim.model.geom_name2id(name)
        if sim.model.geom_type[i] != 6:
            raise ValueError('Expected box collision geometry for this task')
        corners = np.array(list(product([-1, 1], repeat=3)))*sim.model.geom_size[i]
        points.extend(corners@sim.data.geom_xmat[i].reshape(3, 3).T+sim.data.geom_xpos[i])
    return np.min(points, axis=0), np.max(points, axis=0)


def make_move(start, goal, frequency, c):
    """Quintic positional reference with both velocity and per-step bounds."""
    step_limit = min(c['approach_step_size'], c['max_eef_speed_m_s']/frequency)
    count = max(1, int(np.ceil(1.875*np.linalg.norm(goal-start)/step_limit)))
    return start+smoothstep(np.linspace(0, 1, count+1))[1:, None]*(goal-start)


class PickPlace:
    def __init__(self, env, human, settings, mapping, output):
        self.env, self.human, self.c, self.mapping, self.output = env, human, settings, mapping, output
        self.obs = env.reset()
        self.controller = inspect_controller(env)
        self.gripper_contract = verify_gripper(env.env.robots[0], self.controller)
        self.frequency = float(env.env.control_freq)
        self.rotation = Rotation.from_quat(settings['fixed_eef_quaternion_xyzw']).as_matrix()
        self.object_model = env.env.objects_dict[OBJECT]
        self.target_model = env.env.objects_dict[TARGET]
        self.object_geoms = set(self.object_model.contact_geoms)
        grip = env.env.robots[0].gripper
        self.finger_geoms = set(grip.important_geoms['left_finger']+grip.important_geoms['right_finger'])
        self.records, self.events, self.render_errors = [], [], []
        self.phase, self.phase_steps, self.bad_tracking, self.missing_grasp = 'SCENE_SETTLE', 0, 0, 0
        self.acquired, self.lifted, self.grasp_lost, self.transport_complete = False, False, False, False
        self.released, self.completed = False, False
        self.failure_phase = self.failure_reason = None
        self.start = self.placement = self.target = self.offset = None
        self.geometry, self.retargeting = {}, None
        self.nominal_offset = np.array(settings['grasp_offset'])
        self.fixed_target = self.position('robot0_eef_pos')
        self.initial = dict(eef=self.fixed_target.tolist(), object=self.position(OBJECT+'_pos').tolist(),
                            target=self.position(TARGET+'_pos').tolist(), gripper=np.asarray(self.obs['robot0_gripper_qpos']).tolist())

    def position(self, key):
        value = np.asarray(self.obs.get(key, [np.nan]*3), dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ManipulationAbort('Invalid or unavailable observation: '+key)
        return value.copy()

    def contacts(self):
        sim = self.env.env.sim
        pairs, forbidden = [], []
        for contact in sim.data.contact[:sim.data.ncon]:
            names = [sim.model.geom_id2name(int(g)) or '' for g in (contact.geom1, contact.geom2)]
            robot = [n.startswith(('robot0_', 'gripper0_')) for n in names]
            if any(n in self.object_geoms for n in names) or any(robot):
                pairs.append(names)
            if any(robot) and not all(robot):
                allowed = (self.phase not in ('SCENE_SETTLE', 'PREGRASP') and
                           any(n in self.finger_geoms for n in names) and any(n in self.object_geoms for n in names))
                if not allowed: forbidden.append(names)
        return pairs, forbidden

    def snapshot(self, label):
        try:
            from PIL import Image
            rgb = self.env.env.sim.render(width=640, height=480, camera_name='agentview')
            Image.fromarray(rgb[::-1]).save(self.output/(label+'.png'))
        except Exception as exc:
            self.render_errors.append(str(exc))

    def step(self, desired, gripper, desired_object=None, source_time=None):
        """The single simulation step path: bounded feedback, log, then safety checks."""
        if self.phase_steps >= self.c['phase_max_steps'][self.phase]:
            raise ManipulationAbort('Maximum phase steps exceeded')
        actual = self.position('robot0_eef_pos')
        self.position(OBJECT+'_pos'); self.position(TARGET+'_pos')
        low, high = np.array(self.c['workspace_min']), np.array(self.c['workspace_max'])
        if not np.isfinite(desired).all() or np.any(desired < low) or np.any(desired > high):
            raise ManipulationAbort('Desired EEF outside workspace')
        if np.any(actual < low) or np.any(actual > high):
            raise ManipulationAbort('Actual EEF outside workspace')
        delta = desired-actual
        limited = actual+delta*min(1., self.c['max_command_step_m']/max(np.linalg.norm(delta), 1e-12))
        action, clipped = desired_pose_to_action(limited, actual, self.rotation,
            Rotation.from_quat(self.obs['robot0_eef_quat']).as_matrix(), self.controller, self.c['action_clip'], gripper)
        self.obs, reward, done, info = self.env.step(action)
        self.phase_steps += 1
        # Preserve the final observation even if it is invalid; abort after logging.
        actual = np.asarray(self.obs.get('robot0_eef_pos', [np.nan]*3))
        obj = np.asarray(self.obs.get(OBJECT+'_pos', [np.nan]*3))
        target = np.asarray(self.obs.get(TARGET+'_pos', [np.nan]*3))
        contact_pairs, forbidden = self.contacts()
        grasp = bool(self.env.env._check_grasp(self.env.env.robots[0].gripper, self.object_model))
        success = bool(self.env.check_success())
        distance = float(np.linalg.norm(obj-target))
        error = float(np.linalg.norm(desired-actual))
        drift = float(np.linalg.norm(actual-obj-self.offset)) if self.offset is not None else None
        record = dict(step=len(self.records)+1, time=(len(self.records)+1)/self.frequency, phase=self.phase,
            phase_step=self.phase_steps, desired=desired.copy(), actual=actual.copy(), object=obj.copy(),
            desired_object=np.array(desired_object) if desired_object is not None else np.full(3, np.nan),
            target=target.copy(), error_m=error, action=action.copy(), gripper_command=float(gripper),
            gripper_state=np.asarray(self.obs.get('robot0_gripper_qpos', [np.nan, np.nan])).copy(),
            object_quaternion=np.asarray(self.obs.get(OBJECT+'_quat', [np.nan]*4)).copy(),
            object_to_target_distance_m=distance, bilateral_grasp=grasp, grasp_offset_drift_m=drift,
            task_success=success, environment_done_signal=bool(done), source_time=source_time, contacts=contact_pairs, clipped=clipped,
            command_limited=bool(np.linalg.norm(delta) > self.c['max_command_step_m']))
        self.records.append(record)
        if len(self.records) % 100 == 0:
            print(f'{self.phase}: episode step {len(self.records)}, EEF error {error*1000:.2f} mm, grasp={grasp}, object={obj}', flush=True)
        if not np.isfinite([*actual, *obj, *target]).all():
            raise ManipulationAbort('Invalid object, target or EEF observation after step')
        if np.any(actual < low) or np.any(actual > high):
            raise ManipulationAbort('Actual EEF outside workspace')
        if forbidden:
            raise ManipulationAbort('Unexpected robot/scene contact: '+str(forbidden))
        self.bad_tracking = self.bad_tracking+1 if error > self.c['tracking_error_abort_m'] else 0
        if self.bad_tracking >= self.c['tracking_error_abort_steps']:
            raise ManipulationAbort('Excessive sustained EEF tracking error')
        if self.phase in CARRY_PHASES and self.acquired:
            self.missing_grasp = self.missing_grasp+1 if not grasp else 0
            if self.missing_grasp >= self.c['grasp_loss_grace_steps'] or drift > self.c['grasp_offset_drift_limit_m']:
                self.grasp_lost = True
                raise ManipulationAbort('Grasp lost: bilateral contact absent or object-to-EEF offset drift excessive')
        # LIBERO's BDDLBaseDomain.step overwrites returned `done` with task
        # success. A bowl touching the plate can satisfy it while still held.
        # Continue RELEASE/RETRACT unless the underlying simulator truly ended.
        if getattr(self.env.env, 'done', False) or (done and not success):
            raise ManipulationAbort('Environment terminated before state machine completed')

    def hold(self, steps, gripper, desired_object=None):
        for _ in range(steps):
            self.step(self.fixed_target, gripper, desired_object)

    def move(self, goal, gripper, carrying=False):
        path = make_move(self.position('robot0_eef_pos'), np.array(goal), self.frequency, self.c)
        if np.any(path < self.c['workspace_min']) or np.any(path > self.c['workspace_max']):
            raise ManipulationAbort('Planned motion leaves workspace')
        for desired in path:
            self.step(desired, gripper, desired-self.offset if carrying else None)
        self.fixed_target = np.array(goal)
        for _ in range(self.c['move_settle_steps']):
            if np.linalg.norm(self.position('robot0_eef_pos')-goal) <= self.c['position_tolerance_m']:
                return
            self.step(self.fixed_target, gripper, self.fixed_target-self.offset if carrying else None)
        raise ManipulationAbort('Motion failed to reach phase target tolerance')

    def scene_settle(self):
        self.hold(self.c['scene_settle_steps'], -1)
        self.start = self.position(OBJECT+'_pos')
        self.target = self.position(TARGET+'_pos')
        self.placement = self.target+np.array([0, 0, self.c['placement_object_height_offset']])
        omin, omax = collision_bounds(self.env, self.object_model)
        tmin, tmax = collision_bounds(self.env, self.target_model)
        self.geometry = dict(object_start=self.start.tolist(), target_anchor=self.target.tolist(),
            object_collision_min=omin.tolist(), object_collision_max=omax.tolist(),
            target_collision_min=tmin.tolist(), target_collision_max=tmax.tolist(),
            object_origin_to_bottom_m=float(omin[2]-self.start[2]),
            placement_object_origin=self.placement.tolist(),
            placement_bottom_above_target_collision_top_m=float(self.placement[2]+omin[2]-self.start[2]-tmax[2]))
        self.grasp_pose = self.start+self.nominal_offset

    def pregrasp(self):
        pose = self.grasp_pose.copy(); pose[2] = self.start[2]+self.c['pregrasp_height']
        self.move(pose, -1)

    def descend(self):
        self.move(self.grasp_pose, -1)

    def grasp(self):
        self.hold(self.c['grasp_close_steps'], 1)

    def grasp_settle(self):
        self.hold(self.c['grasp_settle_steps'], 1)
        recent = self.records[-self.c['grasp_confirm_steps']:]
        self.acquired = all(r['bilateral_grasp'] for r in recent)
        if not self.acquired:
            raise ManipulationAbort('Grasp not acquired: no sustained contact on both finger pads')
        measured = self.position('robot0_eef_pos')-self.position(OBJECT+'_pos')
        if np.linalg.norm(measured-self.nominal_offset) > self.c['max_grasp_offset_adjustment_m']:
            raise ManipulationAbort('Measured grasp offset differs excessively from configured pose')
        self.offset = measured if self.c['grasp_offset_mode'] == 'measured_after_grasp' else self.nominal_offset.copy()
        self.geometry.update(measured_grasp_offset=measured.tolist(), fixed_grasp_offset=self.offset.tolist())

    def lift(self):
        self.move(self.position('robot0_eef_pos')+np.array([0, 0, self.c['postgrasp_lift_height']]), 1, carrying=True)
        self.lifted = bool(self.position(OBJECT+'_pos')[2]-self.start[2] >= self.c['minimum_object_lift_m'])
        if not self.lifted:
            raise ManipulationAbort('Object did not lift sufficiently off its original support')

    def transport(self):
        times, human, human_goal, _, metadata = self.human
        lifted_start = self.position(OBJECT+'_pos')
        goal = self.placement+np.array([0, 0, self.c['transport_clearance_height']])
        path, progress, self.retargeting = retarget_transport(times, human, human_goal, lifted_start, goal, self.mapping)
        if self.retargeting['endpoint_correction_norm_m'] > self.c['maximum_endpoint_correction_m']:
            raise ManipulationAbort('Human path needs excessive endpoint correction; review mapping')
        grid, objects, factor = build_desired_eef_trajectory(times, path-path[0], path[0], self.frequency,
            self.mapping['time_scale'], self.c['max_eef_speed_m_s'])
        eef = objects+self.offset
        if np.any(eef < self.c['workspace_min']) or np.any(eef > self.c['workspace_max']):
            raise ManipulationAbort('Transport path leaves EEF workspace')
        self.retargeting.update(time_scale_used=factor, transport_hover_goal=goal.tolist())
        np.savetxt(self.output/'transport_reference.csv', np.column_stack([grid, objects, eef]), delimiter=',',
                   header='time,object_x,object_y,object_z,eef_x,eef_y,eef_z', comments='')
        for time, obj, desired in zip(grid, objects, eef):
            self.step(desired, 1, obj, float(metadata['transport_start']+min(time/factor, times[-1])))
        self.fixed_target = eef[-1].copy()
        self.transport_complete = True

    def place(self):
        self.move(self.placement+self.offset, 1, carrying=True)

    def release(self):
        self.hold(self.c['release_open_steps'], -1, self.placement)
        self.released = True

    def release_settle(self):
        self.hold(self.c['release_settle_steps'], -1, self.placement)

    def retract(self):
        self.move(self.fixed_target+np.array([0, 0, self.c['retract_height']]), -1)
        self.hold(self.c['retract_settle_steps'], -1, self.placement)

    def run(self):
        try:
            for phase in PHASES:
                self.phase, self.phase_steps = phase, 0
                self.bad_tracking = 0
                self.events.append(dict(phase=phase, entry_step=len(self.records)+1))
                print('Entering '+phase, flush=True)
                getattr(self, phase.lower())()
                self.events[-1]['exit_step'] = len(self.records)
                self.snapshot(f'{len(self.events):02d}_{phase}')
            self.completed = True
            self.phase = 'DONE'
        except ManipulationAbort as exc:
            self.failure_phase, self.failure_reason = self.phase, str(exc)
            self.snapshot('failure_'+self.phase)
            print(f'ABORT in {self.phase}: {exc}', flush=True)
        return self.report()

    def report(self):
        errors = np.array([r['error_m'] for r in self.records])
        final_object = self.position(OBJECT+'_pos') if np.isfinite(self.obs.get(OBJECT+'_pos', [np.nan]*3)).all() else None
        placement = self.placement
        def in_region(r):
            return placement is not None and np.linalg.norm((r['object']-placement)[:2]) <= self.c['target_region_radius_m'] and abs(r['object'][2]-placement[2]) <= self.c['placement_height_tolerance_m']
        post_release = [r for r in self.records if r['phase'] in ('RELEASE_SETTLE', 'RETRACT')]
        last_hold = [r for r in self.records if r['phase'] == 'RETRACT'][-self.c['retract_settle_steps']:]
        stayed = bool(len(last_hold) == self.c['retract_settle_steps'] and all(in_region(r) and r['task_success'] and not r['bilateral_grasp'] for r in last_hold)) if self.released else None
        final_success = bool(self.records[-1]['task_success']) if self.records else False
        task_completed = bool(self.completed and stayed and final_success)
        failure_phase = self.failure_phase or ('RETRACT' if self.completed and not task_completed else None)
        return dict(completed_sequence=self.completed, task_completed=task_completed,
            failure_phase=failure_phase, failure_reason=self.failure_reason or ('Final post-release success/retention criterion failed' if failure_phase else None),
            gripper_acquired_object=self.acquired, object_lifted=self.lifted,
            remained_grasped_during_transport=(self.transport_complete and not self.grasp_lost) if any(r['phase']=='TRANSPORT' for r in self.records) else None,
            object_reached_target_region=any(in_region(r) for r in self.records),
            object_remained_at_target_after_release=stayed, libero_task_success=final_success,
            libero_success_ever=any(r['task_success'] for r in self.records), grasp_lost=self.grasp_lost,
            eef_tracking_rmse_m=float(np.sqrt(np.nanmean(errors**2))) if len(errors) else None,
            maximum_tracking_error_m=float(np.nanmax(errors)) if len(errors) else None,
            final_object_to_target_anchor_error_m=float(np.linalg.norm(final_object-self.target)) if final_object is not None and self.target is not None else None,
            final_object_to_placement_error_m=float(np.linalg.norm(final_object-placement)) if final_object is not None and placement is not None else None,
            maximum_object_lift_m=float(max(0, max(r['object'][2]-self.start[2] for r in self.records if r['phase']!='SCENE_SETTLE'))) if self.start is not None and any(r['phase']!='SCENE_SETTLE' for r in self.records) else 0.,
            total_episode_steps=len(self.records), phases=self.events, initial_observation=self.initial,
            final_object_position=final_object.tolist() if final_object is not None else None,
            geometry=self.geometry, retargeting=self.retargeting, gripper_contract=self.gripper_contract,
            control_frequency_hz=self.frequency, render_errors=self.render_errors,
            controller={k:v.tolist() for k,v in self.controller.items()},
            final_contacts=self.records[-1]['contacts'] if self.records else [],
            final_gripper_qpos=self.records[-1]['gripper_state'].tolist() if self.records else [],
            final_bilateral_grasp=self.records[-1]['bilateral_grasp'] if self.records else False)


def save_outputs(runner, report, output):
    output.mkdir(parents=True, exist_ok=True)
    fields = ['step', 'time', 'phase', 'phase_step', 'source_time', *['desired_eef_'+a for a in 'xyz'],
              *['actual_eef_'+a for a in 'xyz'], 'eef_error_m', *['actual_object_'+a for a in 'xyz'],
              *['desired_object_'+a for a in 'xyz'], *['target_'+a for a in 'xyz'],
              'gripper_command', 'gripper_qpos_0', 'gripper_qpos_1', 'object_to_target_distance_m',
              'bilateral_grasp', 'grasp_offset_drift_m', 'libero_success', 'environment_done_signal', *['action_'+str(i) for i in range(7)],
              'object_quat_x', 'object_quat_y', 'object_quat_z', 'object_quat_w', 'clipped', 'command_limited', 'contacts']
    with (output/'replay.csv').open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(fields)
        for r in runner.records:
            values = [r['step'], r['time'], r['phase'], r['phase_step'], r['source_time'], *r['desired'], *r['actual'], r['error_m'],
                *r['object'], *r['desired_object'], *r['target'], r['gripper_command'], *r['gripper_state'],
                r['object_to_target_distance_m'], r['bilateral_grasp'], r['grasp_offset_drift_m'], r['task_success'], r['environment_done_signal'],
                *r['action'], *r['object_quaternion'], r['clipped'], r['command_limited'], json.dumps(r['contacts'])]
            writer.writerow(['' if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for v in values])
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if runner.records:
        plot_diagnostics(runner.records, runner.events, output)


def plot_diagnostics(records, events, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.array([r['time'] for r in records])
    obj, actual, desired = [np.array([r[k] for r in records]) for k in ('object', 'actual', 'desired')]
    fig = plt.figure(figsize=(10, 7)); ax = fig.add_subplot(111, projection='3d')
    for points, label in ((obj, 'Actual bowl'), (desired, 'Desired EEF'), (actual, 'Actual EEF')):
        ax.plot(*points.T, label=label)
    ax.scatter(*records[-1]['target'], marker='*', s=100, label='Plate anchor')
    ax.set(xlabel='World x (m)', ylabel='World y (m)', zlabel='World z (m)'); ax.legend()
    fig.tight_layout(); fig.savefig(output/'manipulation_3d.png', dpi=150); plt.close(fig)
    def phase_marks(ax):
        for i, event in enumerate(events):
            index = min(event['entry_step']-1, len(t)-1)
            end = min(events[i+1]['entry_step']-1, len(t)-1) if i+1<len(events) else len(t)-1
            ax.axvspan(t[index], t[end], color=plt.get_cmap('tab20')(i), alpha=.13)
            ax.axvline(t[index], color='gray', linewidth=.5)
            ax.text((t[index]+t[end])/2, 1.02+(i%2)*.10, event['phase'], transform=ax.get_xaxis_transform(), fontsize=7, ha='center')
        ax.set_xlabel('Simulation time (s)'); ax.grid(alpha=.25)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, obj[:, j], label='Actual object')
        ax.plot(t, [r['desired_object'][j] for r in records], '--', label='Desired object')
        ax.plot(t, [r['target'][j] for r in records], ':', label='Target anchor')
        ax.set_ylabel('xyz'[j]+' (m)'); phase_marks(ax)
    axes[0].legend(); fig.tight_layout(); fig.savefig(output/'object_timeseries.png', dpi=150); plt.close(fig)
    for name, values, ylabel in [('eef_tracking_error', [r['error_m']*1000 for r in records], 'EEF error (mm)'),
                                ('object_target_distance', [r['object_to_target_distance_m'] for r in records], 'Object-to-anchor distance (m)')]:
        fig, ax = plt.subplots(figsize=(13, 4)); ax.plot(t, values); ax.set_ylabel(ylabel); phase_marks(ax)
        fig.tight_layout(); fig.savefig(output/(name+'.png'), dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].step(t, [r['gripper_command'] for r in records], where='post'); axes[0].set_ylabel('Command (-1 open, +1 close)')
    qpos = np.array([r['gripper_state'] for r in records])
    axes[1].plot(t, qpos[:, 0], label='Finger 1'); axes[1].plot(t, qpos[:, 1], label='Finger 2'); axes[1].legend(); axes[1].set_ylabel('Finger joint position (m)')
    for ax in axes: phase_marks(ax)
    fig.tight_layout(); fig.savefig(output/'gripper_timeseries.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/libero_pick_place.json')
    parser.add_argument('--mapping-config', type=Path, default=ROOT/'config/libero_retarget.json')
    parser.add_argument('--demo-config', type=Path, default=ROOT/'config/test_007_demo.json')
    parser.add_argument('--demo', type=Path, default=ROOT/'results/test_007_block_only_raw/task_demo')
    parser.add_argument('--output', type=Path, default=ROOT/'results/libero_pick_place_attempt_001')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose an empty output directory; each controlled attempt is preserved')
    c = json.loads(args.config.read_text()); validate_config(c)
    mapping = json.loads(args.mapping_config.read_text())
    annotations = json.loads(args.demo_config.read_text())
    human = load_transport(args.demo/'processed_demo.csv', args.demo/'metadata.json', mapping['max_source_gap_s'], timing_config=annotations)
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = benchmark.get_benchmark_dict()['libero_spatial']().get_task(0)
    if task.name != TASK: raise ValueError('Unexpected task ordering')
    bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    np.random.seed(args.seed)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), use_camera_obs=False, camera_heights=128, camera_widths=128, horizon=100000)
    env.seed(args.seed); args.output.mkdir(parents=True, exist_ok=True)
    try:
        runner = PickPlace(env, human, c, mapping, args.output)
        report = runner.run()
        report.update(task=task.name, seed=args.seed, settings=c, mapping_settings=mapping,
                      demo_annotations=annotations, source_metadata=human[-1], transport_samples=len(human[0]),
                      semantic_timing='Human events select source samples; robot phase durations are independent')
        save_outputs(runner, report, args.output)
    finally:
        env.close()
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_metadata', 'controller', 'settings', 'mapping_settings', 'demo_annotations')}, indent=2))
    if not report['task_completed']: raise SystemExit(2)


if __name__ == '__main__':
    main()
