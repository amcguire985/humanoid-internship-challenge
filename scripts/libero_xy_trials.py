"""Exactly ten fixed-parameter manipulation trials, varying only initial bowl XY.

Preflight uses forward kinematics / static contacts, not physical grasp trials.
Perturbation is applied after the identical nominal SCENE_SETTLE sequence.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from scripted_libero_pick_place import (PickPlace, ManipulationAbort, OBJECT, TARGET,
    collision_bounds, json_safe, save_outputs, validate_config)
from retarget_libero_object import ROOT, TASK, load_transport, retarget_transport

OFFSETS = [(0, 0), (.02, 0), (-.02, 0), (0, .02), (0, -.02), (.02, .02),
           (.02, -.02), (-.02, .02), (-.02, -.02), (.04, 0)]
DEFAULT_OUTPUT = ROOT/'results/libero_xy_trials_10'
SUCCESS = ROOT/'results/libero_pick_place_attempt_003/metrics.json'
CONFIG_PATHS = dict(robot=ROOT/'config/libero_pick_place.json', mapping=ROOT/'config/libero_retarget.json',
                    human=ROOT/'config/test_007_demo.json')
POSE_TOL = 1e-10
COLLISION_TOL = 1e-7
OFFSET_RESOLUTION = 1e-6  # one micrometre along the requested ray


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    settings = {k: json.loads(v.read_text()) for k,v in CONFIG_PATHS.items()}
    successful = json.loads(SUCCESS.read_text())
    if not successful['task_completed']:
        raise ValueError('Reference run must have completed successfully')
    for key, old_key in [('robot', 'settings'), ('mapping', 'mapping_settings'), ('human', 'demo_annotations')]:
        if settings[key] != successful[old_key]:
            raise ValueError('Settings differ from successful run: '+key)
    validate_config(settings['robot'])
    demo = ROOT/'results/test_007_block_only_raw/task_demo'
    human = load_transport(demo/'processed_demo.csv', demo/'metadata.json',
                           settings['mapping']['max_source_gap_s'], timing_config=settings['human'])
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in
              [*CONFIG_PATHS.values(), demo/'processed_demo.csv', demo/'metadata.json',
               ROOT/'scripts/scripted_libero_pick_place.py', ROOT/'scripts/retarget_libero_object.py',
               ROOT/'scripts/replay_libero_transport.py', ROOT/'scripts/process_task_demo.py']}
    return settings, successful, human, hashes


def make_env(seed):
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = benchmark.get_benchmark_dict()['libero_spatial']().get_task(0)
    if task.name != TASK:
        raise ValueError('Unexpected task ordering')
    np.random.seed(seed)
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                            use_camera_obs=False, camera_heights=128, camera_widths=128, horizon=100000)
    env.seed(seed)
    return env


def object_qpos_slice(env):
    name = env.env.objects_dict[OBJECT].joints[-1]
    address = env.env.sim.model.get_joint_qpos_addr(name)
    if not isinstance(address, tuple) or address[1]-address[0] != 7:
        raise ValueError('Expected a seven-coordinate free joint for the manipulated object')
    return slice(*address)


def assert_xy_only(before, after, joint_slice, offset):
    expected = before.copy()
    expected[joint_slice.start:joint_slice.start+2] += offset
    if not np.array_equal(after, expected):
        raise ValueError('Perturbation changed something other than the manipulated object XY')


def largest_valid_on_ray(requested, is_valid, resolution=OFFSET_RESOLUTION):
    """Find the furthest verified point on this short ray; never rotate the offset.

    Descending 1% scan finds the highest sampled valid interval, then bisects
    its upper boundary to the specified metre resolution. Returned point itself
    must pass the full preflight, not just the bisection estimate.
    """
    requested = np.asarray(requested, dtype=float)
    valid, detail = is_valid(requested)
    if valid:
        return requested, detail, dict(reduced=False, requested_validation=detail)
    rejected = detail
    if np.linalg.norm(requested) == 0:
        raise ValueError('Nominal pose failed validation: '+str(detail))
    upper = 1.
    for lower in np.linspace(.99, 0., 100):
        valid, detail = is_valid(requested*lower)
        if valid:
            break
        upper = lower
    else:
        raise ValueError('No valid offset exists on the requested ray')
    while (upper-lower)*np.linalg.norm(requested) > resolution:
        middle = (lower+upper)/2
        valid, test = is_valid(requested*middle)
        if valid:
            lower, detail = middle, test
        else:
            upper = middle
    applied = requested*lower
    valid, detail = is_valid(applied)
    if not valid:
        raise RuntimeError('Validated offset became invalid')
    return applied, detail, dict(reduced=True, requested_validation=rejected,
        scale_factor=float(lower), upper_invalid_scale=float(upper), boundary_resolution_m=resolution,
        search='Highest valid descending 1%-sample interval, boundary refined by bisection; no extrapolation beyond requested offset')


class PoseValidator:
    """Static geometry and joint-limited numerical IK on a separate MjData."""
    def __init__(self, runner):
        import mujoco
        self.mj = mujoco
        self.runner = runner
        self.sim = runner.env.env.sim
        self.model = self.sim.model._model
        self.data = mujoco.MjData(self.model)
        self.nominal_qpos = self.sim.data.qpos.copy()
        self.nominal_qvel = self.sim.data.qvel.copy()
        self.slice = object_qpos_slice(runner.env)
        self.object_ids = {self.sim.model.geom_name2id(n) for n in runner.object_geoms}
        self.table_id = self.sim.model.geom_name2id('table_collision')
        robot = runner.env.env.robots[0]
        self.qids = np.array(robot._ref_joint_pos_indexes)
        self.vids = np.array(robot._ref_joint_vel_indexes)
        self.site = robot.eef_site_id
        self.limits = np.array([self.sim.model.jnt_range[self.sim.model.joint_name2id(n)] for n in robot.robot_model.joints])
        self.rotation = runner.rotation
        self.cache = {}

    def load_pose(self, offset):
        self.data.qpos[:] = self.nominal_qpos
        self.data.qvel[:] = self.nominal_qvel
        self.data.qpos[self.slice.start:self.slice.start+2] += offset
        self.mj.mj_forward(self.model, self.data)

    def bounds(self, ids):
        corners = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                            [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
        points = []
        for i in ids:
            if self.model.geom_type[i] != 6:
                raise ValueError('Expected box collision geometry')
            points.extend((corners*self.model.geom_size[i])@self.data.geom_xmat[i].reshape(3,3).T+self.data.geom_xpos[i])
        return np.min(points, axis=0), np.max(points, axis=0)

    def solve_ik(self, point):
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for iteration in range(150):
            self.mj.mj_forward(self.model, self.data)
            ep = point-self.data.site_xpos[self.site]
            er = Rotation.from_matrix(self.rotation@self.data.site_xmat[self.site].reshape(3,3).T).as_rotvec()
            if np.linalg.norm(ep) <= .001 and np.linalg.norm(er) <= .01:
                return True, float(np.linalg.norm(ep)), float(np.linalg.norm(er)), iteration
            self.mj.mj_jacSite(self.model, self.data, jp, jr, self.site)
            jac = np.vstack([jp[:, self.vids], .15*jr[:, self.vids]])
            error = np.r_[ep, .15*er]
            delta = jac.T@np.linalg.solve(jac@jac.T+1e-5*np.eye(6), error)
            delta *= min(1., .15/max(np.linalg.norm(delta), 1e-12))
            self.data.qpos[self.qids] = np.clip(self.data.qpos[self.qids]+delta, self.limits[:,0]+.001, self.limits[:,1]-.001)
        return False, float(np.linalg.norm(ep)), float(np.linalg.norm(er)), 150

    def __call__(self, offset):
        key = tuple(float(x) for x in offset)
        if key in self.cache:
            return self.cache[key]
        self.load_pose(offset)
        omin, omax = self.bounds(self.object_ids)
        tmin, tmax = self.bounds([self.table_id])
        support = []
        collisions = []
        for c in self.data.contact[:self.data.ncon]:
            ids = [int(c.geom1), int(c.geom2)]
            if not any(g in self.object_ids for g in ids): continue
            other = ids[1] if ids[0] in self.object_ids else ids[0]
            if other in self.object_ids: continue
            pair = dict(geoms=[self.sim.model.geom_id2name(i) for i in ids], distance_m=float(c.dist))
            if other == self.table_id:
                support.append(pair)
            elif c.dist < -COLLISION_TOL:
                collisions.append(pair)
        within = bool(np.all(omin[:2] >= tmin[:2]) and np.all(omax[:2] <= tmax[:2]))
        supported = bool(support and abs(omin[2]-tmax[2]) < 1e-4)
        detail = dict(offset_xy_m=list(key), support_surface='table_collision', support_contact_count=len(support),
            support_surface_z_m=float(tmax[2]), bottom_minus_support_z_m=float(omin[2]-tmax[2]),
            remains_on_support=supported, within_scene_bounds=within,
            object_bounds=[omin.tolist(), omax.tolist()], support_bounds=[tmin.tolist(), tmax.tolist()],
            collision_free=not collisions, invalid_contacts=collisions, intersection_tolerance_m=COLLISION_TOL,
            reachable=None)
        if not supported or not within or collisions:
            self.cache[key] = (False, detail)
            return False, detail
        c = self.runner.c
        start = self.nominal_qpos[self.slice][:3].copy(); start[:2] += offset
        grasp = start+self.runner.nominal_offset
        pre = grasp.copy(); pre[2] = start[2]+c['pregrasp_height']
        lifted = start+np.array([0,0,c['postgrasp_lift_height']])
        goal = self.runner.placement+np.array([0,0,c['transport_clearance_height']])
        times, human, human_goal, _, _ = self.runner.human
        path, _, mapping = retarget_transport(times, human, human_goal, lifted, goal, self.runner.mapping)
        points = np.vstack([pre, grasp, lifted+self.runner.nominal_offset,
                            path+self.runner.nominal_offset, self.runner.placement+self.runner.nominal_offset,
                            self.runner.placement+self.runner.nominal_offset+[0,0,c['retract_height']]])
        if np.any(points < c['workspace_min']) or np.any(points > c['workspace_max']):
            detail.update(reachable=False, reachability_reason='Reference outside configured workspace')
            self.cache[key] = (False, detail); return False, detail
        worst_p = worst_r = 0.
        for i, point in enumerate(points):
            ok, pos_error, rot_error, _ = self.solve_ik(point)
            worst_p, worst_r = max(worst_p, pos_error), max(worst_r, rot_error)
            if not ok:
                detail.update(reachable=False, reachability_reason='Joint-limited IK did not converge', failed_waypoint=i,
                              ik_position_error_m=pos_error, ik_orientation_error_rad=rot_error)
                self.cache[key] = (False, detail); return False, detail
        detail.update(reachable=True, ik_waypoints=len(points), maximum_ik_position_error_m=worst_p,
                      maximum_ik_orientation_error_rad=worst_r,
                      ik_method='Damped Jacobian IK; fixed orientation; joint limits; all human source knots plus pregrasp/grasp/lift/place/retract',
                      reachability_scope='Kinematic feasibility, not a guarantee of dynamic tracking or grasp success')
        self.cache[key] = (True, detail)
        return True, detail


def prepare(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preflight output must be empty')
    output.mkdir(parents=True)
    c, success, human, hashes = inputs()
    env = make_env(success['seed'])
    try:
        setup = output/'nominal_setup'; setup.mkdir()
        runner = PickPlace(env, human, c['robot'], c['mapping'], setup)
        raw_qpos = env.env.sim.data.qpos.copy()
        runner.scene_settle()  # Only the common 20-step settling sequence; no grasp attempt.
        np.testing.assert_allclose(runner.start, success['geometry']['object_start'], atol=POSE_TOL, rtol=0)
        np.testing.assert_allclose(runner.target, success['geometry']['target_anchor'], atol=POSE_TOL, rtol=0)
        validator = PoseValidator(runner)
        np.savez(output/'nominal_state.npz', raw_qpos=raw_qpos,
                 qpos=validator.nominal_qpos, qvel=validator.nominal_qvel)
        nominal = dict(object_start=runner.start.tolist(), target_anchor=runner.target.tolist(),
            object_free_joint_qpos=validator.nominal_qpos[validator.slice].tolist(),
            robot_arm_qpos=validator.nominal_qpos[validator.qids].tolist(),
            robot_gripper_qpos=np.asarray(runner.obs['robot0_gripper_qpos']).tolist(),
            eef_position=runner.position('robot0_eef_pos').tolist(),
            all_object_poses={name:env.env.sim.data.get_joint_qpos(obj.joints[-1]).tolist()
                              for name,obj in env.env.objects_dict.items()},
            initial=runner.initial, settling_steps=c['robot']['scene_settle_steps'])
        manifest = dict(task=TASK, seed=success['seed'], reference_success=str(SUCCESS),
            settings=c, input_hashes=hashes, nominal=nominal, requested_trials=10, trials=[],
            initialization='Same seed-0 reset and identical settling; change only bowl free-joint XY immediately before PREGRASP',
            target_policy='Identical initial plate pose and fixed retargeting goal in all trials; natural contact dynamics unchanged',
            preflight_is_not_a_trial=True)
        for number, request in enumerate(OFFSETS, 1):
            applied, valid, reduction = largest_valid_on_ray(request, validator)
            entry = dict(trial=number, requested_offset_xy_m=list(request), applied_offset_xy_m=applied.tolist(),
                         validation=valid, adjustment=reduction)
            manifest['trials'].append(entry)
            write_json(output/'preflight.json', manifest)
            print(f'Validated {number:02d}: requested {request}, applied {applied.tolist()}, reduced={reduction["reduced"]}', flush=True)
        manifest['all_poses_validated'] = True
        write_json(output/'preflight.json', manifest)
    finally:
        env.close()
    plot_offsets(manifest, output)
    return manifest


class PerturbedPickPlace(PickPlace):
    def __init__(self, *args, trial, nominal, nominal_state, **kwargs):
        self.trial_setup = trial
        self.nominal = nominal
        self.nominal_state = nominal_state
        super().__init__(*args, **kwargs)
        np.testing.assert_allclose(self.env.env.sim.data.qpos, nominal_state['raw_qpos'], atol=POSE_TOL, rtol=0)

    def scene_settle(self):
        super().scene_settle()
        sim = self.env.env.sim
        np.testing.assert_allclose(sim.data.qpos, self.nominal_state['qpos'], atol=POSE_TOL, rtol=0)
        np.testing.assert_allclose(sim.data.qvel, self.nominal_state['qvel'], atol=POSE_TOL, rtol=0)
        before_qpos = sim.data.qpos.copy(); before_qvel = sim.data.qvel.copy()
        address = object_qpos_slice(self.env)
        offset = np.array(self.trial_setup['applied_offset_xy_m'])
        sim.data.qpos[address.start:address.start+2] += offset
        assert_xy_only(before_qpos, sim.data.qpos, address, offset)
        sim.forward()
        np.testing.assert_array_equal(sim.data.qvel, before_qvel)
        self.env.env._update_observables(force=True)
        self.obs = self.env.env._get_observations()
        self.start = self.position(OBJECT+'_pos')
        expected_start = np.array(self.nominal['object_start'])+np.r_[offset,0.]
        np.testing.assert_allclose(self.start, expected_start, atol=POSE_TOL, rtol=0)
        np.testing.assert_allclose(self.position(TARGET+'_pos'), self.nominal['target_anchor'], atol=POSE_TOL, rtol=0)
        self.target = np.array(self.nominal['target_anchor'])  # Identical goal, never re-randomized or adapted.
        self.placement = self.target+np.array([0,0,self.c['placement_object_height_offset']])
        omin, omax = collision_bounds(self.env, self.object_model)
        self.geometry.update(object_start=self.start.tolist(), target_anchor=self.target.tolist(),
            object_collision_min=omin.tolist(), object_collision_max=omax.tolist())
        self.grasp_pose = self.start+self.nominal_offset
        self.initialization_audit = dict(only_bowl_xy_changed=True, object_z_unchanged=True,
            object_orientation_unchanged=True, robot_configuration_unchanged=True,
            target_and_other_object_poses_unchanged=True, velocities_unchanged=True,
            actual_perturbed_object_start=self.start.tolist(), fixed_target_anchor=self.target.tolist(),
            bowl_free_joint_qpos=sim.data.qpos[address].tolist(),
            robot_arm_qpos=sim.data.qpos[self.env.env.robots[0]._ref_joint_pos_indexes].tolist())
        np.savez(self.output/'initial_state.npz', qpos=sim.data.qpos.copy(), qvel=sim.data.qvel.copy())
        write_json(self.output/'initialization_audit.json', self.initialization_audit)


def run_trial(output, number):
    manifest = json.loads((output/'preflight.json').read_text())
    if not manifest.get('all_poses_validated') or len(manifest['trials']) != 10:
        raise ValueError('All ten poses must pass preflight before any trial runs')
    if not 1 <= number <= 10: raise ValueError('Trial index must be 1..10')
    c, success, human, hashes = inputs()
    if hashes != manifest['input_hashes'] or c != manifest['settings']:
        raise ValueError('Inputs changed since preflight; do not mix experiments')
    directory = output/f'trial_{number:02d}'
    if directory.exists(): raise ValueError('Trial already started; no retries permitted: '+str(directory))
    directory.mkdir()
    write_json(directory/'started.json', dict(trial=number, automatic_retry=False))
    trial = manifest['trials'][number-1]
    env = make_env(manifest['seed'])
    interrupted = [False]
    def on_signal(signum, frame):
        interrupted[0] = True
        raise ManipulationAbort(f'Execution interrupted by signal {signum}')
    old_signal = signal.signal(signal.SIGTERM, on_signal)
    try:
        runner = PerturbedPickPlace(env, human, c['robot'], c['mapping'], directory,
            trial=trial, nominal=manifest['nominal'], nominal_state=dict(np.load(output/'nominal_state.npz')))
        report = runner.run()
        report.update(task=TASK, seed=manifest['seed'], trial=number, perturbation=trial,
            settings=c['robot'], mapping_settings=c['mapping'], demo_annotations=c['human'],
            source_metadata=human[-1], transport_samples=len(human[0]),
            initialization_audit=getattr(runner, 'initialization_audit', None), input_hashes=hashes)
        save_outputs(runner, report, directory)
        print(f'TRIAL {number:02d}: success={report["task_completed"]}; failure={report["failure_phase"]}: {report["failure_reason"]}', flush=True)
    finally:
        signal.signal(signal.SIGTERM, old_signal)
        env.close()
    if interrupted[0]: raise SystemExit('Interrupted; this trial was counted and must not be repeated')
    summarize(output)
    return report


def summarize(output):
    manifest = json.loads((output/'preflight.json').read_text())
    reports = []
    for trial in manifest['trials']:
        p = output/f'trial_{trial["trial"]:02d}'/'metrics.json'
        if p.exists(): reports.append(json.loads(p.read_text()))
    rows = []
    for r in reports:
        request, applied = r['perturbation']['requested_offset_xy_m'], r['perturbation']['applied_offset_xy_m']
        rows.append(dict(trial=r['trial'], requested_dx=request[0], requested_dy=request[1], applied_dx=applied[0], applied_dy=applied[1],
            success=r['task_completed'], libero_success=r['libero_task_success'], grasp_acquired=r['gripper_acquired_object'],
            lifted=r['object_lifted'], grasp_lost=r['grasp_lost'], remained_after_release=r['object_remained_at_target_after_release'],
            steps=r['total_episode_steps'], eef_rmse_mm=r['eef_tracking_rmse_m']*1000,
            maximum_eef_error_mm=r['maximum_tracking_error_m']*1000,
            final_target_error_mm=r['final_object_to_target_anchor_error_m']*1000,
            failure_phase=r['failure_phase'], failure_reason=r['failure_reason']))
    if rows:
        with (output/'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output/'summary.json', dict(requested_trials=10, completed_trials=len(reports),
        successful_trials=sum(r['task_completed'] for r in reports), trials=rows))
    return rows


def plot_offsets(manifest, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,7))
    origin = np.array(manifest['nominal']['object_start'])[:2]
    target = np.array(manifest['nominal']['target_anchor'])[:2]
    for trial in manifest['trials']:
        requested = origin+trial['requested_offset_xy_m']
        applied = origin+trial['applied_offset_xy_m']
        ax.scatter(*requested, marker='x', c='gray')
        ax.scatter(*applied, c='tab:blue')
        ax.annotate(str(trial['trial']), applied, xytext=(5,5), textcoords='offset points')
        if trial['adjustment']['reduced']: ax.plot([requested[0],applied[0]],[requested[1],applied[1]],'r--')
    ax.scatter(*origin, facecolors='none', edgecolors='black', s=150, label='Nominal bowl origin')
    ax.scatter(*target, marker='*', s=130, c='orange', label='Fixed plate origin')
    ax.plot([],[], 'x', c='gray', label='Requested bowl origins'); ax.plot([],[], 'o', c='tab:blue', label='Validated bowl origins')
    ax.set(xlabel='World x (m)', ylabel='World y (m)', title='Ten initial XY perturbations (geometry / IK preflight)')
    ax.axis('equal'); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(output/'validated_offsets.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--trial', type=int)
    mode.add_argument('--run-all', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output)
        shutil.copyfile(__file__, args.output/'trial_driver_snapshot.py')
        shutil.copyfile(ROOT/'scripts/scripted_libero_pick_place.py', args.output/'manipulation_snapshot.py')
    elif args.trial:
        run_trial(args.output, args.trial)
    elif args.run_all:
        # Exactly one run per requested index. Existing outputs are never rerun.
        for number in range(1,11):
            if (args.output/f'trial_{number:02d}').exists():
                continue
            run_trial(args.output, number)
    else:
        print(json.dumps(summarize(args.output), indent=2))


if __name__ == '__main__': main()
