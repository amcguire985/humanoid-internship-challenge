"""Two closed-loop learned-policy evaluations; diagnostic logs are not demos."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('LP_NUM_THREADS', '2')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/libero_numba_cache')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/libero_matplotlib')

import numpy as np
import torch
from PIL import Image

from train_libero_bc import ROOT, SmallBC, prepare_images
from libero_xy_trials import inputs, make_env, PerturbedPickPlace, write_json
from scripted_libero_pick_place import OBJECT


def rendering_settings(sim):
    import mujoco
    sim.model._model.vis.quality.offsamples = 0
    sim.model._model.vis.global_.offwidth = 640
    sim.model._model.vis.global_.offheight = 480
    ctx = sim._render_context_offscreen
    ctx.con.free()
    ctx._set_mujoco_context_and_buffers()
    ctx.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    ctx.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0


def proprio(env, obs):
    sim, robot = env.env.sim, env.env.robots[0]
    return np.r_[sim.data.qpos[robot._ref_joint_pos_indexes], sim.data.qvel[robot._ref_joint_vel_indexes],
                 obs['robot0_gripper_qpos'], obs['robot0_gripper_qvel']].astype(np.float32)


def inferred_phase(acquired, lifted, near_target, released):
    if not acquired:
        return 'APPROACH_OR_GRASP'
    if not lifted:
        return 'LIFT'
    if not near_target:
        return 'TRANSPORT'
    if not released:
        return 'PLACE_OR_RELEASE'
    return 'RETRACT_OR_RETENTION'


def evaluate(checkpoint, output, horizon=2400):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = SmallBC()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    c, _, human, hashes = inputs()
    preflight = ROOT/'results/libero_xy_trials_10'
    manifest = json.loads((preflight/'preflight.json').read_text())
    if hashes != manifest['input_hashes']:
        raise ValueError('Original validated inputs changed')
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for number, name in [(1, 'nominal'), (2, 'perturbed_plus_2cm_x')]:
        directory = output/name
        if directory.exists():
            raise ValueError('Refusing to repeat an existing evaluation')
        directory.mkdir()
        env = make_env(manifest['seed'])
        rows = []
        acquired = lifted = near_target = released = success_ever = completed = False
        grasp_streak = retention_streak = 0
        reason = 'Evaluation horizon exhausted'
        try:
            runner = PerturbedPickPlace(env, human, c['robot'], c['mapping'], directory,
                trial=manifest['trials'][number-1], nominal=manifest['nominal'],
                nominal_state=dict(np.load(preflight/'nominal_state.npz')))
            runner.scene_settle()  # Same 20-step initialization and XY-only reset as demonstrations.
            runner.phase = 'POLICY'  # Used only for diagnostic contact classification.
            obs = runner.obs
            rendering_settings(env.env.sim)
            for step in range(horizon):
                rgb = env.env.sim.render(width=128, height=128, camera_name='agentview')[::-1].copy()
                if step % 500 == 0:
                    Image.fromarray(rgb).save(directory/f'agentview_{step:04d}.png')
                with torch.no_grad():
                    prediction = model.predict(prepare_images(rgb[None]), torch.from_numpy(proprio(env, obs)[None]))[0].numpy()
                if not np.isfinite(prediction).all():
                    reason = 'Nonfinite learned action'; break
                action = prediction.astype(np.float64)
                action[:6] = np.clip(action[:6], -.5, .5)
                action[6] = 1. if action[6] >= 0 else -1.
                obs, reward, done, info = env.step(action)
                runner.obs = obs
                grasp = bool(env.env._check_grasp(env.env.robots[0].gripper, runner.object_model))
                grasp_streak = grasp_streak + 1 if grasp else 0
                acquired = acquired or grasp_streak >= c['robot']['grasp_confirm_steps']
                obj = np.asarray(obs[OBJECT+'_pos'])
                eef = np.asarray(obs['robot0_eef_pos'])
                lift_height = float(obj[2]-runner.start[2])
                lifted = lifted or (acquired and lift_height >= c['robot']['minimum_object_lift_m'])
                xy_error = float(np.linalg.norm((obj-runner.placement)[:2]))
                in_region = bool(xy_error <= c['robot']['target_region_radius_m'] and
                                 abs(obj[2]-runner.placement[2]) <= c['robot']['placement_height_tolerance_m'])
                near_target = near_target or (lifted and xy_error <= c['robot']['target_region_radius_m'])
                released = released or (lifted and near_target and action[6] < 0 and not grasp)
                libero_success = bool(env.check_success())
                success_ever = success_ever or libero_success
                retracted = bool(eef[2] >= obj[2]+.08)
                retained = released and in_region and libero_success and not grasp and retracted
                retention_streak = retention_streak + 1 if retained else 0
                phase = inferred_phase(acquired, lifted, near_target, released)
                rows.append(dict(timestep=step, phase=phase, grasp=grasp,
                    libero_success=libero_success, reward=float(reward),
                    object_lift_m=lift_height, object_target_xy_error_m=xy_error,
                    **{f'action_{i}':float(v) for i,v in enumerate(action)},
                    **{f'predicted_{i}':float(v) for i,v in enumerate(prediction)},
                    **{f'eef_{axis}':float(v) for axis,v in zip('xyz',eef)},
                    **{f'object_{axis}':float(v) for axis,v in zip('xyz',obj)}))
                if step % 250 == 0:
                    print(f'{name}: step={step}, phase={phase}, acquired={acquired}, lifted={lifted}, success={libero_success}', flush=True)
                if retention_streak >= c['robot']['retract_settle_steps']:
                    completed = True; reason = None; break
                _, forbidden = runner.contacts()
                if forbidden:
                    reason = 'Unexpected robot/scene contact: '+str(forbidden); break
                if not np.isfinite(np.r_[eef,obj]).all():
                    reason = 'Nonfinite observed state'; break
                if np.any(eef < c['robot']['workspace_min']) or np.any(eef > c['robot']['workspace_max']):
                    reason = 'EEF left validated workspace'; break
                if obj[2] < .82:
                    reason = 'Object fell below support surface'; break
                if getattr(env.env, 'done', False) or (done and not libero_success):
                    reason = 'Environment terminated'; break
            runner.snapshot('final_agentview')
            report = dict(reset=name, trial=number, offset_xy_m=manifest['trials'][number-1]['applied_offset_xy_m'],
                steps=len(rows), horizon=horizon, object_acquired=acquired, object_lifted=lifted,
                object_reached_target=near_target, released=released, task_completed=completed,
                libero_success=bool(env.check_success()), libero_success_ever=success_ever,
                failure_phase=None if completed else inferred_phase(acquired,lifted,near_target,released),
                failure_reason=reason, maximum_object_lift_m=max((r['object_lift_m'] for r in rows),default=0),
                final_object_target_xy_error_m=rows[-1]['object_target_xy_error_m'] if rows else None,
                policy='SmallBC image+18D proprio only; no phase, time, target, or scripted control after initialization',
                phase_semantics='Post-hoc diagnostic milestone, not a scripted policy phase',
                action_postprocessing='Clamp pose input to [-0.5,0.5], threshold predicted gripper at zero to +/-1',
                completion_criterion='Confirmed grasp and lift; release; LIBERO success and target retention without grasp and EEF >=8cm above object for 20 consecutive steps',
                training_distribution_reset=True)
            write_json(directory/'metrics.json', report)
            if rows:
                with (directory/'rollout.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            reports.append(report)
            write_json(output/'summary.json', dict(checkpoint=str(checkpoint.resolve()),evaluations=reports))
            print(json.dumps(report), flush=True)
        finally:
            env.close()
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'results/libero_bc_baseline/policy.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'results/libero_bc_baseline/evaluation')
    parser.add_argument('--horizon', type=int, default=2400)
    args = parser.parse_args()
    evaluate(args.checkpoint,args.output,args.horizon)
