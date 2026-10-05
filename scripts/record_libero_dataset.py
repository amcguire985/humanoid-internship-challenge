"""Record unchanged validated controller runs as success-only robot episodes."""
import argparse
import json
import os
import shutil
import signal
from pathlib import Path

import h5py
import numpy as np

from libero_xy_trials import (ROOT, inputs, make_env, PerturbedPickPlace,
                             write_json, file_hash)
from scripted_libero_pick_place import ManipulationAbort, save_outputs

DEFAULT = ROOT / 'results/libero_robot_dataset'
PREFLIGHT = ROOT / 'results/libero_xy_trials_10'


def append(file, key, value):
    value = np.asarray(value)
    if key not in file:
        file.create_dataset(key, shape=(0, *value.shape), maxshape=(None, *value.shape),
                            dtype=value.dtype, chunks=(1, *value.shape), compression='gzip',
                            compression_opts=1)
    ds = file[key]
    ds.resize(len(ds) + 1, axis=0)
    ds[-1] = value


class RecordingEnv:
    """Intercept exactly the action passed to env.step; never compute control."""
    def __init__(self, env, file, episode_id, resolution):
        self.wrapped, self.file = env, file
        self.episode_id, self.resolution = episode_id, resolution
        self.active = False
        self.runner = None
        self.count = 0
        names = env.env.sim.model.camera_names
        self.cameras = {'agentview_rgb': 'agentview'}
        if 'robot0_eye_in_hand' in names:
            self.cameras['eye_in_hand_rgb'] = 'robot0_eye_in_hand'
        if 'agentview' not in names:
            raise ValueError('Agent-view camera missing')

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def observation(self, obs):
        sim, robot = self.env.sim, self.env.robots[0]
        values = dict(joint_pos=sim.data.qpos[robot._ref_joint_pos_indexes].copy(),
                      joint_vel=sim.data.qvel[robot._ref_joint_vel_indexes].copy(),
                      gripper_qpos=np.asarray(obs['robot0_gripper_qpos']).copy(),
                      gripper_qvel=np.asarray(obs['robot0_gripper_qvel']).copy(),
                      eef_pose=np.r_[obs['robot0_eef_pos'], obs['robot0_eef_quat']],
                      sim_time=np.float64(sim.data.time),
                      libero_success=np.bool_(self.check_success()))
        values['proprio'] = np.concatenate([values[k] for k in
            ('joint_pos', 'joint_vel', 'gripper_qpos', 'gripper_qvel')])
        for key, camera in self.cameras.items():
            values[key] = sim.render(width=self.resolution, height=self.resolution,
                                     camera_name=camera)[::-1].copy()
        for key, value in values.items():
            append(self.file, 'observations/' + key, value)

    def configure_rendering(self):
        # Rendering-only settings: no dynamics, cameras, or control changes.
        import mujoco
        sim = self.env.sim
        sim.model._model.vis.quality.offsamples = 0
        sim.model._model.vis.global_.offwidth = 640
        sim.model._model.vis.global_.offheight = 480
        context = sim._render_context_offscreen
        context.con.free()
        context._set_mujoco_context_and_buffers()
        context.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        context.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.file.attrs['render_settings'] = 'OSMesa; shadows/reflections/MSAA disabled; original cameras and geometry'

    def start(self, obs):
        self.configure_rendering()
        self.observation(obs)
        self.active = True

    def step(self, action):
        if not self.active:
            return self.wrapped.step(action)
        sent = np.asarray(action).copy()
        obs, reward, done, info = self.wrapped.step(action)
        self.observation(obs)
        for key, value in dict(actions=sent, rewards=np.float64(reward),
                               environment_done=np.bool_(done), timestep=np.int64(self.count),
                               episode_id=np.bytes_(self.episode_id),
                               phase=np.bytes_(self.runner.phase),
                               controller_step=np.int64(len(self.runner.records) + 1)).items():
            # Fixed-length byte types must accommodate all later phase names.
            if key == 'phase':
                value = np.asarray(value, dtype='S32')
            append(self.file, key, value)
        self.count += 1
        if self.count % 250 == 0:
            self.file.flush()
            print(f'{self.episode_id}: {self.count} transitions', flush=True)
        return obs, reward, done, info


class RecordedPickPlace(PerturbedPickPlace):
    def scene_settle(self):
        super().scene_settle()
        self.env.start(self.obs)  # The XY reset is complete; no teleport in dataset.


def summarize(output):
    attempts = [json.loads(p.read_text()) for p in sorted((output / 'attempts').glob('*/status.json'))]
    episodes = []
    actions = []
    for p in sorted((output / 'episodes').glob('*.h5')):
        with h5py.File(p, 'r') as f:
            if not f.attrs['complete'] or not f.attrs['episode_success']:
                raise ValueError('Unsuccessful file in accepted episodes')
            actions.append(f['actions'][:])
            episodes.append(dict(episode_id=p.stem, path=str(p.relative_to(output)),
                                 transitions=len(f['actions']), bytes=p.stat().st_size))
            dimensions = {k: list(v.shape[1:]) for k, v in f['observations'].items()}
    result = dict(schema_version=1, attempts=attempts, successful_episodes=len(episodes),
                  total_transitions=sum(e['transitions'] for e in episodes), episodes=episodes)
    if actions:
        a = np.concatenate(actions)
        result.update(observation_dimensions=dimensions, action_dimension=a.shape[1],
                      action_statistics={k: getattr(a, k)(axis=0).tolist() for k in ('min', 'max', 'mean', 'std')},
                      episode_file_bytes=sum(e['bytes'] for e in episodes))
    result['size_on_disk_bytes_excluding_summary'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file() and p.name != 'summary.json')
    write_json(output / 'summary.json', result)
    return result


def run(output, preflight, resolution):
    os.environ.setdefault('LP_NUM_THREADS', '2')
    os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/libero_numba_cache')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/libero_matplotlib')
    c, reference, human, hashes = inputs()
    manifest = json.loads((preflight / 'preflight.json').read_text())
    if not manifest.get('all_poses_validated') or hashes != manifest['input_hashes'] or c != manifest['settings']:
        raise ValueError('Validated preflight inputs no longer match')
    if output.exists():
        raise ValueError('Output already exists; use a new directory (no implicit retries)')
    (output / 'episodes').mkdir(parents=True)
    (output / 'attempts').mkdir()
    shutil.copy2(preflight / 'preflight.json', output / 'preflight.json')
    shutil.copy2(preflight / 'nominal_state.npz', output / 'nominal_state.npz')
    write_json(output / 'provenance.json', dict(input_hashes=hashes, seed=manifest['seed'],
        recording_script_sha256=file_hash(__file__), resolution=resolution,
        reset_boundary='After SCENE_SETTLE and XY perturbation, before PREGRASP',
        selected_trials=[t['trial'] for t in manifest['trials']]))
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    old = signal.signal(signal.SIGTERM, interrupted)
    try:
        for trial in manifest['trials']:
            episode_id = f'episode_{trial["trial"]:03d}'
            directory = output / 'attempts' / episode_id
            directory.mkdir()
            status = dict(episode_id=episode_id, trial=trial, status='incomplete', accepted=False)
            write_json(directory / 'status.json', status)
            temporary = directory / 'episode.partial.h5'
            env = None
            try:
                env = make_env(manifest['seed'])
                with h5py.File(temporary, 'w') as f:
                    wrapped = RecordingEnv(env, f, episode_id, resolution)
                    runner = RecordedPickPlace(wrapped, human, c['robot'], c['mapping'], directory,
                        trial=trial, nominal=manifest['nominal'],
                        nominal_state=dict(np.load(preflight / 'nominal_state.npz')))
                    wrapped.runner = runner
                    f.attrs.update(schema_version=1, complete=False, episode_success=False,
                        episode_id=episode_id, language_instruction=env.language_instruction,
                        control_frequency_hz=runner.frequency,
                        image_convention='RGB uint8 HWC, top-left origin (MuJoCo vertical flip)',
                        eef_pose_convention='world xyz metres + quaternion xyzw',
                        proprio_order='joint_pos[7], joint_vel[7], gripper_qpos[2], gripper_qvel[2]')
                    report = runner.run()
                    report.update(trial=trial, input_hashes=hashes)
                    save_outputs(runner, report, directory)
                    logged = [r for r in runner.records if r['phase'] != 'SCENE_SETTLE']
                    np.testing.assert_array_equal(f['actions'][:], np.array([r['action'] for r in logged]))
                    np.testing.assert_array_equal(f['observations/libero_success'][1:], [r['task_success'] for r in logged])
                    f.attrs['success_metadata_json'] = json.dumps(report)
                    f.attrs['complete'] = bool(report['completed_sequence'])
                    f.attrs['episode_success'] = bool(report['task_completed'])
                    status.update(status='success' if report['task_completed'] else 'failed',
                        accepted=bool(report['task_completed']), transitions=wrapped.count,
                        failure_phase=report['failure_phase'], failure_reason=report['failure_reason'])
                if status['accepted']:
                    from inspect_libero_dataset import validate_episode
                    validate_episode(temporary)
                    temporary.replace(output / 'episodes' / (episode_id + '.h5'))
                else:
                    temporary.unlink()
            except BaseException as exc:
                status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                              accepted=False, failure_reason=f'{type(exc).__name__}: {exc}')
                if temporary.exists():
                    temporary.unlink()
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                print(f'{episode_id}: {status["failure_reason"]}', flush=True)
            finally:
                write_json(directory / 'status.json', status)
                summarize(output)
                if env is not None:
                    env.close()
            print(f'{episode_id}: {status["status"]}', flush=True)
    finally:
        signal.signal(signal.SIGTERM, old)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--preflight', type=Path, default=PREFLIGHT)
    parser.add_argument('--resolution', type=int, default=128)
    args = parser.parse_args()
    run(args.output, args.preflight, args.resolution)
