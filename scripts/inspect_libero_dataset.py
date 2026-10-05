"""Load and validate an episode, then save paired camera/action samples."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def validate_episode(path):
    with h5py.File(path, 'r') as f:
        assert f.attrs['schema_version'] == 1
        assert f.attrs['complete'] and f.attrs['episode_success']
        assert f.attrs['language_instruction']
        n = len(f['actions'])
        assert n > 0 and f['actions'].shape == (n, 7)
        for key in ('rewards', 'environment_done', 'timestep', 'episode_id', 'phase', 'controller_step'):
            assert f[key].shape == (n,), key
        np.testing.assert_array_equal(f['timestep'][:], np.arange(n))
        np.testing.assert_array_equal(f['controller_step'][:], np.arange(n) + 21)
        assert np.all(f['episode_id'][:].astype(str) == f.attrs['episode_id'])
        for key, ds in f['observations'].items():
            assert len(ds) == n + 1, key
            if key.endswith('_rgb'):
                assert ds.ndim == 4 and ds.shape[-1] == 3 and ds.dtype == np.uint8
                assert np.ptp(ds[0]) > 0, 'Blank camera: ' + key
            else:
                assert np.isfinite(ds[:]).all(), key
        obs = f['observations']
        assert 'agentview_rgb' in obs
        for key, dim in dict(joint_pos=7, joint_vel=7, gripper_qpos=2, gripper_qvel=2,
                             proprio=18, eef_pose=7).items():
            assert obs[key].shape == (n + 1, dim), key
        np.testing.assert_array_equal(obs['proprio'][:], np.concatenate([
            obs[k][:] for k in ('joint_pos', 'joint_vel', 'gripper_qpos', 'gripper_qvel')], axis=1))
        np.testing.assert_allclose(np.linalg.norm(obs['eef_pose'][:, 3:], axis=1), 1., atol=1e-6)
        np.testing.assert_allclose(np.diff(obs['sim_time'][:]), 1. / f.attrs['control_frequency_hz'], atol=1e-8)
        a = f['actions'][:]
        assert np.isfinite(a).all() and np.max(np.abs(a[:, :6])) <= .5
        assert np.isin(a[:, 6], [-1., 1.]).all()
        report = json.loads(f.attrs['success_metadata_json'])
        assert report['task_completed'] and report['completed_sequence']
        assert report['object_remained_at_target_after_release'] and obs['libero_success'][-1]
        assert report['total_episode_steps'] == n + 20
        return dict(episode_id=f.attrs['episode_id'], transitions=n,
                    language_instruction=f.attrs['language_instruction'],
                    observations={k: list(v.shape) for k, v in obs.items()},
                    action_shape=list(a.shape))


def load_transition(file, timestep):
    """For BC use observation + action; next_observation is also available."""
    return dict(observation={k: v[timestep] for k, v in file['observations'].items()},
                action=file['actions'][timestep],
                next_observation={k: v[timestep + 1] for k, v in file['observations'].items()},
                language_instruction=file.attrs['language_instruction'],
                episode_id=file.attrs['episode_id'], timestep=timestep,
                episode_success=bool(file.attrs['episode_success']),
                success_after_action=bool(file['observations/libero_success'][timestep + 1]))


def save_samples(path, output):
    from PIL import Image, ImageDraw
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    with h5py.File(path, 'r') as f:
        for t in np.linspace(0, len(f['actions']) - 1, 5, dtype=int):
            sample = load_transition(f, int(t))
            images = [Image.fromarray(v) for k, v in sample['observation'].items() if k.endswith('_rgb')]
            width = max(sum(im.width for im in images), 760)
            canvas = Image.new('RGB', (width, images[0].height + 70), 'white')
            x = 0
            for im in images:
                canvas.paste(im, (x, 0)); x += im.width
            ImageDraw.Draw(canvas).text((4, images[0].height + 4),
                f'{sample["episode_id"]} t={t} phase={f["phase"][t].decode()}\n'
                f'action={np.array2string(sample["action"], precision=5)}\n'
                'Left: agent view. Right: wrist. Both BEFORE this action.', fill='black')
            filename = f'sample_{t:05d}.png'
            canvas.save(output / filename)
            samples.append(dict(timestep=int(t), image=filename, action=sample['action'].tolist(),
                                eef_pose=sample['observation']['eef_pose'].tolist(),
                                sim_time=float(sample['observation']['sim_time'])))
    (output / 'samples.json').write_text(json.dumps(samples, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--samples', type=Path, default=Path('results/libero_dataset_samples'))
    args = parser.parse_args()
    print(json.dumps(validate_episode(args.episode), indent=2))
    save_samples(args.episode, args.samples)
