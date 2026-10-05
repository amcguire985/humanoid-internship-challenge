"""Lossless, dependency-light SmolVLA tensor export; optional native LeRobot export."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'observation.images.camera1'
STATE = 'observation.state'


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    episodes = []
    for number in (1, 2):
        path = ROOT / f'results/libero_robot_dataset/episodes/episode_{number:03d}.h5'
        directory = output / path.stem
        directory.mkdir()
        with h5py.File(path) as f:
            assert f.attrs['complete'] and f.attrs['episode_success']
            n = len(f['actions'])
            assert f['observations/agentview_rgb'].shape == (n+1,128,128,3)
            assert f['observations/proprio'].shape == (n+1,18)
            assert f['actions'].shape == (n,7)
            for name, value in {
                IMAGE: f['observations/agentview_rgb'][:-1],
                STATE: f['observations/proprio'][:-1].astype('float32'),
                'action': f['actions'][:].astype('float32'),
                'timestamp': np.arange(n, dtype='float64') / 20,
                'source_timestep': f['timestep'][:],
                'controller_step': f['controller_step'][:],
                'observation.sim_time': f['observations/sim_time'][:-1],
                'next.success': f['observations/libero_success'][1:],
                'next.done': f['environment_done'][:],
                'terminal.rgb': f['observations/agentview_rgb'][-1],
                'terminal.state': f['observations/proprio'][-1],
            }.items():
                np.save(directory / (name+'.npy'), value, allow_pickle=False)
            episodes.append(dict(directory=path.stem, source=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), length=n,
                task=str(f.attrs['language_instruction']), success=True,
                success_metadata=json.loads(f.attrs['success_metadata_json']),
                proprio_order=str(f.attrs['proprio_order'])))
    manifest = dict(format='smolvla_numpy_v1', fps=20, total_frames=sum(e['length'] for e in episodes),
        image_key=IMAGE, state_key=STATE, action_key='action', episodes=episodes,
        alignment='Observation t predicts action t; terminal observation retained separately, never used as a target.',
        gripper='-1 open; +1 close; raw LIBERO OSC actions, no joint-space reinterpretation')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"Exported {manifest['total_frames']} transitions to {output}")


def native_export(source, output):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    manifest=json.loads((source/'manifest.json').read_text())
    features={IMAGE:dict(dtype='image',shape=(128,128,3),names=['height','width','channels']),
              STATE:dict(dtype='float32',shape=(18,),names=None),
              'action':dict(dtype='float32',shape=(7,),names=None),
              'next.success':dict(dtype='bool',shape=(1,),names=None),
              'next.done':dict(dtype='bool',shape=(1,),names=None)}
    dataset=LeRobotDataset.create(repo_id='local/human_libero_two_demos',root=output,
                                 fps=20,robot_type='libero_panda_osc',features=features,use_videos=False)
    for episode in manifest['episodes']:
        directory=source/episode['directory']
        arrays={k:np.load(directory/(k+'.npy'),mmap_mode='r') for k in features}
        for t in range(episode['length']):
            frame={k:np.array(v[t],copy=True).reshape(features[k]['shape']) for k,v in arrays.items()}
            frame['task']=episode['task']
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Native loader round-trip, including episode boundaries.
    loaded=LeRobotDataset(repo_id='local/human_libero_two_demos',root=output)
    offset=0
    for episode in manifest['episodes']:
        directory=source/episode['directory']
        arrays={k:np.load(directory/(k+'.npy'),mmap_mode='r') for k in (IMAGE,STATE,'action')}
        for t in [0,episode['length']//2,episode['length']-1]:
            sample=loaded[offset+t]
            np.testing.assert_array_equal((sample[IMAGE].permute(1,2,0).numpy()*255).round().astype('uint8'),arrays[IMAGE][t])
            for k in (STATE,'action'): np.testing.assert_array_equal(sample[k].numpy(),arrays[k][t])
            assert sample['task']==episode['task']
        offset+=episode['length']
    print('Native LeRobot conversion and sample round-trip passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/libero_smolvla/dataset')
    parser.add_argument('--native-from',type=Path)
    args=parser.parse_args()
    if args.native_from: native_export(args.native_from,args.output)
    else: prepare(args.output)
