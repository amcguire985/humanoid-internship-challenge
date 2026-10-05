"""Check exported samples against original synchronized HDF5, including grasp and boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from prepare_libero_smolvla import ROOT, IMAGE, STATE


def validate(root):
    manifest=json.loads((root/'manifest.json').read_text())
    assert [e['directory'] for e in manifest['episodes']]==['episode_001','episode_002']
    checks=[]
    for episode in manifest['episodes']:
        path=ROOT/episode['source']; directory=root/episode['directory']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==episode['sha256']
        with h5py.File(path) as f:
            n=episode['length']; actions=np.load(directory/'action.npy',mmap_mode='r')
            assert len(actions)==n and episode['task']==f.attrs['language_instruction']
            switches=np.flatnonzero(np.diff(actions[:,6]))+1
            samples=sorted(set([0,1,n//2,n-1]+[int(t+d) for t in switches for d in (-1,0,1)]))
            for key,original in [(IMAGE,'observations/agentview_rgb'),(STATE,'observations/proprio'),('action','actions')]:
                array=np.load(directory/(key+'.npy'),mmap_mode='r')
                assert len(array)==n
                for t in samples: np.testing.assert_array_equal(array[t],f[original][t].astype(array.dtype))
            np.testing.assert_array_equal(np.load(directory/'source_timestep.npy'),f['timestep'][:])
            np.testing.assert_array_equal(np.load(directory/'controller_step.npy'),f['controller_step'][:])
            np.testing.assert_allclose(np.diff(np.load(directory/'observation.sim_time.npy')),.05,atol=1e-9)
            np.testing.assert_array_equal(np.load(directory/'next.success.npy'),f['observations/libero_success'][1:])
            np.testing.assert_array_equal(np.load(directory/'next.done.npy'),f['environment_done'][:])
            np.testing.assert_array_equal(np.load(directory/'terminal.rgb.npy'),f['observations/agentview_rgb'][-1])
            np.testing.assert_array_equal(np.load(directory/'terminal.state.npy'),f['observations/proprio'][-1])
            assert episode['success'] and episode['success_metadata']['task_completed']
            checks.append(dict(episode=episode['directory'],length=n,sample_indices=samples,passed=True))
    assert sum(e['length'] for e in manifest['episodes'])==4230
    report=dict(passed=True,transitions=4230,episodes=checks)
    (root.parent/'dataset_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset',type=Path)
    validate(parser.parse_args().dataset)
