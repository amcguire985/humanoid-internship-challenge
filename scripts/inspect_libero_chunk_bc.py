"""Report chunk predictions around commanded close and measured physical grasp."""
import argparse
import csv
import json
from pathlib import Path
import h5py
import numpy as np


def inspect(output):
    audit=json.loads((output/'gripper_audit.json').read_text())
    saved=np.load(output/'training_predictions.npz')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,1,figsize=(11,7))
    offset=0
    for ax,episode in zip(axes,audit['episodes']):
        path=Path(episode['path'])
        rows=list(csv.DictReader((path.parents[1]/'attempts'/path.stem/'replay.csv').open()))
        with h5py.File(path) as f:
            indices={int(s):t for t,s in enumerate(f['controller_step'][:])}
        grasp=next(r for r in rows if r['bilateral_grasp']=='True')
        t=indices[int(grasp['step'])]
        episode['first_physical_grasp_transition']=t
        episode['pre_physical_grasp_observation']=dict(t=t-1,demonstrated=float(saved['target'][offset+t-1,0,6]),predicted=saved['predicted'][offset+t-1,:,6].tolist())
        event=episode['grasp_windows'][0]['close_transition']
        x=np.arange(event-16,t+10)
        ax.step(x,saved['target'][offset+x,0,6],where='post',label='Demonstrated current command')
        for k in (0,3,11):
            ax.plot(x,saved['predicted'][offset+x,k,6],label=f'Predicted action t+{k}')
        ax.axvline(t,color='gray',linestyle=':',label='First physical grasp (post-action)')
        ax.axhline(0,color='black',linewidth=.5)
        ax.set(title=path.stem,xlabel='Observation/action index',ylabel='Gripper command'); ax.legend(fontsize=8)
        offset+=episode['transitions']
    fig.tight_layout(); fig.savefig(output/'gripper_around_grasp.png'); plt.close(fig)
    (output/'gripper_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps([dict(episode=e['episode_id'],physical_grasp=e['first_physical_grasp_transition'],pre_physical_grasp=e['pre_physical_grasp_observation'],pre_close=e['grasp_windows'][0]['rows'][10],on_close=e['grasp_windows'][0]['rows'][11]) for e in audit['episodes']],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('output',type=Path)
    inspect(parser.parse_args().output)
