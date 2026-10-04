"""Compare gentle position smoothing on existing gap-filled centre trajectories."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def smooth_positions(times, positions, method, window=5, tau=.03):
    """Comparison interface using the shared timestamp-aware position smoother."""
    from smooth_centers import smooth_positions as smooth_centres
    strength = tau if method == 'ema' else window
    return smooth_centres(times, positions, method, strength,
                          max_span=None, shrink_window=True)


def run(source):
    source=Path(source)
    with source.open(newline='',encoding='utf-8') as handle:
        reader=csv.DictReader(handle)
        fields=list(reader.fieldnames)
        rows=list(reader)
    times=np.array([float(r['time_s']) for r in rows])
    if len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Need finite strictly increasing timestamps')
    p=np.array([[float(r[a+'_m']) if r[a+'_m'] else np.nan for a in 'xyz'] for r in rows])
    variants={'gap_filled':p,'savgol_5':smooth_positions(times,p,'savgol',window=5),
              'savgol_7':smooth_positions(times,p,'savgol',window=7),
              'ema_0.03s':smooth_positions(times,p,'ema',tau=.03)}
    output=source.parent/'gentle_smoothing'
    output.mkdir(parents=True,exist_ok=True)
    valid=np.isfinite(p).all(axis=1)
    observed=np.array([r['pose_source']=='observed' for r in rows]) & valid
    report=dict(source=str(source),gap_fill_settings=json.loads((source.parent/'metrics.json').read_text()),
                settings={'savgol_windows_frames':[5,7],'polynomial_order':2,'ema_tau_s':.03},
                notes=['No additional gap filling or outlier rejection.',
                       'Quadratic fits use actual timestamps, are centred offline, and use asymmetric windows at segment ends.',
                       'EMA is causal and may delay motion; it resets after missing frames or large timestamp jumps.',
                       'Missing positions remain blank. Centre orientation is unavailable.',
                       'Signal changes measure filter strength, not accuracy.'],variants={})
    for name,positions in variants.items():
        delta=np.linalg.norm(positions[valid]-p[valid],axis=1)
        report['variants'][name]=dict(available_frames=int(valid.sum()),
            rms_change_mm=float(np.sqrt(np.mean(delta**2))*1000),
            max_change_mm=float(np.max(delta)*1000),
            rms_change_observed_mm=float(np.sqrt(np.mean(np.sum((positions[observed]-p[observed])**2,axis=1)))*1000))
        if name=='gap_filled':
            continue
        with (output/(name+'.csv')).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=[*fields,'smoothing_method'])
            writer.writeheader()
            for i,row in enumerate(rows):
                record=dict(row,smoothing_method=name)
                if valid[i]:
                    record.update({a+'_m':float(positions[i,j]) for j,a in enumerate('xyz')})
                    record['distance_m']=float(np.linalg.norm(positions[i]))
                writer.writerow(record)
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    titles=['Gap-filled, no smoothing','SG: 5 frames, order 2','SG: 7 frames, order 2','EMA: 0.03 s']
    colors=['tab:gray','tab:green','tab:blue','tab:orange']
    # Choose the largest local second difference with three available samples for a reproducible detail view.
    score=np.linalg.norm(np.diff(p,n=2,axis=0),axis=1)
    peak=int(np.nanargmax(score)+1) if np.isfinite(score).any() else int(np.flatnonzero(valid)[0])
    zoom=(max(times[0],times[peak]-1.5),min(times[-1],times[peak]+1.5))
    report['detail_interval_s']=list(zoom)
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for filename,limits in [('comparison.png',None),('detail.png',zoom)]:
        fig,axes=plt.subplots(3,4,figsize=(19,9),sharex=True,sharey='row')
        for col,((name,positions),title,color) in enumerate(zip(variants.items(),titles,colors)):
            for axis_index in range(3):
                ax=axes[axis_index,col]
                ax.plot(times,p[:,axis_index]*100,color='0.7',linewidth=.8,label='Gap-filled baseline')
                ax.plot(times,positions[:,axis_index]*100,color=color,linewidth=1,label=title)
                ax.grid(alpha=.25)
                if col==0:
                    ax.set_ylabel('xyz'[axis_index]+' (cm)')
                if axis_index==0:
                    ax.set_title(title)
                if axis_index==2:
                    ax.set_xlabel('Time (s)')
                if limits:
                    ax.set_xlim(*limits)
        fig.suptitle(source.parent.parent.name.replace('_',' ')+' - gentle smoothing; 10-frame gap fill')
        fig.tight_layout()
        fig.savefig(output/filename,dpi=150)
        plt.close(fig)
    print(str(output),json.dumps(report['variants']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory',type=Path)
    run(parser.parse_args().trajectory)

