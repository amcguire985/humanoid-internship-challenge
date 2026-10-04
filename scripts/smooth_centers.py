"""Gentle position-only smoothing of already gap-filled centre trajectories."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def smooth_positions(times, positions, method, strength, max_span=.15, shrink_window=False):
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (len(times), 3) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Need Nx3 positions and finite strictly increasing timestamps')
    if method not in ('ema', 'savgol') or not np.isfinite(strength) or strength <= 0:
        raise ValueError('Select ema or savgol with positive strength')
    if method == 'savgol' and (int(strength) != strength or int(strength) < 3 or int(strength) % 2 != 1):
        raise ValueError('Savitzky-Golay window must be an odd integer >=3')
    if max_span is None:
        max_span = 3 * np.median(np.diff(times))
    if not np.isfinite(max_span) or max_span <= 0:
        raise ValueError('Maximum adjacent timestamp span must be finite and positive')
    result = positions.copy()
    valid = np.flatnonzero(np.isfinite(positions).all(axis=1))
    cuts = np.flatnonzero((np.diff(valid) > 1) | (np.diff(times[valid]) > max_span)) + 1
    for ids in np.split(valid, cuts):
        if len(ids) < 2:
            continue
        if method == 'ema':
            for previous, current in zip(ids[:-1], ids[1:]):
                alpha = -np.expm1(-(times[current] - times[previous]) / strength)
                result[current] = (1-alpha)*result[previous] + alpha*positions[current]
        else:
            width = int(strength)
            if shrink_window:
                width = min(width, len(ids) if len(ids) % 2 else len(ids) - 1)
            if width < 3 or len(ids) < width:
                continue
            for i, current in enumerate(ids):
                start = min(max(i-width//2, 0), len(ids)-width)
                window = ids[start:start+width]
                offsets = times[window] - times[current]
                # Local quadratic least squares equals SG on uniform timestamps,
                # and handles decoder timing directly without resampling.
                offsets /= np.max(np.abs(offsets))
                design = np.column_stack((np.ones(width), offsets, offsets**2))
                result[current] = np.linalg.lstsq(design, positions[window], rcond=None)[0][0]
    return result


def run(source, output):
    from track_apriltags import plot_trajectory
    source, output = Path(source), Path(output)
    if source.parent.resolve() == output.resolve():
        raise ValueError('Use a separate output directory')
    with source.open(newline='',encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = list(reader)
    times = np.array([float(r['time_s']) for r in rows])
    p = np.array([[float(r[a+'_m']) if r.get(a+'_m') else np.nan for a in 'xyz'] for r in rows])
    valid = np.isfinite(p).all(axis=1)
    specs = {'savgol_5': ('savgol',5), 'savgol_7': ('savgol',7), 'ema_0.03s': ('ema',.03), 'ema_0.05s': ('ema',.05)}
    output.mkdir(parents=True,exist_ok=True)
    report = dict(source=str(source),frames=len(rows),available_frames=int(valid.sum()),
                  additional_gap_filling=False,outlier_rejection=False,max_adjacent_span_s=.15,
                  notes=['Position-only smoothing; provenance labels retained.',
                         'Filters reset at unfilled gaps and timestamp jumps above 0.15 seconds.',
                         'Quadratic local fits use actual timestamps; short segments pass through.',
                         'SG uses future samples; EMA is causal but its input gap filling uses future samples.',
                         'Displacement from input measures filter changes, not physical accuracy.'],variants={})
    variants = {'gap-filled':p}
    for name,(method,strength) in specs.items():
        filtered = smooth_positions(times,p,method,strength)
        variants[name] = filtered
        difference = np.linalg.norm(filtered[valid]-p[valid],axis=1)*1000
        observed = valid & np.array([r.get('pose_source')=='observed' for r in rows])
        report['variants'][name] = dict(method=method,strength=strength,
             rms_change_mm=float(np.sqrt(np.mean(difference**2))) if len(difference) else None,
             max_change_mm=float(np.max(difference)) if len(difference) else None,
             observed_rms_change_mm=float(np.sqrt(np.mean(np.sum((filtered[observed]-p[observed])**2,axis=1)))*1000) if observed.any() else None)
        records=[]
        for i,r in enumerate(rows):
            record=dict(r,smoothing_method=name)
            if valid[i]:
                record.update({a+'_m':float(filtered[i,j]) for j,a in enumerate('xyz')})
                record['distance_m']=float(np.linalg.norm(filtered[i]))
            records.append(record)
        with (output/(name+'.csv')).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=[*fields,'smoothing_method'])
            writer.writeheader()
            writer.writerows(records)
        destination=output/name
        destination.mkdir(exist_ok=True)
        plotting=[dict(r,time_s=times[i],**{a+'_m':filtered[i,j] for j,a in enumerate('xyz')}) for i,r in enumerate(records)]
        plot_trajectory(plotting,destination,source.parent.parent.name.replace('_',' ')+f' ({name})','ID0 world')
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True)
    for name,values in variants.items():
        for i,axis in enumerate(axes):
            axis.plot(times,values[:,i]*100,label=name,linewidth=.9,alpha=.8)
    for i,axis in enumerate(axes):
        axis.set_ylabel('xyz'[i]+' (cm)')
        axis.grid(alpha=.3)
    axes[0].legend(ncol=3)
    axes[0].set_title(source.parent.parent.name+': gentle smoothing comparison')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=160)
    plt.close(fig)
    print(json.dumps(report['variants'],indent=2),flush=True)
    print(f'Saved {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    run(args.trajectory,args.output or args.trajectory.parent/'gentle_smoothing')
