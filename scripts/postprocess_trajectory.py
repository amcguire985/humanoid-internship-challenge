"""Offline gap filling and EMA/SLERP smoothing of relative AprilTag poses."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def interpolate_rotation(a, b, fraction):
    return Slerp([0, 1], Rotation.from_quat([a, b]))([fraction]).as_quat()[0]


def fill_gaps(times, positions, quaternions, max_frames=3, max_span=0.15):
    """Fill bounded gaps only; max_span includes both observed endpoints."""
    p, q = positions.copy(), quaternions.copy()
    valid = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    labels = np.where(valid, 'observed', 'missing').astype(object)
    anchors = np.flatnonzero(valid)
    for left, right in zip(anchors[:-1], anchors[1:]):
        if 0 < right-left-1 <= max_frames and times[right]-times[left] <= max_span:
            for i in range(left+1, right):
                fraction = (times[i]-times[left])/(times[right]-times[left])
                p[i] = (1-fraction)*p[left]+fraction*p[right]
                q[i] = interpolate_rotation(q[left], q[right], fraction)
                labels[i] = 'interpolated'
    return p, q, labels


def smooth(times, positions, quaternions, tau, max_span=0.15):
    p, q = positions.copy(), quaternions.copy()
    for i in range(1, len(times)):
        if (not np.isfinite(p[i-1:i+1]).all() or
                not np.isfinite(q[i-1:i+1]).all() or times[i]-times[i-1] > max_span):
            continue
        alpha = -np.expm1(-(times[i]-times[i-1])/tau)
        p[i] = (1-alpha)*p[i-1]+alpha*positions[i]
        q[i] = interpolate_rotation(q[i-1], quaternions[i], alpha)
    return p, q


def metrics(times, p, q, observed, known_distance=None, max_span=0.15):
    """Compare jitter on the same observed frames; derivatives never cross gaps."""
    valid = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    selected = valid & observed
    result = {'available_frames': int(valid.sum()), 'observed_frames_scored': int(selected.sum())}
    if not selected.any():
        return result
    xyz = p[selected]
    rotations = Rotation.from_quat(q[selected])
    angles = (rotations.mean().inv()*rotations).magnitude()
    result.update(position_std_mm=(xyz.std(axis=0)*1000).tolist(),
                  mean_position_m=xyz.mean(axis=0).tolist(),
                  orientation_rms_deg=float(np.rad2deg(np.sqrt(np.mean(angles**2)))))
    if known_distance is not None:
        result['distance_rmse_mm'] = float(np.sqrt(np.mean((np.linalg.norm(xyz,axis=1)-known_distance)**2))*1000)
    velocity = np.full((len(times)-1,3),np.nan)
    angular = []
    for i in range(1,len(times)):
        dt = times[i]-times[i-1]
        if valid[i-1] and valid[i] and dt <= max_span:
            velocity[i-1] = (p[i]-p[i-1])/dt
            angular.append((Rotation.from_quat(q[i-1]).inv()*Rotation.from_quat(q[i])).magnitude()/dt)
    good = np.isfinite(velocity).all(axis=1)
    if good.any():
        result['rms_speed_m_s'] = float(np.sqrt(np.mean(np.sum(velocity[good]**2,axis=1))))
        result['rms_angular_speed_deg_s'] = float(np.rad2deg(np.sqrt(np.mean(np.array(angular)**2))))
    adjacent = good[:-1] & good[1:]
    if adjacent.any():
        midpoint_dt = (times[2:]-times[:-2])/2
        acc = np.diff(velocity,axis=0)[adjacent]/midpoint_dt[adjacent,None]
        result['rms_acceleration_m_s2'] = float(np.sqrt(np.mean(np.sum(acc**2,axis=1))))
    return result


def run(source, output, taus, max_frames, max_span, known_distance):
    with source.open(newline='',encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if rows and 'relative_rx_rad' not in rows[0]:
        if taus:
            raise ValueError('Position-only centre trajectories require --no-smoothing')
        return run_positions(source, output, rows, max_frames, max_span)
    times = np.array([float(r['time_s']) for r in rows])
    if len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Need at least two finite, strictly increasing timestamps')
    p = np.full((len(rows),3),np.nan)
    q = np.full((len(rows),4),np.nan)
    for i,r in enumerate(rows):
        if r['status'] in ('tracked', 'inferred_hand'):
            p[i] = [float(r[a+'_m']) for a in 'xyz']
            q[i] = Rotation.from_rotvec([float(r['relative_r'+a+'_rad']) for a in 'xyz']).as_quat()
    observed = np.array([r["status"] == "tracked" for r in rows])
    inferred = np.array([r["status"] == "inferred_hand" for r in rows])
    filled_p, filled_q, labels = fill_gaps(times,p,q,max_frames,max_span)
    labels[inferred] = "inferred_hand"
    variants = {'raw':(p,q), 'interpolated':(filled_p,filled_q)}
    for tau in taus:
        variants[f'ema_{tau:g}s'] = smooth(times,filled_p,filled_q,tau,max_span)
    if output.resolve()==source.parent.resolve():
        raise ValueError('Use a separate output directory to preserve raw outputs')
    output.mkdir(parents=True,exist_ok=True)
    report = {'source':str(source), 'settings':{'tau_seconds':taus,'max_missing_frames':max_frames,
              'max_endpoint_span_s':max_span,'known_distance_m':known_distance},
              'frames':len(rows),'raw_pose_fraction':float(observed.mean()),
              'inferred_hand_frames':int(inferred.sum()),
              'interpolated_frames':int(np.sum(labels=='interpolated')),
              'processed_pose_fraction':float(np.mean(labels!='missing')),
              'notes':['Offline interpolation uses future observations.',
                       'Jitter metrics use original observed timestamps and assume static relative pose.',
                       'Smoothness uses adjacent available poses, including interpolation; it is not accuracy.'],
              'variants':{}}
    for name,(vp,vq) in variants.items():
        m = metrics(times,vp,vq,observed,known_distance,max_span)
        m['mean_shift_from_raw_mm'] = ((vp[observed].mean(axis=0)-p[observed].mean(axis=0))*1000).tolist() if observed.any() else None
        report['variants'][name] = m
        if name=='raw': continue
        with (output/(name+'.csv')).open('w',newline='',encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['frame','time_s','source_status','pose_source','x_m','y_m','z_m','qx','qy','qz','qw'])
            for i,r in enumerate(rows):
                values = [*vp[i],*vq[i]]
                writer.writerow([r['frame'],times[i],r['status'],labels[i],
                                 *[float(v) if np.isfinite(v) else '' for v in values]])
    # Deterministic holdouts measure reconstruction consistency, not ground truth.
    report['holdout_interpolation'] = {}
    for length in [1,2,3]:
        errors, angular_errors = [], []
        for left in range(0,len(rows)-length-1,length+3):
            right = left+length+1
            if not observed[left:right+1].all() or times[right]-times[left]>max_span: continue
            hp,hq = p[left:right+1].copy(),q[left:right+1].copy()
            hp[1:-1],hq[1:-1] = np.nan,np.nan
            hp,hq,_ = fill_gaps(times[left:right+1],hp,hq,max_frames,max_span)
            if not np.isfinite(hp).all(): continue
            errors.extend(np.sum((hp[1:-1]-p[left+1:right])**2,axis=1))
            angular_errors.extend((Rotation.from_quat(hq[1:-1]).inv()*Rotation.from_quat(q[left+1:right])).magnitude()**2)
        report['holdout_interpolation'][str(length)] = {'frames_scored':len(errors),
            'position_rmse_mm':float(np.sqrt(np.mean(errors))*1000) if errors else None,
            'orientation_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(angular_errors)))) if errors else None}
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(4,1,figsize=(12,10),sharex=True)
    anchor = Rotation.from_quat(q[np.flatnonzero(observed)[0]]) if observed.any() else Rotation.identity()
    for name,(vp,vq) in variants.items():
        for i in range(3): axes[i].plot(times,vp[:,i]*1000,label=name,linewidth=1,alpha=0.8)
        valid = np.isfinite(vq).all(axis=1)
        angles = np.full(len(times),np.nan)
        angles[valid] = np.rad2deg((anchor.inv()*Rotation.from_quat(vq[valid])).magnitude()) if valid.any() else []
        axes[3].plot(times,angles,label=name,linewidth=1)
    for axis,label in zip(axes,['x (mm)','y (mm)','z (mm)','Rotation from start (deg)']):
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    axes[0].legend(ncol=3)
    axes[0].set_title(source.parent.name + ' - raw and processed relative poses')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=160)
    plt.close(fig)
    print(f'{source.parent.name}: {observed.sum()}/{len(rows)} observed, {report["interpolated_frames"]} filled; saved {output}',flush=True)



def fill_position_gaps(times, positions, max_frames=10, max_span=0.4):
    """Compatibility interface: use the shared position-only gap filler."""
    from gap_fill_centers import fill_position_gaps as fill_positions
    filled, labels, _ = fill_positions(times, positions, max_frames, max_span)
    return filled, labels


def run_positions(source, output, rows, max_frames, max_span):
    times = np.array([float(r['time_s']) for r in rows])
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Need at least two finite, strictly increasing timestamps')
    if output.resolve() == source.parent.resolve():
        raise ValueError('Use a separate output directory to preserve raw outputs')
    positions = np.full((len(rows), 3), np.nan)
    for i, row in enumerate(rows):
        if row['status'] == 'tracked':
            positions[i] = [float(row[axis + '_m']) for axis in 'xyz']
    filled, labels = fill_position_gaps(times, positions, max_frames, max_span)
    output.mkdir(parents=True, exist_ok=True)
    fields = ['frame', 'time_s', 'source_status', 'pose_source', 'x_m', 'y_m', 'z_m']
    # Retain observed-source diagnostics; interpolated rows inherit no tag identity.
    diagnostics = [key for key in rows[0] if key not in ('frame', 'time_s', 'status', 'x_m', 'y_m', 'z_m', 'distance_m')]
    fields += ['distance_m', *diagnostics]
    processed = []
    for i, row in enumerate(rows):
        record = dict(frame=row['frame'], time_s=row['time_s'], source_status=row['status'], pose_source=labels[i])
        for j, axis in enumerate('xyz'):
            record[axis + '_m'] = float(filled[i, j]) if np.isfinite(filled[i, j]) else ''
        record['distance_m'] = float(np.linalg.norm(filled[i])) if np.isfinite(filled[i]).all() else ''
        record.update({key: row[key] if labels[i] == 'observed' else '' for key in diagnostics})
        processed.append(record)
    with (output / 'interpolated.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(processed)
    observed = labels == 'observed'
    report = dict(source=str(source), frames=len(rows), observed_frames=int(observed.sum()),
                  interpolated_frames=int(np.sum(labels == 'interpolated')),
                  remaining_missing_frames=int(np.sum(labels == 'missing')),
                  processed_pose_fraction=float(np.mean(labels != 'missing')),
                  settings=dict(max_missing_frames=max_frames, max_endpoint_span_s=max_span, smoothing=False),
                  notes=['Offline linear position interpolation uses bounding future observations.',
                         'Observed positions are unchanged. Leading, trailing and over-limit gaps remain blank.',
                         'No orientation is available in centre trajectories.'])
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for j, axis in enumerate(axes):
        axis.plot(times, filled[:, j] * 100, label='gap-filled (no smoothing)', linewidth=1)
        axis.plot(times, positions[:, j] * 100, label='raw', linewidth=1, alpha=.65)
        interpolated = labels == 'interpolated'
        axis.scatter(times[interpolated], filled[interpolated, j] * 100, color='orange', s=8, label='interpolated')
        axis.set_ylabel('xyz'[j] + ' (cm)')
        axis.grid(alpha=.3)
    axes[0].legend()
    axes[0].set_title(source.parent.name + ' ? raw and gap-filled positions')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)
    print(f"{source.parent.name}: {report['observed_frames']} observed, {report['interpolated_frames']} filled, {report['remaining_missing_frames']} missing; saved {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--tau',type=float,nargs='+',default=[0.03,0.05,0.10])
    parser.add_argument('--no-smoothing', action='store_true', help='Only fill gaps; preserve observed samples exactly')
    parser.add_argument('--max-gap-frames',type=int,default=3)
    parser.add_argument('--max-gap-span',type=float,default=0.15,help='Maximum seconds between bounding observations')
    parser.add_argument('--known-distance',type=float,help='Measured tag-centre distance in metres for static validation')
    a = parser.parse_args()
    if (any(not np.isfinite(t) or t<=0 for t in a.tau) or a.max_gap_frames<0 or
        not np.isfinite(a.max_gap_span) or a.max_gap_span<=0 or
        (a.known_distance is not None and (not np.isfinite(a.known_distance) or a.known_distance<=0))):
        parser.error('Time constants, span, and distance must be positive; gap frames must be nonnegative')
    run(a.trajectory,a.output or a.trajectory.parent/'postprocessed',[] if a.no_smoothing else a.tau,a.max_gap_frames,a.max_gap_span,a.known_distance)


if __name__=='__main__':
    main()
