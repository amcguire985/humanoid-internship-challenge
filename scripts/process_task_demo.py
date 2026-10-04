"""Convert cached world-relative AprilTag observations into a task-level demo.

No wrist poses, contact inference, robot replay or training. All lengths are metres.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from describe_object_trajectory import compute_velocity, motion_runs
from filter_trajectory import reject_spikes, segments
from gap_fill_centers import fill_position_gaps
from smooth_centers import smooth_positions

CONVENTION = ('W is ID0 centre: +X decoded corner0 to corner1; +Y corner3 to corner0; '
              '+Z out of printed face. Standard41 corners reordered [1,0,3,2] for OpenCV IPPE. '
              'W_T_tag = inverse(C_T_W) @ C_T_tag. World ID1 is registered to ID0. '
              'World axes are not gravity-calibrated. Object centre = face centre minus edge/2 '
              'along outward tag Z; common object orientation is unspecified.')
DEFAULTS = dict(goal_object_offset=[0., 0., 0.], goal_offset_frame='goal_tag',
                goal_offset_verified=False, goal_coplanar=False, goal_center_height=0., vertical_axis='z', vertical_sign=1,
                vertical_verified=False, calibration_verified=False,
                max_reprojection_error=3., max_speed=2., max_gap_frames=3,
                max_gap_span=.15, smoothing_window=7, stationary_speed=.03,
                vertical_speed=.03, min_motion_duration=.2, stationary_duration=.3,
                goal_radius=.05, max_goal_scatter=.02, min_coverage=.9, phase_overrides={})


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def validate_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown settings: {sorted(unknown)}')
    c = {**DEFAULTS, **config}
    if c['goal_offset_frame'] != 'goal_tag' or c['vertical_axis'] not in 'xyz' or len(c['vertical_axis']) != 1:
        raise ValueError('Offset frame must be goal_tag; select vertical x, y or z')
    if not isinstance(c['goal_coplanar'], bool) or not np.isfinite(c['goal_center_height']) or c['goal_center_height'] < 0:
        raise ValueError('goal_coplanar must be boolean; goal_center_height finite and nonnegative')
    if c['goal_coplanar'] and c['goal_object_offset'][2] != 0:
        raise ValueError('Coplanar goal-area offset must have z=0; use goal_center_height for object height')
    if c['vertical_sign'] not in (-1, 1):
        raise ValueError('Vertical sign must be -1 or 1')
    if np.asarray(c['goal_object_offset']).shape != (3,) or not np.isfinite(c['goal_object_offset']).all():
        raise ValueError('Goal offset must have three finite values')
    for key in ('max_reprojection_error', 'max_speed', 'max_gap_span', 'stationary_speed',
                'vertical_speed', 'min_motion_duration', 'stationary_duration', 'goal_radius', 'max_goal_scatter'):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not isinstance(c['max_gap_frames'], int) or c['max_gap_frames'] < 0:
        raise ValueError('max_gap_frames must be a nonnegative integer')
    w = c['smoothing_window']
    if not isinstance(w, int) or w < 3 or w % 2 != 1:
        raise ValueError('smoothing_window must be odd and >=3')
    if not 0 < c['min_coverage'] <= 1:
        raise ValueError('min_coverage must be in (0,1]')
    allowed = {'pickup_time', 'transport_start', 'transport_end', 'release_time'}
    if not isinstance(c['phase_overrides'], dict) or set(c['phase_overrides']) - allowed:
        raise ValueError('Unknown phase override')
    for value in c['phase_overrides'].values():
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError('Phase overrides must be null or finite nonnegative seconds')
    return c


def clean_trajectory(times, raw, errors, c):
    """Reuse isolated position spike rejection, bounded filling and local quadratic SG."""
    p = raw.copy()
    error_rejected = np.isfinite(p).all(axis=1) & (~np.isfinite(errors) | (errors > c['max_reprojection_error']))
    p[error_rejected] = np.nan
    # Identity quaternions only adapt the existing position/rotation API; no orientation is inferred.
    q = np.tile([0., 0., 0., 1.], (len(times), 1))
    p, _, spikes, reasons = reject_spikes(times, p, q, speed=c['max_speed'], max_span=c['max_gap_span'])
    reasons[error_rejected] = 'reprojection_error'
    filled, provenance, gaps = fill_position_gaps(times, p, c['max_gap_frames'], c['max_gap_span'])
    smoothed = smooth_positions(times, filled, 'savgol', c['smoothing_window'], max_span=c['max_gap_span'])
    return p, smoothed, provenance, error_rejected | spikes, reasons, gaps


def compute_goal_position(rows, offset, max_error=3., coplanar=False, center_height=0., anchor_before=None):
    """Register a fixed target, optionally enforcing its known world-XY plane.

    In coplanar mode only yaw and XY translation are measured; tilt and Z drift
    cannot move the goal off W's z=0 plane. Offset locates the surface region;
    center_height locates the desired block centre above that surface.
    """
    candidates, surface_points, residuals = [], [], []
    for row in rows:
        if (row['status'] != 'tracked' or not row.get('x_m') or not row.get('error_px')
                or not np.isfinite(float(row['error_px'])) or float(row['error_px']) > max_error):
            continue
        if anchor_before is not None and float(row['time_s']) >= anchor_before:
            continue
        p = np.array([float(row[a + '_m']) for a in 'xyz'])
        rv = np.array([float(row['relative_r' + a + '_rad']) for a in 'xyz'])
        if not np.isfinite([*p, *rv]).all():
            continue
        rotation = cv2.Rodrigues(rv)[0]
        if coplanar:
            # Nearest proper in-plane rotation uses both measured in-plane axes.
            yaw = np.arctan2(rotation[1, 0]-rotation[0, 1], rotation[0, 0]+rotation[1, 1])
            co, si = np.cos(yaw), np.sin(yaw)
            residuals.append(dict(z=float(p[2]), tilt=float(np.arccos(np.clip(rotation[2,2],-1,1)))))
            rotation = np.array([[co,-si,0.],[si,co,0.],[0.,0.,1.]])
            p[2] = 0.
        surface = p + rotation @ np.asarray(offset)
        surface_points.append(surface)
        candidates.append(surface + rotation @ np.array([0.,0.,center_height]))
    if not candidates:
        raise ValueError('No accepted world-relative goal poses in the anchor interval; cannot anchor the demonstration')
    candidates = np.array(candidates)
    goal = np.median(candidates, axis=0)
    distances = np.linalg.norm(candidates - goal, axis=1)
    info = dict(accepted_samples=len(candidates),
                method='Fixed componentwise median of transformed goal observations; yaw-only / z=0 projection' if coplanar
                       else 'Componentwise median of transformed fixed-goal observations',
                coplanar_with_world=coplanar, goal_area_center=np.median(surface_points,axis=0).tolist(),
                desired_object_center_height_m=center_height, anchor_before_time=anchor_before,
                stationary_during_manipulation=True,
                scatter_p95_m=float(np.percentile(distances, 95)), max_scatter_m=float(distances.max()))
    if residuals:
        info.update(raw_plane_z_median_m=float(np.median([r['z'] for r in residuals])),
                    raw_plane_tilt_median_deg=float(np.rad2deg(np.median([r['tilt'] for r in residuals]))))
    return goal, info


def goal_planar_distance(positions, goal, c):
    """Centre-based circular region membership; height is reported separately."""
    horizontal = [0, 1] if c['goal_coplanar'] else [j for j in range(3) if j != 'xyz'.index(c['vertical_axis'])]
    return np.linalg.norm((np.asarray(positions)-goal)[...,horizontal],axis=-1)


def trajectory_velocity(t, p, max_span):
    v = np.full_like(p, np.nan)
    speed = np.full(len(t), np.nan)
    for ids in segments(t, np.isfinite(p).all(axis=1), max_span):
        if len(ids) >= 2:
            v[ids], speed[ids] = compute_velocity(t[ids], p[ids], np.r_[True, np.zeros(len(ids)-1, dtype=bool)])
    return v, speed


def segment_task_phases(t, p, goal, c):
    """Sustained object movement with quiet brackets; signed height velocity labels lift/lower."""
    v, speed = trajectory_velocity(t, p, c['max_gap_span'])
    valid = np.isfinite(p).all(axis=1)
    breaks = np.r_[True, (~valid[:-1]) | (~valid[1:]) | (np.diff(t) > c['max_gap_span'])]
    moving = np.isfinite(speed) & (speed > c['stationary_speed'])
    quiet = np.isfinite(speed) & ~moving
    states = np.where(moving, 'moving', np.where(quiet, 'quiet', 'invalid'))
    runs = motion_runs(t, states, breaks)
    sustained = [r for r in runs if r['phase'] == 'moving' and r['end_time_s']-r['start_time_s'] >= c['min_motion_duration']]
    bounds = dict(pickup_time=None, transport_start=None, transport_end=None, release_time=None)
    notes = []
    if sustained:
        first, last = sustained[0]['start_index'], sustained[-1]['end_index']
        bounds['pickup_time'], bounds['release_time'] = float(t[first]), float(t[last])
        before = [r for r in runs if r['phase'] == 'quiet' and r['end_index'] == first-1
                  and not breaks[first] and r['end_time_s']-r['start_time_s'] >= c['stationary_duration']]
        after = [r for r in runs if r['phase'] == 'quiet' and r['start_index'] == last+1
                 and not breaks[last+1] and r['end_time_s']-r['start_time_s'] >= c['stationary_duration']]
        if not before:
            notes.append('Pickup lacks a sustained adjacent stationary period; estimated motion onset only.')
        if not after:
            notes.append('Release lacks a sustained adjacent stationary period; estimated motion end only.')
        else:
            bounds['release_time'] = float(t[last+1])
            if goal_planar_distance(p[last+1],goal,c) > c['goal_radius']:
                notes.append('Post-motion object is outside the goal radius; placement is uncertain.')
        axis = 'xyz'.index(c['vertical_axis'])
        vz = v[:, axis] * c['vertical_sign']
        active = (np.arange(len(t)) >= first) & (np.arange(len(t)) <= last) & moving
        lifts = np.flatnonzero(active & (vz > c['vertical_speed']))
        lowers = np.flatnonzero(active & (vz < -c['vertical_speed']))
        # Only the initial lift and final lowering define transport bounds; intermediate reversals stay transport.
        early_lifts = lifts[t[lifts] <= t[first] + max(.5, .25*(t[last]-t[first]))]
        late_lowers = lowers[t[lowers] >= t[first] + .5*(t[last]-t[first])]
        bounds['transport_start'] = float(t[early_lifts[-1]]) if len(early_lifts) else float(t[first])
        bounds['transport_end'] = float(t[late_lowers[0]]) if len(late_lowers) else float(t[last])
        if not len(early_lifts): notes.append('Initial lift not distinguished from transport.')
        if not len(late_lowers): notes.append('Final lowering not distinguished from transport.')
        elif goal_planar_distance(p[late_lowers[0]],goal,c) > c['goal_radius']:
            notes.append('Lowering begins away from the goal region; placement label is based on height motion only.')
    else:
        notes.append('No sustained object motion detected; task timestamps unavailable.')
    estimated = bounds.copy()
    bounds.update(c['phase_overrides'])
    known = [bounds[k] for k in ('pickup_time', 'transport_start', 'transport_end', 'release_time') if bounds[k] is not None]
    if any(value < t[0] or value > t[-1] for value in known) or any(b < a for a, b in zip(known[:-1], known[1:])):
        raise ValueError('Phase timestamps must be ordered pickup <= transport_start <= transport_end <= release within recording')
    labels = np.full(len(t), 'stationary', dtype=object)
    pickup, ts, te, release = [bounds[k] for k in ('pickup_time', 'transport_start', 'transport_end', 'release_time')]
    if pickup is not None and release is not None:
        labels[t < pickup] = 'stationary_pre_pickup'
        labels[(t >= pickup) & (t <= release)] = 'transport'
        labels[t > release] = 'stationary_post_release'
        if ts is not None: labels[(t >= pickup) & (t < ts)] = 'pickup_lift'
        if te is not None: labels[(t > te) & (t <= release)] = 'placement_lowering'
        labels[((t < pickup) | (t > release)) & moving] = 'unclassified_motion'
    labels[~valid] = 'invalid'
    progress = np.full(len(t), np.nan)
    if ts is not None and te is not None and te > ts:
        mask = valid & (t >= ts) & (t <= te)
        progress[mask] = (t[mask]-ts)/(te-ts)
    return labels, speed, bounds, estimated, progress, notes


def missing_gap_metrics(t, valid):
    ids = np.flatnonzero(~valid)
    gaps = np.split(ids, np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []
    records = []
    for ids in gaps:
        a, b = int(ids[0]), int(ids[-1])
        span = t[min(b+1, len(t)-1)]-t[max(a-1, 0)]
        records.append(dict(first_index=a, last_index=b, frames=len(ids), span_s=float(span)))
    return dict(longest_missing_gap_frames=max((r['frames'] for r in records), default=0),
                longest_missing_gap_s=max((r['span_s'] for r in records), default=0.), gaps=records)


def compute_task_metrics(t, p, goal, speed, bounds, c):
    ids = np.flatnonzero(np.isfinite(p).all(axis=1))
    if not len(ids): raise ValueError('No object positions retained after cleaning')
    start, final = p[ids[0]], p[ids[-1]]
    ts, te = bounds['transport_start'], bounds['transport_end']
    transport = np.zeros(len(t), dtype=bool) if ts is None or te is None else ((t>=ts)&(t<=te))
    valid = np.isfinite(p).all(axis=1)
    edge = valid[:-1]&valid[1:]&transport[:-1]&transport[1:]&(np.diff(t)<=c['max_gap_span'])
    distances = np.linalg.norm(np.diff(p, axis=0), axis=1)[edge]
    durations = np.diff(t)[edge]
    axis = 'xyz'.index(c['vertical_axis'])
    lift_ids = valid & transport
    lift = float(max(0., np.max((p[lift_ids, axis]-start[axis])*c['vertical_sign']))) if lift_ids.any() else None
    pickup, release = bounds['pickup_time'], bounds['release_time']
    manipulation = np.zeros(len(t), dtype=bool) if pickup is None or release is None else ((t>=pickup)&(t<=release))
    manipulation_edges = valid[:-1]&valid[1:]&manipulation[:-1]&manipulation[1:]&(np.diff(t)<=c['max_gap_span'])
    manipulation_path = np.linalg.norm(np.diff(p,axis=0),axis=1)[manipulation_edges].sum()
    return dict(pickup_to_release_path_length=float(manipulation_path) if pickup is not None and release is not None else None,
                pickup_to_release_duration=float(release-pickup) if pickup is not None and release is not None else None,
                object_start_position=start.tolist(), goal_position=goal.tolist(), final_object_position=final.tolist(),
                start_sample_time=float(t[ids[0]]), final_sample_time=float(t[ids[-1]]),
                task_displacement=(goal-start).tolist(), start_to_goal_distance=float(np.linalg.norm(goal-start)),
                final_placement_error=float(np.linalg.norm(final-goal)),
                final_planar_placement_error=float(goal_planar_distance(final,goal,c)),
                final_inside_goal_region=bool(goal_planar_distance(final,goal,c)<=c['goal_radius']),
                final_distance_outside_goal_region=float(max(0.,goal_planar_distance(final,goal,c)-c['goal_radius'])),
                final_goal_height_error=float(abs(final[2]-goal[2])) if c['goal_coplanar'] else None,
                max_lift_height=lift,
                max_lift_height_entire_clip=float(max(0., np.nanmax((p[:,axis]-start[axis])*c['vertical_sign']))),
                transport_path_length=float(distances.sum()) if ts is not None and te is not None else None,
                transport_duration=float(te-ts) if ts is not None and te is not None else None,
                transport_observed_duration=float(durations.sum()),
                average_transport_speed=float(distances.sum()/durations.sum()) if durations.sum()>0 else None,
                maximum_transport_speed=float(np.max(distances/durations)) if len(distances) else None,
                transport_complete=bool(ts is not None and te is not None and te>ts and
                                        transport.sum()>=2 and np.isfinite(p[transport]).all() and
                                        np.all(np.diff(t[transport])<=c['max_gap_span'])))


def export_processed_demo(output, t, raw, cleaned, p, goal, labels, speed, progress, provenance, rejected, reasons, metrics, bounds, config=None):
    c=validate_config(config or {})
    start = np.array(metrics['object_start_position'])
    fields = ['time', *['object_'+a for a in 'xyz'], *['object_rel_start_'+a for a in 'xyz'],
              *['object_rel_goal_'+a for a in 'xyz'], 'speed', 'phase', 'valid_measurement',
              'valid_processed', 'goal_planar_distance', 'inside_goal_region', 'interpolated', 'outlier_rejected', 'rejection_reason', 'transport_progress',
              *['raw_object_'+a for a in 'xyz'], *['goal_'+a for a in 'xyz']]
    def finite(value): return float(value) if np.isfinite(value) else ''
    records = []
    for i in range(len(t)):
        radial=goal_planar_distance(p[i],goal,c)
        r = dict(time=float(t[i]), speed=finite(speed[i]), phase=labels[i],
                 goal_planar_distance=finite(radial), inside_goal_region=int(radial<=c['goal_radius']) if np.isfinite(radial) else '',
                 valid_measurement=int(np.isfinite(raw[i]).all() and not rejected[i]),
                 valid_processed=int(np.isfinite(p[i]).all()), interpolated=int(provenance[i]=='interpolated'),
                 outlier_rejected=int(rejected[i]), rejection_reason=reasons[i], transport_progress=finite(progress[i]))
        for prefix, values in [('object_',p[i]), ('object_rel_start_',p[i]-start),
                               ('object_rel_goal_',p[i]-goal), ('raw_object_',raw[i]), ('goal_',goal)]:
            r.update({prefix+a:finite(values[j]) for j,a in enumerate('xyz')})
        records.append(r)
    pickup, release = bounds['pickup_time'], bounds['release_time']
    manipulation = [r for r in records if pickup is not None and release is not None and pickup <= r['time'] <= release]
    for name, selected in [('processed_demo.csv',records), ('transport.csv',manipulation)]:
        with (output/name).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(selected)
    for name, values in [('raw_trajectory.csv',raw), ('cleaned_trajectory.csv',cleaned)]:
        with (output/name).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle); writer.writerow(['time','x_world','y_world','z_world'])
            writer.writerows([float(t[i]),*[finite(v) for v in values[i]]] for i in range(len(t)))


def plot_demo(output, t, raw, p, goal, speed, labels, bounds, c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids = np.flatnonzero(np.isfinite(p).all(axis=1))
    groups = segments(t, np.isfinite(p).all(axis=1), c['max_gap_span'])
    fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection='3d')
    for group in groups: ax.plot(*p[group].T,color='tab:blue')
    for point,color,label in [(p[ids[0]],'green','Start'),(goal,'orange','Goal'),(p[ids[-1]],'red','Final')]:
        ax.scatter(*point,color=color,label=label,s=65)
    ax.set(xlabel='World x (m)',ylabel='World y (m)',zlabel='World z (m)',title='Cleaned object-centre trajectory')
    theta=np.linspace(0,2*np.pi,129)
    if c['goal_coplanar']:
        ax.plot(goal[0]+c['goal_radius']*np.cos(theta),goal[1]+c['goal_radius']*np.sin(theta),
                np.zeros_like(theta),color='orange',label='Goal area (table plane)')
    extent=np.ptp(np.vstack((p[ids],goal)),axis=0); ax.set_box_aspect(np.maximum(extent,.03))
    ax.legend(); fig.tight_layout(); fig.savefig(output/'trajectory_3d.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for j,ax in enumerate(axes):
        ax.plot(t,raw[:,j],color='gray',alpha=.4,label='Raw')
        for n,group in enumerate(groups): ax.plot(t[group],p[group,j],color='tab:blue',label=f"Cleaned / SG{c['smoothing_window']}" if n==0 else None)
        ax.axhline(goal[j],color='orange',linestyle='--',label='Goal'); ax.set_ylabel('xyz'[j]+' (m)'); ax.grid(alpha=.3)
    axes[0].legend(); axes[-1].set_xlabel('Time (s)'); fig.tight_layout(); fig.savefig(output/'position_time.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5))
    for group in groups: ax.plot(t[group],speed[group],color='black',linewidth=.9)
    ax.axhline(c['stationary_speed'],linestyle='--',color='gray',label='Stationary threshold')
    for key,value in bounds.items():
        if value is not None: ax.axvline(value,label=f'{key}: {value:.2f}s',alpha=.7)
    ax.set(xlabel='Time (s)',ylabel='Speed (m/s)',title='Approximate object-motion boundaries'); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(output/'speed_phases.png',dpi=160); plt.close(fig)
    horizontal=[j for j in range(3) if j!='xyz'.index(c['vertical_axis'])]
    fig,ax=plt.subplots(figsize=(8,7))
    for group in groups: ax.plot(p[group,horizontal[0]],p[group,horizontal[1]],color='tab:blue')
    for point,color,label in [(p[ids[0]],'green','Start'),(goal,'orange','Goal'),(p[ids[-1]],'red','Final')]:
        ax.scatter(*point[horizontal],color=color,label=label,s=65)
    from matplotlib.patches import Circle
    ax.add_patch(Circle(goal[horizontal],c['goal_radius'],fill=False,color='orange',label='Goal area'))
    ax.set(xlabel='xyz'[horizontal[0]]+' (m)',ylabel='xyz'[horizontal[1]]+' (m)',title='Top-down projection (configured height axis)'); ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(output/'top_down.png',dpi=160); plt.close(fig)


def run(source, output, config):
    c=validate_config(config)
    source,output=Path(source),Path(output)
    summary=json.loads((source/'summary.json').read_text(encoding='utf-8-sig'))
    groups=summary.get('tag_groups',{})
    if len(groups.get('target',[]))!=1: raise ValueError('Tracking run must configure exactly one target tag')
    goal_id=groups['target'][0]
    object_path=source/'object/cube_center/trajectory.csv'
    goal_rows=read_csv(source/f'target/id{goal_id}/trajectory.csv')
    rows=read_csv(object_path)
    t=np.array([float(r['time_s']) for r in rows])
    if len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0): raise ValueError('Need finite increasing frame timestamps')
    if len(rows)!=summary['frames_processed'] or len(goal_rows)!=len(rows): raise ValueError('Mismatched tracking frame counts')
    if any(int(r['frame'])!=i or int(goal_rows[i]['frame'])!=i or abs(float(goal_rows[i]['time_s'])-t[i])>1e-6 for i,r in enumerate(rows)):
        raise ValueError('Object and goal frames/timestamps must align')
    if any((output/name).resolve()==object_path.resolve() for name in ('raw_trajectory.csv','processed_demo.csv')) or output.resolve()==source.resolve():
        raise ValueError('Use a separate processed output directory')
    raw=np.array([[float(r[a+'_m']) if r.get(a+'_m') else np.nan for a in 'xyz'] for r in rows])
    errors=np.array([float(r['error_px']) if r.get('error_px') else np.nan for r in rows])
    cleaned,p,provenance,rejected,reasons,gaps=clean_trajectory(t,raw,errors,c)
    goal,goal_info=compute_goal_position(goal_rows,c['goal_object_offset'],c['max_reprojection_error'],
                                         c['goal_coplanar'],c['goal_center_height'])
    if c['goal_coplanar']:
        # Use pre-pickup observations to lock geometry before the block moves.
        _,_,seed_bounds,_,_,_=segment_task_phases(t,p,goal,c)
        pickup=seed_bounds['pickup_time']
        if pickup is not None:
            try:
                goal,goal_info=compute_goal_position(goal_rows,c['goal_object_offset'],c['max_reprojection_error'],
                                                     True,c['goal_center_height'],anchor_before=pickup)
                goal_info['anchor_source']='pre_pickup'
            except ValueError:
                release=seed_bounds['release_time']
                active_rows=[r for r in goal_rows if pickup<=float(r['time_s'])<=release]
                goal,goal_info=compute_goal_position(active_rows,c['goal_object_offset'],c['max_reprojection_error'],
                                                     True,c['goal_center_height'])
                goal_info['anchor_source']='stationary_goal_during_manipulation'
    labels,speed,bounds,estimated,progress,notes=segment_task_phases(t,p,goal,c)
    metrics=compute_task_metrics(t,p,goal,speed,bounds,c)
    detections=read_csv(source/'detections.csv')
    rates={}
    group_gaps={}
    for group in ('world','object','target'):
        frame_ids={int(r['frame']) for r in detections if int(r['tag_id']) in groups.get(group,[])}
        name='goal' if group=='target' else group
        rates[name]=len(frame_ids)/len(t)
        group_gaps[name]=missing_gap_metrics(t,np.array([i in frame_ids for i in range(len(t))]))
    raw_valid=np.isfinite(raw).all(axis=1); retained=np.isfinite(cleaned).all(axis=1); valid=np.isfinite(p).all(axis=1)
    per_id={str(tag):len({int(r['frame']) for r in detections if int(r['tag_id'])==tag})/len(t)
            for ids in groups.values() for tag in ids}
    quality=dict(detection_rates=rates, detection_rates_by_id=per_id, detection_missing_gaps=group_gaps, raw_world_relative_object_rate=float(raw_valid.mean()),
                 retained_measurement_percentage=100*float(retained.sum()/raw_valid.sum()) if raw_valid.any() else 0.,
                 processed_coverage_percentage=100*float(valid.mean()), rejected_frames=int(rejected.sum()),
                 interpolated_frames=int(np.sum(provenance=='interpolated')),
                 raw_missing=missing_gap_metrics(t,raw_valid), filtered_missing=missing_gap_metrics(t,valid),
                 goal_estimation=goal_info)
    flags=notes.copy()
    if not c['calibration_verified']: flags.append('Landscape still-photo calibration / crop has not been validated for this video recording mode.')
    if not c['vertical_verified']: flags.append('Height axis is a configured proxy, not a verified gravity direction.')
    if not c['goal_offset_verified']: flags.append('Desired centre offset from goal tag is a geometry assumption.')
    if valid.mean()<c['min_coverage']: flags.append('Processed object coverage below configured minimum.')
    if not metrics['transport_complete']: flags.append('Transport interval is unavailable or contains missing data/time gaps.')
    if goal_info['scatter_p95_m']>c['max_goal_scatter']: flags.append('Fixed-goal observation scatter exceeds configured tolerance.')
    if not metrics['final_inside_goal_region']: flags.append('Final position lies outside configured goal radius.')
    quality['review_flags']=flags
    quality['usable_for_retargeting']=not flags
    report={**metrics,**bounds, 'automatic_phase_estimates':estimated, 'quality':quality,
            'coordinate_frame_convention':CONVENTION,'parameters':c,'interpolation_gaps':gaps,
            'goal_region':dict(shape='circle',radius_m=c['goal_radius'],diameter_m=2*c['goal_radius'],
                               center_world=goal_info['goal_area_center'],center_offset_goal_tag=c['goal_object_offset'],
                               membership='Projected block centre within radius; no footprint or contact test',
                               plane='W z=0' if c['goal_coplanar'] else 'Configured horizontal axes'),
            'tracking_configuration': {key:summary.get(key) for key in ('tag_groups','family','pose_tag_sizes_m','world_registration','registered_world_ids','effective_camera_matrix','distortion_coefficients','cube_center')},
            'sources':dict(tracking=str(source),object=str(object_path),video=summary['video'],calibration=summary['calibration']),
            'notes':['Raw inputs preserved. Object positions are cube centres, goal is a fixed robust anchor.',
                     'Goal offset locates the surface circle in goal-tag coordinates; goal_center_height separately locates the desired block centre. Coplanar mode enforces W z=0 and yaw-only orientation and locks the anchor before pickup.',
                     'valid_measurement means accepted measured input; object columns may be smoothed. Raw columns preserve measurements.',
                     'Pickup/release are object-motion estimates, not grasp/contact detections; override in phase_overrides.',
                     'transport.csv covers pickup through release, including invalid rows. transport_progress normalizes the inner transport phase and is blank outside it or at unavailable samples.',
                     'transport_* metrics use transport_start through transport_end; pickup_to_release_* cover the full carrying interval. Paths/speeds never bridge missing samples; duration includes gaps.',
                     'Lift values are relative to first retained object position and the configured signed height axis.']}
    output.mkdir(parents=True,exist_ok=True)
    export_processed_demo(output,t,raw,cleaned,p,goal,labels,speed,progress,provenance,rejected,reasons,metrics,bounds,c)
    (output/'metadata.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plot_demo(output,t,raw,p,goal,speed,labels,bounds,c)
    print(json.dumps({**metrics,**bounds,'quality':quality},indent=2,allow_nan=False),flush=True)
    print(f'Saved task demonstration to {output}',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tracking',type=Path)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    run(args.tracking,args.output or args.tracking/'task_demo',json.loads(args.config.read_text(encoding='utf-8-sig')))


if __name__=='__main__': main()
