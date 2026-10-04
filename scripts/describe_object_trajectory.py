"""Describe measured object motion in task space; no robot control or frame mapping."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def load_trajectory(path):
    """Read metres/seconds, skip unavailable rows, and retain breaks between observations."""
    with Path(path).open(newline='',encoding='utf-8-sig') as handle:
        reader=csv.DictReader(handle)
        fields=reader.fieldnames or []
        keys=('time','x','y','z') if all(k in fields for k in ('time','x','y','z')) else ('time_s','x_m','y_m','z_m')
        if not all(k in fields for k in keys):
            raise ValueError('Expected time,x,y,z or time_s,x_m,y_m,z_m')
        rows=list(reader)
    times,positions,indices=[],[],[]
    for index,row in enumerate(rows):
        if any(not row[k].strip() for k in keys):
            continue
        values=np.array([float(row[k]) for k in keys])
        if not np.isfinite(values).all():
            continue
        times.append(values[0]); positions.append(values[1:]); indices.append(index)
    times=np.array(times); positions=np.array(positions)
    if len(times)<2 or np.any(np.diff(times)<=0):
        raise ValueError('Need at least two valid samples with strictly increasing timestamps')
    # Missing rows and unusually large timestamp gaps are not traversed.
    cadence=float(np.median(np.diff(times)))
    breaks=np.r_[True,(np.diff(indices)>1)|(np.diff(times)>3*cadence)]
    return times,positions,breaks,dict(input_rows=len(rows),valid_rows=len(times),skipped_rows=len(rows)-len(times),
                                    max_adjacent_span_s=3*cadence,segments=int(breaks.sum()))


def normalize_trajectory(positions):
    """Translate the first valid object position to the origin; axes stay unchanged."""
    return positions-positions[0]


def compute_velocity(times,positions,breaks):
    """Central finite differences inside segments; one-sided at ends, singleton unknown."""
    velocity=np.full_like(positions,np.nan)
    for ids in np.split(np.arange(len(times)),np.flatnonzero(breaks)[1:]):
        if len(ids)>=2:
            velocity[ids]=np.gradient(positions[ids],times[ids],axis=0)
    return velocity,np.linalg.norm(velocity,axis=1)


def motion_runs(times,labels,breaks):
    """Return contiguous phase intervals without crossing data gaps."""
    starts=np.r_[0,np.flatnonzero((labels[1:]!=labels[:-1])|breaks[1:])+1]
    ends=np.r_[starts[1:]-1,len(times)-1]
    return [dict(phase=str(labels[a]),start_time_s=float(times[a]),end_time_s=float(times[b]),
                 start_index=int(a),end_index=int(b)) for a,b in zip(starts,ends)]


def segment_motion(times,velocity,speed,breaks,stationary_speed=.03,vertical_speed=.03,
                   min_motion_duration=.2,vertical_axis=2):
    """Detect sustained moving bounds, then classify by speed and signed vertical velocity."""
    moving=np.isfinite(speed)&(speed>stationary_speed)
    runs=motion_runs(times,np.where(moving,'moving','quiet'),breaks)
    sustained=[r for r in runs if r['phase']=='moving' and r['end_time_s']-r['start_time_s']>=min_motion_duration]
    labels=np.full(len(times),'stationary',dtype=object)
    pickup=placement=None
    if sustained:
        first,last=sustained[0]['start_index'],sustained[-1]['end_index']
        labels[:first]='stationary_before_pickup'
        labels[first:last+1]='stationary_during_transport'
        labels[last+1:]='stationary_after_placement'
        active=moving.copy(); active[:first]=False; active[last+1:]=False
        labels[active]='horizontal_transport'
        labels[active&(velocity[:,vertical_axis]>vertical_speed)]='lift'
        labels[active&(velocity[:,vertical_axis]<-vertical_speed)]='lower'
        # Require visible quiet samples adjacent to each transition; a gap/end is inconclusive.
        if first>0 and not breaks[first] and np.isfinite(speed[first-1]) and not moving[first-1]:
            pickup=float(times[first])
        if last+1<len(times) and not breaks[last+1] and np.isfinite(speed[last+1]) and not moving[last+1]:
            placement=float(times[last+1])
    labels[~np.isfinite(speed)]='unknown'
    return labels,pickup,placement,motion_runs(times,labels,breaks)


def compute_metrics(times,positions,relative,breaks,pickup,placement,vertical_axis=2):
    """Path sums include only adjacent observations within a continuous segment."""
    horizontal=[a for a in range(3) if a!=vertical_axis]
    steps=np.diff(positions,axis=0)[~breaks[1:]]
    return dict(total_horizontal_displacement_m=float(np.linalg.norm(relative[-1,horizontal])),
                horizontal_path_length_m=float(np.linalg.norm(steps[:,horizontal],axis=1).sum()),
                maximum_vertical_lift_m=float(np.max(relative[:,vertical_axis])),
                total_3d_path_length_m=float(np.linalg.norm(steps,axis=1).sum()),
                trajectory_duration_s=float(times[-1]-times[0]),start_time_s=float(times[0]),end_time_s=float(times[-1]),
                start_position_m=positions[0].tolist(),end_position_m=positions[-1].tolist(),
                approximate_pickup_time_s=pickup,approximate_placement_time_s=placement)


def plot_results(times,positions,relative,speed,labels,breaks,runs,output):
    """Plot measured coordinates, relative 3D motion and speed with all phase boundaries."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    segments=np.split(np.arange(len(times)),np.flatnonzero(breaks)[1:])
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for a,ax in enumerate(axes):
        for ids in segments: ax.plot(times[ids],positions[ids,a])
        ax.set_ylabel('xyz'[a]+' (m)'); ax.grid(alpha=.3)
    axes[0].set_title('Input processed positions (before normalization)')
    axes[-1].set_xlabel('Time (s)'); fig.tight_layout(); fig.savefig(output/'raw_positions.png',dpi=160); plt.close(fig)
    fig=plt.figure(figsize=(9,8)); ax=fig.add_subplot(111,projection='3d')
    for ids in segments: ax.plot(*relative[ids].T,color='tab:blue')
    ax.scatter(*relative[0],color='green',s=70,label='Start'); ax.scatter(*relative[-1],color='red',s=70,label='End')
    ext=np.ptp(relative,axis=0); ax.set_box_aspect(np.maximum(ext,.01))
    ax.set(xlabel='x relative (m)',ylabel='y relative (m)',zlabel='z relative (m)',title='Object trajectory relative to initial position')
    ax.legend(); fig.tight_layout(); fig.savefig(output/'trajectory_3d.png',dpi=160); plt.close(fig)
    colors={'stationary':'gray','stationary_before_pickup':'gray','stationary_after_placement':'gray',
            'stationary_during_transport':'gold','lift':'green','horizontal_transport':'blue','lower':'red','unknown':'black'}
    fig,ax=plt.subplots(figsize=(15,6))
    for ids in segments: ax.plot(times[ids],speed[ids],color='black',linewidth=.8)
    seen=set()
    for run in runs:
        phase=run['phase']; start=run['start_time_s']; end=run['end_time_s']
        ax.axvspan(start,end,color=colors[phase],alpha=.16,label=phase if phase not in seen else None)
        ax.axvline(start,color=colors[phase],linewidth=.5,alpha=.5); seen.add(phase)
    ax.set(xlabel='Time (s)',ylabel='Speed (m/s)',title='Threshold-based motion phases')
    ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=3); fig.tight_layout(); fig.savefig(output/'speed_phases.png',dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--stationary-speed',type=float,default=.03,help='m/s; samples below this are quiet')
    parser.add_argument('--vertical-speed',type=float,default=.03,help='m/s; signed threshold for lift/lower')
    parser.add_argument('--min-motion-duration',type=float,default=.2,help='Seconds of continuous movement required for task bounds')
    parser.add_argument('--vertical-axis',choices=list('xyz'),default='z')
    args=parser.parse_args()
    if any(not np.isfinite(v) or v<=0 for v in (args.stationary_speed,args.vertical_speed,args.min_motion_duration)):
        parser.error('Thresholds must be finite and positive')
    times,positions,breaks,info=load_trajectory(args.trajectory)
    relative=normalize_trajectory(positions)
    velocity,speed=compute_velocity(times,positions,breaks)
    axis='xyz'.index(args.vertical_axis)
    labels,pickup,placement,runs=segment_motion(times,velocity,speed,breaks,args.stationary_speed,args.vertical_speed,args.min_motion_duration,axis)
    output=args.output or args.trajectory.parent/'task_space'
    if (output/'trajectory.csv').resolve()==args.trajectory.resolve():
        parser.error('Output must not overwrite the input CSV')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'trajectory.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle); writer.writerow(['time','x_rel','y_rel','z_rel','speed','phase'])
        for t,p,s,label in zip(times,relative,speed,labels): writer.writerow([t,*p,float(s) if np.isfinite(s) else '',label])
    report=compute_metrics(times,positions,relative,breaks,pickup,placement,axis)
    report.update(source=str(args.trajectory),loading=info,phase_intervals=runs,
                  thresholds=dict(stationary_speed_m_s=args.stationary_speed,vertical_speed_m_s=args.vertical_speed,
                                  min_motion_duration_s=args.min_motion_duration,vertical_axis=args.vertical_axis),
                  notes=['Threshold phases are approximate; no contact or grasp is inferred.',
                         'Horizontal displacement is net start-to-end displacement; horizontal path is reported separately.',
                         'Paths and velocities never connect missing-data segments. Duration includes unavailable intervals.',
                         'Stationary labels describe quiet samples outside sustained movement; brief noise bursts there are ignored.',
                         'World vertical must match the selected axis. No LIBERO mapping or robot control is applied.'])
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plot_results(times,positions,relative,speed,labels,breaks,runs,output)
    print(f'Saved {len(times)} samples and {len(runs)} phase intervals to {output}',flush=True)


if __name__=='__main__': main()
