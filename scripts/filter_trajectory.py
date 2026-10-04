"""Offline isolated-spike rejection, short-gap filling, and pose filtering."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter, savgol_coeffs, butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp
from postprocess_trajectory import fill_gaps


def valid_pose(p, q):
    return np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)


def reject_spikes(t, p, q, speed=2.0, angular_speed=720.0, max_span=0.15):
    """Reject isolated excursions, not persistent steps or sustained fast motion.

    Both incident speeds must exceed the limit and the neighbor-to-neighbor
    speed must be below it. Decisions use raw data in a single pass.
    """
    valid = valid_pose(p, q)
    rejected = np.zeros(len(t), dtype=bool)
    reasons = np.full(len(t), '', dtype=object)
    for i in range(1, len(t)-1):
        if not valid[i-1:i+2].all() or t[i+1]-t[i-1] > max_span:
            continue
        dt = np.diff(t[i-1:i+2])
        v = np.linalg.norm(np.diff(p[i-1:i+2], axis=0), axis=1)/dt
        bridge = np.linalg.norm(p[i+1]-p[i-1])/(t[i+1]-t[i-1])
        r = Rotation.from_quat(q[i-1:i+2])
        w = np.rad2deg((r[:-1].inv()*r[1:]).magnitude())/dt
        bridge_w = np.rad2deg((r[0].inv()*r[2]).magnitude())/(t[i+1]-t[i-1])
        flags = []
        if np.all(v > speed) and bridge <= speed: flags.append('position_spike')
        if np.all(w > angular_speed) and bridge_w <= angular_speed: flags.append('rotation_spike')
        rejected[i] = bool(flags)
        reasons[i] = '+'.join(flags)
    cleaned_p, cleaned_q = p.copy(), q.copy()
    cleaned_p[rejected] = np.nan
    cleaned_q[rejected] = np.nan
    return cleaned_p, cleaned_q, rejected, reasons


def segments(t, valid, max_span):
    indices = np.flatnonzero(valid)
    if not len(indices): return []
    cuts = np.flatnonzero((np.diff(indices)>1) | (np.diff(t[indices])>max_span))+1
    return np.split(indices, cuts)


def filter_segments(t, p, q, window=7, order=2, cutoff=None, max_span=0.15):
    """Resample each available segment uniformly, filter, then restore timestamps.

    SG rotations use a separate tangent space centered on each input rotation.
    Butterworth uses continuous-sign unit quaternions and renormalization.
    Never filter through unfilled gaps. Short segments pass through unchanged.
    """
    result_p, result_q = p.copy(), q.copy()
    notes = []
    for ids in segments(t, valid_pose(p,q), max_span):
        ts = t[ids]
        if len(ids)<3:
            notes.append({'start_frame':int(ids[0]),'reason':'fewer than three samples'})
            continue
        dt = np.median(np.diff(ts))
        grid = np.linspace(ts[0],ts[-1],max(3,int(round((ts[-1]-ts[0])/dt))+1))
        gp = np.column_stack([np.interp(grid,ts,p[ids,a]) for a in range(3)])
        gr = Slerp(ts,Rotation.from_quat(q[ids]))(grid)
        if cutoff is None:
            width = min(window,len(grid) if len(grid)%2 else len(grid)-1)
            if width<=order:
                notes.append({'start_frame':int(ids[0]),'reason':'too short for polynomial order'})
                continue
            fp = savgol_filter(gp,width,order,axis=0,mode='interp')
            fq = gr.as_quat().copy()
            for i in range(len(grid)):
                start = min(max(i-width//2,0),len(grid)-width)
                local = (gr[i].inv()*gr[start:start+width]).as_rotvec()
                # A local chart near pi is ambiguous: retain that rotation.
                if np.max(np.linalg.norm(local,axis=1))>=np.deg2rad(150):
                    notes.append({'time_s':float(grid[i]),'reason':'rotation window spans >=150 degrees'})
                    continue
                weights = savgol_coeffs(width,order,pos=i-start,use='dot')
                fq[i] = (gr[i]*Rotation.from_rotvec(weights@local)).as_quat()
        else:
            fs = 1/(grid[1]-grid[0])
            if cutoff>=fs/2: raise ValueError('Butterworth cutoff must be below segment Nyquist frequency')
            # Order 2 has default sosfiltfilt padding length 9; do not reduce it.
            if len(grid)<=10:
                notes.append({'start_frame':int(ids[0]),'reason':'too short for Butterworth padding'})
                continue
            sos = butter(2,cutoff,fs=fs,output='sos')
            fp = sosfiltfilt(sos,gp,axis=0)
            signed = gr.as_quat().copy()
            for i in range(1,len(signed)):
                if np.dot(signed[i-1],signed[i])<0: signed[i]*=-1
            fq = sosfiltfilt(sos,signed,axis=0)
            norms = np.linalg.norm(fq,axis=1)
            small = norms<1e-8
            fq[small] = signed[small]
            fq /= np.linalg.norm(fq,axis=1)[:,None]
        result_p[ids] = np.column_stack([np.interp(ts,grid,fp[:,a]) for a in range(3)])
        result_q[ids] = Slerp(grid,Rotation.from_quat(fq))(ts).as_quat()
    return result_p, result_q, notes


def gap_records(t, mask, rejected):
    records = []
    indices = np.flatnonzero(mask)
    if not len(indices): return records
    for ids in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
        left,right = int(ids[0]),int(ids[-1])
        records.append({'start_frame':left,'end_frame':right,'frames':len(ids),
                        'endpoint_span_s':float(t[right+1]-t[left-1]) if left>0 and right+1<len(t) else None,
                        'rejected_frames':int(rejected[ids].sum())})
    return records


def evaluate(t,p,q,raw_p,raw_q,common,stationary,max_span):
    valid = valid_pose(p,q)
    edge = valid[:-1] & valid[1:] & (np.diff(t)<=max_span)
    distances = np.linalg.norm(np.diff(p,axis=0),axis=1)
    speeds = distances[edge]/np.diff(t)[edge]
    common_edge = edge & common[:-1] & common[1:]
    both = valid & valid_pose(raw_p,raw_q)
    accepted = both & common
    result = {'available_frames':int(valid.sum()),'path_edges':int(edge.sum()),
              'path_length_m':float(distances[edge].sum()),
              'common_edge_path_length_m':float(distances[common_edge].sum()),
              'common_edges':int(common_edge.sum()),
              'peak_velocity_m_s':float(speeds.max()) if len(speeds) else None,
              'stationary':[]}
    for label,mask in [('all_raw',both),('accepted_raw',accepted)]:
        result['max_deviation_'+label+'_mm'] = float(np.linalg.norm(p[mask]-raw_p[mask],axis=1).max()*1000) if mask.any() else None
        result['max_orientation_deviation_'+label+'_deg'] = float(np.rad2deg((Rotation.from_quat(raw_q[mask]).inv()*Rotation.from_quat(q[mask])).magnitude()).max()) if mask.any() else None
    for start,end in stationary:
        mask = accepted & (t>=start) & (t<=end)
        entry = {'start_s':start,'end_s':end,'samples':int(mask.sum())}
        if mask.sum()>=2:
            rotations = Rotation.from_quat(q[mask])
            entry['position_std_mm'] = (p[mask].std(axis=0)*1000).tolist()
            entry['orientation_rms_deg'] = float(np.rad2deg(np.sqrt(np.mean((rotations.mean().inv()*rotations).magnitude()**2))))
        result['stationary'].append(entry)
    return result


def run(a):
    with a.trajectory.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    t=np.array([float(r['time_s']) for r in rows])
    if len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Need finite, strictly increasing timestamps')
    p=np.full((len(t),3),np.nan);q=np.full((len(t),4),np.nan)
    for i,r in enumerate(rows):
        if r['status'] in ('tracked','inferred_hand'):
            p[i]=[float(r[k+'_m']) for k in 'xyz']
            rv=np.array([float(r['relative_r'+k+'_rad']) for k in 'xyz'])
            if not np.isfinite(p[i]).all() or not np.isfinite(rv).all():
                raise ValueError(f'Nonfinite accepted pose at row {i}')
            q[i]=Rotation.from_rotvec(rv).as_quat()
    cp,cq,rejected,reasons=reject_spikes(t,p,q,a.max_speed,a.max_angular_speed,a.max_gap_span)
    fp,fq,labels=fill_gaps(t,cp,cq,a.max_gap_frames,a.max_gap_span)
    for i,r in enumerate(rows):
        if labels[i]=='observed' and r['status']=='inferred_hand': labels[i]='inferred_hand'
    variants={'raw':(p,q),'cleaned':(cp,cq),'interpolated':(fp,fq)}
    sp,sq,sg_notes=filter_segments(t,fp,fq,a.window,a.order,max_span=a.max_gap_span)
    variants['savgol']=(sp,sq)
    notes={'savgol':sg_notes}
    if a.butterworth_hz is not None:
        bp,bq,bnotes=filter_segments(t,fp,fq,cutoff=a.butterworth_hz,max_span=a.max_gap_span)
        variants['butterworth']=(bp,bq);notes['butterworth']=bnotes
    out=a.output or a.trajectory.parent/'offline_filtered'
    if out.resolve()==a.trajectory.parent.resolve(): raise ValueError('Use a separate output folder')
    out.mkdir(parents=True,exist_ok=True)
    common=valid_pose(cp,cq)
    stationary=a.stationary or []
    report={'source':str(a.trajectory),'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            'rejected_frames':int(rejected.sum()),'interpolated_frames':int(np.sum(labels=='interpolated')),
            'interpolated_gaps':gap_records(t,labels=='interpolated',rejected),
            'unfilled_gaps':gap_records(t,labels=='missing',rejected),'filter_passthrough_notes':notes,
            'metric_notes':['Stationary jitter uses explicit intervals and common accepted raw samples, including any hand-inferred poses.',
                            'Path length and velocity exclude gaps. Compare common-edge path length to avoid changes in coverage.',
                            'Maximum deviation includes rejected raw spikes; accepted_raw reports deviation excluding them.',
                            'Isolated spike rejection does not detect sustained bad poses. Inferred object poses are not measurements.'],
            'variants':{}}
    for name,(vp,vq) in variants.items():
        report['variants'][name]=evaluate(t,vp,vq,p,q,common,stationary,a.max_gap_span)
        if name=='raw':continue
        with (out/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f)
            w.writerow(['frame','time_s','source_status','pose_source','outlier_rejected','rejection_reason','x_m','y_m','z_m','qx','qy','qz','qw'])
            for i,r in enumerate(rows):
                label=labels[i] if name!='cleaned' else ('missing' if not valid_pose(vp[i:i+1],vq[i:i+1])[0] else 'inferred_hand' if r['status']=='inferred_hand' else 'observed')
                w.writerow([r['frame'],t[i],r['status'],label,int(rejected[i]),reasons[i],*[float(v) if np.isfinite(v) else '' for v in [*vp[i],*vq[i]]]])
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True)
    valid=valid_pose(p,q)
    anchor=Rotation.from_quat(q[np.flatnonzero(valid)[0]]) if valid.any() else Rotation.identity()
    for name in ['raw','interpolated','savgol','butterworth']:
        if name not in variants:continue
        vp,vq=variants[name]
        for j in range(3):axes[j].plot(t,vp[:,j]*1000,label=name,alpha=.8,linewidth=1)
        good=valid_pose(vp,vq);angles=np.full(len(t),np.nan)
        if good.any():angles[good]=np.rad2deg((anchor.inv()*Rotation.from_quat(vq[good])).magnitude())
        axes[3].plot(t,angles,label=name,linewidth=1)
    for j in range(3):axes[j].scatter(t[rejected],p[rejected,j]*1000,marker='x',color='red',label='rejected' if j==0 else None,zorder=5)
    for ax,label in zip(axes,['x (mm)','y (mm)','z (mm)','Angle from start (deg)']):
        ax.set_ylabel(label);ax.grid(alpha=.3)
        for gap in report['interpolated_gaps']:ax.axvspan(t[gap['start_frame']],t[gap['end_frame']],color='orange',alpha=.15)
    axes[0].legend(ncol=5);axes[0].set_title(a.trajectory.parent.name+' - offline pose filtering')
    axes[-1].set_xlabel('Time (s); shaded spans are filled gaps')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print(f'{a.trajectory}: rejected {rejected.sum()}, filled {report["interpolated_frames"]} in {len(report["interpolated_gaps"])} gaps; {out}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-speed',type=float,default=2.,help='Isolated-spike threshold in m/s')
    parser.add_argument('--max-angular-speed',type=float,default=720.,help='Isolated-spike threshold in degrees/s')
    parser.add_argument('--max-gap-frames',type=int,default=3)
    parser.add_argument('--max-gap-span',type=float,default=.15)
    parser.add_argument('--window',type=int,default=7,help='SG samples at the segment median cadence; odd')
    parser.add_argument('--order',type=int,default=2)
    parser.add_argument('--butterworth-hz',type=float,help='Optional order-2 forward-backward comparison cutoff')
    parser.add_argument('--stationary',type=float,nargs=2,action='append',metavar=('START','END'),help='Known stationary interval in seconds; repeatable')
    a=parser.parse_args()
    values=[a.max_speed,a.max_angular_speed,a.max_gap_span]+([] if a.butterworth_hz is None else [a.butterworth_hz])
    if any(not np.isfinite(v) or v<=0 for v in values) or a.max_gap_frames<0 or a.window<3 or a.window%2!=1 or not 0<=a.order<a.window:
        parser.error('Positive thresholds, nonnegative gap count, odd window >=3, and 0 <= order < window required')
    if any(not np.isfinite([s,e]).all() or s<0 or e<=s for s,e in a.stationary or []):
        parser.error('Stationary intervals require finite 0 <= start < end')
    run(a)


if __name__=='__main__': main()
