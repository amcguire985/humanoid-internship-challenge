"""Extract multiple transfers using existing AprilTag trajectory cleaning and phase APIs."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from process_task_demo import (read_csv, validate_config, clean_trajectory, compute_goal_position,
    trajectory_velocity, segment_task_phases, compute_task_metrics, export_processed_demo, CONVENTION)
from filter_trajectory import segments


def detect(t,p,speed,c):
    rests=[]
    for ids in segments(t,np.isfinite(speed)&(speed<=c['stationary_speed']),c['max_gap_span']):
        if t[ids[-1]]-t[ids[0]] < c['stationary_duration']: continue
        a=dict(first=int(ids[0]),last=int(ids[-1]),position=np.median(p[ids],axis=0))
        if rests and np.linalg.norm((a['position']-rests[-1]['position'])[:2])<.025:
            between=p[rests[-1]['last']:a['first']+1]
            if np.nanmax(np.linalg.norm(between-rests[-1]['position'],axis=1))<.06:
                rests[-1]['last']=a['last']; continue
        rests.append(a)
    return rests


def plot(dest,t,raw,p,speed,start,end,bounds,c,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.3))
    for ids in segments(t,np.isfinite(p).all(axis=1),c['max_gap_span']):
        axes[0].plot(p[ids,0]*100,p[ids,1]*100)
        axes[1].plot(t[ids],(p[ids,2]-start[2])*100)
        axes[2].plot(t[ids],speed[ids])
    axes[0].scatter([start[0]*100,end[0]*100],[start[1]*100,end[1]*100],c=['green','red'])
    axes[0].set(xlabel='World x (cm)',ylabel='World y (cm)'); axes[0].axis('equal')
    axes[1].plot(t,(raw[:,2]-start[2])*100,color='gray',alpha=.25)
    axes[1].set(xlabel='Video time (s)',ylabel='Lift above start (cm)')
    axes[2].set(xlabel='Video time (s)',ylabel='Speed (m/s)')
    for key,value in bounds.items():
        if value is not None:
            for ax in axes[1:]: ax.axvline(value,alpha=.5,label=key)
    axes[2].legend(fontsize=6)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(dest/'diagnostic.png',dpi=150); plt.close(fig)


def run(source,output,config_path):
    cfg=json.loads(config_path.read_text(encoding='utf-8-sig')); c=validate_config(cfg['processing'])
    summary=json.loads((source/'summary.json').read_text())
    rows=read_csv(source/'object/cube_center/trajectory.csv')
    t=np.array([float(r['time_s']) for r in rows])
    if len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0): raise ValueError('Invalid timestamps')
    raw=np.array([[float(r[a+'_m']) if r.get(a+'_m') else np.nan for a in 'xyz'] for r in rows])
    errors=np.array([float(r['error_px']) if r.get('error_px') else np.nan for r in rows])
    cleaned,p,provenance,rejected,reasons,gaps=clean_trajectory(t,raw,errors,c)
    target,target_info=compute_goal_position(read_csv(source/'target/id2/trajectory.csv'),c['goal_object_offset'],
        c['max_reprojection_error'],c['goal_coplanar'],c['goal_center_height'])
    _,speed=trajectory_velocity(t,p,c['max_gap_span']); rests=detect(t,p,speed,c)
    automatic=[]
    for left,right in zip(rests[:-1],rests[1:]):
        if np.linalg.norm((right['position']-left['position'])[:2])<.08: continue
        at_start=np.linalg.norm((left['position']-target)[:2])<=c['goal_radius']
        at_end=np.linalg.norm((right['position']-target)[:2])<=c['goal_radius']
        automatic.append(dict(start_time=float(t[left['first']]),end_time=float(t[right['last']]),
            pickup_time=float(t[left['last']]),release_time=float(t[right['first']]),
            direction='target -> object' if at_start and not at_end else 'object -> target' if at_end and not at_start else 'unresolved',
            start=left['position'].tolist(),end=right['position'].tolist()))
    output.mkdir(parents=True,exist_ok=True)
    np.savez(output/'cleaned_full_trajectory.npz',time=t,raw=raw,cleaned=cleaned,smoothed=p,provenance=provenance,rejected=rejected)
    (output/'automatic_candidates.json').write_text(json.dumps(dict(target=target.tolist(),candidates=automatic,
        rests=[dict(start_time=float(t[a['first']]),end_time=float(t[a['last']]),position=a['position'].tolist()) for a in rests]),indent=2)+'\n')
    reports=[]
    for number,item in enumerate(cfg.get('demonstrations') or automatic,1):
        ids=np.flatnonzero((t>=item['start_time'])&(t<=item['end_time']))
        if len(ids)<2: raise ValueError('Empty demo')
        tt,pp=t[ids],p[ids]; start=np.array(item['start']); end=np.array(item['end'])
        cc={**c,'phase_overrides':{},'phase_override_reason':item.get('review_reason')}
        _,_,_,estimated,_,_=segment_task_phases(tt,pp,end,cc)
        cc['phase_overrides']={**dict(pickup_time=item['pickup_time'],release_time=item['release_time'], transport_start=max(item['pickup_time'],min(estimated['transport_start'] or item['pickup_time'],item['release_time'])), transport_end=max(item['pickup_time'],min(estimated['transport_end'] or item['release_time'],item['release_time']))),**item.get('phase_overrides',{})}
        labels,ss,bounds,_,progress,notes=segment_task_phases(tt,pp,end,cc)
        metrics=compute_task_metrics(tt,pp,end,ss,bounds,cc)
        delta=end-start; distance=float(np.linalg.norm(delta[:2]))
        if distance<1e-6: raise ValueError('Zero task displacement')
        basis=delta[:2]/distance; rel=pp-start
        norm=np.column_stack((rel[:,:2]@basis/distance,rel[:,:2]@np.array([-basis[1],basis[0]])/distance,rel[:,2]*c['vertical_sign']/distance))
        active=(tt>=bounds['pickup_time'])&(tt<=bounds['release_time']); valid=np.isfinite(pp).all(axis=1)
        edge=valid[:-1]&valid[1:]&active[:-1]&active[1:]&(np.diff(tt)<=c['max_gap_span'])
        lift=float(max(0,np.nanmax((pp[active,2]-start[2])*c['vertical_sign'])))
        flags=[]
        if not np.all(valid[active]): flags.append('Unfilled tracking gap during pickup-to-release; reject for retargeting.')
        if np.any(np.diff(tt[active])>c['max_gap_span']): flags.append('Unresolved timestamp gap.')
        if item['direction']=='unresolved': flags.append('Target direction unresolved; review segmentation.')
        if lift<.015: flags.append('No reliable lift; review segmentation.')
        report={**metrics,**bounds,'id':f'demo_{number:03d}','direction':item['direction'],
            'object_start_position':start.tolist(),'goal_position':end.tolist(),'task_displacement':delta.tolist(),
            'start_to_goal_distance':float(np.linalg.norm(delta)),'planar_distance_m':distance,
            'world_xy_direction_deg':float(np.degrees(np.arctan2(delta[1],delta[0]))),'maximum_lift_m':lift,
            'path_length_m':float(np.linalg.norm(np.diff(pp,axis=0),axis=1)[edge].sum()),
            'automatic_phase_estimates':estimated,'phase_review_notes':notes,'rejection_reasons':flags,'reject':bool(flags),
            'missing_active_frames':int(np.sum(~valid[active])),'interpolated_frames':int(np.sum(provenance[ids]=='interpolated')),
            'parameters':cc,'selection':item,'target_anchor':target.tolist(),'coordinate_frame_convention':CONVENTION,
            'source_video':summary['video'],'source_tracking':str(source),
            'usable_for_retargeting':not flags and c['calibration_verified'] and c['vertical_verified'],
            'geometry_review_required':not(c['calibration_verified'] and c['vertical_verified']),
            'phase_boundary_sources':{k:('manual_override' if k in item.get('phase_overrides',{}) else 'reviewed_motion_bracket' if k in ('pickup_time','release_time') else 'automatic_clamped_to_carry_interval') for k in bounds},
            'active_tracking_gaps':[dict(start_time_s=float(tt[g[0]]),end_time_s=float(tt[g[-1]]),missing_frames=len(g),missing_duration_s=float(tt[g[-1]]-tt[g[0]]+np.median(np.diff(tt)))) for g in segments(tt,active & ~valid,c['max_gap_span'])],
            'normalization':'Start origin; longitudinal toward measured endpoint, lateral left, signed world Z; divide lengths by planar displacement. Missing rows preserved. No endpoint correction.',
            'limitations':['Motion timings do not measure grasp/contact.','Inherited landscape calibration/crop and world Z gravity direction remain unverified.']}
        dest=output/report['id']; dest.mkdir(exist_ok=True)
        export_processed_demo(dest,tt,raw[ids],cleaned[ids],pp,end,labels,ss,progress,provenance[ids],rejected[ids],reasons[ids],report,bounds,cc)
        with (dest/'normalized_task_trajectory.csv').open('w',newline='') as handle:
            writer=csv.writer(handle); writer.writerow(['video_time_s','normalized_time','longitudinal','lateral','lift','valid'])
            for j in range(len(tt)):
                writer.writerow([tt[j],(tt[j]-bounds['pickup_time'])/(bounds['release_time']-bounds['pickup_time']),*[float(v) if np.isfinite(v) else '' for v in norm[j]],int(valid[j])])
        (dest/'metadata.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        plot(dest,tt,raw[ids],pp,ss,start,end,bounds,cc,report['id']+' '+item['direction']); reports.append(report)
    result=dict(source=str(source),target_estimation=target_info,demonstrations=reports,complete_demonstrations=len(reports),
        accepted=sum(not r['reject'] for r in reports),usable_for_retargeting=sum(r['usable_for_retargeting'] for r in reports),cleaning_gaps=gaps,config=str(config_path))
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(id=r['id'],direction=r['direction'],distance=r['start_to_goal_distance'],lift=r['maximum_lift_m'],reject=r['reject'],missing=r['missing_active_frames']) for r in reports],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('tracking',type=Path)
    parser.add_argument('--config',required=True,type=Path); parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args(); run(a.tracking,a.output,a.config)
