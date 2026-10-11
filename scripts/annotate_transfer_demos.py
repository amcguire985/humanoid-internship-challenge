"""Editable per-demo contact annotations, diagnostics and rollout input validation."""
import argparse
import hashlib
import csv
import json
from pathlib import Path
import numpy as np
from process_task_demo import EVENT_BOUNDARIES, validate_config, resolve_phase_bounds, segment_task_phases
from retarget_libero_object import load_transport

FIELDS=tuple(EVENT_BOUNDARIES)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def initialize(demo,path):
    """Never overwrite an existing manual annotation file."""
    if path.exists(): return read_json(path)
    m=read_json(demo/'metadata.json')
    a=dict(schema_version=1,demo_id=m['id'],time_basis='absolute source video seconds',
           source_video=m['source_video'],**{key:None for key in FIELDS},notes='',
           suggested_motion_brackets={key:m[boundary] for key,boundary in EVENT_BOUNDARIES.items()},
           automatic_estimates=m['automatic_phase_estimates'])
    write_json(path,a); return a

def resolve(t,metadata,annotations):
    """Manual fields always win. Invalid manual order fails rather than being repaired."""
    allowed={'schema_version','demo_id','time_basis','source_video','notes',
             'suggested_motion_brackets','automatic_estimates',*FIELDS}
    if set(annotations)-allowed: raise ValueError('Unknown annotation fields: '+str(set(annotations)-allowed))
    if annotations.get('demo_id')!=metadata['id']: raise ValueError('Annotation/demo identity mismatch')
    c=validate_config({**metadata['parameters'],'phase_overrides':{},
                       **{key:annotations.get(key) for key in FIELDS}})
    # Existing reviewed motion brackets remain fallback annotations, never contact ground truth.
    baseline={key:metadata[key] for key in EVENT_BOUNDARIES.values()}
    bounds=resolve_phase_bounds(t,baseline,c)
    sources={boundary:dict(source='manual' if annotations.get(key) is not None else 'existing_motion_estimate',
             requested_time_seconds=annotations.get(key),effective_time_seconds=bounds[boundary])
             for key,boundary in EVENT_BOUNDARIES.items()}
    return bounds,sources,c

def diagnostic(output,t,p,speed,metadata,annotations,bounds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(11,8),sharex=True)
    for j in range(3):
        axes[j].plot(t,p[:,j],color='black',lw=.9); axes[j].set_ylabel('xyz'[j]+' (m)')
    axes[3].plot(t,speed,color='black',lw=.9); axes[3].set_ylabel('Speed (m/s)')
    colors={'pickup_time':'green','release_time':'red','transport_start':'tab:blue','transport_end':'tab:orange'}
    for boundary,value in metadata['automatic_phase_estimates'].items():
        if value is not None:
            for ax in axes: ax.axvline(value,color=colors[boundary],ls='--',alpha=.5,label='automatic '+boundary)
    for field,boundary in EVENT_BOUNDARIES.items():
        value=annotations.get(field)
        if value is not None:
            for ax in axes: ax.axvline(value,color=colors[boundary],ls='-',lw=1.8,label='manual '+field)
    for boundary,value in bounds.items():
        if value is not None:
            for ax in axes: ax.axvline(value,color=colors[boundary],ls=':',alpha=.4,label='effective '+boundary)
    for ax in axes: ax.grid(alpha=.2)
    axes[0].legend(fontsize=6,ncol=3); axes[-1].set_xlabel('Absolute video time (s)')
    fig.suptitle(metadata['id']+' '+metadata['direction']+' — grasp/release review')
    fig.tight_layout(); fig.savefig(output/'annotation_diagnostic.png',dpi=150); plt.close(fig)

def prepare(demo,annotation_path,output):
    m=read_json(demo/'metadata.json'); a=initialize(demo,annotation_path)
    with (demo/'processed_demo.csv').open(newline='',encoding='utf-8-sig') as handle:
        reader=csv.DictReader(handle); fields=reader.fieldnames; rows=list(reader)
    t=np.array([float(r['time']) for r in rows]); p=np.array([[float(r['object_'+k]) if r['object_'+k] else np.nan for k in 'xyz'] for r in rows])
    speed=np.array([float(r['speed']) if r['speed'] else np.nan for r in rows])
    bounds,sources,c=resolve(t,m,a); reasons=[]
    if any(a.get(k) is None for k in ('grasp_time_seconds','release_time_seconds')):
        reasons.append('Manual grasp_time_seconds and release_time_seconds are required before rollout.')
    b,e=bounds['transport_start'],bounds['transport_end']
    if b is None or e is None or e<=b: reasons.append('Transport interval must have positive duration.')
    carry=(t>=bounds['pickup_time'])&(t<=bounds['release_time'])
    if (not np.isfinite(p[carry]).all() or np.any(np.diff(t[carry])>c['max_gap_span'])
            or any(row['valid_processed'] != '1' for row, selected in zip(rows, carry) if selected)):
        reasons.append('Pickup-to-release trajectory contains missing measurements or large timestamp gaps.')
    if m['direction'] not in ('object -> target','target -> object'): reasons.append('Direction unresolved.')
    updated={**m,**bounds,'manual_event_annotations':{k:a.get(k) for k in FIELDS},
             'phase_boundary_sources':sources,'annotation_file':annotation_path.as_posix(),
             'automatic_phase_estimates':m['automatic_phase_estimates'],
             'manual_contact_annotations_present':all(a.get(k) is not None for k in ('grasp_time_seconds','release_time_seconds')),
             'annotation_notes':a.get('notes','')}
    output.mkdir(parents=True,exist_ok=True)
    # Update phase labels from resolved annotations while preserving samples/validity.
    phase_config={**c,'phase_overrides':bounds}
    labels,_,_,_,progress,_=segment_task_phases(t,p,np.array(m['goal_position']),phase_config)
    for i,row in enumerate(rows):
        row['phase']=labels[i]; row['transport_progress']=float(progress[i]) if np.isfinite(progress[i]) else ''
    with (output/'processed_demo.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (output/'transport.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        writer.writerows(r for r,time in zip(rows,t) if bounds['pickup_time']<=time<=bounds['release_time'])
    write_json(output/'metadata.json',updated)
    if not reasons:
        try: load_transport(output/'processed_demo.csv',output/'metadata.json',c['max_gap_span'])
        except ValueError as exc: reasons.append(str(exc))
    report=dict(demo_id=m['id'],direction=m['direction'],start_position=m['object_start_position'],
        end_goal_position=m['goal_position'],start_to_goal_distance_m=m['start_to_goal_distance'],
        maximum_lift_m=m['maximum_lift_m'],grasp_time_seconds=bounds['pickup_time'],
        release_time_seconds=bounds['release_time'],transport_start_time_seconds=b,
        transport_end_time_seconds=e,transport_duration_seconds=e-b if b is not None and e is not None else None,
        suitable_for_retargeting=not reasons,source_tracking_and_timing_suitable=not reasons,
        suitability_scope="Source samples and timing only; rollout_inputs.json includes geometry and annotation-review gates.",
        blocking_reasons=reasons,phase_boundary_sources=sources,
        automatic_estimates=m['automatic_phase_estimates'],manual_annotations={k:a.get(k) for k in FIELDS},
        geometry_limitations=m.get('limitations',[]),source_demo=demo.as_posix(),annotation_path=annotation_path.as_posix(),prepared_demo=output.as_posix(),
        annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        direction_mapping='Align this demo start-to-end task axis with LIBERO object-start-to-target; preserve temporal order and lift/lateral shape, including target -> object strategies.')
    write_json(output/'validation.json',report); diagnostic(output,t,p,speed,m,a,bounds)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demos',type=Path,default=Path('results/data_003_demos'))
    parser.add_argument('--annotations',type=Path,default=Path('config/data_003_annotations'))
    parser.add_argument('--output',type=Path,default=Path('results/data_003_annotated'))
    args=parser.parse_args()
    reports=[prepare(d,args.annotations/(d.name+'.json'),args.output/d.name) for d in sorted(args.demos.glob('demo_*')) if (d/'metadata.json').exists()]
    write_json(args.output/'validation.json',reports); print(json.dumps(reports,indent=2))
