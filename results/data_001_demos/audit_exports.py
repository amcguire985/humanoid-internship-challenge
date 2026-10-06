import csv,json
from pathlib import Path
import numpy as np
out=Path('results/data_001_demos')
reports=[]
for dest in sorted(out.glob('demo_*')):
 m=json.loads((dest/'metadata.json').read_text()); reports.append(m)
 rows=list(csv.DictReader((dest/'processed_demo.csv').open()))
 nrows=list(csv.DictReader((dest/'normalized_task_trajectory.csv').open()))
 assert len(rows)==len(nrows)
 t=np.array([float(r['time']) for r in rows]); valid=np.array([r['valid_processed']=='1' for r in rows])
 p=np.array([[float(r['object_'+a]) if r['object_'+a] else np.nan for a in 'xyz'] for r in rows])
 active=(t>=m['pickup_time'])&(t<=m['release_time'])
 assert np.sum(active&~valid)==m['missing_active_frames']
 assert m['reject']==bool(m['rejection_reasons'])
 for i,r in enumerate(nrows):
  assert (r['valid']=='1')==valid[i]
  if not valid[i]: assert all(r[k]=='' for k in ('longitudinal','lateral','lift'))
 edge=valid[:-1]&valid[1:]&active[:-1]&active[1:]&(np.diff(t)<=.15)
 path=np.linalg.norm(np.diff(p,axis=0),axis=1)[edge].sum()
 assert abs(path-m['path_length_m'])<1e-10
 assert all((dest/name).exists() for name in ('diagnostic.png','transport.csv','raw_trajectory.csv','cleaned_trajectory.csv'))
lines=['# data_001: seven video-complete transfers','',
 'Seven complete transfers were confirmed by video review. Only demo_005 has complete cleaned pickup-to-release coverage; the other six are rejected for retargeting. All remain behind the inherited calibration/height review gate.','',
 '| Demo | Direction | Carry interval (s) | Distance (cm) | XY direction (deg) | Maximum observed lift (cm) | Observed path (cm) | Missing carry frames | Longest gap (s) | Decision |',
 '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
for r in reports:
 longest=max([g['missing_duration_s'] for g in r.get('active_tracking_gaps',[])],default=0)
 lines.append(f"| [{r['id']}]({r['id']}/diagnostic.png) | {r['direction']} | {r['pickup_time']:.2f}-{r['release_time']:.2f} | {100*r['start_to_goal_distance']:.1f} | {r['world_xy_direction_deg']:.1f} | {100*r['maximum_lift_m']:.1f} | {100*r['path_length_m']:.1f} | {r['missing_active_frames']} | {longest:.2f} | {'Reject: tracking gaps' if r['reject'] else 'Candidate: geometry/timing review'} |")
lines+=['','Distances use robust measured resting endpoints; target-tag anchor is stored separately. Directions are in the existing ID0 world frame. Lift is signed world-Z relative to the measured start; its gravity alignment is unverified. Rejected trajectories have partial paths and observed lift maxima, which can underestimate unseen motion.',
 '', 'Automatic segmentation merged the first three transfers and failed to identify some target endpoints. Video review supplies seven explicit windows/directions. Inner transport phase estimates remain automatic height heuristics and need review; pickup/release are approximate motion brackets, not contact detection. Edit ../../config/data_001_demos.json and rerun the documented extractor.',
 '', 'Outputs preserve missing rows and never connect a path across them. Normalized coordinates preserve those same missing rows. Interpolation is limited to 3 missing frames / 0.15-second endpoint spans. No LIBERO execution or policy training was performed.',
 '', 'Validation: all seven CSV/normalization exports audited for matching validity, missing carry counts, no-gap path sums, and required artifacts. Three new splitter tests and ten existing task-processing tests passed.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[4:13])); print('Export audit passed for seven demonstrations.')
