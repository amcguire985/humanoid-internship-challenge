import csv,json
from pathlib import Path
import numpy as np
out=Path('results/data_002_demos')
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

summary=json.loads((out/'summary.json').read_text())
assert len(reports)==7 and len(summary['demonstrations'])==7
assert all(r['direction']==('object -> target' if i%2==0 else 'target -> object') for i,r in enumerate(reports))
lines=['# data_002 transfer extraction','',f"Seven complete transfers confirmed in video. {sum(not r['reject'] for r in reports)} pass the tracking/segmentation checks; {sum(r['reject'] for r in reports)} are rejected. Calibration/crop and height-axis validation remain pending.",'',
'| Demo | Direction | Pickup-release (s) | Distance (cm) | XY heading (deg) | Observed lift (cm) | Observed path (cm) | Missing carry frames | Longest gap (s) | Decision |',
'|---|---|---|---:|---:|---:|---:|---:|---:|---|']
for r in reports:
 longest=max((g['missing_duration_s'] for g in r['active_tracking_gaps']),default=0)
 lines.append(f"| [{r['id']}]({r['id']}/diagnostic.png) | {r['direction']} | {r['pickup_time']:.2f}-{r['release_time']:.2f} | {100*r['start_to_goal_distance']:.1f} | {r['world_xy_direction_deg']:.1f} | {100*r['maximum_lift_m']:.1f} | {100*r['path_length_m']:.1f} | {r['missing_active_frames']} | {longest:.2f} | {'Reject' if r['reject'] else 'Review geometry/timings'} |")
lines+=['','All lift values are maxima over available observations, not verified maximum physical lift. Gaps can hide the peak; paths include only valid adjacent edges and are partial when tracking is missing. Distances use robust measured resting endpoints; the fixed target-tag anchor is stored separately. XY headings use ID0 world axes.',
'', 'Automatic segmentation did not recover all seven transfers because of missing observations and pose jitter. Explicit split windows/directions were checked using the 2-second video contact sheets and a denser 1-second placement review. The final transfer includes visible release before the clip ends. Pickup/release annotations are approximate motion/video brackets, not grasp/contact ground truth. Inner transport boundaries retain automatic height heuristics and remain easy to override.',
'', 'Each demo directory contains metadata.json, processed_demo.csv (authoritative smoothed coordinates/validity), raw_trajectory.csv, cleaned_trajectory.csv (pre-interpolation accepted measurements), transport.csv (pickup-through-release including missing samples), normalized_task_trajectory.csv, and diagnostic.png. Normalization translates the start to the origin, aligns longitudinal XY toward the measured end, uses lateral-left and signed world-Z axes, and divides lengths by planar displacement. No endpoint correction is applied.',
'', 'Raw tracking: ../data_002_raw. Whole-video diagnostics: full_overview.png, video_contact_sheet_1.jpg, video_contact_sheet_2.jpg, and placement_review.jpg. Per-demo timing overrides and endpoint windows: ../../config/data_002_demos.json.',
'', 'Reproduce exports:', '', '```powershell', '.\\.venv\\Scripts\\python.exe scripts/extract_transfer_demos.py results/data_002_raw --config config/data_002_demos.json --output results/data_002_demos', '```',
'', 'Tracking used the same Standard41h12 configuration as data_001: world 0/1, target 2, object 4/5/6, 60 mm full-pattern stationary tags, 40 mm object tags, 45 mm cube, landscape calibration with center-crop. Cleaning reused existing reprojection/spike rejection, at most 3-frame/0.15-second bounded interpolation and SG7 smoothing within continuous segments. Large gaps remain missing.',
'', 'Validation: CSV lengths and normalized validity match; missing carry counts and path sums were independently recomputed without crossing gaps; all required artifacts exist. No LIBERO execution or policy training was performed.']
for r in reports:
 lines+=['',r['id']+': '+'; '.join(r['rejection_reasons'])]
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[4:13])); print('Independent export audit passed for seven demonstrations.')
