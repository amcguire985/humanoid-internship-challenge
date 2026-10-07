import csv,json
from pathlib import Path
import numpy as np
out=Path('results/data_003_demos')
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
assert len(reports)==3 and len(summary['demonstrations'])==3
assert all(r['direction']==('object -> target' if i%2==0 else 'target -> object') for i,r in enumerate(reports))
d=np.load(out/'cleaned_full_trajectory.npz'); rv=np.isfinite(d['raw']).all(axis=1); pv=np.isfinite(d['smoothed']).all(axis=1)
lines=['# data_003 transfer extraction','',f"Three video-complete transfers. {sum(not r['reject'] for r in reports)} pass tracking/segmentation checks; {sum(r['reject'] for r in reports)} are rejected. Calibration/crop, vertical direction and phase/contact semantics remain pending review.",'',
f"Raw and cleaned full-video coverage: {rv.sum()}/{len(rv)} ({100*rv.mean():.1f}%) raw, {pv.sum()}/{len(pv)} ({100*pv.mean():.1f}%) cleaned. Cleaning rejected {int(d['rejected'].sum())} observations.",'',
'| Demo | Direction | Pickup-release (s) | Distance (cm) | XY heading (deg) | Observed max lift (cm) | Path (cm) | Missing carry frames | Decision |',
'|---|---|---|---:|---:|---:|---:|---:|---|']
for r in reports:
 lines.append(f"| [{r['id']}]({r['id']}/diagnostic.png) | {r['direction']} | {r['pickup_time']:.2f}-{r['release_time']:.2f} | {100*r['start_to_goal_distance']:.1f} | {r['world_xy_direction_deg']:.1f} | {100*r['maximum_lift_m']:.1f} | {100*r['path_length_m']:.1f} | {r['missing_active_frames']} | {'Reject' if r['reject'] else 'Gap-free candidate'} |")
lines+=['', 'Distances use robust measured resting start/end positions. The fixed target-tag goal anchor is stored separately. XY headings use the existing ID0 world frame. Maximum lift is measured signed world-Z displacement relative to the resting start, not a verified gravity height. Pose jitter, particularly while lowering/at rest, contributes to path length; full coverage does not establish metric accuracy.',
'', 'The clip contains three alternating completed transfers including the final release. Automatic stationary segmentation merged them because resting position jitter exceeds the stationary-speed threshold. The saved configuration records explicit video-reviewed splits/directions and resting endpoint medians. Pickup/release are approximate motion/video annotations, not measured contact. Inner transport boundaries retain automatic height heuristics, may omit late lowering/holding, and need review before robot use. Full pickup-through-release trajectories are preserved.',
'', 'Each demo directory contains metadata.json, processed_demo.csv (authoritative smoothed coordinates with validity/provenance), transport.csv (pickup-through-release), raw_trajectory.csv, cleaned_trajectory.csv (pre-interpolation accepted measurements), normalized_task_trajectory.csv, and diagnostic.png. Normalized coordinates translate the start to the origin, align longitudinal XY with measured task displacement, use lateral-left and signed world-Z axes, and divide lengths by planar task distance. No endpoint correction or invented motion.',
'', 'Review images: [whole-video positions](full_overview.png), [video contact sheet](video_contact_sheet_1.jpg), [placement/release review](placement_review.jpg). Original [annotated tracking video](../data_003_raw/annotated.mp4). Timing overrides: ../../config/data_003_demos.json.',
'', 'Reproduce:', '', '```powershell', '.\\.venv\\Scripts\\python.exe scripts/extract_transfer_demos.py results/data_003_raw --config config/data_003_demos.json --output results/data_003_demos', '```',
'', 'Unchanged tracker configuration: Standard41h12 world 0/1, target 2, object 4/5/6; 60 mm world/target and 40 mm object full-pattern tags; 45 mm cube; landscape calibration with center-crop. Cleaning uses existing reprojection/spike rejection, bounded short-gap interpolation (3 frames / 0.15-second endpoint span), and SG7 smoothing within continuous segments.',
'', 'Validation: independent export audit checked artifact presence, matching CSV/normalization validity, missing carry counts and path sums without crossing gaps. No LIBERO execution or policy training. All candidates remain behind the inherited calibration/height review gate.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:12])); print('Independent export audit passed for three demonstrations.')
