import csv,json
from pathlib import Path
import numpy as np
out=Path('results/test_008_demos')
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


assert len(reports)==1 and reports[0]['direction']=='object -> target'
r=reports[0]; tracking=json.loads(Path('results/test_008_raw/summary.json').read_text())
d=np.load(out/'cleaned_full_trajectory.npz'); raw_valid=np.isfinite(d['raw']).all(axis=1); valid=np.isfinite(d['smoothed']).all(axis=1)
longest=max((g['missing_duration_s'] for g in r['active_tracking_gaps']),default=0)
lines=['# test_008 shutter-speed-500 tracking test','',
'Input: videos/test__008_shutterspeed500.MOV. One complete object -> target transfer confirmed in half-second video review, including release before the clip ends.','',
f"Raw cube-centre coverage: {raw_valid.sum()}/{len(raw_valid)} ({100*raw_valid.mean():.1f}%). Cleaned coverage: {valid.sum()}/{len(valid)} ({100*valid.mean():.1f}%).",'',
'| Metric | Result |','|---|---|',
f"| Start-to-end distance | {100*r['start_to_goal_distance']:.1f} cm |",
f"| World-XY heading | {r['world_xy_direction_deg']:.1f} degrees |",
f"| Maximum observed world-Z lift | {100*r['maximum_lift_m']:.1f} cm |",
f"| Observed path length (does not cross gaps) | {100*r['path_length_m']:.1f} cm |",
f"| Pickup / transport start / transport end / release | {r['pickup_time']:.3f} / {r['transport_start']:.3f} / {r['transport_end']:.3f} / {r['release_time']:.3f} s |",
f"| Missing carry frames | {r['missing_active_frames']} |",
f"| Longest missing carry interval | {longest:.3f} s |",
f"| Retargeting decision | {'Reject: tracking gaps' if r['reject'] else 'Candidate pending geometry/timing review'} |",'',
'Individual tracking gaps:', '', '| Start (s) | End (s) | Missing frames | Missing duration (s) |','|---:|---:|---:|---:|']
for g in r['active_tracking_gaps']:
 lines.append(f"| {g['start_time_s']:.3f} | {g['end_time_s']:.3f} | {g['missing_frames']} | {g['missing_duration_s']:.3f} |")
lines+=['', 'Diagnostic: [compact trajectory/height/speed plot](demo_001/diagnostic.png). Whole-video motion: [overview](full_overview.png). Video review: [half-second contact sheet](phase_review.jpg). Original annotated tracking: [video](../test_008_raw/annotated.mp4).',
'', 'Exported demo_001 contains processed_demo.csv (smoothed positions with validity/provenance), transport.csv (pickup through release, missing rows retained), raw_trajectory.csv, cleaned_trajectory.csv (accepted measurements before filling/smoothing), normalized_task_trajectory.csv, metadata.json, and diagnostic.png. Resting endpoints are robust measured medians; the fixed target-tag anchor is retained separately.',
'', 'Automatic stationary detection did not confirm a stable final rest because final poses are missing/noisy. The complete transfer and direction were verified in video and an explicit split was saved. Pickup is motion onset; release is an approximate annotation. Inner transport estimates remain automatic height heuristics. Edit ../../config/test_008_demos.json to override timing.',
'', 'Reproduce:', '', '```powershell', '.\\.venv\\Scripts\\python.exe scripts/extract_transfer_demos.py results/test_008_raw --config config/test_008_demos.json --output results/test_008_demos', '```',
'', 'Tracking and cleaning use unchanged data_001/data_002 settings. No shutter-speed setting was inferred from camera metadata; 500 is the filename label. Motion is never invented across large tracking gaps. Interpolation is bounded by 3 missing frames and 0.15-second endpoint span; SG7 smoothing resets at unresolved gaps. Normalized coordinates retain the same missing rows.',
'', 'Lift is relative to the start in configured signed world-Z, whose gravity alignment is unverified. Calibration/crop validation remains pending. Observed maximum lift can miss a peak hidden by gaps; observed path length is partial. This short clip cannot isolate the effect of shutter speed because recording length, motion and visible tags differ from the prior videos.',
'', 'Validation: export audit recomputed validity, missing carry counts and no-gap path sums, and verified all artifacts. No LIBERO execution or training.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:18])); print('Single-demo export audit passed.')
