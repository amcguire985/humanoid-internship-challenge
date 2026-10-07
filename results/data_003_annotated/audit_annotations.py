import csv,json,hashlib
from pathlib import Path
source=Path('results/data_003_demos'); output=Path('results/data_003_annotated'); evidence=[]
for i,(grasp,release) in enumerate(((5.30,18.50),(24.50,36.50),(41.10,55.00)),1):
 name=f'demo_{i:03d}'; original=json.loads((source/name/'metadata.json').read_text()); updated=json.loads((output/name/'metadata.json').read_text()); v=json.loads((output/name/'validation.json').read_text())
 assert updated['manual_event_annotations']['grasp_time_seconds']==grasp
 assert updated['manual_event_annotations']['release_time_seconds']==release
 assert updated['automatic_phase_estimates']==original['automatic_phase_estimates']
 assert v['suitable_for_retargeting']; assert '\\' not in v['prepared_demo']
 original_rows=list(csv.DictReader((source/name/'processed_demo.csv').open())); updated_rows=list(csv.DictReader((output/name/'processed_demo.csv').open()))
 assert len(original_rows)==len(updated_rows)
 for a,b in zip(original_rows,updated_rows):
  for key in a:
   if key not in ('phase','transport_progress'): assert a[key]==b[key],key
 for row in updated_rows:
  time=float(row['time'])
  if updated['transport_start']<=time<=updated['transport_end']: assert row['phase']=='transport'
  if time<updated['pickup_time'] or time>updated['release_time']: assert row['phase'] not in ('pickup_lift','transport','placement_lowering')
 assert (output/name/'annotation_diagnostic.png').exists()
 evidence.append(dict(demo_id=name,manual_grasp=grasp,manual_release=release,observations_unchanged=len(updated_rows),automatic_estimates_preserved=True,portable_paths=True))
(output/'validation_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(evidence,indent=2))
