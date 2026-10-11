from pathlib import Path
p=Path('results/data_003_annotated/audit_annotations.py'); s=p.read_text()
s=s.replace(" assert updated_rows[0]['phase']=='stationary_before_pickup'", " for row in updated_rows:\n  time=float(row['time'])\n  if updated['transport_start']<=time<=updated['transport_end']: assert row['phase']=='transport'\n  if time<updated['pickup_time'] or time>updated['release_time']: assert row['phase'] not in ('pickup_lift','transport','placement_lowering')")
p.write_text(s)
