import json
from pathlib import Path
for i,(grasp,release) in enumerate(((5.30,18.50),(24.50,36.50),(41.10,55.00)),1):
 path=Path(f'config/data_003_annotations/demo_{i:03d}.json'); a=json.loads(path.read_text())
 a.update(grasp_time_seconds=grasp,release_time_seconds=release,notes='Manual grasp/degrasp annotations provided by the user on 2026-10-06. Demo 1 5:30 interpreted as 5.30 absolute video seconds. Future rollouts require manual annotations.')
 path.write_text(json.dumps(a,indent=2)+'\n')
