import json,numpy as np
from pathlib import Path
out=Path('results/data_002_demos'); d=np.load(out/'cleaned_full_trajectory.npz'); t=d['time']; p=d['smoothed']
cfg=json.loads(Path('config/data_002_demos.json').read_text(encoding='utf-8-sig'))
windows=[(0,13.9,3.835,12.5,(.5,3.5),(10,13)),(13.9,24,14.2,21,(10,13),(21,22.5)),(24,39,26.2,35,(21,22.5),(36,39)),(39,55,41.5,51.6,(36,39),(51,53)),(55,67,56.3,65,(53,54.5),(65,66.5)),(67,79.5,67.8,77,(65,66.5),(75,77)),(79.5,91.4,79.7,90.2,(78,79),(90.3,91.3))]
cfg['demonstrations']=[]
for i,(a,b,pick,release,sw,ew) in enumerate(windows):
 snap=lambda v:float(t[np.argmin(abs(t-v))])
 def median(w):
  pts=p[(t>=w[0])&(t<=w[1])]; pts=pts[np.isfinite(pts).all(axis=1)]
  if not len(pts): raise ValueError(f'No endpoint measurements in {w}')
  return np.median(pts,axis=0).tolist()
 cfg['demonstrations'].append(dict(start_time=snap(a),end_time=snap(b),pickup_time=snap(pick),release_time=snap(release),start=median(sw),end=median(ew),direction='object -> target' if i%2==0 else 'target -> object',review_reason='Seven complete alternating transfers confirmed in 2-second video contact sheets and 1-second placement review. Approximate pickup-to-release brackets include visible carrying and release. Automatic stationary segmentation is incomplete due occlusion and pose jitter. Timing annotations are reviewable, not contact ground truth.',phase_overrides={}))
Path('config/data_002_demos.json').write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps(cfg['demonstrations'],indent=2))
