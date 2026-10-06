import json,numpy as np
from pathlib import Path
cfg_path=Path('config/test_008_demos.json'); cfg=json.loads(cfg_path.read_text(encoding='utf-8-sig'))
d=np.load('results/test_008_demos/cleaned_full_trajectory.npz'); t=d['time']; p=d['smoothed']
def median(a,b):
 pts=p[(t>=a)&(t<=b)]; pts=pts[np.isfinite(pts).all(axis=1)]
 if not len(pts): raise ValueError('No observed endpoint samples')
 return np.median(pts,axis=0).tolist()
snap=lambda v:float(t[np.argmin(abs(t-v))])
cfg['demonstrations']=[dict(start_time=float(t[0]),end_time=float(t[-1]),pickup_time=snap(2.167),release_time=snap(6.75),start=median(.2,2),end=median(6.8,7.65),direction='object -> target',phase_overrides={},review_reason='One complete object-to-target transfer confirmed in half-second video review, including visible release before the end. Pickup is measured-motion onset; release is an approximate video annotation at 6.75s. Stationary endpoint detection failed due noisy/missing placed poses; robust final-resting endpoint uses 6.8-7.65s observations.')]
cfg_path.write_text(json.dumps(cfg,indent=2)+'\n')
