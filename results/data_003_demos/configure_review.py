import json,numpy as np
from pathlib import Path
path=Path('config/data_003_demos.json'); cfg=json.loads(path.read_text(encoding='utf-8-sig'))
d=np.load('results/data_003_demos/cleaned_full_trajectory.npz'); t=d['time']; p=d['smoothed']
snap=lambda v:float(t[np.argmin(abs(t-v))])
def median(w):
 pts=p[(t>=w[0])&(t<=w[1])]; pts=pts[np.isfinite(pts).all(axis=1)]
 if not len(pts): raise ValueError('Missing resting endpoint')
 return np.median(pts,axis=0).tolist()
windows=[(0,22,5.5,19.2,(.5,3.5),(20,22)),(22,40,25.6,36.8,(20,22),(38,40)),(40,float(t[-1]),42.6,56.6,(38,40),(57,57.6))]
cfg['demonstrations']=[]
for i,(a,b,pick,release,sw,ew) in enumerate(windows):
 cfg['demonstrations'].append(dict(start_time=snap(a),end_time=snap(b),pickup_time=snap(pick),release_time=snap(release),start=median(sw),end=median(ew),direction='object -> target' if i%2==0 else 'target -> object',phase_overrides={},review_reason='Three complete alternating transfers independently verified in video contact sheet and denser placement review; final release is visible before clip ends. Pickup is approximate measured lift onset; release is approximate video annotation. Stationary detector merges transfers because resting position jitter exceeds its threshold. Endpoints use measured resting medians. Inner transport phases remain automatic and need semantic review.'))
path.write_text(json.dumps(cfg,indent=2)+'\n')
print(dict(raw_observed=int(np.isfinite(d['raw']).all(axis=1).sum()),cleaned_valid=int(np.isfinite(p).all(axis=1).sum()),rejected=int(d['rejected'].sum())))

