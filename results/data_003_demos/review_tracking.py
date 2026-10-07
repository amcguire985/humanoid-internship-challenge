import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path('scripts').resolve()))
from process_task_demo import read_csv,validate_config,clean_trajectory,trajectory_velocity,compute_goal_position
from extract_transfer_demos import detect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
source=Path('results/data_003_raw'); out=Path('results/data_003_demos')
c=validate_config(json.loads(Path('config/data_003_demos.json').read_text(encoding='utf-8-sig'))['processing'])
rows=read_csv(source/'object/cube_center/trajectory.csv'); t=np.array([float(r['time_s']) for r in rows])
raw=np.array([[float(r[a+'_m']) if r.get(a+'_m') else np.nan for a in 'xyz'] for r in rows]); errors=np.array([float(r['error_px']) if r.get('error_px') else np.nan for r in rows])
cleaned,p,provenance,rejected,reasons,gaps=clean_trajectory(t,raw,errors,c)
np.savez(out/'cleaned_full_trajectory.npz',time=t,raw=raw,cleaned=cleaned,smoothed=p,provenance=provenance,rejected=rejected)
_,speed=trajectory_velocity(t,p,c['max_gap_span']); rests=detect(t,p,speed,c)
rest_report=[dict(start_time=float(t[a['first']]),end_time=float(t[a['last']]),position=a['position'].tolist()) for a in rests]
(out/'rest_review.json').write_text(json.dumps(rest_report,indent=2)+'\n'); print(json.dumps(rest_report,indent=2),flush=True)
fig,axes=plt.subplots(3,1,figsize=(17,8),sharex=True)
for j,ax in enumerate(axes):
 ax.plot(t,p[:,j]); ax.set_ylabel('xyz'[j]+' (m)'); ax.grid(); ax.set_xticks(np.arange(0,t[-1]+2,2))
axes[-1].set_xlabel('Video time (s)'); fig.tight_layout(); fig.savefig(out/'full_overview.png'); plt.close(fig)

