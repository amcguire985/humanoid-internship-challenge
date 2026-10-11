import json,numpy as np
from pathlib import Path
path=Path('scripts/extract_transfer_demos.py')
s=path.read_text(encoding='utf-8-sig')
s=s.replace("cc['phase_overrides']={**dict(pickup_time=item['pickup_time'],release_time=item['release_time']),**item.get('phase_overrides',{})}","cc['phase_overrides']={**dict(pickup_time=item['pickup_time'],release_time=item['release_time'], transport_start=max(item['pickup_time'],min(estimated['transport_start'] or item['pickup_time'],item['release_time'])), transport_end=max(item['pickup_time'],min(estimated['transport_end'] or item['release_time'],item['release_time']))),**item.get('phase_overrides',{})}")
path.write_text(s,encoding='utf-8')
d=np.load('results/data_001_demos/cleaned_full_trajectory.npz'); t=d['time']; p=d['smoothed']
cfg=json.loads(Path('config/data_001_demos.json').read_text(encoding='utf-8-sig'))
# Resting windows checked against the video contact sheet and measured motion.
windows=[(0,16.5,4.135,11.0,(0.3,3.5),(12,15)),(16.5,27,17.2,24.5,(16.5,17),(25,27)),(27,41,28.7,36.217,(27,28),(37,40)),(41,51,42.82,48.155,(40,42),(49,51)),(51,61,52.525,57.46,(49,51),(59,61)),(61,71.5,63.4,68.0,(59,61),(69,71)),(71.5,80.3,72.933,78.57,(69,71),(78.57,80.2))]
cfg['demonstrations']=[]
for i,(a,b,pick,release,sw,ew) in enumerate(windows):
 snap=lambda v:float(t[np.argmin(abs(t-v))])
 median=lambda w:np.nanmedian(p[(t>=w[0])&(t<=w[1])],axis=0).tolist()
 cfg['demonstrations'].append(dict(start_time=snap(a),end_time=snap(b),pickup_time=snap(pick),release_time=snap(release),start=median(sw),end=median(ew),direction='object -> target' if i%2==0 else 'target -> object',review_reason='Transfer identity/direction reviewed in video contact sheet and full measured trajectory. Motion onset/end timings are approximate; edit phase_overrides for contact-reviewed boundaries. Early quiet detection failed due pose jitter; seven transfers selected explicitly.',phase_overrides={}))
Path('config/data_001_demos.json').write_text(json.dumps(cfg,indent=2)+'\n')
