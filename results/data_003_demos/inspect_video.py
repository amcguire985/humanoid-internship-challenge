import cv2,numpy as np,matplotlib,json
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
out=Path('results/data_003_demos'); cap=cv2.VideoCapture('videos/data_003.MOV'); cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
fps=cap.get(cv2.CAP_PROP_FPS); frames=cap.get(cv2.CAP_PROP_FRAME_COUNT); duration=frames/fps
print(dict(fps=fps,frames=frames,duration_s=duration),flush=True)
seconds=np.arange(0,duration,2)
for page in range((len(seconds)+39)//40):
 fig,axes=plt.subplots(5,8,figsize=(20,13))
 for ax,sec in zip(axes.ravel(),seconds[page*40:(page+1)*40]):
  cap.set(cv2.CAP_PROP_POS_MSEC,float(sec*1000)); ok,frame=cap.read()
  if ok: ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
  ax.set_title(str(sec)+'s'); ax.axis('off')
 fig.tight_layout(); fig.savefig(out/f'video_contact_sheet_{page+1}.jpg',dpi=110); plt.close(fig)
cap.release()

