import cv2,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
out=Path('results/data_001_demos'); d=np.load(out/'cleaned_full_trajectory.npz'); t=d['time']; p=d['smoothed']
fig,ax=plt.subplots(3,1,figsize=(16,8),sharex=True)
for j in range(3):
 ax[j].plot(t,p[:,j]); ax[j].set_ylabel('xyz'[j]); ax[j].grid(); ax[j].set_xticks(np.arange(0,81,2))
fig.tight_layout(); fig.savefig(out/'full_overview.png'); plt.close(fig)
cap=cv2.VideoCapture('videos/data_001.MOV'); cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
fig,axes=plt.subplots(5,8,figsize=(20,13))
for ax,sec in zip(axes.ravel(),range(0,80,2)):
 cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000); ok,frame=cap.read()
 if ok: ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
 ax.set_title(str(sec)+'s'); ax.axis('off')
cap.release(); fig.tight_layout(); fig.savefig(out/'video_contact_sheet.jpg',dpi=110)
