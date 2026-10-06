import cv2,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cap=cv2.VideoCapture('videos/test__008_shutterspeed500.MOV'); cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
fig,axes=plt.subplots(4,4,figsize=(16,10))
for ax,sec in zip(axes.ravel(),np.arange(0,8,.5)):
 cap.set(cv2.CAP_PROP_POS_MSEC,float(sec*1000)); ok,frame=cap.read()
 if ok: ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
 ax.set_title(f'{sec:.1f}s'); ax.axis('off')
cap.release(); fig.tight_layout(); fig.savefig('results/test_008_demos/phase_review.jpg',dpi=120)
