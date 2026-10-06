import cv2,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cap=cv2.VideoCapture('videos/data_002.MOV'); cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
fig,axes=plt.subplots(7,5,figsize=(15,16))
for row,begin in enumerate([9,19,33,47,63,75,87]):
 for col in range(5):
  sec=begin+col; cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000); ok,frame=cap.read(); ax=axes[row,col]
  if ok: ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
  ax.set_title(f'{sec}s'); ax.axis('off')
cap.release(); fig.tight_layout(); fig.savefig('results/data_002_demos/placement_review.jpg',dpi=110)
