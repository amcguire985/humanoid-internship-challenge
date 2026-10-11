import cv2,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cap=cv2.VideoCapture('videos/data_003.MOV'); cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
fig,axes=plt.subplots(3,6,figsize=(18,9))
for row,seconds in enumerate(([15,16,17,18,19,20],[33,34,35,36,37,38],[53,54,55,56,57,57.5])):
 for ax,sec in zip(axes[row],seconds):
  cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000); ok,frame=cap.read()
  if ok: ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
  ax.set_title(f'{sec}s'); ax.axis('off')
cap.release(); fig.tight_layout(); fig.savefig('results/data_003_demos/placement_review.jpg',dpi=120)
