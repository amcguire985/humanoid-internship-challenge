"""Overlay smoothed world-frame centres using current-frame stationary-tag camera poses."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np


def load_rows(path):
    with path.open(newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def run(root, video_output=None):
    root=Path(root)
    summary=json.loads((root/'summary.json').read_text())
    trajectories={body:load_rows(root/path/'gap_filled_10_frames/gentle_smoothing/savgol_7.csv')
                  for body,path in [('Hand',Path('hand/hand_center')),('Cube',Path('object/cube_center'))]}
    layout={int(k):np.array(v) for k,v in summary['world_tag_transforms'].items()}
    matrix=np.array(summary['effective_camera_matrix'])
    distortion=np.array(summary['distortion_coefficients'])
    world_by_frame={}
    for record in load_rows(root/'detections.csv'):
        if record['status']!='observed' or not record['world_source_id'] or record['tag_id']!=record['world_source_id']:
            continue
        tag=int(record['tag_id'])
        camera=np.eye(4)
        camera[:3,3]=[float(record['camera_'+a+'_m']) for a in 'xyz']
        camera[:3,:3]=cv2.Rodrigues(np.array([float(record['camera_relative_'+a+'_rad']) for a in ('rx','ry','rz')]))[0]
        world_by_frame[int(record['frame'])]=camera@np.linalg.inv(layout[tag])
    from video_paths import resolve_video, output_video
    video=resolve_video(summary['video'])
    cap=cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
    fps=cap.get(cv2.CAP_PROP_FPS)
    ok,frame=cap.read()
    if not ok or not np.isfinite(fps) or fps<=0:
        raise RuntimeError('Cannot decode source video')
    height,width=frame.shape[:2]
    destination=output_video(root, 'sg7_gap_filled_annotated.mp4', video_output)
    writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Cannot open MP4 writer')
    colors={'Hand':(0,210,255),'Cube':(255,210,0)}
    counts={body:dict(observed=0,interpolated=0,missing=0) for body in trajectories}
    histories={body:[] for body in trajectories}
    index=0
    def project(points,world):
        points=np.asarray(points,dtype=float).reshape(-1,3)
        camera=(world[:3,:3]@points.T).T+world[:3,3]
        image=cv2.projectPoints(camera,np.zeros(3),np.zeros(3),matrix,distortion)[0].reshape(-1,2)
        return image,camera[:,2]>0
    try:
        while ok:
            if index>=summary['frames_processed']:
                raise RuntimeError('Source video has more frames than the trajectory')
            world=world_by_frame.get(index)
            panel=frame[:200].copy()
            panel[:]=(20,20,20)
            frame[:200]=cv2.addWeighted(frame[:200],.25,panel,.75,0)
            time_s=float(trajectories['Hand'][index]['time_s'])
            cv2.putText(frame,f'{time_s:.2f}s | SG 7 frames | gap fill <=10 frames',(16,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            cv2.putText(frame,'Filled circle: observed | ring: interpolated | trail: last 1 s',(16,61),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,220,220),1)
            for body,rows in trajectories.items():
                row=rows[index]
                if int(row['frame'])!=index:
                    raise RuntimeError('CSV frame indices do not match video')
                source=row['pose_source']
                color=colors[body]
                counts[body][source]+=1
                line=97 if body=='Hand' else 132
                if source=='missing':
                    histories[body]=[]
                    text=f'{body}: unavailable'
                else:
                    xyz=np.array([float(row[a+'_m']) for a in 'xyz'])
                    histories[body].append((time_s,xyz))
                    histories[body]=[(t,p) for t,p in histories[body] if time_s-t<=1]
                    text=f'{body}: {source} | x={xyz[0]*100:+.1f} y={xyz[1]*100:+.1f} z={xyz[2]*100:+.1f} cm'
                    if world is not None:
                        pixels,positive=project([p for _,p in histories[body]],world)
                        inside=positive & np.isfinite(pixels).all(axis=1) & (pixels[:,0]>=0) & (pixels[:,0]<width) & (pixels[:,1]>=200) & (pixels[:,1]<height)
                        for j in range(1,len(pixels)):
                            if inside[j-1:j+1].all():
                                cv2.line(frame,tuple(np.rint(pixels[j-1]).astype(int)),tuple(np.rint(pixels[j]).astype(int)),color,2,cv2.LINE_AA)
                        if inside[-1]:
                            point=tuple(np.rint(pixels[-1]).astype(int))
                            cv2.circle(frame,point,10,color,-1 if source=='observed' else 3,cv2.LINE_AA)
                            cv2.putText(frame,body,tuple((np.array(point)+[14,-12]).tolist()),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
                    else:
                        text+=' | world projection unavailable'
                cv2.putText(frame,text,(16,line),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
            cv2.putText(frame,'Positions in ID0 world frame; both centres use 45 mm cube geometry',(16,170),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,220,220),1)
            writer.write(frame)
            if index==int(fps*5):
                cv2.imwrite(str(root/'sg7_gap_filled_preview.jpg'),frame)
            index+=1
            if index%200==0:
                print(f'Overlaid {index} frames',flush=True)
            ok,frame=cap.read()
    finally:
        cap.release()
        writer.release()
    if index!=summary['frames_processed']:
        raise RuntimeError('Source video ended before trajectory')
    report=dict(video=str(video),output=str(destination),frames=index,fps=fps,frame_size=[width,height],
                smoothing='savgol_7',max_gap_frames=10,counts=counts,
                projection='Current-frame stationary-tag camera pose; historical world centres reprojected into current camera.',
                audio=False)
    (root/'sg7_gap_filled_overlay.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Saved {destination}: {index} frames; {counts}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory',type=Path)
    parser.add_argument("--video-output", type=Path, help="External overlay MP4 path")
    args=parser.parse_args()
    run(args.run_directory, args.video_output)
