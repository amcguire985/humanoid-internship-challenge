"""Estimate fixed tag mounting rotations from co-visible raw object tags.

Object +Z is the outward normal of the explicitly specified top-face tag.
Gravity direction and camera-calibration validity require independent confirmation.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def estimate(path, top_tag=6, object_tags=(4,5,6), max_scatter_deg=15):
    frames={}
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            tag=int(row['tag_id'])
            if tag not in object_tags or row['status']!='observed' or not row.get('world_relative_rx_rad'): continue
            frames.setdefault(int(row['frame']),{}).setdefault(tag,[]).append(Rotation.from_rotvec([float(row['world_relative_'+axis+'_rad']) for axis in ('rx','ry','rz')]).as_matrix())
    edges={}; reports={}
    for a in object_tags:
        for b in object_tags:
            if a>=b: continue
            candidates=[tags[a][0].T@tags[b][0] for tags in frames.values() if len(tags.get(a,[]))==len(tags.get(b,[]))==1]
            if len(candidates)<10: continue
            rotations=Rotation.from_matrix(np.array(candidates))
            mean=rotations.mean(); deviations=np.degrees((rotations*mean.inv()).magnitude())
            # A robust first pass removes pose ambiguities; retained fit is audited.
            keep=deviations<=max_scatter_deg
            if keep.sum()<10 or keep.mean()<.8: raise ValueError('Inconsistent mounting pair '+str((a,b)))
            mean=rotations[keep].mean(); residual=np.degrees((rotations[keep]*mean.inv()).magnitude())
            edges[a,b]=mean.as_matrix(); edges[b,a]=mean.as_matrix().T
            reports[str((a,b))]=dict(samples=len(candidates),retained=int(keep.sum()),discarded=int((~keep).sum()),p95_scatter_deg=float(np.percentile(residual,95)))
    mounts={top_tag:np.eye(3)}
    for _ in object_tags:
        for (a,b),matrix in edges.items():
            if a in mounts and b not in mounts: mounts[b]=mounts[a]@matrix
    if set(mounts)!=set(object_tags): raise ValueError('Not enough co-visible tag observations to connect object mounting frames')
    closure={str((a,b)):float(np.degrees(Rotation.from_matrix(mounts[a]@edge@mounts[b].T).magnitude())) for (a,b),edge in edges.items() if a<b}
    if max(closure.values(),default=0)>max_scatter_deg: raise ValueError('Mounting graph closure is inconsistent')
    reports['graph_closure_error_deg']=closure
    return mounts,reports


def gravity_rotation(up):
    up=np.asarray(up,float)
    if up.shape!=(3,) or not np.isfinite(up).all() or np.linalg.norm(up)<1e-6: raise ValueError('Invalid gravity-up vector')
    up=up/np.linalg.norm(up)
    x=np.array([1.,0,0]); x-=np.dot(x,up)*up
    if np.linalg.norm(x)<.1: x=np.array([0.,1,0]); x-=np.dot(x,up)*up
    x/=np.linalg.norm(x); y=np.cross(up,x)
    return np.stack([x,y,up])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detections',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--top-tag',type=int,default=6)
    parser.add_argument('--gravity-up-world',type=float,nargs=3,default=[0,0,1])
    parser.add_argument('--confirm-top-face',action='store_true'); parser.add_argument('--confirm-gravity',action='store_true'); parser.add_argument('--confirm-camera-calibration',action='store_true')
    args=parser.parse_args(); mounts,report=estimate(args.detections,args.top_tag)
    config=dict(gravity_verified=args.confirm_gravity,mounting_verified=args.confirm_top_face,calibration_verified=args.confirm_camera_calibration,gravity_from_world_rotation=gravity_rotation(args.gravity_up_world).tolist(),object_from_tag_rotations={str(k):v.tolist() for k,v in mounts.items()},top_tag=args.top_tag,mounting_fit=report,object_axis_definition='Object +Z = specified top tag outward normal; XY = decoded top tag axes. Fixed face transforms estimated from co-visible observations; no human hand orientation.',warning=None if args.confirm_gravity and args.confirm_top_face and args.confirm_camera_calibration else 'Unconfirmed geometry: use only a draft until all setup facts are verified')
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(config,indent=2)+'\n'); print(json.dumps(config,indent=2))
