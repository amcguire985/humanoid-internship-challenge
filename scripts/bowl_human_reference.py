"""Verified gravity-aligned human object references; no hidden time scaling."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def rotation(value):
    value = np.asarray(value, float)
    if value.shape != (3, 3) or not np.isfinite(value).all() or not np.allclose(value.T @ value, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(value), 1):
        raise ValueError('Expected a proper rotation matrix')
    return value


class Reference:
    def __init__(self, times, positions, rotations, metadata=None):
        self.t = np.asarray(times, float)
        self.p = np.asarray(positions, float)
        self.r = np.asarray([rotation(r) for r in rotations])
        if len(self.t) < 3 or self.p.shape != (len(self.t), 3) or len(self.r) != len(self.t) or not np.isfinite(self.p).all() or not np.isfinite(self.t).all() or self.t[0] != 0 or np.any(np.diff(self.t) <= 0):
            raise ValueError('Need finite increasing relative times and matched object poses')
        self.metadata = metadata or {}

    def sample(self, times):
        times = np.clip(np.asarray(times, float), 0, self.t[-1])
        positions = np.column_stack([np.interp(times, self.t, self.p[:, i]) for i in range(3)])
        orientations = Slerp(self.t, Rotation.from_matrix(self.r))(times).as_matrix()
        return positions, orientations

    def aligned(self, start, goal):
        delta = self.p[-1] - self.p[0]
        target = np.asarray(goal) - start
        if np.linalg.norm(delta[:2]) < .01 or np.linalg.norm(target[:2]) < .01:
            raise ValueError('Transport endpoints require horizontal displacement')
        yaw = np.arctan2(target[1], target[0]) - np.arctan2(delta[1], delta[0])
        basis = Rotation.from_euler('z', yaw).as_matrix()
        scale = np.linalg.norm(target[:2]) / np.linalg.norm(delta[:2])
        offsets = (self.p - self.p[0]) @ basis.T
        offsets[:, :2] *= scale
        u = self.t / self.t[-1]
        blend = 10*u**3 - 15*u**4 + 6*u**5
        residual = target - offsets[-1]
        p = np.asarray(start) + offsets + blend[:, None]*residual
        return Reference(self.t, p, basis @ self.r, {**self.metadata, 'yaw_rad': float(yaw), 'planar_scale': float(scale), 'endpoint_residual_m': residual.tolist(), 'timing_scale': 1, 'vertical_scale': 1, 'alignment': 'Yaw, planar scaling, quintic endpoint residual; gravity-relative tilt preserved'})

    def save(self, path):
        q = Rotation.from_matrix(self.r).as_quat()
        v = np.gradient(self.p, self.t, axis=0, edge_order=2)
        a = np.gradient(v, self.t, axis=0, edge_order=2)
        np.savetxt(path, np.column_stack([self.t, self.p, q, v, a]), delimiter=',', comments='', header='time_s,x_m,y_m,z_m,qx,qy,qz,qw,vx_m_s,vy_m_s,vz_m_s,ax_m_s2,ay_m_s2,az_m_s2')
        Path(str(path)+'.json').write_text(json.dumps(self.metadata, indent=2)+'\n')

    @classmethod
    def load(cls, path):
        meta = json.loads(Path(str(path)+'.json').read_text())
        if meta.get('audit_only'): raise ValueError('Audit-only draft cannot be used for simulation')
        for key in ('gravity_verified','mounting_verified','calibration_verified'):
            if meta.get(key) is not True:
                raise ValueError('Reference requires verified '+key)
        rows = np.genfromtxt(path, delimiter=',', names=True)
        return cls(rows['time_s'], np.column_stack([rows[k] for k in ('x_m','y_m','z_m')]), Rotation.from_quat(np.column_stack([rows[k] for k in ('qx','qy','qz','qw')])).as_matrix(), meta)


def prepare(detections, centers, calibration, start, end, output, frequency=20, audit_unverified=False):
    cfg = json.loads(Path(calibration).read_text())
    for key in ('gravity_verified','mounting_verified','calibration_verified'):
        if cfg.get(key) is not True and not audit_unverified:
            raise ValueError('Calibration is not verified: '+key)
    gravity = rotation(cfg['gravity_from_world_rotation'])
    mounts = {int(k): rotation(v) for k,v in cfg['object_from_tag_rotations'].items()}
    if not np.isfinite([start,end,frequency]).all() or end<=start or frequency<=0:
        raise ValueError('Invalid transport timing or frequency')
    poses = {}
    with Path(detections).open(newline='') as f:
        for row in csv.DictReader(f):
            tag = int(row['tag_id'])
            if tag not in mounts or row['status'] != 'observed' or not row.get('world_relative_rx_rad'):
                continue
            r = Rotation.from_rotvec([float(row['world_relative_'+axis+'_rad']) for axis in ('rx','ry','rz')]).as_matrix()
            poses.setdefault(int(row['frame']), []).append((float(row['error_px']), gravity @ r @ mounts[tag].T))
    t,p,r = [],[],[]
    rejected_candidates=0
    with Path(centers).open(newline='') as f:
        for row in csv.DictReader(f):
            time = float(row['time_s'])
            # Include neighbors to interpolate the exact reviewed boundaries.
            if time < start-.15 or time > end+.15:
                continue
            if not all(row.get(axis+'_m') for axis in 'xyz') or int(row['frame']) not in poses:
                if start<=time<=end:
                    raise ValueError('Unresolved position/orientation gap inside transport')
                continue
            t.append(time); p.append(gravity @ np.array([float(row[axis+'_m']) for axis in 'xyz']))
            candidates=poses[int(row['frame'])]
            if r:
                continuous=[candidate for candidate in candidates if np.degrees(Rotation.from_matrix(candidate[1]@r[-1].T).magnitude())<=30]
                rejected_candidates+=len(candidates)-len(continuous)
                if not continuous: raise ValueError('No temporally consistent object orientation at source time '+str(time))
                candidates=continuous
            r.append(min(candidates,key=lambda x:x[0])[1])
    if len(t)<3 or t[0]>start or t[-1]<end or np.max(np.diff(t))>.15:
        raise ValueError('Incomplete boundary coverage or source timestamp gap')
    meta={**cfg, 'audit_only': bool(audit_unverified), 'source_detections_sha256': hashlib.sha256(Path(detections).read_bytes()).hexdigest(), 'source_centers_sha256': hashlib.sha256(Path(centers).read_bytes()).hexdigest(), 'source_start_s': start, 'source_end_s': end, 'smoothing': 'None; linear positions and quaternion SLERP', 'timing_scale': 1, 'control_frequency_hz': frequency, 'orientation_candidate_rejections': rejected_candidates, 'pose_selection': 'Lowest reprojection error among measured face poses within 30 degrees of previous accepted object orientation; no orientation interpolation across missing source frames'}
    jumps=np.degrees((Rotation.from_matrix(np.asarray(r)[1:])*Rotation.from_matrix(np.asarray(r)[:-1]).inv()).magnitude())
    meta['source_max_orientation_step_deg']=float(jumps.max())
    meta['orientation_jumps_over_30deg']=int((jumps>30).sum())
    if np.any(jumps>30):
        raise ValueError('Raw orientation jumps exceed 30 degrees per source sample; review pose ambiguities before using reference')
    source=Reference(np.array(t)-t[0],p,r,meta)
    grid=np.unique(np.r_[np.arange(0,end-start,1/frequency),end-start])
    p,r=source.sample(grid+start-t[0]); ref=Reference(grid,p,r,meta)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); ref.save(output)
    preview(ref,output.with_suffix('.png')); return ref


def preview(ref, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    axes[0].plot(ref.t,ref.p-ref.p[0]); axes[0].set_ylabel('Object offset (m)')
    axes[1].plot(ref.t,np.degrees(np.arccos(np.clip(ref.r[:,2,2],-1,1)))); axes[1].set_ylabel('Gravity tilt (deg)')
    v=np.gradient(ref.p,ref.t,axis=0,edge_order=2)
    axes[2].plot(ref.t,np.linalg.norm(np.gradient(v,ref.t,axis=0,edge_order=2),axis=1)); axes[2].set(ylabel='Reference acceleration (m/s2)',xlabel='Unscaled human time (s)')
    fig.tight_layout(); fig.savefig(output); plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('detections','centers','calibration','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--audit-unverified',action='store_true',help='Export a clearly unverified draft for inspection; the simulator rejects it')
    parser.add_argument('--start',type=float,required=True); parser.add_argument('--end',type=float,required=True)
    args=parser.parse_args(); prepare(**vars(args))
