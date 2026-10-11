"""Gravity-relative orientation shared by simulator and future human evaluation."""
import numpy as np


def compute_upright_deviation_deg(rotation, local_up=(0, 0, 1), world_up=(0, 0, 1)):
    """rotation maps object-frame vectors into world coordinates; yaw is irrelevant."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError('Expected a finite 3x3 object-to-world rotation')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError('Expected a proper rotation matrix')
    local, world = np.asarray(local_up, dtype=float), np.asarray(world_up, dtype=float)
    if local.shape != (3,) or world.shape != (3,) or not np.isfinite(np.r_[local, world]).all() or min(np.linalg.norm(local), np.linalg.norm(world)) == 0:
        raise ValueError('Up axes must be finite nonzero 3-vectors')
    up = rotation @ (local / np.linalg.norm(local))
    return float(np.degrees(np.arccos(np.clip(up @ (world / np.linalg.norm(world)), -1, 1))))


def get_mug_orientation(env, object_name='porcelain_mug_1'):
    """Root-body object-to-world matrix and MuJoCo quaternion w,x,y,z.

    robosuite observation quaternions instead use x,y,z,w.
    Porcelain mug local +Z points from bottom_site toward top_site.
    """
    base = env.env
    obj = base.objects_dict[object_name]
    body_id = base.sim.model.body_name2id(obj.root_body)
    return (np.array(base.sim.data.body_xmat[body_id]).reshape(3, 3).copy(),
            np.array(base.sim.data.body_xquat[body_id]).copy())
