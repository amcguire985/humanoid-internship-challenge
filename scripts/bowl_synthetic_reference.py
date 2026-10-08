"""Oracle position-only diagnostic in the existing phone Reference schema."""
import numpy as np
from bowl_human_reference import Reference, rotation


def generate(start, goal, grasp_rotation, duration=10.8, frequency=20.):
    """Minimum-jerk straight path to a supplied placement approach position.

    Endpoint geometry is supplied by the simulator at confirmed grasp. No
    release commands, extra waypoints, time compression or orientation targets.
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    r = rotation(grasp_rotation)
    if start.shape != (3,) or goal.shape != (3,) or not np.isfinite([start, goal]).all():
        raise ValueError('Expected finite 3D endpoints')
    if not np.isfinite([duration, frequency]).all() or duration <= 0 or frequency <= 0 or duration*frequency < 2:
        raise ValueError('Need positive duration and at least two sampling intervals')
    if np.linalg.norm((goal-start)[:2]) < .01:
        raise ValueError('Existing alignment requires horizontal displacement >= 1 cm')
    t = np.linspace(0., duration, int(np.ceil(duration*frequency))+1)
    u = t/duration
    s = 10*u**3 - 15*u**4 + 6*u**5
    p = start + s[:, None]*(goal-start)
    p[0], p[-1] = start, goal
    return Reference(t, p, np.repeat(r[None], len(t), axis=0), dict(
        source='synthetic_oracle_position_only',
        gravity_verified=True, mounting_verified=True, calibration_verified=True,
        verification_basis='Simulator world frame and object body pose; not phone calibration measurements',
        ground_truth_used='Confirmed-grasp bowl position/rotation and existing physical plate placement target',
        duration_s=float(duration), control_frequency_hz=float(frequency),
        timing_scale=1, orientation_strategy='Constant grasp rotation; transport orientation correction disabled',
        path='Straight quintic minimum-jerk; no extra lift or lowering/release',
        start_m=start.tolist(), approach_target_m=goal.tolist()))
