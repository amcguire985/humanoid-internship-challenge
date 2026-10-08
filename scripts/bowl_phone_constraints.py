"""Phone-derived transport constraints; no path or demonstration clock tracking."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from bowl_human_reference import Reference
from smooth_centers import smooth_positions
from bowl_transport_guidance import opening_axis_rotation_vector


def extract(reference, window=7, percentile=95., acceleration_margin=1., tilt_margin_deg=0.):
    if window < 3 or window % 2 != 1 or window > len(reference.t) or not 0 < percentile < 100:
        raise ValueError('Invalid smoothing window or percentile')
    if not np.isfinite([acceleration_margin, tilt_margin_deg]).all() or acceleration_margin <= 0 or tilt_margin_deg < 0:
        raise ValueError('Invalid explicit margins')
    if np.max(np.diff(reference.t)) > .15:
        raise ValueError('Reference has unresolved timestamp gaps')
    p=smooth_positions(reference.t, reference.p, 'savgol', window)
    v=np.gradient(p, reference.t, axis=0, edge_order=2)
    a=np.linalg.norm(np.gradient(v, reference.t, axis=0, edge_order=2),axis=1)
    # Exclude the local-fit and derivative boundary neighborhood.
    edge=window//2+2
    interior=a[edge:-edge]
    if len(interior)<3: raise ValueError('Transport interval too short')
    tilt=np.degrees(np.arccos(np.clip(reference.r[:,2,2],-1,1)))
    measured=float(np.percentile(interior,percentile))
    allowed=float(np.percentile(tilt,percentile))
    if measured <= 0 or allowed+tilt_margin_deg >= 180: raise ValueError('Degenerate constraints')
    return dict(acceleration_limit_m_s2=measured*acceleration_margin,
        allowable_tilt_deg=allowed+tilt_margin_deg, gravity_target=[0,0,1],
        measured_acceleration_percentile_m_s2=measured, measured_tilt_percentile_deg=allowed,
        acceleration_margin_multiplier=acceleration_margin, tilt_margin_deg=tilt_margin_deg,
        percentile=percentile, smoothing_method='Existing smooth_positions savgol local quadratic actual-time fit',
        smoothing_window_samples=window, excluded_boundary_samples_each_end=edge,
        acceleration_method='Two numpy.gradient derivatives on actual seconds; vector norm; transport interior percentile',
        tilt_method='acos(world_from_object[2,2]); +Z opening axis relative to world gravity',
        transport_duration_s=float(reference.t[-1]), sample_count=len(reference.t),
        interpretation='Empirical processed phone motion, not sensor accuracy or a dynamics guarantee; percentile/window are analysis choices')


def export(source, output):
    source,output=Path(source),Path(output)
    constraints=extract(Reference.load(source))
    constraints.update(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_metadata_sha256=hashlib.sha256(Path(str(source)+'.json').read_bytes()).hexdigest())
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(constraints,indent=2,allow_nan=False)+'\n')
    print(json.dumps(constraints,indent=2)); return constraints


class MotionConstraints:
    def __init__(self, constraints, controller, settings, initial_action, dt):
        self.c=controller; self.s=settings.validate(); self.limits=constraints
        a=float(constraints['acceleration_limit_m_s2']); tilt=float(constraints['allowable_tilt_deg'])
        if not np.isfinite([a,tilt,dt]).all() or a<=0 or not 0<=tilt<180 or dt<=0:
            raise ValueError('Invalid constraints/control timestep')
        self.previous_velocity=self.decode(np.asarray(initial_action,float)[:3])/dt
        self.previous_rotation=np.zeros(3)

    def decode(self, action):
        c=self.c; lo=c['input_min'][:3]; hi=c['input_max'][:3]
        return (np.clip(action,lo,hi)-lo)/(hi-lo)*(c['output_max'][:3]-c['output_min'][:3])+c['output_min'][:3]

    def encode(self, delta):
        c=self.c
        return (delta-c['output_min'][:3])/(c['output_max'][:3]-c['output_min'][:3])*(c['input_max'][:3]-c['input_min'][:3])+c['input_min'][:3]

    def apply(self, policy_action, bowl_rotation, dt, elapsed):
        action=np.asarray(policy_action,float)
        if action.shape!=(7,) or not np.isfinite(action).all() or not np.isfinite([dt,elapsed]).all() or dt<=0 or elapsed<0:
            raise ValueError('Invalid action or control time')
        result=np.clip(action,self.c['action_min'],self.c['action_max'])
        proposed=self.decode(result[:3])/dt
        change=proposed-self.previous_velocity; norm=float(np.linalg.norm(change))
        fraction=min(1.,self.limits['acceleration_limit_m_s2']*dt/max(norm,1e-15))
        velocity=self.previous_velocity+fraction*change
        result[:3]=self.encode(velocity*dt)
        actual_velocity=self.decode(result[:3])/dt
        commanded=float(np.linalg.norm(actual_velocity-self.previous_velocity)/dt)
        self.previous_velocity=actual_velocity
        error=opening_axis_rotation_vector(bowl_rotation)
        angle=float(np.linalg.norm(error)); excess=max(0.,angle-np.radians(self.limits['allowable_tilt_deg']))
        # Existing OSC rotation scaling and existing blend/gain/cap/ramp/slew.
        swing=error*(excess/max(angle,1e-15))*min(1.,elapsed/self.s.ramp_s)
        c=self.c
        rotation_action=swing/(c['output_max'][3:]-c['output_min'][3:])*(c['input_max'][3:]-c['input_min'][3:])
        desired=np.clip(self.s.orientation_blend*self.s.orientation_gain*rotation_action,-self.s.correction_limit,self.s.correction_limit)
        if not self.s.orientation_enabled: desired=np.zeros(3)
        correction=np.clip(desired,self.previous_rotation-self.s.correction_slew_per_s*dt,self.previous_rotation+self.s.correction_slew_per_s*dt)
        result[3:6]=np.clip(result[3:6]+correction,c['action_min'][3:6],c['action_max'][3:6])
        self.previous_rotation=result[3:6]-np.clip(action[3:6],c['action_min'][3:6],c['action_max'][3:6])
        return result,dict(transport_mode='phone_constraints',constraints_active=True,control_dt_s=float(dt),
            proposed_cartesian_velocity_m_s=proposed.tolist(),commanded_cartesian_velocity_m_s=actual_velocity.tolist(),
            proposed_commanded_acceleration_m_s2=norm/dt,commanded_acceleration_m_s2=commanded,
            acceleration_limit_m_s2=self.limits['acceleration_limit_m_s2'],acceleration_limiter_fraction=fraction,
            acceleration_limiter_active=bool(fraction<1.),tilt_excess_deg=float(np.degrees(excess)),
            constraint_rotation_correction=self.previous_rotation.tolist(),constraint_action_change=(result-action).tolist(),
            commanded_acceleration_definition='Scaled OSC translation delta / control dt; not measured EE or bowl acceleration')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();export(args.reference,args.output)
