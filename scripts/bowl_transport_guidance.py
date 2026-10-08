"""Small bounded human object-pose intervention for world-frame delta OSC."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from replay_libero_transport import desired_pose_to_action
from bowl_human_reference import rotation


def pose(position, orientation):
    result=np.eye(4); result[:3,:3]=rotation(orientation); result[:3,3]=np.asarray(position,float)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite pose')
    return result



def opening_axis_rotation_vector(bowl_rotation, gravity_up=(0.,0.,1.)):
    """Shortest world-frame swing of object +Z onto gravity; no yaw objective."""
    opening=rotation(bowl_rotation)[:,2]
    gravity=np.asarray(gravity_up,float)
    if gravity.shape!=(3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity)<1e-12:
        raise ValueError('Invalid gravity-up direction')
    gravity=gravity/np.linalg.norm(gravity)
    cross=np.cross(opening,gravity); sine=float(np.linalg.norm(cross))
    cosine=float(np.clip(np.dot(opening,gravity),-1,1))
    if sine<1e-12:
        if cosine>0: return np.zeros(3)
        # Exact inversion has no unique shortest swing; choose a deterministic axis.
        basis=np.eye(3)[int(np.argmin(np.abs(opening)))]
        axis=np.cross(opening,basis); axis/=np.linalg.norm(axis)
        return np.pi*axis
    return np.arctan2(sine,cosine)*cross/sine


def physical_placement_goal(base, bowl_object, plate_object):
    """Use collision boxes, never asset top/bottom extent sites, for placement."""
    sim=base.sim
    bowl_id=sim.model.body_name2id(bowl_object.root_body)
    plate_id=sim.model.body_name2id(plate_object.root_body)
    bowl_position=np.asarray(sim.data.body_xpos[bowl_id])
    bowl_rotation=np.asarray(sim.data.body_xmat[bowl_id]).reshape(3,3)
    plate_position=np.asarray(sim.data.body_xpos[plate_id])
    bottoms=[]
    for name in bowl_object.contact_geoms:
        geom=sim.model.geom_name2id(name)
        if int(sim.model.geom_type[geom])!=6: raise ValueError('Placement audit requires bowl collision boxes')
        local_position=bowl_rotation.T@(sim.data.geom_xpos[geom]-bowl_position)
        local_rotation=bowl_rotation.T@sim.data.geom_xmat[geom].reshape(3,3)
        bottoms.append(float(local_position[2]-np.abs(local_rotation[2])@sim.model.geom_size[geom]))
    candidates=[]
    for name in plate_object.contact_geoms:
        geom=sim.model.geom_name2id(name)
        if int(sim.model.geom_type[geom])!=6: raise ValueError('Placement audit requires plate collision boxes')
        center=np.asarray(sim.data.geom_xpos[geom]); orient=sim.data.geom_xmat[geom].reshape(3,3)
        top=float(center[2]+np.abs(orient[2])@sim.model.geom_size[geom])
        candidates.append((float(np.linalg.norm(center[:2]-plate_position[:2])),top,name))
    if not bottoms or not candidates: raise ValueError('Missing bowl/plate collision geometry')
    distance,surface,name=min(candidates,key=lambda item:item[0])
    if distance>.02: raise ValueError('No central plate collision support surface')
    goal=plate_position.copy(); goal[2]=surface-min(bottoms)
    return goal,dict(method='Central plate collision-box top minus upright bowl collision-box bottom; ignores extent sites',plate_support_geom=name,plate_surface_z_m=surface,bowl_upright_bottom_offset_m=min(bottoms),placement_goal_m=goal.tolist())


@dataclass
class Settings:
    position_blend: float=.25
    orientation_blend: float=.35
    correction_limit: float=.15
    correction_slew_per_s: float=.6
    ramp_s: float=1.
    position_gain: float=.6
    orientation_gain: float=.4
    slip_translation_m: float=.025
    slip_rotation_deg: float=20.
    lift_m: float=.02
    confirm_steps: int=3
    grasp_timeout_steps: int=180
    loss_steps: int=3
    release_radius_m: float=.06
    release_height_m: float=.07
    orientation_enabled: bool=True

    def validate(self):
        for name in ('position_blend','orientation_blend'):
            if not 0<=getattr(self,name)<=1: raise ValueError('Invalid blend')
        for name in ('correction_limit','correction_slew_per_s','ramp_s','position_gain','orientation_gain','slip_translation_m','slip_rotation_deg','lift_m','release_radius_m','release_height_m'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0: raise ValueError('Invalid setting '+name)
        if self.correction_limit>1: raise ValueError('Correction limit exceeds normalized action range')
        for name in ('confirm_steps','grasp_timeout_steps','loss_steps'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1: raise ValueError('Invalid count '+name)
        if type(self.orientation_enabled) is not bool: raise ValueError('orientation_enabled must be boolean')
        return self


class Phases:
    def __init__(self, settings):
        self.s=settings; self.phase='APPROACH'; self.confirm=0; self.loss=0
        self.start=None; self.end=None; self.failure=None; self.grasp_loss=None

    def update(self, step, time, grasp, lift, closing, released, success):
        previous=self.phase
        if success:
            self.phase='DONE'
            if self.start is not None and self.end is None: self.end=(step,time)
        elif self.phase in ('APPROACH','GRASP'):
            if closing: self.phase='GRASP'
            self.confirm=self.confirm+1 if grasp is True and lift>=self.s.lift_m else 0
            if self.confirm>=self.s.confirm_steps:
                self.phase='TRANSPORT'; self.start=(step,time)
            elif step>=self.s.grasp_timeout_steps:
                self.phase='FAILED'; self.failure='grasp_timeout'
        elif self.phase=='TRANSPORT':
            if released:
                self.phase='RELEASE'; self.end=(step-1,time)
            else:
                self.loss=self.loss+1 if grasp is not True else 0
                if self.loss>=self.s.loss_steps:
                    self.grasp_loss=(step-self.loss+1,time)
                    self.phase='FAILED'; self.failure='confirmed_grasp_loss'; self.end=(step,time)
        return previous!=self.phase


class Guidance:
    def __init__(self, reference, bowl_pose, eef_pose, goal, controller, settings):
        self.s=settings.validate(); self.reference=reference.aligned(bowl_pose[:3,3],goal)
        self.gripper_bowl=np.linalg.inv(eef_pose)@bowl_pose
        self.controller=controller; self.previous_correction=np.zeros(6)
        self.reference_velocity=np.gradient(self.reference.p,self.reference.t,axis=0,edge_order=2)
        self.reference_acceleration=np.gradient(self.reference_velocity,self.reference.t,axis=0,edge_order=2)

    def apply(self, policy_action, bowl_pose, eef_pose, elapsed, dt):
        s=self.s; action=np.asarray(policy_action,float)
        if action.shape!=(7,) or not np.isfinite(action).all() or not np.isfinite([elapsed,dt]).all() or elapsed<0 or dt<=0:
            raise ValueError('Invalid action or control time')
        relative=np.linalg.inv(eef_pose)@bowl_pose
        drift_position=float(np.linalg.norm(relative[:3,3]-self.gripper_bowl[:3,3]))
        drift_angle=float(np.degrees(Rotation.from_matrix(relative[:3,:3]@self.gripper_bowl[:3,:3].T).magnitude()))
        info={'slip_translation_m':drift_position,'slip_rotation_deg':drift_angle,'slip_detected':drift_position>s.slip_translation_m or drift_angle>s.slip_rotation_deg}
        p,r=self.reference.sample([elapsed])
        velocity=[float(np.interp(elapsed,self.reference.t,self.reference_velocity[:,i])) for i in range(3)]
        acceleration=[float(np.interp(elapsed,self.reference.t,self.reference_acceleration[:,i])) for i in range(3)]
        info.update(desired_bowl_velocity_m_s=velocity,desired_bowl_acceleration_m_s2=acceleration,desired_bowl_position_m=p[0].tolist(),desired_bowl_rotation=r[0].tolist(),reference_elapsed_s=float(elapsed),reference_finished=bool(elapsed>=self.reference.t[-1]))
        if info['slip_detected']:
            info['guidance_disabled_reason']='rigid_grasp_transform_drift'
            return action.copy(),info
        ramp=np.clip(elapsed/s.ramp_s,0,1)
        # Fade tracking at the goal to let the policy place and release naturally.
        distance=np.linalg.norm(bowl_pose[:2,3]-self.reference.p[-1,:2])
        terminal=np.clip(distance/s.release_radius_m,0,1)
        weight=float(ramp*terminal)
        # Human orientation is recorded for audit, but arbitrary tag yaw/tilt is not a goal.
        # Keep actual yaw free; apply the shortest gravity-aligning swing to the bowl
        # and the actual gripper together, avoiding frozen-transform twist correction.
        opening_error=opening_axis_rotation_vector(bowl_pose[:3,:3])
        swing=Rotation.from_rotvec(ramp*opening_error).as_matrix()
        desired_rotation=swing@bowl_pose[:3,:3]
        target=pose(p[0],desired_rotation)@np.linalg.inv(self.gripper_bowl)
        target[:3,:3]=swing@eef_pose[:3,:3]
        info.update(orientation_mode='gravity_opening_axis',guidance_mode='transport',
            orientation_target_bowl_rotation=desired_rotation.tolist(),
            orientation_target_opening_axis=desired_rotation[:,2].tolist(),
            opening_axis_error_world_rad=opening_error.tolist(),
            human_orientation_used_for_rotation=False)

        tracking,_=desired_pose_to_action(target[:3,3],eef_pose[:3,3],target[:3,:3],eef_pose[:3,:3],self.controller,clip=1,gripper=action[6])
        # Position targets advance on unscaled human time. This influences both speed and
        # acceleration through feedback, without pretending to impose a physics-level limit.
        gain=np.r_[np.full(3,s.position_gain),np.full(3,s.orientation_gain)]
        blend=np.r_[np.full(3,s.position_blend),np.full(3,s.orientation_blend if s.orientation_enabled else 0)]
        correction=np.clip(weight*blend*(gain*tracking[:6]-action[:6]),-s.correction_limit,s.correction_limit)
        correction=np.clip(correction,self.previous_correction-s.correction_slew_per_s*dt,self.previous_correction+s.correction_slew_per_s*dt)
        if not s.orientation_enabled: correction[3:]=0
        self.previous_correction=correction.copy()
        result=action.copy(); result[:6]+=correction
        # Keep the bowl held until it is near the plate, then leave release to SmolVLA.
        safe_release=distance<=s.release_radius_m and abs(bowl_pose[2,3]-self.reference.p[-1,2])<=s.release_height_m
        if result[6]<0 and not safe_release: result[6]=1
        result=np.clip(result,self.controller['action_min'],self.controller['action_max'])
        info.update(correction=correction.tolist(),guidance_weight=weight,release_allowed=bool(safe_release),gripper_gate_applied=bool(action[6]<0 and not safe_release),desired_eef_pose=target.tolist())
        return result,info


    def release_orientation(self, policy_action, dt):
        """Slew the last rotation correction to zero; leave translation/gripper alone."""
        action=np.asarray(policy_action,float)
        if action.shape!=(7,) or not np.isfinite(action).all() or not np.isfinite(dt) or dt<=0:
            raise ValueError('Invalid release action or control time')
        previous=self.previous_correction[3:].copy()
        maximum_change=self.s.correction_slew_per_s*dt
        correction=previous-np.clip(previous,-maximum_change,maximum_change)
        self.previous_correction=np.r_[np.zeros(3),correction]
        result=action.copy()
        result[3:6]=np.clip(action[3:6]+correction,self.controller['action_min'][3:6],self.controller['action_max'][3:6])
        return result,dict(correction=self.previous_correction.tolist(),guidance_mode='release_orientation_fade',
            orientation_mode='gravity_opening_axis',guidance_active=bool(np.any(np.abs(correction)>1e-12)),
            human_orientation_used_for_rotation=False,release_rotation_correction=correction.tolist())
