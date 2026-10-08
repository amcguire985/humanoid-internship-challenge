"""Shared B/C geometry placement using world-relative OSC, no policy blending."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from bowl_transport_guidance import pose, opening_axis_rotation_vector
from replay_libero_transport import desired_pose_to_action

@dataclass(frozen=True)
class PlacementSettings:
    approach_radius_m: float = .09
    approach_height_m: float = .025
    align_tolerance_m: float = .012
    height_tolerance_m: float = .004
    speed_tolerance_m_s: float = .025
    tilt_tolerance_deg: float = 8.
    confirm_steps: int = 3
    translation_step_m: float = .003
    rotation_step_rad: float = .02
    align_timeout_steps: int = 65
    lower_timeout_steps: int = 65
    release_timeout_steps: int = 25
    verify_timeout_steps: int = 35
    withdraw_m: float = .06

    def validate(self):
        for name,value in vars(self).items():
            if not np.isfinite(value) or value<=0: raise ValueError('Invalid placement setting '+name)
            if name.endswith('_steps') and type(value) is not int: raise ValueError('Expected integer '+name)
        return self

class Placement:
    def __init__(self, goal, gripper_bowl, controller, settings=None):
        self.goal=np.asarray(goal,float); self.relative=np.asarray(gripper_bowl,float)
        self.controller=controller; self.s=(settings or PlacementSettings()).validate()
        self.phase='TRANSPORT'; self.subphase=None; self.count=0; self.loss=0
        self.enter_step=None; self.enter_reason=None; self.phase_step=0; self.failure=None
        self.last_position=None; self.speed=0.; self.release_aperture=None
        self.withdraw_target=None; self.hold_eef=None
        self.align_setpoint=None; self.align_target=None
        self.lower_setpoint=None; self.lower_target=None
        self.lower_history=[]; self.lower_supported=False; self.lower_contact_count=0

    def switch(self, phase, subphase, row):
        self.align_setpoint=None; self.align_target=None
        if phase=='PLACE' and subphase=='ALIGN':
            self.align_setpoint=np.asarray(row['eef_pose'],float)[:3,3].copy()
        self.lower_setpoint=None; self.lower_target=None
        self.lower_history=[]; self.lower_supported=False; self.lower_contact_count=0
        if phase=='PLACE' and subphase=='LOWER':
            eef=np.asarray(row['eef_pose'],float)
            r=np.asarray(row['bowl_rotation_world_from_object'],float)
            swing=Rotation.from_rotvec(opening_axis_rotation_vector(r)).as_matrix()
            self.lower_target=(pose(self.goal,swing@r)@np.linalg.inv(self.relative))[:3,3].copy()
            self.lower_setpoint=eef[:3,3].copy()
        self.phase=phase; self.subphase=subphase; self.phase_step=row['timestep']; self.count=0

    def observe(self, row, supported, slip, dt):
        p=np.asarray(row['bowl_position_m']); eef=np.asarray(row['eef_pose'])
        self.speed=0. if self.last_position is None else float(np.linalg.norm(p-self.last_position)/dt)
        self.last_position=p.copy(); distance=float(np.linalg.norm(p[:2]-self.goal[:2]))
        height=float(p[2]-self.goal[2]); stable=self.speed<=self.s.speed_tolerance_m_s
        grasp=row['grasp'] is True
        if self.phase=='TRANSPORT':
            eligible=grasp and not slip and distance<=self.s.approach_radius_m and height>=self.s.approach_height_m
            self.count=self.count+1 if eligible else 0
            if self.count>=self.s.confirm_steps:
                self.enter_step=row['timestep']; self.enter_reason='confirmed_grasp_inside_approach_region_above_support'
                self.relative=np.linalg.inv(eef)@pose(p,row['bowl_rotation_world_from_object'])
                self.align_height=max(p[2],self.goal[2]+self.s.approach_height_m)
                self.switch('PLACE','ALIGN',row)
        elif self.phase=='PLACE':
            self.loss=0 if grasp else self.loss+1
            if slip or self.loss>=3:
                self.failure='placement_slip' if slip else 'placement_grasp_loss'; self.switch('FAILED',self.subphase,row)
            elif self.subphase=='ALIGN':
                ready=distance<=self.s.align_tolerance_m and row['tilt_deg']<=self.s.tilt_tolerance_deg and stable
                self.count=self.count+1 if ready else 0
                if self.count>=self.s.confirm_steps: self.switch('PLACE','LOWER',row)
            else:
                self.lower_supported=bool(supported)
                unexpected_contact=supported and height>self.s.height_tolerance_m
                self.lower_contact_count=self.lower_contact_count+1 if unexpected_contact else 0
                self.lower_history.append((row['timestep'],float(p[2])))
                window=3*self.s.confirm_steps
                self.lower_history=self.lower_history[-(window+1):]
                tracking=float(np.linalg.norm(self.lower_setpoint-eef[:3,3])) if self.lower_setpoint is not None else 0.
                capacity=np.minimum(self.controller['output_max'][:3],-self.controller['output_min'][:3])
                lead=min(4*self.s.translation_step_m,float(np.min(capacity)))
                stalled=(len(self.lower_history)==window+1 and
                    self.lower_history[0][1]-self.lower_history[-1][1]<=.1*self.s.height_tolerance_m and
                    height>self.s.height_tolerance_m and tracking>=lead-self.s.translation_step_m)
                reason=None
                if self.lower_contact_count>=self.s.confirm_steps: reason='placement_lower_unexpected_support_contact'
                elif (tracking>lead+self.s.translation_step_m and height>self.s.height_tolerance_m and
                      np.dot(eef[:3,3]-self.lower_setpoint,eef[:3,3]-self.lower_target)>0):
                    reason='placement_lower_excessive_tracking_error'
                elif stalled: reason='placement_lower_tracking_stall'
                if reason:
                    self.failure=reason; self.switch('FAILED','LOWER',row)
                ready=distance<=self.s.align_tolerance_m and abs(height)<=self.s.height_tolerance_m and stable and supported and grasp and row['tilt_deg']<=self.s.tilt_tolerance_deg
                self.count=self.count+1 if ready else 0
                if self.phase=='PLACE' and self.count>=self.s.confirm_steps:
                    self.release_aperture=row['gripper_aperture_m']; self.hold_eef=eef.copy(); self.switch('RELEASE','OPEN',row)
        elif self.phase=='RELEASE':
            opened=row['gripper_aperture_m']>=min(.07,self.release_aperture+.02) and not grasp
            self.count=self.count+1 if opened else 0
            if self.count>=self.s.confirm_steps:
                self.withdraw_target=eef.copy(); self.withdraw_target[2,3]+=self.s.withdraw_m
                self.switch('VERIFY','WITHDRAW',row)
        elif self.phase=='VERIFY':
            # Require withdrawal/settling before declaring controller completion.
            if row['libero_success'] and row['timestep']-self.phase_step>=10 and eef[2,3]>=self.withdraw_target[2,3]-.01:
                self.switch('DONE','SETTLED',row)
        limit={'ALIGN':self.s.align_timeout_steps,'LOWER':self.s.lower_timeout_steps,'OPEN':self.s.release_timeout_steps,'WITHDRAW':self.s.verify_timeout_steps}.get(self.subphase)
        if self.phase not in ('DONE','FAILED') and limit is not None and row['timestep']-self.phase_step>=limit:
            self.failure='placement_'+self.subphase.lower()+'_timeout'; self.switch('FAILED',self.subphase,row)
        return dict(placement_subphase=self.subphase,placement_entry_timestep=self.enter_step,placement_transition_reason=self.enter_reason,plate_target_position_m=self.goal.tolist(),bowl_plate_horizontal_error_m=distance,bowl_support_height_error_m=height,bowl_speed_m_s=self.speed,bowl_plate_contact=bool(supported),placement_slip=bool(slip),release_allowed=self.phase in ('RELEASE','VERIFY','DONE'))

    def action(self, bowl, eef):
        if self.phase=='RELEASE': target=self.hold_eef.copy(); gripper=-1.
        elif self.phase in ('VERIFY','DONE'): target=self.withdraw_target.copy(); gripper=-1.
        elif self.phase=='FAILED': return np.r_[np.zeros(6),-1. if self.release_aperture is not None else 1.],dict(placement_control=True)
        else:
            swing=Rotation.from_rotvec(opening_axis_rotation_vector(bowl[:3,:3])).as_matrix()
            p=self.goal.copy()
            if self.subphase=='ALIGN': p[2]=self.align_height
            target=pose(p,swing@bowl[:3,:3])@np.linalg.inv(self.relative)
            target[:3,:3]=swing@eef[:3,:3]; gripper=1.
        delta=target[:3,3]-eef[:3,3]; norm=np.linalg.norm(delta)
        limited=eef[:3,3]+delta*min(1.,self.s.translation_step_m/max(norm,1e-12))
        align_info={}
        if self.phase=='PLACE' and self.subphase=='ALIGN':
            if self.align_setpoint is None: self.align_setpoint=eef[:3,3].copy()
            if self.align_target is None: self.align_target=target[:3,3].copy()
            target[:3,3]=self.align_target
            # Advance only the persistent target. Bound accumulated tracking error
            # to four existing increments (12 mm), also respecting OSC capacity.
            capacity=np.minimum(self.controller['output_max'][:3],-self.controller['output_min'][:3])
            lead=min(4*self.s.translation_step_m,float(np.min(capacity)))
            if lead<=0: raise ValueError('ALIGN requires translation scaling straddling zero')
            remaining=self.align_target-self.align_setpoint
            distance=np.linalg.norm(remaining)
            candidate=self.align_setpoint+remaining*min(1.,self.s.translation_step_m/max(distance,1e-12))
            offset=candidate-eef[:3,3]; error=np.linalg.norm(offset)
            capped=error>lead
            if capped: candidate=eef[:3,3]+offset*(lead/error)
            # When the measured EEF has passed the target, stop commanding beyond it.
            if np.linalg.norm(self.align_target-eef[:3,3])<=lead and np.dot(candidate-self.align_target,candidate-eef[:3,3])>0:
                candidate=self.align_target.copy()
            self.align_setpoint=candidate.copy(); limited=candidate
            align_info=dict(placement_align_setpoint_m=candidate.tolist(),
                placement_align_fixed_target_m=self.align_target.tolist(),
                placement_align_lead_limit_m=lead,placement_align_lead_capped=bool(capped),
                placement_align_tracking_error_m=float(np.linalg.norm(candidate-eef[:3,3])))

        if self.phase=='PLACE' and self.subphase=='LOWER':
            target[:3,3]=self.lower_target
            remaining=self.lower_target-self.lower_setpoint
            distance=np.linalg.norm(remaining)
            candidate=self.lower_setpoint+remaining*min(1.,self.s.translation_step_m/max(distance,1e-12))
            capacity=np.minimum(self.controller['output_max'][:3],-self.controller['output_min'][:3])
            lead=min(4*self.s.translation_step_m,float(np.min(capacity)))
            if lead<=0: raise ValueError('LOWER requires translation scaling straddling zero')
            offset=candidate-eef[:3,3]; error=np.linalg.norm(offset)
            capped=error>lead
            if capped: candidate=eef[:3,3]+offset*(lead/error)
            if np.linalg.norm(self.lower_target-eef[:3,3])<=lead and np.dot(candidate-self.lower_target,candidate-eef[:3,3])>0:
                candidate=self.lower_target.copy()
            # Physical endpoint is a floor, including anti-windup projection.
            candidate[2]=max(candidate[2],self.lower_target[2])
            if self.lower_supported: candidate[2]=max(candidate[2],eef[2,3])
            self.lower_setpoint=candidate.copy(); limited=candidate
            align_info.update(placement_lower_setpoint_m=candidate.tolist(),
                placement_lower_fixed_target_m=self.lower_target.tolist(),
                placement_lower_lead_limit_m=lead,placement_lower_lead_capped=bool(capped),
                placement_lower_tracking_error_m=float(np.linalg.norm(candidate-eef[:3,3])),
                placement_lower_contact_hold=bool(self.lower_supported),
                placement_lower_stall_window_steps=3*self.s.confirm_steps,
                placement_lower_stall_min_descent_m=.1*self.s.height_tolerance_m)
        vector=Rotation.from_matrix(target[:3,:3]@eef[:3,:3].T).as_rotvec(); norm=np.linalg.norm(vector)
        rot=Rotation.from_rotvec(vector*min(1.,self.s.rotation_step_rad/max(norm,1e-12))).as_matrix()@eef[:3,:3]
        action,clipped=desired_pose_to_action(limited,eef[:3,3],rot,eef[:3,:3],self.controller,clip=1.,gripper=gripper)
        return action,dict(placement_control=True,placement_command_target_eef=target.tolist(),placement_action_clipped=bool(clipped),guidance_active=False,**align_info)
