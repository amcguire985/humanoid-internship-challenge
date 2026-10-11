# Placement progress diagnosis from saved rollouts

No controller behavior, gains, thresholds, phone data or checkpoint changed.
No simulation run. Sources: supplied trajectory (7).jsonl / experiment.json /
phone_constraints.json and earlier bowl_synthetic_results.zip. Earlier real-phone
evidence is documented in BOWL_SHARED_PLACEMENT.md; its raw trace was not
reanalysed here.

## Exact transition and progress

Latest constraints run enters PLACE/ALIGN at observation step 84 (4.70 s).
That row still records the final transport action. First placement action is
step 85 (4.75 s). There is no literal stop or sustained backwards XY movement:
plate distance decreases monotonically from grasp through failure at 149.
The discontinuity is a sharp reduction in command/progress at 85.

| Step | Phase/action source | XY distance mm | Command toward plate mm | Observed progress toward plate mm |
|---|---|---:|---:|---:|
| 84 | final TRANSPORT | 64.700 | 33.584 | 8.890 |
| 85 | first PLACE/ALIGN | 60.329 | 2.950 | 4.400 |
| 100 | PLACE/ALIGN | 42.446 | 2.723 | 0.843 |
| 120 | PLACE/ALIGN | 28.710 | 2.478 | 0.598 |
| 149 | final ALIGN action / FAILED observation | 14.070 | 1.929 | 0.421 |

Projection uses previous bowl-to-plate XY direction. Actions decoded with saved
OSC input/output scaling. Observation is post-action; commands belong to the
preceding observation. 149 misses 12 mm tolerance by 2.07 mm; tilt 0.419 degrees
and speed 0.012715 m/s already pass. No LOWER or RELEASE occurs.
Synthetic D1 enters PLACE at 93 / first placement command 94, also monotonically
approaches plate, and times out at 158 with 47.743 mm error. Its first placement
command toward plate is 2.966 mm; final observed progress is 0.589 mm per step.
D1 transport at step 93 does command away (projection -3.662 mm), while the bowl
still approaches (+1.115 mm). This is before placement commands take over.

## Target and grasp transform

Physical bowl support goal: [0.07160356, 0.20039043, 0.90698797] m.
At latest handoff, bowl minus goal: approximately
[-0.0275948, -0.05852052, 0.14142812] m.
The precise XY displacement is available in trajectory.jsonl; 64.700 mm norm.
ALIGN deliberately holds handoff bowl height 1.04841610 m, not support height.
First desired EEF translation: reconstructed via logged placement target, with
rigid transform inverse, predicts bowl target [0.071604,0.200390,1.048416] m.
No target sign inversion is apparent.

EEF-from-bowl relative translation at PLACE entry, i.e. bowl origin expressed
in EEF frame: [-0.03700548,-0.00534805,+0.06367640] m.
Code captures inverse(T_world_eef) @ T_world_bowl on handoff and computes
T_world_eef_target = T_world_bowl_target @ inverse(T_eef_bowl).
Then EEF orientation is composed with the shortest world gravity swing.
Logged target composed back with the frozen transform differs from intended
bowl target by under about 0.5 mm at the final sample; this small drift does not
explain the 14 mm remaining XY error.

At 85 raw policy delta XYZ = [+15.878,+20.490,-10.108] mm;
executed placement delta = [+0.903,+2.836,-0.376] mm.
Raw policy is entirely overridden in placement. At 149 raw =
[+11.373,+21.125,-16.677] mm; executed = [-1.378,+1.350,-2.297] mm.
Negative world X here points toward plate because the bowl has crossed plate X;
it is not backwards relative to the destination. Both project toward plate.

## Upward motion and geometry limits of the evidence

From 84 to peak height at 125, bowl rises 19.927 mm despite generally downward
placement Z commands. EEF itself rises 11.638 mm; changing world bowl-to-EEF
Z offset contributes 8.288 mm as the held bowl is righted. This demonstrates
orientation/translation dynamics coupling, not an upward placement target.
ALIGN holds height, so support-relative height remains large until LOWER.

The existing support goal uses central plate collision-box top (0.90857797 m)
minus upright bowl collision-box bottom offset (+0.001590 m), not visual extent
markers. This is internally consistent with saved geometry audit; logs do not
independently establish all meshes, collision-free reachability, or IK feasibility.
Target is finite and gripper-scaled commands are not action-clipped. Stable grasp
and small relative drift argue against slip as the cause. Absence of bowl/plate
contact does not exclude arm/table or other collisions. No joint limits, forces,
OSC internal goals or torques are present to exclude those causes.

## Most likely cause and smallest proposed correction

Most likely: placement resets its 3 mm bounded position goal relative to the
*current measured EEF pose every control step*, combined with simultaneous
uprighting. Thus it commands a very small persistent tracking error rather than
advancing a placement setpoint independently. Observed XY motion is substantially
smaller than that increment; the fixed 65-step alignment window expires before
centering. This shared placement behavior explains recurrence across inputs;
exact dynamics attribution remains a hypothesis, not proven from these logs.

Smallest proposed correction: retain a placement Cartesian command/setpoint
across ALIGN steps and advance it toward the same geometrically computed target
at the existing 3 mm increment, rather than restarting every increment from
measured pose. Preserve target, gains, thresholds, rotation cap and timeout.
Do not implement until internal OSC goal semantics are checked: accumulating a
setpoint can increase tracking error if the robot is obstructed.

If one diagnostic rollout is authorized, additional required logging is:
pre-action EEF and bowl poses; full and limited placement targets; commanded
world XYZ/rotation deltas; OSC actual goal_pos/goal_ori before/after set_goal and
at physics substeps; joint qpos/qvel and joint-limit margins; applied torques and
actuator saturation; all contacting geom pairs, normals and contact forces;
EEF linear/angular velocity; frozen grasp transform; phase readiness booleans.
Existing proposed/executed actions and post-step poses must remain to pair them.
These distinguish moving-setpoint lag, rotation coupling, joint limits and
collision rather than merely extending the timeout.

Plots from saved logs only: results/bowl_placement_diagnosis/latest/distance_constraints.png
and distance_D1.png; lines mark state transitions and show 12 mm tolerance.
