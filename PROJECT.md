# Project Plan

## Requirements

- Personally collected data must materially affect the approach.
- Robot behavior must be demonstrated in simulation.
- Results must be measurable and clearly presented.
- Implementation should remain simple enough to complete within one week.

## Key Risks

| Risk | Why it matters | Derisking step |
|---|---|---|
| Monocular pose is too noisy | Bad supervision | Test fiducial-based 6-DoF pose accuracy |
| Camera motion corrupts estimates | Unstable trajectories | Use fixed world-reference marker |
| Marker occlusion | Missing pose data | Test wrist / forearm placement |
| Human and robot workspaces differ | Poor retargeting | Use normalized Cartesian trajectories |
| Human images do not transfer visually | Domain gap | Recreate trajectories in simulation |
| VLA training is too expensive | Schedule risk | Keep trajectory replay / simple policy baseline viable |

## Near-Term Plan

1. Calibrate phone camera.
2. Record fixed + moving fiducial markers.
3. Measure 3D pose accuracy and jitter.
4. Convert human motion into Panda task-space actions.
5. Replay trajectory in LIBERO.
6. Add learning only after the data pipeline works.