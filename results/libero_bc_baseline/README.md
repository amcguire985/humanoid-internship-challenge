# Minimal BC result: action fitting succeeded; task reproduction failed

The two selected complete successful demonstrations contain 4,230 transitions (2,109 + 2,121). Exact action and post-action EEF/gripper alignment against controller logs passed; timestamps are contiguous at 20 Hz. Each episode stores T+1 synchronized observations for T actions. Agent and wrist RGB are uint8 128×128×3; proprioception is 18D, separate EEF pose is 7D, and action is 7D. Only agent RGB and proprioception were used for training. Episode 3 completed during the prior interruption but was excluded; further collection was stopped.

Action ranges (normalized LIBERO input):

| Channel | Minimum | Maximum |
|---|---:|---:|
| dx | -0.04211297 | 0.06442154 |
| dy | -0.03780552 | 0.21687525 |
| dz | -0.28123784 | 0.07969921 |
| rx | -0.00841299 | 0.00436558 |
| ry | -0.00853199 | 0.00844421 |
| rz | -0.00237651 | 0.00177772 |
| gripper | -1.00000000 | 1.00000000 |

The lightweight CNN + proprioception MLP has 162,111 parameters and was trained once on CPU, using Adam (0.001), batch size 128, and seed 0. Training stopped at the predefined normalized-MSE threshold after 35 epochs; no hyperparameter sweep or evaluation-driven retraining was performed.

- Standardized training action MSE: 0.987094 → 0.024130.
- Raw training action MSE: 0.141738 → 0.000897.
- Training gripper-sign accuracy: 99.86%.
- Saved-checkpoint predictions match training predictions within 2.4e-7.

| Evaluation reset | Steps | Object acquired | Task completed | LIBERO success (ever/final) | Failure phase |
|---|---:|---|---|---|---|
| nominal | 2400 | False | False | False/False | APPROACH_OR_GRASP |
| perturbed_plus_2cm_x | 2400 | False | False | False/False | APPROACH_OR_GRASP |

Both evaluations exhausted their 2,400-step horizon before acquiring or lifting the object. Both resets were seen in training; this was a reproduction test, not a generalization claim. Initial rendered RGB and full simulator qpos/qvel match the corresponding demonstrations exactly. The policy received no scripted actions after reset preparation. Failure phases are inferred diagnostic milestones, not policy inputs.

**Conclusion:** the robot-native demonstrations are valid supervised-learning data, and a small policy can fit their recorded actions. This experiment did **not** demonstrate successful autonomous task reproduction (0/2). Low one-step training error was insufficient for closed-loop control. Distribution drift and insufficient feedback accuracy are plausible explanations, not isolated causes established by this experiment.

Files: [dataset audit](dataset_audit.json), [training metrics](training.json), [checkpoint](policy.pt), [action reproduction plot](action_reproduction.png), [evaluation metrics](evaluation/summary.json), and [reproduction guide](../../docs/LIBERO_BC_BASELINE.md).

Validation: 25 LIBERO tests passed. The broader prior repository test run had one unrelated missing-dependency error (`pupil_apriltags`); no retargeting or manipulation source/configuration was changed.
