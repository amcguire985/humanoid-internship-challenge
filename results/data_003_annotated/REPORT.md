# data_003 manual annotation validation

User-confirmed grasp/release annotations applied. All three pass timing order and gap checks. Retargeting/controller feasibility and physical rollouts are deferred to Colab. Optional transport fields remain automatic/reviewed motion estimates and need review because they end early.

| Demo | Direction | Start XYZ (m) | End XYZ (m) | Distance (cm) | Max lift (cm) | Manual grasp/release (s) | Transport interval (s) | Transport duration (s) |
|---|---|---|---|---:|---:|---|---|---:|
| [demo_001](demo_001/annotation_diagnostic.png) | object -> target | -0.2092, 0.2882, -0.0915 | -0.3570, 0.0499, -0.0789 | 28.1 | 21.6 | 5.30 / 18.50 | 5.503-11.138 | 5.635 |
| [demo_002](demo_002/annotation_diagnostic.png) | target -> object | -0.3570, 0.0499, -0.0789 | -0.3512, 0.3500, -0.1236 | 30.3 | 12.9 | 24.50 / 36.50 | 26.678-31.182 | 4.503 |
| [demo_003](demo_003/annotation_diagnostic.png) | object -> target | -0.3512, 0.3500, -0.1236 | -0.3455, 0.0543, -0.0704 | 30.0 | 18.9 | 41.10 / 55.00 | 43.453-47.088 | 3.635 |

All values use the inherited ID0 world frame; calibration/crop and gravity alignment remain unverified. User annotations are preserved exactly in config and metadata; execution boundaries snap to nearest source frames. Automatic estimates remain unchanged. The reverse-direction demo is preserved as a distinct strategy and aligns its task axis to the same LIBERO bowl-to-plate task.

No physical rollout has run, so there are no new successes or physical failures. The currently combined baseline has two validated episodes and 4,230 transitions (episode_001: 2,109; episode_002: 2,121), both from test_007. Rollouts, recording and any mapping feasibility checks run only in Colab; no SmolVLA training.

See [Colab handoff](../../docs/DATA_003_COLAB.md).
