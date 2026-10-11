# data_003 reviewed rollout preparation

Prepared on Windows without simulator execution. All three source demonstrations and nominal mappings pass the configured gates. All four event times were manually supplied by the user; automatic estimates are preserved separately.

| Demo | Direction | Distance (cm) | Max human lift (cm) | Grasp | Transport start/end | Release | Source transport duration |
|---|---|---:|---:|---:|---|---:|---:|
| demo_001 | object -> target | 28.07 | 21.59 | 5.30 | 5.90 / 16.70 | 18.50 | 10.805 s |
| demo_002 | target -> object | 30.34 | 12.92 | 24.50 | 25.40 / 35.50 | 36.50 | 10.072 s |
| demo_003 | object -> target | 30.04 | 18.93 | 41.10 | 42.50 / 53.20 | 55.00 | 10.705 s |

Start and goal positions are in each per-demo `validation.json`. Effective times snap to source frames, while exact manual times remain in configs.

The shared validated controller and mapping configuration are unchanged. Per-demo TRANSPORT timeouts are 5,000 / 5,500 / 8,000 steps to accommodate the existing 0.02 m/s speed limiting. No path was shortened, reversed or interpolated across missing observations.

Nominal endpoint correction is 0.81 / 0.48 / 0.84 cm, all below the existing 6 cm limit. Predicted transport steps are 4,411 / 4,822 / 7,136. Nominal EEF paths remain in the configured workspace. This is preparation validation, not physical-success evidence.

Future accepted episodes: demo_001 -> episode_004, demo_002 -> episode_005, demo_003 -> episode_006. One durable attempt per demo, with failures excluded from the success-only dataset. Baseline episodes 001/002 contribute 4,230 transitions; baseline 003 is excluded.

See [the Colab guide](../../docs/DATA_003_COLAB.md) for the exact single command and output locations.
