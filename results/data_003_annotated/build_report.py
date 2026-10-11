import json
from pathlib import Path
out=Path('results/data_003_annotated'); reports=json.loads((out/'validation.json').read_text())
lines=['# data_003 manual annotation validation','', 'User-confirmed grasp/release annotations applied. All three pass timing order and gap checks. Retargeting/controller feasibility and physical rollouts are deferred to Colab. Optional transport fields remain automatic/reviewed motion estimates and need review because they end early.','',
'| Demo | Direction | Start XYZ (m) | End XYZ (m) | Distance (cm) | Max lift (cm) | Manual grasp/release (s) | Transport interval (s) | Transport duration (s) |',
'|---|---|---|---|---:|---:|---|---|---:|']
for r in reports:
 xyz=lambda v:', '.join(f'{x:.4f}' for x in v)
 a=r['manual_annotations']; lines.append(f"| [{r['demo_id']}]({r['demo_id']}/annotation_diagnostic.png) | {r['direction']} | {xyz(r['start_position'])} | {xyz(r['end_goal_position'])} | {100*r['start_to_goal_distance_m']:.1f} | {100*r['maximum_lift_m']:.1f} | {a['grasp_time_seconds']:.2f} / {a['release_time_seconds']:.2f} | {r['transport_start_time_seconds']:.3f}-{r['transport_end_time_seconds']:.3f} | {r['transport_duration_seconds']:.3f} |")
lines+=['','All values use the inherited ID0 world frame; calibration/crop and gravity alignment remain unverified. User annotations are preserved exactly in config and metadata; execution boundaries snap to nearest source frames. Automatic estimates remain unchanged. The reverse-direction demo is preserved as a distinct strategy and aligns its task axis to the same LIBERO bowl-to-plate task.',
'', 'No physical rollout has run, so there are no new successes or physical failures. The currently combined baseline has two validated episodes and 4,230 transitions (episode_001: 2,109; episode_002: 2,121), both from test_007. Rollouts, recording and any mapping feasibility checks run only in Colab; no SmolVLA training.',
'', 'See [Colab handoff](../../docs/DATA_003_COLAB.md).']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
readiness=Path('results/data_003_robot_rollouts/run_readiness.json'); readiness.parent.mkdir(exist_ok=True)
readiness.write_text(json.dumps(dict(status='deferred_to_colab_by_user',manual_annotations_present=True,
 requested_rollouts_per_demo=1,automatic_retries=False,actual_rollouts=0,prepared_demos=[r['prepared_demo'] for r in reports],
 execution_runtime='/content/micromamba/envs/libero/bin/python',MUJOCO_GL='osmesa',colab_repo='/content/humanoid-internship-challenge',
 pending='Review optional transport windows and run unchanged-controller mapping preflight in Colab before any physical attempt.'),indent=2)+'\n')
