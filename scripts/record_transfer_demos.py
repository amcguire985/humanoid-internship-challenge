"""One recorded nominal LIBERO rollout per manually annotated human demo; no retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from annotate_transfer_demos import read_json,write_json


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def combine(base,new,output):
    """Select exactly the requested baseline episodes 1/2 plus accepted new episodes."""
    import h5py
    from inspect_libero_dataset import validate_episode
    selected=[(base/'episodes'/f'episode_{i:03d}.h5','test_007', 'existing') for i in (1,2)]
    for path in sorted((new/'episodes').glob('*.h5')):
        with h5py.File(path) as f:
            human=str(f.attrs['human_demo_id'])
        selected.append((path,human,'data_003'))
    output.mkdir(parents=True,exist_ok=True); (output/'episodes').mkdir(exist_ok=True)
    episodes=[]
    for source,human,origin in selected:
        check=validate_episode(source); digest=sha256(source); dest=output/'episodes'/source.name
        if dest.exists():
            if sha256(dest)!=digest: raise ValueError('Dataset episode collision: '+str(dest))
        else: shutil.copy2(source,dest)
        episodes.append(dict(episode_id=check['episode_id'],path=dest.relative_to(output).as_posix(),
            source_path=source.as_posix(),sha256=digest,transitions=check['transitions'],
            human_demo_id=human,source_video='videos/data_003.MOV' if origin=='data_003' else 'videos/test_007_block_only.MOV',origin=origin))
    inventory={p.name for p in (output/'episodes').glob('*.h5')}
    if inventory!={Path(e['path']).name for e in episodes}: raise ValueError('Unexpected episodes in combined dataset')
    result=dict(schema_version=1,successful_episodes=len(episodes),total_transitions=sum(e['transitions'] for e in episodes),
        episodes=episodes,selected_existing_episodes=[1,2],new_successful_episodes=sum(e['origin']=='data_003' for e in episodes),
        new_attempts=[read_json(p) for p in sorted((new/'attempts').glob('*/status.json'))],
        format='Existing robot-native HDF5: N actions / N+1 observations; no policy training.')
    write_json(output/'summary.json',result); return result


def run(prepared,output,base,combined):
    import importlib.util
    required=('libero','robosuite','mujoco','h5py')
    missing=[name for name in required if importlib.util.find_spec(name) is None]
    validations=[read_json(p) for p in sorted(prepared.glob('demo_*/validation.json'))]
    from annotate_transfer_demos import prepare
    validations=[prepare(Path(v['source_demo']),Path(v['annotation_path']),Path(v['prepared_demo']))
        if sha256(v['annotation_path'])!=v['annotation_sha256'] else v for v in validations]
    eligible=[v for v in validations if v['suitable_for_retargeting']]
    write_json(output/'run_readiness.json',dict(validations=validations,missing_runtime_modules=missing,
        requested_rollouts_per_demo=1,automatic_retries=False,status='blocked' if missing or os.name=='nt' or len(eligible)!=len(validations) else 'ready',execution_runtime='Google Colab /content/micromamba/envs/libero/bin/python; Windows preparation only'))
    if missing or os.name=='nt':
        result=combine(base,output,combined)
        print(json.dumps(dict(blocked_by_runtime=missing,windows_preparation_only=os.name=='nt',dataset=result),indent=2)); return
    # Importing these modules changes no controller, robot or task settings.
    import h5py
    from libero_xy_trials import make_env
    from scripted_libero_pick_place import PickPlace,save_outputs,validate_config
    from record_libero_dataset import RecordingEnv
    from retarget_libero_object import load_transport
    from inspect_libero_dataset import validate_episode
    robot=read_json(Path('config/libero_pick_place.json')); validate_config(robot)
    mapping=read_json(Path('config/libero_retarget.json'))
    class NominalRecording(PickPlace):
        def scene_settle(self):
            super().scene_settle(); self.env.start(self.obs)
    (output/'episodes').mkdir(parents=True,exist_ok=True)
    for v in eligible:
        from plan_transfer_rollouts import plan
        preview=plan(Path(v['prepared_demo']))
        if not preview['mapping_suitable']:
            write_json(output/'rejected'/(v['demo_id']+'.json'),preview)
            print('Mapping preflight rejected '+v['demo_id']+': '+str(preview['reasons']),flush=True)
            continue
        demo_id=v['demo_id']; episode_id=f"episode_{int(demo_id.split('_')[-1])+3:03d}"
        directory=output/'attempts'/demo_id
        if directory.exists():
            print('Already attempted; no retry: '+demo_id,flush=True); continue
        directory.mkdir(parents=True)
        status=dict(human_demo_id=demo_id,episode_id=episode_id,direction=v['direction'],status='started',
                    accepted=False,rollout_limit=1,first_physical_failure=None)
        write_json(directory/'status.json',status)
        env=None; runner=None; temporary=directory/'episode.partial.h5'
        try:
            demo=Path(v['prepared_demo'])
            if sha256(v['annotation_path'])!=v['annotation_sha256']: raise ValueError('Annotation changed after validation; rerun preparation.')
            human=load_transport(demo/'processed_demo.csv',demo/'metadata.json',mapping['max_source_gap_s'])
            env=make_env(0)
            with h5py.File(temporary,'w') as f:
                wrapped=RecordingEnv(env,f,episode_id,128)
                runner=NominalRecording(wrapped,human,robot,mapping,directory); wrapped.runner=runner
                f.attrs.update(schema_version=1,complete=False,episode_success=False,episode_id=episode_id,
                    human_demo_id=demo_id,human_direction=v['direction'],human_annotation_sha256=sha256(v['annotation_path']),
                    human_trajectory_sha256=sha256(demo/'processed_demo.csv'),language_instruction=env.language_instruction,
                    control_frequency_hz=runner.frequency,image_convention='RGB uint8 HWC, top-left origin (MuJoCo vertical flip)',
                    eef_pose_convention='world xyz metres + quaternion xyzw',
                    proprio_order='joint_pos[7], joint_vel[7], gripper_qpos[2], gripper_qvel[2]')
                report=runner.run()
                report.update(human_demo_id=demo_id,human_direction=v['direction'],source_metadata=human[-1],
                    annotation_file=v['annotation_path'],settings=robot,mapping_settings=mapping,seed=0,
                    controller_sha256=sha256('scripts/scripted_libero_pick_place.py'))
                save_outputs(runner,report,directory)
                logged=[r for r in runner.records if r['phase']!='SCENE_SETTLE']
                np.testing.assert_array_equal(f['actions'][:],np.array([r['action'] for r in logged]))
                np.testing.assert_array_equal(f['observations/libero_success'][1:],[r['task_success'] for r in logged])
                f.attrs['success_metadata_json']=json.dumps(report)
                f.attrs['complete']=bool(report['completed_sequence']); f.attrs['episode_success']=bool(report['task_completed'])
                status.update(status='success' if report['task_completed'] else 'failed',accepted=bool(report['task_completed']),
                    transitions=wrapped.count,first_physical_failure=dict(phase=report['failure_phase'],reason=report['failure_reason']) if report['failure_phase'] else None)
            if status['accepted']:
                validate_episode(temporary); temporary.replace(output/'episodes'/(episode_id+'.h5'))
            # Failed recordings remain in attempts for diagnostics, never accepted into training.
        except Exception as exc:
            status.update(status='error',accepted=False,error=f'{type(exc).__name__}: {exc}')
            if runner is not None and runner.records:
                report=runner.report(); save_outputs(runner,report,directory)
                status['first_physical_failure']=dict(phase=report['failure_phase'],reason=report['failure_reason']) if report['failure_phase'] else None
        finally:
            write_json(directory/'status.json',status)
            if env is not None: env.close()
            combine(base,output,combined)
        print(json.dumps(status),flush=True)
    print(json.dumps(combine(base,output,combined),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=Path('results/data_003_annotated'))
    parser.add_argument('--output',type=Path,default=Path('results/data_003_robot_rollouts'))
    parser.add_argument('--base',type=Path,default=Path('results/libero_robot_dataset'))
    parser.add_argument('--combined',type=Path,default=Path('results/libero_robot_dataset_multi_demo'))
    parser.add_argument('--combine-only',action='store_true')
    a=parser.parse_args()
    if a.combine_only: print(json.dumps(combine(a.base,a.output,a.combined),indent=2))
    else: run(a.prepared,a.output,a.base,a.combined)
