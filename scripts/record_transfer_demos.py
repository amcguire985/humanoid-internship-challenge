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


def combine(base,new,output,expected=None):
    """Select exactly the requested baseline episodes 1/2 plus accepted new episodes."""
    import h5py
    from inspect_libero_dataset import validate_episode
    expected=expected or {f'episode_{i+3:03d}':f'demo_{i:03d}' for i in range(1,4)}
    selected=[(base/'episodes'/f'episode_{i:03d}.h5','test_007', 'existing') for i in (1,2)]
    humans=set()
    for path in sorted((new/'episodes').glob('*.h5')):
        with h5py.File(path) as f:
            human=str(f.attrs['human_demo_id'])
            if expected.get(path.stem)!=human or human in humans:
                raise ValueError('Unexpected or duplicate new episode/source demo: '+str(path))
            if str(f.attrs['episode_id'])!=path.stem:
                raise ValueError('Episode identity/file mismatch: '+str(path))
        humans.add(human)
        selected.append((path,human,'data_003'))
    output.mkdir(parents=True,exist_ok=True); (output/'episodes').mkdir(exist_ok=True)
    episodes=[]
    for source,human,origin in selected:
        check=validate_episode(source); digest=sha256(source); dest=output/'episodes'/source.name
        if dest.exists():
            if sha256(dest)!=digest: raise ValueError('Dataset episode collision: '+str(dest))
        else: shutil.copy2(source,dest)
        provenance={}
        if origin=='data_003':
            with h5py.File(source) as file:
                provenance=dict(human_direction=str(file.attrs['human_direction']),
                    human_provenance=json.loads(file.attrs['human_provenance_json']))
        episodes.append(dict(**provenance,episode_id=check['episode_id'],path=dest.relative_to(output).as_posix(),
            source_path=source.as_posix(),sha256=digest,transitions=check['transitions'],
            human_demo_id=human,source_video='videos/data_003.MOV' if origin=='data_003' else 'videos/test_007_block_only.MOV',origin=origin))
    inventory={p.name for p in (output/'episodes').glob('*.h5')}
    if inventory!={Path(e['path']).name for e in episodes}: raise ValueError('Unexpected episodes in combined dataset')
    result=dict(schema_version=1,successful_episodes=len(episodes),total_transitions=sum(e['transitions'] for e in episodes),
        episodes=episodes,selected_existing_episodes=[1,2],new_successful_episodes=sum(e['origin']=='data_003' for e in episodes),
        new_attempts=[read_json(p) for p in sorted((new/'attempts').glob('*/status.json'))],
        rejected_demos=[read_json(p) for p in sorted((new/'rejected').glob('*.json'))],
        format='Existing robot-native HDF5: N actions / N+1 observations; no policy training.')
    write_json(output/'summary.json',result); return result


def run(config_path,prepare_only=False):
    from prepare_transfer_rollouts import prepare_bundle, claim_attempt
    config, bundle=prepare_bundle(config_path)
    prepared=Path(config['prepared_root']); output=Path(config['rollout_output'])
    base=Path(config['baseline_dataset']); combined=Path(config['combined_dataset'])
    expected={r['episode_id']:r['demo_id'] for r in config['demos']}
    eligible=[r for r in bundle['demos'] if r['ready_for_rollout']]
    rejected=[r for r in bundle['demos'] if not r['ready_for_rollout']]
    for entry in rejected:
        write_json(output/'rejected'/(entry['demo_id']+'.json'),entry)
    # Remove only stale rejection reports for inputs that now pass review/preflight.
    for entry in eligible:
        (output/'rejected'/(entry['demo_id']+'.json')).unlink(missing_ok=True)
    readiness=dict(inputs=bundle,requested_rollouts_per_demo=1,automatic_retries=False,
        status='ready' if not rejected else 'needs_annotation_or_mapping_review',
        execution_runtime='Google Colab /content/micromamba/envs/libero/bin/python; Windows preparation only',
        simulation_executed=False)
    if prepare_only:
        write_json(output/'run_readiness.json',readiness)
        print(json.dumps(bundle,indent=2)); return bundle
    # Never import simulator modules on Windows, even when they happen to be installed.
    if os.name=='nt':
        readiness['status']='windows_execution_prohibited'
        write_json(output/'run_readiness.json',readiness)
        raise RuntimeError('Windows supports --prepare-only or --combine-only. Run rollouts in Colab.')
    result=combine(base,output,combined,expected)
    if not eligible:
        write_json(output/'run_readiness.json',readiness)
        print(json.dumps(dict(no_rollouts_attempted=True,review_required=rejected,dataset=result),indent=2))
        return result
    import importlib.util
    required=('libero','robosuite','mujoco','h5py')
    missing=[name for name in required if importlib.util.find_spec(name) is None]
    readiness['missing_runtime_modules']=missing
    if missing:
        readiness['status']='missing_colab_runtime_dependencies'
        write_json(output/'run_readiness.json',readiness)
        raise RuntimeError('Use the validated Colab LIBERO interpreter; missing: '+str(missing))
    write_json(output/'run_readiness.json',readiness)
    # Importing these modules changes no controller, robot or task settings.
    import h5py
    from libero_xy_trials import make_env
    from scripted_libero_pick_place import PickPlace,save_outputs,validate_config
    from record_libero_dataset import RecordingEnv
    from retarget_libero_object import load_transport
    from inspect_libero_dataset import validate_episode
    validate_config(read_json(Path(config['robot_config'])))
    class NominalRecording(PickPlace):
        def scene_settle(self):
            super().scene_settle(); self.env.start(self.obs)
    (output/'episodes').mkdir(parents=True,exist_ok=True)
    for entry in eligible:
        v=entry['validation']
        robot=entry['robot_settings']; mapping=entry['mapping_settings']; validate_config(robot)
        demo_id=entry['demo_id']; episode_id=entry['episode_id']
        directory=output/'attempts'/demo_id
        if not claim_attempt(directory):
            print('Already attempted; no retry: '+demo_id,flush=True); continue
        status=dict(human_demo_id=demo_id,episode_id=episode_id,direction=v['direction'],status='started',
                    accepted=False,rollout_limit=1,first_physical_failure=None,
                    provenance=entry['provenance'],task=config['task'],seed=config['seed'])
        write_json(directory/'status.json',status)
        env=None; runner=None; temporary=directory/'episode.partial.h5'
        write_json(directory/'retargeting_input.json',entry)
        try:
            demo=Path(v['prepared_demo'])
            if (sha256(config['robot_config'])!=entry['provenance']['robot_config_sha256']
                    or sha256(config['mapping_config'])!=entry['provenance']['mapping_config_sha256']
                    or sha256(config_path)!=entry['provenance']['manifest_sha256']
                    or sha256(v['annotation_path'])!=v['annotation_sha256']
                    or sha256(demo/'processed_demo.csv')!=entry['provenance']['prepared_trajectory_sha256']
                    or sha256(demo/'metadata.json')!=entry['provenance']['prepared_metadata_sha256']):
                raise ValueError('Prepared input changed after preflight; rerun preparation.')
            human=load_transport(demo/'processed_demo.csv',demo/'metadata.json',mapping['max_source_gap_s'])
            env=make_env(config['seed'])
            if env.language_instruction.replace('_',' ').strip()!=config['task'].replace('_',' ').strip():
                raise ValueError('Runtime task does not match the configured bowl-to-plate task')
            with h5py.File(temporary,'w') as f:
                wrapped=RecordingEnv(env,f,episode_id,config['image_resolution'])
                runner=NominalRecording(wrapped,human,robot,mapping,directory); wrapped.runner=runner
                f.attrs.update(schema_version=1,complete=False,episode_success=False,episode_id=episode_id,
                    human_demo_id=demo_id,human_direction=v['direction'],human_annotation_sha256=sha256(v['annotation_path']),
                    human_trajectory_sha256=sha256(demo/'processed_demo.csv'),
                    source_video=entry['provenance']['source_video'],source_human_demo=entry['source_demo'],
                    human_provenance_json=json.dumps(entry['provenance']),task=config['task'],seed=config['seed'],
                    human_manual_annotations_json=json.dumps(v['manual_annotations']),
                    robot_settings_json=json.dumps(robot),mapping_settings_json=json.dumps(mapping),
                    language_instruction=env.language_instruction,
                    control_frequency_hz=runner.frequency,image_convention='RGB uint8 HWC, top-left origin (MuJoCo vertical flip)',
                    eef_pose_convention='world xyz metres + quaternion xyzw',
                    proprio_order='joint_pos[7], joint_vel[7], gripper_qpos[2], gripper_qvel[2]')
                report=runner.run()
                report.update(human_demo_id=demo_id,human_direction=v['direction'],source_metadata=human[-1],
                    annotation_file=v['annotation_path'],settings=robot,mapping_settings=mapping,seed=config['seed'],
                    provenance=entry['provenance'],human_validation=v,retargeting_input=entry,
                    controller_sha256=sha256('scripts/scripted_libero_pick_place.py'))
                save_outputs(runner,report,directory)
                logged=[r for r in runner.records if r['phase']!='SCENE_SETTLE']
                np.testing.assert_array_equal(f['actions'][:],np.array([r['action'] for r in logged]))
                np.testing.assert_array_equal(f['observations/libero_success'][1:],[r['task_success'] for r in logged])
                f.attrs['success_metadata_json']=json.dumps(report)
                f.attrs['complete']=bool(report['completed_sequence']); f.attrs['episode_success']=bool(report['task_completed'])
                status.update(status='success' if report['task_completed'] else 'failed',accepted=bool(report['task_completed']),
                    transitions=wrapped.count,first_physical_failure=dict(phase=report['failure_phase'],reason=report['failure_reason'],controller_step=len(runner.records)) if report['failure_phase'] else None)
            if status['accepted']:
                validate_episode(temporary); temporary.replace(output/'episodes'/(episode_id+'.h5'))
            # Failed recordings remain in attempts for diagnostics, never accepted into training.
        except Exception as exc:
            status.update(status='error',accepted=False,error=f'{type(exc).__name__}: {exc}')
            if runner is not None and runner.records:
                try:
                    report=runner.report(); save_outputs(runner,report,directory)
                    status['first_physical_failure']=dict(phase=report['failure_phase'],reason=report['failure_reason'],controller_step=len(runner.records)) if report['failure_phase'] else None
                except Exception as diagnostic_error:
                    status['diagnostic_error']=str(diagnostic_error)
        except KeyboardInterrupt:
            status.update(status='interrupted',accepted=False,error='Interrupted; attempt is consumed, no retry.')
            raise
        finally:
            write_json(directory/'status.json',status)
            try:
                if env is not None: env.close()
            finally:
                combine(base,output,combined,expected)
        print(json.dumps(status),flush=True)
    readiness['status']='finished'
    readiness['simulation_executed']=bool(list((output/'attempts').glob('*/status.json')))
    readiness['attempts']=[read_json(p) for p in sorted((output/'attempts').glob('*/status.json'))]
    write_json(output/'run_readiness.json',readiness)
    print(json.dumps(combine(base,output,combined,expected),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--combine-only',action='store_true')
    parser.add_argument('--config',type=Path,default=Path('config/data_003_rollouts.json'))
    parser.add_argument('--prepare-only',action='store_true',help='Refresh human inputs/plots/preflight only; no simulator imports or execution')
    a=parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    if a.combine_only:
        from prepare_transfer_rollouts import load_config
        config=load_config(a.config)
        result=combine(Path(config['baseline_dataset']),Path(config['rollout_output']),Path(config['combined_dataset']),
            {entry['episode_id']:entry['demo_id'] for entry in config['demos']})
        print(json.dumps(result,indent=2))
    else: run(a.config,a.prepare_only)
