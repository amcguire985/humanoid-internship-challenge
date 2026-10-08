"""One frozen B rollout paired with an existing successful phone-constraints C."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

FROZEN_REVISION='08d37c3d36720498d552711463e0f53992c53386'
ALGORITHM='shared_geometry_placement_v7_persistent_withdraw'


def read(path): return json.loads(Path(path).read_text())
def save(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_c(c):
    summary=read(c/'summary.json');manifest=read(c/'experiment.json')
    if not all(summary.get(k) is True for k in ('libero_success','placement_success','controller_completed')):
        raise ValueError('Select the successful completed C run, not an earlier failed run')
    if manifest.get('condition')!='C' or manifest.get('transport_mode')!='phone_constraints' or manifest.get('guidance_algorithm')!=ALGORITHM:
        raise ValueError('C intervention/controller version mismatch')
    if not manifest.get('phone_constraints'): raise ValueError('C lacks extracted constraints')
    init=read(c/'ground_truth/episode_000/initialization.json')
    if init['seed']!=0 or init['init_state_index']!=0: raise ValueError('C is not the documented seed/state 0')
    return summary,manifest,init


def compare(c,b,output):
    cs,cm,ci=validate_c(c);bs=read(b/'summary.json');bm=read(b/'experiment.json')
    bi=read(b/'ground_truth/episode_000/initialization.json')
    for key in ('seed','init_state_index','init_state_sha256','bddl_sha256','local_opening_axis'):
        if ci[key]!=bi[key]: raise ValueError('Initial-state mismatch: '+key)
    for key in ('checkpoint_sha256','settings','placement_settings','guidance_algorithm','policy_instruction_transport'):
        if cm[key]!=bm[key]: raise ValueError('Unintended B/C difference: '+key)
    if bm['condition']!='B' or bm['transport_mode']!='trajectory': raise ValueError('B mode mismatch')
    cc=read(c/'ground_truth/episode_000/controller.json');bc=read(b/'ground_truth/episode_000/controller.json')
    if cc!=bc: raise ValueError('OSC scaling/action bound mismatch')
    # Frozen manifest settings alone are insufficient: verify B added no intervention.
    rows=[json.loads(x) for x in (b/'ground_truth/episode_000/trajectory.jsonl').read_text().splitlines()]
    if any(r.get('constraints_active') or r.get('guidance_active') for r in rows):
        raise ValueError('Unexpected B transport intervention')
    fields=('libero_success','placement_success','controller_completed','max_tilt_deg',
        'max_acceleration_m_s2','transport_duration_s','grasp_lost','failure_reason')
    lines=['| Metric | B: phone constraints off | C: phone constraints on |','|---|---|---|']
    for key in fields:
        def show(value): return 'unavailable' if value is None else str(value)
        lines.append(f'| {key} | {"None" if key=="failure_reason" and bs.get(key) is None else show(bs.get(key))} | {"None" if key=="failure_reason" and cs.get(key) is None else show(cs.get(key))} |')
    lines+=['','Single matched initial-state pair; transport intervals can differ. No general reliability or causal-effect claim.']
    text='\n'.join(lines)+'\n';(output/'comparison.md').write_text(text)
    save(output/'comparison.json',dict(B=bs,C=cs,matching_verified=True,C_path=str(c),B_path=str(b),
        difference='Phone acceleration limiter and excess-tilt correction disabled in B; shared safeguards retained',
        delta_C_minus_B={k:cs[k]-bs[k] if cs.get(k) is not None and bs.get(k) is not None else None for k in ('max_tilt_deg','max_acceleration_m_s2','transport_duration_s')}))
    print(text,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--c-output',type=Path,required=True)
    args=parser.parse_args();c=args.c_output.resolve();_,cm,ci=validate_c(c)
    repo=Path(__file__).resolve().parents[1]
    base=Path('/content/drive/MyDrive/humanoid_results')
    output=Path(tempfile.mkdtemp(prefix='bowl_matched_B_',dir=base));b=output/'B'
    print('Matched B output folder:',output,flush=True)
    with tempfile.TemporaryDirectory(prefix='bowl_frozen_B_') as tmp:
        frozen=Path(tmp)/'source';frozen.mkdir()
        archive=Path(tmp)/'source.tar'
        with archive.open('wb') as f:
            subprocess.run(['git','archive',FROZEN_REVISION],cwd=repo,stdout=f,check=True)
        with tarfile.open(archive) as t: t.extractall(frozen,filter='data')
        guidance=output/'frozen_guidance.json';placement=output/'frozen_placement.json'
        save(guidance,cm['settings']);save(placement,cm['placement_settings'])
        checkpoint=Path('/content/smolvla-libero-official-checkpoint')
        for name,expected in cm['checkpoint_sha256'].items():
            with (checkpoint/name).open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=expected: raise ValueError('Checkpoint differs from C: '+name)
        baseline=Path(cm['baseline_root'])
        expected={line.split('==',1)[0].lower().replace('_','-'):line.split('==',1)[1] for line in (baseline/'packages.txt').read_text().splitlines() if '==' in line}
        for name in ('hf-libero','robosuite','mujoco','torch','torchvision','transformers','numpy','gymnasium'):
            if metadata.version(name)!=expected[name]: raise ValueError('Baseline environment mismatch: '+name)
        if read(frozen/'config/smolvla/bowl_hybrid.json')!=cm['settings'] or read(frozen/'config/smolvla/bowl_placement.json')!=cm['placement_settings']:
            raise ValueError('Saved C settings differ from frozen working configuration')
        import numpy as np
        from libero.libero import benchmark,get_libero_path
        suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)
        state=np.asarray(suite.get_task_init_states(2)[0])
        if hashlib.sha256(state.tobytes()).hexdigest()!=ci['init_state_sha256']:
            raise ValueError('Runtime task initial state differs from saved C')
        task=suite.get_task(2)
        bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        if hashlib.sha256(bddl.read_bytes()).hexdigest()!=ci['bddl_sha256']:
            raise ValueError('Runtime BDDL differs from saved C')
        sources={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (frozen/'scripts').glob('*.py')}
        plan=dict(frozen_revision=FROZEN_REVISION,C_path=str(c),checkpoint_sha256=cm['checkpoint_sha256'],
            shared_settings=cm['settings'],shared_placement_settings=cm['placement_settings'],
            shared_algorithm=ALGORITHM,expected_initial_state=ci,frozen_source_sha256=sources,
            intended_differences=dict(condition=['C','B'],transport_intervention=['phone_constraints','none'],
                acceleration_limiter=[True,False],excess_tilt_correction=[True,False]),
            unchanged='Checkpoint, grasp/original prompt, liquid transport prompt, OSC action bounds, shared guards, placement/release/withdrawal, gains, tolerances, timeouts, seed/state',
            historical_source_limit='C did not save source hashes; frozen source uses matching algorithm/configuration and known working revision. Actual state/controller/checkpoint matching verified after B.')
        save(output/'configuration_diff_before_rollout.json',plan)
        print(json.dumps(plan['intended_differences'],indent=2),flush=True)
        command=[sys.executable,str(frozen/'scripts/evaluate_bowl_hybrid.py'),
            '--hybrid-condition','B','--transport-mode','trajectory',
            '--guidance-config',str(guidance),'--placement-config',str(placement),'--baseline-root',str(baseline),
            '--policy.path='+str(checkpoint),'--policy.device=cuda','--policy.load_vlm_weights=false',
            '--env.type=libero','--env.task=libero_spatial','--env.task_ids=[2]',
            '--env.control_mode=relative','--env.init_states=true','--env.hard_reset=true',
            '--env.observation_height=256','--env.observation_width=256','--env.max_parallel_tasks=1',
            '--eval.batch_size=1','--eval.n_episodes=1','--seed=0','--output_dir='+str(b)]
        save(output/'command.json',command)
        # Exactly one evaluator invocation, no retries, no C rerun.
        with (output/'B.log').open('w') as log:
            process=subprocess.Popen(command,cwd=frozen,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout: print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        if code: raise RuntimeError(f'B evaluator exited {code}; inspect {output}/B.log; do not automatically retry')
    compare(c,b,output)
    print('Results and comparison:',output,flush=True)

if __name__=='__main__': main()
