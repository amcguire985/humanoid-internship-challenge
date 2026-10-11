"""Frozen five-pair B/C evaluation; no retries or controller changes."""
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile
from run_matched_b import FROZEN_REVISION, read, save, validate_c

METRICS=('libero_success','grasp_success','placement_success','controller_completed','max_tilt_deg','max_acceleration_m_s2','transport_duration_s','grasp_lost','failure_reason')


def patch_selection(frozen):
    """Change reset state selection/reporting only, with exact-source guards."""
    p=frozen/'scripts/evaluate_bowl_liquid.py'
    text=p.read_text()
    old='        index=int(self.env.init_state_id % len(self.env._init_states))'
    assert text.count(old)==1
    text=text.replace(old,"        self.env.init_state_id=int(os.environ.get('BOWL_BATCH_STATE', '0'))\n"+old)
    p.write_text(text)
    p=frozen/'scripts/evaluate_bowl_hybrid.py';text=p.read_text()
    old="if initialization['seed']!=0 or initialization['init_state_index']!=0: raise ValueError('Unexpected matched state')"
    assert text.count(old)==1
    text=text.replace(old,"if initialization['seed']!=0 or initialization['init_state_index']!=int(os.environ.get('BOWL_BATCH_STATE','0')): raise ValueError('Unexpected matched state')")
    text=text.replace("'Matched seed 0 and initial state 0; grasp snapshots are diagnostic, not restored'","'Matched seed 0 and initial state '+os.environ.get('BOWL_BATCH_STATE','0')+'; grasp snapshots are diagnostic, not restored'")
    p.write_text(text)


def reduction(b,c):
    return None if b is None or c is None or b==0 else 100*(b-c)/b


def summarize(records):
    result={}
    for condition in 'BC':
        rows=[r for r in records if r['condition']==condition]
        result[condition]={'scheduled_rollouts':len(rows),'observed_successes':sum(r.get('libero_success') is True for r in rows),
            'task_success_rate_all_scheduled':sum(r.get('libero_success') is True for r in rows)/len(rows),
            'unavailable_success_outcomes':sum(r.get('libero_success') is None for r in rows)}
        for key in ('max_acceleration_m_s2','max_tilt_deg','transport_duration_s'):
            values=[r[key] for r in rows if r.get(key) is not None and math.isfinite(r[key])]
            result[condition][key]={'available_n':len(values),'mean':statistics.mean(values) if values else None,'median':statistics.median(values) if values else None}
    return result


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def reports(root,records):
    write_csv(root/'rollouts.csv',records)
    pairs=[]
    for state in sorted({r['initial_state_index'] for r in records}):
        b,c=[next(r for r in records if r['initial_state_index']==state and r['condition']==k) for k in 'BC']
        row={'initial_state_index':state}
        for key in METRICS:
            row['B_'+key]=b.get(key);row['C_'+key]=c.get(key)
        for key in ('max_acceleration_m_s2','max_tilt_deg','transport_duration_s'):
            row[key+'_B_minus_C']=None if b.get(key) is None or c.get(key) is None else b[key]-c[key]
            row[key+'_reduction_percent']=reduction(b.get(key),c.get(key))
        pairs.append(row)
    write_csv(root/'paired_comparison.csv',pairs)
    stats=summarize(records)
    stats['C_lower_acceleration_pairs']=sum(p['max_acceleration_m_s2_B_minus_C'] is not None and p['max_acceleration_m_s2_B_minus_C']>0 for p in pairs)
    stats['acceleration_complete_pairs']=sum(p['max_acceleration_m_s2_B_minus_C'] is not None for p in pairs)
    stats['aggregate_B_minus_C']={}
    for key in ('max_acceleration_m_s2','max_tilt_deg','transport_duration_s'):
        stats['aggregate_B_minus_C'][key]={}
        for statistic in ('mean','median'):
            b=stats['B'][key][statistic];c=stats['C'][key][statistic]
            stats['aggregate_B_minus_C'][key][statistic]={'absolute':None if b is None or c is None else b-c,'percent_reduction':reduction(b,c)}
    stats['interpretation']='Five prespecified pairs are exploratory. Unavailable transport metrics excluded from numeric aggregates, never zero-filled; all scheduled runs retained.'
    save(root/'aggregate.json',stats)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,label,name in [('max_acceleration_m_s2','Maximum sampled bowl acceleration (m/s²)','paired_acceleration'),('max_tilt_deg','Maximum intact-grasp transport tilt (degrees)','paired_tilt')]:
        fig,ax=plt.subplots(figsize=(8,4))
        for i,p in enumerate(pairs):
            vals=[p['B_'+key],p['C_'+key]]
            for j,v in enumerate(vals):
                if v is None: ax.text(i+(j-.5)*.15,0,'N/A',ha='center')
                else: ax.scatter(i+(j-.5)*.15,v,color=['tab:blue','tab:orange'][j],label='BC'[j] if i==0 else None)
            if all(v is not None for v in vals): ax.plot([i-.075,i+.075],vals,color='gray')
        ax.set(xticks=range(len(pairs)),xticklabels=[p['initial_state_index'] for p in pairs],xlabel='Initial-state index',ylabel=label);
        if ax.collections: ax.legend()
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(root/(name+'.'+ext),dpi=200)
        plt.close(fig)
    # Preselected first pair, regardless of success or direction of effect.
    first= min(r['initial_state_index'] for r in records)
    fig,ax=plt.subplots(figsize=(8,4))
    for condition in 'BC':
        r=next(r for r in records if r['initial_state_index']==first and r['condition']==condition)
        path=Path(r['output'])/'ground_truth/episode_000/metrics.csv'
        if not path.exists():continue
        with path.open() as f: rows=list(csv.DictReader(f))
        valid=[v for v in rows if v['acceleration_m_s2'] not in ('','None','nan')]
        start=r.get('transport_start_time_s')
        if valid and start is not None:ax.plot([float(v['sim_time_s'])-start for v in valid],[float(v['acceleration_m_s2']) for v in valid],label=condition)
    ax.set(xlabel='Time since each condition entered TRANSPORT (s)',ylabel='Sampled bowl acceleration (m/s²)',title=f'Prespecified first pair: state {first} (missing traces omitted)')
    if ax.lines:ax.legend()
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(root/('acceleration_overlay.'+ext),dpi=200)
    plt.close(fig)
    columns=['State','Mode','Task','Grasp','Place','Complete','Accel m/s²','Tilt °','Duration s','Failure']
    cells=[[r['initial_state_index'],r['condition'],r.get('libero_success'),r.get('grasp_success'),r.get('placement_success'),r.get('controller_completed'),r.get('max_acceleration_m_s2'),r.get('max_tilt_deg'),r.get('transport_duration_s'),r.get('failure_reason')] for r in records]
    cells=[['N/A' if v is None else f'{v:.3f}' if isinstance(v,float) else str(v) for v in row] for row in cells]
    fig,ax=plt.subplots(figsize=(16,4));ax.axis('off');table=ax.table(cellText=cells,colLabels=columns,loc='center');table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.5);fig.tight_layout()
    fig.savefig(root/'rollout_table.png',dpi=200,bbox_inches='tight');plt.close(fig)
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+['| '+' | '.join(row)+' |' for row in cells]
    (root/'report.md').write_text('\n'.join(lines)+'\n\n'+json.dumps(stats,indent=2)+'\n',encoding='utf-8')


def verify_pair(b,c):
    for filename,keys in [('experiment.json',('checkpoint_sha256','settings','placement_settings','guidance_algorithm','policy_instruction_transport')),('ground_truth/episode_000/initialization.json',('seed','init_state_index','init_state_sha256','bddl_sha256','local_opening_axis'))]:
        bm,cm=read(b/filename),read(c/filename)
        for key in keys:
            if bm[key]!=cm[key]:raise ValueError('Unintended pair difference: '+key)
    if read(b/'ground_truth/episode_000/controller.json')!=read(c/'ground_truth/episode_000/controller.json'):raise ValueError('OSC mismatch')
    rows=[json.loads(x) for x in (b/'ground_truth/episode_000/trajectory.jsonl').read_text().splitlines()]
    if any(r.get('constraints_active') or r.get('guidance_active') for r in rows):raise ValueError('B intervention active')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c-output',type=Path,required=True)
    parser.add_argument('--human-reference',type=Path,required=True)
    parser.add_argument('--states',type=int,nargs=5,default=[1,2,3,4,5])
    parser.add_argument('--excluded-states',type=int,nargs='+',default=[0])
    args=parser.parse_args()
    if len(set(args.states))!=5 or set(args.states)&set(args.excluded_states):raise ValueError('Need five distinct states outside known tuning states')
    _,cm,ci=validate_c(args.c_output)
    from libero.libero import benchmark,get_libero_path
    import numpy as np
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no rollouts launched')
    suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0);states=suite.get_task_init_states(2)
    if any(i<0 or i>=len(states) for i in args.states):raise ValueError('Requested states unavailable; no substitution based on outcomes')
    checkpoint=Path('/content/smolvla-libero-official-checkpoint');baseline=Path(cm['baseline_root'])
    for name,expected in cm['checkpoint_sha256'].items():
        with (checkpoint/name).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=expected:raise ValueError('Checkpoint changed: '+name)
    packages={line.split('==',1)[0].lower().replace('_','-'):line.split('==',1)[1] for line in (baseline/'packages.txt').read_text().splitlines() if '==' in line}
    for name in ('hf-libero','robosuite','mujoco','torch','torchvision','transformers','numpy','gymnasium'):
        if metadata.version(name)!=packages[name]:raise ValueError('Environment changed: '+name)
    task=suite.get_task(2);bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    if hashlib.sha256(bddl.read_bytes()).hexdigest()!=ci['bddl_sha256']:raise ValueError('Task geometry changed')
    if not __import__('shutil').which('ffmpeg'):raise RuntimeError('ffmpeg missing; restore the working runtime before running')
    root=Path(tempfile.mkdtemp(prefix='bowl_final_matched_BC_',dir='/content/drive/MyDrive/humanoid_results'))
    Path('/content/bowl_final_matched_last_output.txt').write_text(str(root))
    print('Result folder:',root,flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        frozen=Path(tmp)/'source';frozen.mkdir();archive=Path(tmp)/'source.tar'
        with archive.open('wb') as f:subprocess.run(['git','archive',FROZEN_REVISION],cwd=Path(__file__).resolve().parents[1],stdout=f,check=True)
        with tarfile.open(archive) as t:t.extractall(frozen,filter='data')
        for filename,key in [('bowl_hybrid.json','settings'),('bowl_placement.json','placement_settings')]:
            if read(frozen/'config/smolvla'/filename)!=cm[key]:raise ValueError('Frozen C settings mismatch')
        sys.path.insert(0,str(frozen/'scripts'))
        from bowl_phone_constraints import extract
        from bowl_human_reference import Reference
        limits=extract(Reference.load(args.human_reference))
        for key,value in limits.items():
            if cm['phone_constraints'].get(key)!=value:raise ValueError('Phone constraint changed: '+key)
        for p in [args.human_reference,Path(str(args.human_reference)+'.json')]:
            key='source_sha256' if p==args.human_reference else 'source_metadata_sha256'
            if hashlib.sha256(p.read_bytes()).hexdigest()!=cm['phone_constraints'][key]:raise ValueError('Phone source changed')
        original={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (frozen/'scripts').glob('*.py')}
        patch_selection(frozen)
        patched={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (frozen/'scripts').glob('*.py')}
        plan={'revision':FROZEN_REVISION,'original_source_hashes':original,'batch_source_hashes':patched,'selection_only_adapter':'Explicit init_state_id before original reset; state-index reporting assertion/text. Controller code unchanged.', 'checkpoint_sha256':cm['checkpoint_sha256'],'settings':cm['settings'],'placement_settings':cm['placement_settings'],'phone_constraints':cm['phone_constraints'],'seed':0,'states':args.states,'excluded_states':args.excluded_states,'tuning_history_limit':'Only state 0 documented as tuned; add any other known tuning states to --excluded-states before execution.','historical_source_limit':'Historical C lacks source hashes; known working revision and saved settings matched. New B/C both use identical archived source.', 'intended_differences':{'condition':['B','C'],'transport_mode':['trajectory (B has no trajectory intervention)','phone_constraints'],'phone_acceleration_limiter':[False,True],'phone_excess_tilt_correction':[False,True]},'state_hashes':{str(i):hashlib.sha256(np.asarray(states[i]).tobytes()).hexdigest() for i in args.states},'order':[['B','C'] if i%2==0 else ['C','B'] for i in range(5)]}
        save(root/'configuration_diff_before_rollout.json',plan)
        print(json.dumps(plan['intended_differences'],indent=2),flush=True)
        records=[]
        for pair,state in enumerate(args.states):
            for condition in plan['order'][pair]:
                output=root/f'state_{state:03d}'/condition;output.parent.mkdir(exist_ok=True)
                cmd=[sys.executable,str(frozen/'scripts/evaluate_bowl_hybrid.py'),'--hybrid-condition',condition,'--transport-mode','phone_constraints' if condition=='C' else 'trajectory','--guidance-config',str(frozen/'config/smolvla/bowl_hybrid.json'),'--placement-config',str(frozen/'config/smolvla/bowl_placement.json'),'--baseline-root',str(baseline),'--policy.path='+str(checkpoint),'--policy.device=cuda','--policy.load_vlm_weights=false','--env.type=libero','--env.task=libero_spatial','--env.task_ids=[2]','--env.control_mode=relative','--env.init_states=true','--env.hard_reset=true','--env.observation_height=256','--env.observation_width=256','--env.max_parallel_tasks=1','--eval.batch_size=1','--eval.n_episodes=1','--seed=0','--output_dir='+str(output)]
                if condition=='C':cmd+=['--human-reference',str(args.human_reference.resolve())]
                save(output.parent/(condition+'_command.json'),cmd)
                env=os.environ.copy();env['BOWL_BATCH_STATE']=str(state)
                print(f'Pair {pair+1}/5 state {state} condition {condition}',flush=True)
                with (output.parent/(condition+'.log')).open('w') as log:
                    code=subprocess.run(cmd,cwd=frozen,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                record={'initial_state_index':state,'seed':0,'condition':condition,'output':str(output),'exit_code':code,'transport_start_time_s':None,**dict.fromkeys(METRICS)}
                if (output/'summary.json').exists():
                    summary=read(output/'summary.json');record.update({k:summary.get(k) for k in METRICS});record['grasp_success']=summary.get('transport_detected')
                    init=read(output/'ground_truth/episode_000/initialization.json')
                    if init['init_state_index']!=state or init['seed']!=0 or init['init_state_sha256']!=plan['state_hashes'][str(state)]:raise ValueError('Actual starting state mismatch; batch stopped')
                    rows=[json.loads(x) for x in (output/'ground_truth/episode_000/trajectory.jsonl').read_text().splitlines()]
                    start=summary.get('transport_start_timestep')
                    if start is not None:record['transport_start_time_s']=rows[start]['sim_time_s']
                if code or not (output/'summary.json').exists():record['failure_reason']=record.get('failure_reason') or f'evaluator_error_exit_{code}; inspect log'
                records.append(record);save(root/'progress.json',records)
            b,c=[root/f'state_{state:03d}'/k for k in 'BC']
            if all((p/'experiment.json').exists() for p in (b,c)):
                verify_pair(b,c);save(b.parent/'matching_verified.json',{'matching_verified':True})
            else:save(b.parent/'matching_verified.json',{'matching_verified':False,'reason':'Missing rollout artifacts; retained as unavailable, not silently matched'})
        reports(root,records)
    print('Batch complete; results:',root,flush=True)

if __name__=='__main__':main()
