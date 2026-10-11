"""Geometry-only preview of the unchanged retargeter against validated nominal scene."""
from pathlib import Path
import numpy as np
from annotate_transfer_demos import read_json,write_json
from retarget_libero_object import load_transport,retarget_transport
from replay_libero_transport import build_desired_eef_trajectory


def plan(prepared, robot_config='config/libero_pick_place.json', mapping_config='config/libero_retarget.json', nominal_preflight='results/libero_xy_trials_10/preflight.json', transport_step_budget=None):
    robot=read_json(Path(robot_config)); mapping=read_json(Path(mapping_config))
    nominal=read_json(Path(nominal_preflight))['nominal']
    start=np.array(nominal['object_start'])+np.array([0,0,robot['postgrasp_lift_height']])
    goal=np.array(nominal['target_anchor'])+np.array([0,0,robot['placement_object_height_offset']+robot['transport_clearance_height']])
    human=load_transport(prepared/'processed_demo.csv',prepared/'metadata.json',mapping['max_source_gap_s'])
    times,positions,endpoint,_,metadata=human
    path,progress,report=retarget_transport(times,positions,endpoint,start,goal,mapping)
    grid,objects,factor=build_desired_eef_trajectory(times,path-path[0],path[0],20,mapping['time_scale'],robot['max_eef_speed_m_s'])
    budget=robot['phase_max_steps']['TRANSPORT'] if transport_step_budget is None else transport_step_budget
    if not isinstance(budget,int) or isinstance(budget,bool) or budget<=0:
        raise ValueError('Transport step budget must be a positive integer')
    # Configured offset is a preview only; live controller measures its actual offset after grasp.
    eef=objects+np.array(robot['grasp_offset']); reasons=[]
    if report['endpoint_correction_norm_m']>robot['maximum_endpoint_correction_m']:
        reasons.append('Transport window ends too far from its goal: endpoint correction exceeds unchanged controller limit.')
    if np.any(eef<robot['workspace_min']) or np.any(eef>robot['workspace_max']): reasons.append('Nominal mapped EEF path leaves configured workspace.')
    if len(grid)>budget: reasons.append('Speed-limited transport exceeds configured per-demo phase-step budget.')
    np.savetxt(prepared/'nominal_mapped_transport.csv',np.column_stack((grid,objects,eef)),delimiter=',',
               header='time,object_x,object_y,object_z,eef_x,eef_y,eef_z',comments='')
    result=dict(demo_id=metadata['id'],direction=metadata['direction'],nominal_preview_only=True,
        retargeting=report,time_scale_used=factor,transport_steps=len(grid),robot_duration_seconds=float(grid[-1]),
        mapping_suitable=not reasons,reasons=reasons,
        transport_step_budget=budget,baseline_transport_step_budget=robot['phase_max_steps']['TRANSPORT'],
        robot_max_eef_speed_m_s=robot['max_eef_speed_m_s'],
        maximum_endpoint_correction_m=robot['maximum_endpoint_correction_m'],
        limitation='Nominal scene and configured offset only; live grasp state and measured offset may differ. Not a physical rollout or IK certificate.')
    write_json(prepared/'mapping_preview.json',result); return result

if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepared',type=Path,default=Path('results/data_003_annotated'))
    a=parser.parse_args(); reports=[plan(d) for d in sorted(a.prepared.glob('demo_*'))]
    write_json(a.prepared/'mapping_previews.json',reports); print(json.dumps(reports,indent=2))
