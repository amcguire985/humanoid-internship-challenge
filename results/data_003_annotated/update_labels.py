from pathlib import Path
p=Path('scripts/annotate_transfer_demos.py'); s=p.read_text()
s=s.replace('validate_config, resolve_phase_bounds','validate_config, resolve_phase_bounds, segment_task_phases')
s=s.replace("    # Preserve all samples/validity. These metadata drive the existing loader; no filtering or interpolation.\n", "    # Update phase labels from resolved annotations while preserving samples/validity.\n    phase_config={**c,'phase_overrides':bounds}\n    labels,_,_,_,progress,_=segment_task_phases(t,p,np.array(m['goal_position']),phase_config)\n    for i,row in enumerate(rows):\n        row['phase']=labels[i]; row['transport_progress']=float(progress[i]) if np.isfinite(progress[i]) else ''\n")
s=s.replace("    write_json(output/'metadata.json',updated)", "    with (output/'transport.csv').open('w',newline='',encoding='utf-8') as handle:\n        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()\n        writer.writerows(r for r,time in zip(rows,t) if bounds['pickup_time']<=time<=bounds['release_time'])\n    write_json(output/'metadata.json',updated)")
p.write_text(s)
