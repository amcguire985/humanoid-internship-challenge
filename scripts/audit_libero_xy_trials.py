"""Audit the completed ten-trial experiment from saved states and step logs."""
import argparse
import json
from pathlib import Path

import numpy as np
from libero_xy_trials import DEFAULT_OUTPUT, OFFSETS, inputs, write_json


def audit(output):
    manifest = json.loads((output / 'preflight.json').read_text())
    baseline = np.load(output / 'nominal_state.npz')
    settings, _, _, hashes = inputs()
    assert hashes == manifest['input_hashes']
    assert settings == manifest['settings']
    assert manifest['all_poses_validated']
    directories = sorted(output.glob('trial_[0-9][0-9]'))
    assert len(directories) == 10, 'Exactly ten started trials required'
    # Locate the unique free joint using the saved nominal seven-coordinate pose.
    bowl = np.array(manifest['nominal']['object_free_joint_qpos'])
    candidates = [i for i in range(len(baseline['qpos'])-6)
                  if np.array_equal(baseline['qpos'][i:i+7], bowl)]
    assert len(candidates) == 1
    index = candidates[0]
    target = np.array(manifest['nominal']['target_anchor'])
    goal = (target + [0, 0, settings['robot']['placement_object_height_offset']]
            + [0, 0, settings['robot']['transport_clearance_height']])
    rows = []
    for number, directory in enumerate(directories, 1):
        assert directory.name == f'trial_{number:02d}'
        report = json.loads((directory / 'metrics.json').read_text())
        trial = manifest['trials'][number-1]
        assert report['trial'] == number
        assert trial['requested_offset_xy_m'] == list(OFFSETS[number-1])
        for key in ['remains_on_support', 'within_scene_bounds', 'collision_free', 'reachable']:
            assert trial['validation'][key]
        assert report['input_hashes'] == hashes
        assert report['settings'] == settings['robot']
        assert report['mapping_settings'] == settings['mapping']
        assert report['demo_annotations'] == settings['human']
        state = np.load(directory / 'initial_state.npz')
        expected = baseline['qpos'].copy()
        expected[index:index+2] += trial['applied_offset_xy_m']
        np.testing.assert_allclose(state['qpos'], expected, atol=1e-10, rtol=0)
        np.testing.assert_allclose(state['qvel'], baseline['qvel'], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(report['geometry']['target_anchor'], target)
        records = [json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines()]
        assert len(records) == report['total_episode_steps']
        assert [r['step'] for r in records] == list(range(1, len(records)+1))
        reached_transport = report['retargeting'] is not None
        if reached_transport:
            lift_end = [r for r in records if r['phase'] == 'LIFT'][-1]
            np.testing.assert_array_equal(report['retargeting']['sim_start'], lift_end['object'])
            np.testing.assert_array_equal(report['retargeting']['sim_goal'], goal)
            reference = np.loadtxt(directory / 'transport_reference.csv', delimiter=',', skiprows=1)
            np.testing.assert_allclose(reference[0,1:4], lift_end['object'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(reference[-1,1:4], goal, atol=1e-12, rtol=0)
        active = [r for r in records if r['phase'] != 'SCENE_SETTLE']
        plate_drift = max(np.linalg.norm(np.array(r['target'])-target) for r in active)
        rows.append(dict(trial=number, initialization_passed=True, settings_unchanged=True,
                         maximum_initial_qpos_residual=float(np.max(np.abs(state['qpos']-expected))),
                         transport_retargeted_from_actual_lifted_pose=reached_transport,
                         fixed_retargeting_goal=True, maximum_physical_plate_translation_m=float(plate_drift)))
    result = dict(passed=True, exactly_ten_trials=True, no_retries=True,
                  initial_state_tolerance=1e-10, trials=rows)
    write_json(output / 'final_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    print(json.dumps(audit(parser.parse_args().output), indent=2))
