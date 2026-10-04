"""Manipulation control-flow/safety contracts without running a physical grasp."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from scripted_libero_pick_place import PickPlace, ManipulationAbort, make_move, validate_config, OBJECT, TARGET
from process_task_demo import validate_config as validate_demo_config, resolve_phase_bounds
from retarget_libero_object import load_transport

ROOT = Path(__file__).resolve().parents[1]


class PickPlaceTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((ROOT/'config/libero_pick_place.json').read_text())

    def runner(self):
        r = PickPlace.__new__(PickPlace)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        r.output = Path(temp.name)
        r.c = self.c
        r.phase, r.phase_steps, r.frequency = 'TRANSPORT', 0, 20.
        r.records = []; r.acquired = True; r.grasp_lost = False; r.missing_grasp = 0; r.bad_tracking = 0
        r.offset = np.array([0, 0, .1]); r.rotation = np.eye(3)
        obs = {'robot0_eef_pos': np.array([0, 0, 1.]), 'robot0_eef_quat': np.array([0, 0, 0, 1.]),
               OBJECT+'_pos': np.array([0, 0, .9]), TARGET+'_pos': np.array([.1, 0, .9]),
               'robot0_gripper_qpos': np.array([.01, -.01]), OBJECT+'_quat': np.array([0, 0, 0, 1.])}
        r.obs = obs
        r.env = SimpleNamespace(step=Mock(return_value=(obs, 0, False, {})), check_success=lambda:False,
            env=SimpleNamespace(_check_grasp=lambda *args:False, robots=[SimpleNamespace(gripper=None)]))
        r.object_model = None; r.contacts = lambda:([], [])
        r.controller = dict(input_min=-np.ones(6), input_max=np.ones(6), output_min=-np.array([.05]*3+[.5]*3),
                            output_max=np.array([.05]*3+[.5]*3), action_min=-np.ones(7), action_max=np.ones(7))
        return r

    def test_motion_reference_has_bounded_speed_and_exact_endpoint(self):
        start, end = np.array([0, 0, 1.]), np.array([.2, -.15, .93])
        p = make_move(start, end, 20., self.c)
        np.testing.assert_allclose(p[-1], end)
        self.assertLessEqual(np.linalg.norm(np.diff(np.vstack([start, p]), axis=0), axis=1).max(), self.c['approach_step_size']+1e-12)
        validate_config(self.c)

    def test_missing_grasp_aborts_after_grace_and_keeps_final_log(self):
        r = self.runner()
        for _ in range(self.c['grasp_loss_grace_steps']-1): r.step(np.array([0, 0, 1.]), 1)
        with self.assertRaisesRegex(ManipulationAbort, 'Grasp lost'): r.step(np.array([0, 0, 1.]), 1)
        self.assertTrue(r.grasp_lost)
        self.assertEqual(len(r.records), self.c['grasp_loss_grace_steps'])
        self.assertTrue(all(row['gripper_command']==1 for row in r.records))

    def test_invalid_object_aborts_before_another_command(self):
        r = self.runner(); r.obs[OBJECT+'_pos'] = np.full(3, np.nan)
        with self.assertRaisesRegex(ManipulationAbort, 'Invalid or unavailable'): r.step(np.array([0, 0, 1.]), 1)
        r.env.step.assert_not_called()

    def test_unexpected_contact_is_logged_then_aborted(self):
        r = self.runner(); r.contacts = lambda:([['robot0_link', 'table']], [['robot0_link', 'table']])
        with self.assertRaisesRegex(ManipulationAbort, 'Unexpected robot/scene contact'): r.step(np.array([0, 0, 1.]), 1)
        self.assertEqual(len(r.records), 1)
        self.assertTrue(r.records[-1]['contacts'])

    def test_phase_budget_prevents_additional_steps(self):
        r = self.runner(); r.phase_steps = self.c['phase_max_steps']['TRANSPORT']
        with self.assertRaisesRegex(ManipulationAbort, 'Maximum phase steps'): r.step(np.array([0, 0, 1.]), 1)
        r.env.step.assert_not_called()

    def test_libero_success_signal_does_not_skip_release(self):
        r = self.runner()
        r.env.step.return_value = (r.obs, 1, True, {})
        r.env.check_success = lambda:True
        r.step(np.array([0, 0, 1.]), 1)
        self.assertTrue(r.records[-1]['task_success'])
        self.assertTrue(r.records[-1]['environment_done_signal'])
        # A real simulator termination must still stop execution.
        r.env.env.done = True
        with self.assertRaisesRegex(ManipulationAbort, 'Environment terminated'):
            r.step(np.array([0, 0, 1.]), 1)

    def test_named_manual_events_win_without_rewriting_automatic_estimates(self):
        t = np.arange(0, 12.01, .01)
        automatic = dict(pickup_time=3.8, transport_start=4.8, transport_end=6.4, release_time=8.1)
        original = automatic.copy()
        c = validate_demo_config(dict(phase_overrides={'transport_end':8.}, grasp_time_seconds=3.77,
            release_time_seconds=10.5, transport_start_time_seconds=4.337, transport_end_time_seconds=10.307))
        resolved = resolve_phase_bounds(t, automatic, c)
        self.assertAlmostEqual(resolved['transport_start'],4.34)
        self.assertAlmostEqual(resolved['transport_end'],10.31)
        self.assertEqual(automatic, original)
        c['release_time_seconds'] = 4.
        with self.assertRaises(ValueError): resolve_phase_bounds(t, automatic, c)

    def test_runtime_loader_honors_edited_config_even_with_cached_metadata(self):
        demo = ROOT/'results/test_007_block_only_raw/task_demo'
        c = json.loads((ROOT/'config/test_007_demo.json').read_text())
        c['transport_end_time_seconds'] = 9.7
        t, p, _, _, metadata = load_transport(demo/'processed_demo.csv', demo/'metadata.json', timing_config=c)
        self.assertAlmostEqual(metadata['transport_end'],9.706666666666667)
        self.assertAlmostEqual(t[-1],metadata['transport_end']-metadata['transport_start'])
        self.assertLess(metadata['automatic_phase_estimates']['transport_end'],6.5)


if __name__ == '__main__':
    unittest.main()
